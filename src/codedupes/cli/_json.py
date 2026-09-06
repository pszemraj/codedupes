"""JSON serializers for CLI output."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from typing import Any

from codedupes.models import (
    AnalysisResult,
    CodeUnit,
    DuplicatePair,
    ExtractionDiagnostic,
    HybridDuplicate,
)
from codedupes.semantic import EmbeddingRunStats


def _embedding_stats_to_dict(stats: EmbeddingRunStats | None) -> dict[str, Any] | None:
    """Convert optional embedding telemetry to a JSON-safe mapping.

    :param stats: Optional embedding run telemetry.
    :return: Serialized telemetry, or ``None`` when semantic work did not run.
    """
    return asdict(stats) if stats is not None else None


def _language_counts(units: list[CodeUnit]) -> dict[str, int]:
    """Count extracted units by canonical language.

    :param units: Extracted code units.
    :return: Counts keyed by canonical language.
    """
    return dict(sorted(Counter(unit.language for unit in units).items()))


def _diagnostic_to_dict(diagnostic: ExtractionDiagnostic) -> dict[str, Any]:
    """Convert an extraction diagnostic to a JSON-safe mapping.

    :param diagnostic: Diagnostic to serialize.
    :return: Serialized diagnostic fields.
    """
    return {
        "file": str(diagnostic.file_path),
        "language": diagnostic.language,
        "severity": diagnostic.severity,
        "code": diagnostic.code,
        "message": diagnostic.message,
        "line": diagnostic.lineno,
        "end_line": diagnostic.end_lineno,
    }


def _unit_to_dict(unit: CodeUnit) -> dict[str, Any]:
    """Convert a code unit to a JSON-serializable summary.

    :param unit: Code unit to serialize.
    :return: Serialized unit fields.
    """
    return {
        "name": unit.name,
        "qualified_name": unit.qualified_name,
        "type": unit.unit_type.name.lower(),
        "language": unit.language,
        "dialect": unit.dialect,
        "native_kind": unit.native_kind,
        "file": str(unit.file_path),
        "line": unit.lineno,
        "end_line": unit.end_lineno,
        "start_byte": unit.start_byte,
        "end_byte": unit.end_byte,
        "start_column": unit.start_column,
        "end_column": unit.end_column,
        "statement_count": unit.statement_count,
        "is_public": unit.is_public,
        "is_exported": unit.is_exported,
    }


def check_result_to_json(
    result: AnalysisResult,
    *,
    show_all: bool,
    fail_on: str,
    exit_code: int,
) -> dict[str, Any]:
    """Serialize one check result using the normalized schema-v2 graph shape.

    :param result: Analysis result to serialize.
    :param show_all: Include raw edge lists in combined mode.
    :param fail_on: Finding policy selected for this run.
    :param exit_code: Exit code computed from the selected policy.
    :return: Schema-v2 check payload.
    """
    units: dict[str, dict[str, Any]] = {}

    def ref(unit: CodeUnit) -> str:
        """Register a unit once and return its stable node identifier.

        :param unit: Unit referenced by a finding.
        :return: Stable unit identifier.
        """
        units.setdefault(unit.uid, _unit_to_dict(unit))
        return unit.uid

    def hybrid_edge(duplicate: HybridDuplicate) -> dict[str, Any]:
        """Serialize one hybrid duplicate as an edge between unit identifiers.

        :param duplicate: Hybrid duplicate to serialize.
        :return: Serialized hybrid edge.
        """
        return {
            "unit_a": ref(duplicate.unit_a),
            "unit_b": ref(duplicate.unit_b),
            "tier": duplicate.tier,
            "confidence": duplicate.confidence,
            "has_exact": duplicate.has_exact,
            "semantic_similarity": duplicate.semantic_similarity,
            "jaccard_similarity": duplicate.jaccard_similarity,
            "weak_identifier_jaccard": duplicate.weak_identifier_jaccard,
            "statement_count_ratio": duplicate.statement_count_ratio,
        }

    def raw_edge(duplicate: DuplicatePair) -> dict[str, Any]:
        """Serialize one raw duplicate as an edge between unit identifiers.

        :param duplicate: Raw duplicate to serialize.
        :return: Serialized raw edge.
        """
        return {
            "unit_a": ref(duplicate.unit_a),
            "unit_b": ref(duplicate.unit_b),
            "similarity": duplicate.similarity,
            "method": duplicate.method,
        }

    combined_mode = result.analysis_mode == "combined"
    if combined_mode:
        duplicates = [hybrid_edge(duplicate) for duplicate in result.hybrid_duplicates]
    else:
        duplicates = [
            raw_edge(duplicate)
            for duplicate in result.traditional_duplicates + result.semantic_duplicates
        ]

    output: dict[str, Any] = {
        "schema_version": 2,
        "analysis_mode": result.analysis_mode,
        "summary": {
            "total_units": len(result.units),
            "units_by_language": _language_counts(result.units),
            "hybrid_duplicates": len(result.hybrid_duplicates),
            "potentially_unused": len(result.potentially_unused),
            "raw_traditional_duplicates": len(result.traditional_duplicates),
            "raw_semantic_duplicates": len(result.semantic_duplicates),
            "semantic_fallback": result.semantic_fallback,
            "semantic_fallback_reason": result.semantic_fallback_reason,
            "extraction_diagnostics": len(result.extraction_diagnostics),
            "semantic_diagnostics": len(result.semantic_diagnostics),
            "unused_supported_languages": list(result.unused_supported_languages),
            "unused_excluded_units": result.unused_excluded_units,
            "embeddings": _embedding_stats_to_dict(result.embedding_stats),
            "fail_on": fail_on,
            "exit_code": exit_code,
        },
        "duplicates": duplicates,
        "potentially_unused": [ref(unit) for unit in result.potentially_unused],
        "extraction_diagnostics": [
            _diagnostic_to_dict(diagnostic) for diagnostic in result.extraction_diagnostics
        ],
        "semantic_diagnostics": [
            _diagnostic_to_dict(diagnostic) for diagnostic in result.semantic_diagnostics
        ],
    }
    if combined_mode and show_all:
        output["traditional_duplicates"] = [
            raw_edge(duplicate) for duplicate in result.traditional_duplicates
        ]
        output["semantic_duplicates"] = [
            raw_edge(duplicate) for duplicate in result.semantic_duplicates
        ]
    output["units"] = units
    return output


def print_check_json(
    result: AnalysisResult,
    *,
    show_all: bool,
    fail_on: str,
    exit_code: int,
) -> None:
    """Output check results as schema-v2 JSON.

    :param result: Analysis result to serialize.
    :param show_all: Include raw duplicate edges in combined mode.
    :param fail_on: Finding policy selected for this run.
    :param exit_code: Exit code computed from the selected policy.
    :return: ``None``.
    """
    print(
        json.dumps(
            check_result_to_json(
                result,
                show_all=show_all,
                fail_on=fail_on,
                exit_code=exit_code,
            ),
            indent=2,
            sort_keys=True,
        )
    )


def search_result_to_json(
    query: str,
    results: list[tuple[CodeUnit, float]],
    extraction_diagnostics: list[ExtractionDiagnostic],
    semantic_diagnostics: list[ExtractionDiagnostic],
    indexed_units: int,
    embedding_stats: EmbeddingRunStats | None,
) -> dict[str, Any]:
    """Serialize semantic search results using schema-v2 unit references.

    :param query: Original search query.
    :param results: Ranked unit and score pairs.
    :param extraction_diagnostics: Diagnostics from corpus extraction.
    :param semantic_diagnostics: Units skipped by semantic indexing.
    :param indexed_units: Number of indexed corpus units.
    :param embedding_stats: Optional indexing telemetry.
    :return: Schema-v2 search payload.
    """
    units: dict[str, dict[str, Any]] = {}

    def ref(unit: CodeUnit) -> str:
        """Register a search result unit once and return its identifier.

        :param unit: Search-result unit to register.
        :return: Stable unit identifier.
        """
        units.setdefault(unit.uid, _unit_to_dict(unit))
        return unit.uid

    serialized_results = [{"unit": ref(unit), "score": float(score)} for unit, score in results]
    return {
        "schema_version": 2,
        "query": query,
        "summary": {
            "indexed_units": indexed_units,
            "results": len(results),
            "embeddings": _embedding_stats_to_dict(embedding_stats),
        },
        "results": serialized_results,
        "units": units,
        "extraction_diagnostics": [
            _diagnostic_to_dict(diagnostic) for diagnostic in extraction_diagnostics
        ],
        "semantic_diagnostics": [
            _diagnostic_to_dict(diagnostic) for diagnostic in semantic_diagnostics
        ],
    }


def print_search_json(
    query: str,
    results: list[tuple[CodeUnit, float]],
    extraction_diagnostics: list[ExtractionDiagnostic],
    semantic_diagnostics: list[ExtractionDiagnostic],
    indexed_units: int,
    embedding_stats: EmbeddingRunStats | None,
) -> None:
    """Output search results as schema-v2 JSON.

    :param query: Original search query.
    :param results: Ranked unit and score pairs.
    :param extraction_diagnostics: Diagnostics from corpus extraction.
    :param semantic_diagnostics: Units skipped by semantic indexing.
    :param indexed_units: Number of indexed corpus units.
    :param embedding_stats: Optional indexing telemetry.
    :return: ``None``.
    """
    print(
        json.dumps(
            search_result_to_json(
                query,
                results,
                extraction_diagnostics,
                semantic_diagnostics,
                indexed_units,
                embedding_stats,
            ),
            indent=2,
            sort_keys=True,
        )
    )
