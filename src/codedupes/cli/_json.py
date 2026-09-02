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
    """Convert optional embedding telemetry to a JSON-safe mapping."""
    return asdict(stats) if stats is not None else None


def _language_counts(units: list[CodeUnit]) -> dict[str, int]:
    """Count extracted units by canonical language."""
    return dict(sorted(Counter(unit.language for unit in units).items()))


def _diagnostic_to_dict(diagnostic: ExtractionDiagnostic) -> dict[str, Any]:
    """Convert an extraction diagnostic to a JSON-safe mapping."""
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
    """Convert a code unit to a JSON-serializable summary."""
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


def _dup_to_dict(dup: DuplicatePair) -> dict[str, Any]:
    """Convert a duplicate pair to a JSON-serializable mapping."""
    return {
        "unit_a": _unit_to_dict(dup.unit_a),
        "unit_b": _unit_to_dict(dup.unit_b),
        "similarity": dup.similarity,
        "method": dup.method,
    }


def _hybrid_dup_to_dict(dup: HybridDuplicate) -> dict[str, Any]:
    """Convert a hybrid duplicate pair for JSON output."""
    return {
        "unit_a": _unit_to_dict(dup.unit_a),
        "unit_b": _unit_to_dict(dup.unit_b),
        "tier": dup.tier,
        "confidence": dup.confidence,
        "has_exact": dup.has_exact,
        "semantic_similarity": dup.semantic_similarity,
        "jaccard_similarity": dup.jaccard_similarity,
        "weak_identifier_jaccard": dup.weak_identifier_jaccard,
        "statement_count_ratio": dup.statement_count_ratio,
    }


def print_check_json_combined(result: AnalysisResult, *, show_all: bool) -> None:
    """Output combined-mode check results as JSON."""
    output: dict[str, Any] = {
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
        },
        "extraction_diagnostics": [
            _diagnostic_to_dict(diagnostic) for diagnostic in result.extraction_diagnostics
        ],
        "semantic_diagnostics": [
            _diagnostic_to_dict(diagnostic) for diagnostic in result.semantic_diagnostics
        ],
        "hybrid_duplicates": [
            _hybrid_dup_to_dict(duplicate) for duplicate in result.hybrid_duplicates
        ],
        "potentially_unused": [_unit_to_dict(unit) for unit in result.potentially_unused],
    }
    if show_all:
        output["traditional_duplicates"] = [
            _dup_to_dict(duplicate) for duplicate in result.traditional_duplicates
        ]
        output["semantic_duplicates"] = [
            _dup_to_dict(duplicate) for duplicate in result.semantic_duplicates
        ]

    print(json.dumps(output, indent=2, sort_keys=True))


def print_check_json_raw(result: AnalysisResult) -> None:
    """Output raw single-method check results as JSON."""
    output = {
        "analysis_mode": result.analysis_mode,
        "summary": {
            "total_units": len(result.units),
            "units_by_language": _language_counts(result.units),
            "traditional_duplicates": len(result.traditional_duplicates),
            "semantic_duplicates": len(result.semantic_duplicates),
            "potentially_unused": len(result.potentially_unused),
            "semantic_fallback": result.semantic_fallback,
            "semantic_fallback_reason": result.semantic_fallback_reason,
            "extraction_diagnostics": len(result.extraction_diagnostics),
            "semantic_diagnostics": len(result.semantic_diagnostics),
            "unused_supported_languages": list(result.unused_supported_languages),
            "unused_excluded_units": result.unused_excluded_units,
            "embeddings": _embedding_stats_to_dict(result.embedding_stats),
        },
        "extraction_diagnostics": [
            _diagnostic_to_dict(diagnostic) for diagnostic in result.extraction_diagnostics
        ],
        "semantic_diagnostics": [
            _diagnostic_to_dict(diagnostic) for diagnostic in result.semantic_diagnostics
        ],
        "traditional_duplicates": [
            _dup_to_dict(duplicate) for duplicate in result.traditional_duplicates
        ],
        "semantic_duplicates": [
            _dup_to_dict(duplicate) for duplicate in result.semantic_duplicates
        ],
        "potentially_unused": [_unit_to_dict(unit) for unit in result.potentially_unused],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


def print_search_json(
    query: str,
    results: list[tuple[CodeUnit, float]],
    semantic_diagnostics: list[ExtractionDiagnostic],
    indexed_units: int,
    embedding_stats: EmbeddingRunStats | None,
) -> None:
    """Output search results as JSON."""
    payload = {
        "query": query,
        "indexed_units": indexed_units,
        "summary": {
            "indexed_units": indexed_units,
            "embeddings": _embedding_stats_to_dict(embedding_stats),
        },
        "results": [{"score": float(score), **_unit_to_dict(unit)} for unit, score in results],
        "semantic_diagnostics": [
            _diagnostic_to_dict(diagnostic) for diagnostic in semantic_diagnostics
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
