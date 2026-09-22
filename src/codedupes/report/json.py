"""JSON serializers for check and search reports."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from typing import Any

from codedupes.models import (
    CodeUnit,
    DuplicatePair,
    ExtractionDiagnostic,
    HybridDuplicate,
)
from codedupes.semantic import EmbeddingRunStats

from .selection import (
    ExactFamily,
    FailOnPolicy,
    FileSearchResult,
    ReportSelection,
    assign_unit_ids,
    collect_units,
    hidden_only_failure,
)

SCHEMA_VERSION = 4


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


def unit_to_dict(
    unit: CodeUnit, *, include_source: bool = False, source_lines: int | None = None
) -> dict[str, Any]:
    """Convert a code unit to a JSON-serializable summary.

    :param unit: Code unit to serialize.
    :param include_source: Whether to add a bounded ``source`` field.
    :param source_lines: Max lines to keep, or ``None`` for no bound (needs ``include_source``).
    :return: Serialized unit fields, including the in-run ``uid``.
    """
    payload: dict[str, Any] = {
        "uid": unit.uid,
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
    if unit.suppressions:
        payload["suppressions"] = sorted(unit.suppressions)
    if include_source:
        lines, omitted = unit.source_lines(source_lines)
        payload["source"] = "\n".join(lines)
        payload["source_lines_omitted"] = omitted
    return payload


def _unit_nodes(
    units: list[CodeUnit],
    ids: dict[str, str],
    *,
    include_source: bool = False,
    source_lines: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Serialize referenced units once each, keyed by their report-local id.

    :param units: Distinct referenced units in report order.
    :param ids: Mapping from unit uid to report-local id.
    :param include_source: Whether to add a bounded ``source`` field to every unit.
    :param source_lines: Maximum source lines to keep per unit, or ``None`` for no bound.
    :return: Node map keyed by report-local id.
    """
    return {
        ids[unit.uid]: unit_to_dict(unit, include_source=include_source, source_lines=source_lines)
        for unit in units
    }


def _hybrid_edge(duplicate: HybridDuplicate, ids: dict[str, str]) -> dict[str, Any]:
    """Serialize one hybrid duplicate as an edge between report-local ids.

    :param duplicate: Hybrid duplicate to serialize.
    :param ids: Mapping from unit uid to report-local id.
    :return: Serialized hybrid edge.
    """
    return {
        "unit_a": ids[duplicate.unit_a.uid],
        "unit_b": ids[duplicate.unit_b.uid],
        "tier": duplicate.tier,
        "score": duplicate.score,
        "semantic_similarity": duplicate.semantic_similarity,
        "jaccard_similarity": duplicate.jaccard_similarity,
        "weak_identifier_jaccard": duplicate.weak_identifier_jaccard,
        "statement_count_ratio": duplicate.statement_count_ratio,
    }


def _family_record(family: ExactFamily, ids: dict[str, str]) -> dict[str, Any]:
    """Serialize one exact family over report-local ids.

    :param family: Exact family to serialize.
    :param ids: Mapping from unit uid to report-local id.
    :return: Serialized family: fingerprint method, member ids, and size.
    """
    return {
        "method": family.method,
        "members": [ids[unit.uid] for unit in family.members],
        "lines": family.lines,
        "redundant_lines": family.redundant_lines,
    }


def _raw_edge(duplicate: DuplicatePair, ids: dict[str, str]) -> dict[str, Any]:
    """Serialize one raw duplicate as an edge between report-local ids.

    :param duplicate: Raw duplicate to serialize.
    :param ids: Mapping from unit uid to report-local id.
    :return: Serialized raw edge.
    """
    return {
        "unit_a": ids[duplicate.unit_a.uid],
        "unit_b": ids[duplicate.unit_b.uid],
        "similarity": duplicate.similarity,
        "method": duplicate.method,
    }


def check_result_to_json(
    selection: ReportSelection,
    *,
    fail_on: FailOnPolicy,
    exit_code: int,
    strict_unused: bool,
    include_source: bool = False,
    source_lines: int | None = None,
) -> dict[str, Any]:
    """Serialize one selected check report using the normalized graph shape.

    :param selection: Findings selected for this report.
    :param fail_on: Finding policy selected for this run.
    :param exit_code: Exit code computed from the selected policy.
    :param strict_unused: Whether unused findings counted under the failure policy.
    :param include_source: Whether to add a bounded ``source`` field to every unit.
    :param source_lines: Maximum source lines to keep per unit, or ``None`` for no bound.
    :return: Check payload.
    """
    result = selection.result
    combined = result.analysis_mode == "combined"
    ids = assign_unit_ids(selection.units)
    duplicates = [
        _hybrid_edge(pair, ids) if isinstance(pair, HybridDuplicate) else _raw_edge(pair, ids)
        for pair in selection.duplicates
    ]
    exact_families = [_family_record(family, ids) for family in selection.exact_families]

    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_mode": result.analysis_mode,
        "summary": {
            "total_units": len(result.units),
            "units_by_language": _language_counts(result.units),
            # Finding counts: an exact family counts once, every other tier per pair.
            "hybrid_duplicates": selection.total_findings if combined else 0,
            "reported_duplicates": selection.reported_findings,
            "omitted_review_duplicates": len(selection.omitted_review),
            "truncated_duplicates": selection.truncated_findings,
            "truncated_by_tier": dict(selection.truncated_by_tier),
            "max_duplicates": selection.policy.max_duplicates,
            "actionable_duplicates": selection.actionable_findings,
            "reported_actionable_duplicates": selection.reported_actionable_findings,
            "duplicates_by_tier": dict(selection.duplicates_by_tier),
            "exact_family_members": selection.exact_family_members,
            "potentially_unused": len(result.potentially_unused),
            "reported_unused": len(selection.potentially_unused),
            "truncated_unused": len(selection.truncated_unused),
            "max_unused": selection.policy.max_unused,
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
            "strict_unused": strict_unused,
            "exit_code": exit_code,
            "hidden_only_failure": sorted(
                hidden_only_failure(selection, policy=fail_on, strict_unused=strict_unused)
            ),
        },
        "exact_families": exact_families,
        "duplicates": duplicates,
        "potentially_unused": [ids[unit.uid] for unit in selection.potentially_unused],
        "extraction_diagnostics": [
            _diagnostic_to_dict(diagnostic) for diagnostic in result.extraction_diagnostics
        ],
        "semantic_diagnostics": [
            _diagnostic_to_dict(diagnostic) for diagnostic in result.semantic_diagnostics
        ],
        "unused_diagnostics": [
            _diagnostic_to_dict(diagnostic) for diagnostic in result.unused_diagnostics
        ],
    }
    if selection.traditional_duplicates is not None:
        output["traditional_duplicates"] = [
            _raw_edge(pair, ids) for pair in selection.traditional_duplicates
        ]
    if selection.semantic_duplicates is not None:
        output["semantic_duplicates"] = [
            _raw_edge(pair, ids) for pair in selection.semantic_duplicates
        ]
    output["units"] = _unit_nodes(
        selection.units, ids, include_source=include_source, source_lines=source_lines
    )
    return output


def search_result_to_json(
    query: str,
    results: list[tuple[CodeUnit, float]],
    indexed_units: int,
    embedding_stats: EmbeddingRunStats | None,
    *,
    extraction_diagnostics: list[ExtractionDiagnostic],
    semantic_diagnostics: list[ExtractionDiagnostic],
    file_results: list[FileSearchResult] | None = None,
) -> dict[str, Any]:
    """Serialize semantic search results using normalized unit references.

    :param query: Original search query.
    :param results: Ranked unit and score pairs.
    :param indexed_units: Number of indexed corpus units.
    :param embedding_stats: Optional indexing telemetry.
    :param extraction_diagnostics: Diagnostics from corpus extraction.
    :param semantic_diagnostics: Warnings from semantic indexing.
    :param file_results: Ranked file results, or ``None`` for unit-level output.
    :return: Search payload.
    """
    if file_results is None:
        referenced = collect_units(unit for unit, _ in results)
        ids = assign_unit_ids(referenced)
        serialized_results: list[dict[str, Any]] = [
            {"unit": ids[unit.uid], "score": float(score)} for unit, score in results
        ]
    else:
        referenced = collect_units(unit for result in file_results for unit, _ in result.matches)
        ids = assign_unit_ids(referenced)
        serialized_results = [
            {
                "file": str(result.file_path),
                "score": float(result.score),
                "matching_units": result.matching_units,
                "matches": [
                    {"unit": ids[unit.uid], "score": float(score)} for unit, score in result.matches
                ],
            }
            for result in file_results
        ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "query": query,
        "summary": {
            "indexed_units": indexed_units,
            "results": len(serialized_results),
            "embeddings": _embedding_stats_to_dict(embedding_stats),
        },
        "results": serialized_results,
        "units": _unit_nodes(referenced, ids),
        "extraction_diagnostics": [
            _diagnostic_to_dict(diagnostic) for diagnostic in extraction_diagnostics
        ],
        "semantic_diagnostics": [
            _diagnostic_to_dict(diagnostic) for diagnostic in semantic_diagnostics
        ],
    }
    if file_results is not None:
        payload["result_level"] = "file"
    return payload


def to_json_text(payload: dict[str, Any]) -> str:
    """Render a report payload as the CLI's canonical JSON text.

    :param payload: Serialized report.
    :return: Indented JSON with sorted keys.
    """
    return json.dumps(payload, indent=2, sort_keys=True)
