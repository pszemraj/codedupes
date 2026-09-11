"""Report visibility policy, finding selection, and finding exit policy.

Analysis results stay complete; this module decides which findings a report
emits and which findings make ``check`` fail. JSON, terminal, and exit-code
paths all read the same :class:`ReportSelection` so they cannot disagree.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from codedupes.models import (
    HYBRID_TIERS,
    AnalysisMode,
    AnalysisResult,
    CodeUnit,
    DuplicatePair,
    HybridDuplicate,
    HybridTier,
)

FailOnPolicy = Literal["actionable", "all", "none"]
# Groups of findings a report can hide while the exit code still counts them.
HiddenGroup = Literal["review", "truncated"]
# Tiers with deterministic structural/token corroboration; the only ones that
# fail the default ``actionable`` policy.
ACTIONABLE_TIERS: frozenset[HybridTier] = frozenset(
    {"exact", "traditional_near", "hybrid_confirmed"}
)
# Tiers withheld from default reports; ``--include-review`` restores them.
WITHHELD_TIERS: frozenset[HybridTier] = frozenset({"semantic_review"})


@dataclass(frozen=True)
class ReportPolicy:
    """Visibility policy for one rendered report."""

    include_review: bool = False
    show_all: bool = False
    # Cap on emitted duplicate pairs, applied after the review filter in
    # analyzer order so the highest-confidence pairs survive. ``None`` = no cap.
    max_duplicates: int | None = None

    def __post_init__(self) -> None:
        """Reject a cap that would emit nothing.

        :raises ValueError: If ``max_duplicates`` is below one.
        """
        if self.max_duplicates is not None and self.max_duplicates < 1:
            raise ValueError("max_duplicates must be at least 1 or None.")

    @property
    def shows_review(self) -> bool:
        """Return whether withheld review pairs are listed.

        :return: ``True`` when ``include_review`` or ``show_all`` is set.
        """
        return self.include_review or self.show_all


@dataclass(frozen=True)
class ReportSelection:
    """Findings chosen for one report, derived from a complete analysis result.

    ``duplicates`` + ``omitted_review`` + ``truncated`` is the complete duplicate
    list the analyzer produced for the selected mode.
    """

    result: AnalysisResult
    policy: ReportPolicy
    mode: AnalysisMode
    duplicates: list[HybridDuplicate] | list[DuplicatePair]
    omitted_review: list[HybridDuplicate]
    truncated: list[HybridDuplicate] | list[DuplicatePair]
    duplicates_by_tier: dict[HybridTier, int]
    traditional_duplicates: list[DuplicatePair] | None
    semantic_duplicates: list[DuplicatePair] | None
    potentially_unused: list[CodeUnit]
    units: list[CodeUnit]


@dataclass
class FileSearchResult:
    """One ranked file with its score and up to three contributing code units."""

    file_path: Path
    score: float
    matching_units: int
    matches: list[tuple[CodeUnit, float]]


def unit_sort_key(unit: CodeUnit) -> tuple[str, int, str]:
    """Return the report ordering key for a unit: file, source offset, then uid.

    :param unit: Unit to order.
    :return: Sort key that reads top-to-bottom within a file and is total.
    """
    return (str(unit.file_path), unit.start_byte, unit.uid)


def collect_units(*groups: Iterable[CodeUnit]) -> list[CodeUnit]:
    """Deduplicate referenced units by uid and order them for report-local ids.

    :param groups: Iterables of referenced units.
    :return: Distinct units sorted by :func:`unit_sort_key`.
    """
    by_uid: dict[str, CodeUnit] = {}
    for group in groups:
        for unit in group:
            by_uid.setdefault(unit.uid, unit)
    return sorted(by_uid.values(), key=unit_sort_key)


def assign_unit_ids(units: Sequence[CodeUnit]) -> dict[str, str]:
    """Map unit uids to report-local ids ``u0``, ``u1``, ... in sequence order.

    :param units: Units in their report order.
    :return: Mapping from ``CodeUnit.uid`` to report-local id.
    """
    return {unit.uid: f"u{index}" for index, unit in enumerate(units)}


def _pair_units(pairs: Iterable[HybridDuplicate | DuplicatePair]) -> Iterable[CodeUnit]:
    """Yield both endpoints of every pair.

    :param pairs: Duplicate pairs.
    :return: Endpoint units in pair order.
    """
    for pair in pairs:
        yield pair.unit_a
        yield pair.unit_b


def select_findings(result: AnalysisResult, policy: ReportPolicy | None = None) -> ReportSelection:
    """Apply a visibility policy to a complete analysis result.

    :param result: Complete analysis result; never mutated.
    :param policy: Visibility policy for the report, defaults to the default report policy.
    :return: Findings to emit plus the counts needed to describe what was withheld.
    """
    policy = policy or ReportPolicy()
    mode = result.analysis_mode
    duplicates_by_tier: dict[HybridTier, int] = dict.fromkeys(HYBRID_TIERS, 0)
    omitted_review: list[HybridDuplicate] = []
    traditional: list[DuplicatePair] | None = None
    semantic: list[DuplicatePair] | None = None
    duplicates: list[HybridDuplicate] | list[DuplicatePair]
    truncated: list[HybridDuplicate] | list[DuplicatePair] = []

    if mode == "combined":
        duplicates_by_tier.update(Counter(pair.tier for pair in result.hybrid_duplicates))
        if policy.shows_review:
            duplicates = list(result.hybrid_duplicates)
        else:
            shown: list[HybridDuplicate] = []
            for pair in result.hybrid_duplicates:
                (omitted_review if pair.tier in WITHHELD_TIERS else shown).append(pair)
            duplicates = shown
        if policy.show_all:
            traditional = list(result.traditional_duplicates)
            semantic = list(result.semantic_duplicates)
    else:
        duplicates = result.traditional_duplicates + result.semantic_duplicates

    # The cap keeps a prefix of the analyzer's ranking; the raw ``--show-all``
    # lists are diagnostic and stay complete.
    if policy.max_duplicates is not None:
        truncated = duplicates[policy.max_duplicates :]
        duplicates = duplicates[: policy.max_duplicates]

    units = collect_units(
        _pair_units(duplicates),
        _pair_units(traditional or ()),
        _pair_units(semantic or ()),
        result.potentially_unused,
    )
    return ReportSelection(
        result=result,
        policy=policy,
        mode=mode,
        duplicates=duplicates,
        omitted_review=omitted_review,
        truncated=truncated,
        duplicates_by_tier=duplicates_by_tier,
        traditional_duplicates=traditional,
        semantic_duplicates=semantic,
        potentially_unused=list(result.potentially_unused),
        units=units,
    )


def _findings_fail(
    duplicates: Sequence[HybridDuplicate | DuplicatePair],
    unused: Sequence[CodeUnit],
    *,
    combined: bool,
    policy: FailOnPolicy,
    strict_unused: bool,
) -> bool:
    """Evaluate one finding policy over explicit duplicate and unused lists.

    :param duplicates: Duplicate findings under consideration.
    :param unused: Unused findings under consideration.
    :param combined: Whether ``duplicates`` are hybrid pairs carrying tiers.
    :param policy: Selected finding policy.
    :param strict_unused: Whether unused findings are strict rather than heuristic.
    :return: Whether these findings require exit code one.
    """
    if policy == "none":
        return False
    failing = list(duplicates)
    if combined and policy == "actionable":
        failing = [
            pair
            for pair in failing
            if isinstance(pair, HybridDuplicate) and pair.tier in ACTIONABLE_TIERS
        ]
    failing_unused = [] if policy == "actionable" and not strict_unused else list(unused)
    return bool(failing or failing_unused)


def run_should_fail(
    result: AnalysisResult,
    *,
    policy: FailOnPolicy,
    strict_unused: bool,
) -> bool:
    """Return whether the complete analysis result should make ``check`` exit one.

    The verdict is computed on every finding the analyzer produced, so report
    visibility never changes the exit status.

    :param result: Completed analysis result.
    :param policy: Selected finding policy.
    :param strict_unused: Whether unused findings are strict rather than heuristic.
    :return: Whether findings require exit code one.
    """
    combined = result.analysis_mode == "combined"
    duplicates: Sequence[HybridDuplicate | DuplicatePair] = (
        result.hybrid_duplicates
        if combined
        else result.traditional_duplicates + result.semantic_duplicates
    )
    return _findings_fail(
        duplicates,
        result.potentially_unused,
        combined=combined,
        policy=policy,
        strict_unused=strict_unused,
    )


def hidden_only_failure(
    selection: ReportSelection,
    *,
    policy: FailOnPolicy,
    strict_unused: bool,
) -> frozenset[HiddenGroup]:
    """Return which hidden finding groups fail when every emitted finding passes.

    The exit code is decided on the complete result, so a run can fail over pairs
    the report withheld (``review``) or cut with ``max_duplicates``
    (``truncated``). Renderers use the answer to name what the reader cannot see.

    :param selection: Report selection derived from the complete result.
    :param policy: Selected finding policy.
    :param strict_unused: Whether unused findings are strict rather than heuristic.
    :return: Failing hidden groups; empty when the run passes or an emitted finding already fails it.
    """
    combined = selection.mode == "combined"
    if _findings_fail(
        selection.duplicates,
        selection.potentially_unused,
        combined=combined,
        policy=policy,
        strict_unused=strict_unused,
    ):
        return frozenset()
    hidden: dict[HiddenGroup, Sequence[HybridDuplicate | DuplicatePair]] = {
        "review": selection.omitted_review,
        "truncated": selection.truncated,
    }
    return frozenset(
        group
        for group, pairs in hidden.items()
        if _findings_fail(pairs, (), combined=combined, policy=policy, strict_unused=strict_unused)
    )


def group_file_results(results: list[tuple[CodeUnit, float]], top_k: int) -> list[FileSearchResult]:
    """Group matching units into files ranked by their strongest unit score.

    :param results: All unit matches above the search threshold.
    :param top_k: Maximum number of distinct files to return.
    :return: Ranked files with at most three contributing units each.
    """
    grouped: dict[Path, list[tuple[CodeUnit, float]]] = {}
    for unit, score in results:
        grouped.setdefault(unit.file_path, []).append((unit, score))

    files = []
    for path, matches in grouped.items():
        matches.sort(key=lambda match: (-match[1], match[0].lineno, match[0].uid))
        files.append(FileSearchResult(path, matches[0][1], len(matches), matches[:3]))
    return sorted(files, key=lambda result: (-result.score, str(result.file_path)))[:top_k]
