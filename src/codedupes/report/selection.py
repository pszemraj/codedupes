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
# Groups omitted from the primary duplicate list that the exit code still
# counts. Only withheld review pairs qualify: the primary list ranks actionable
# tiers first and the cap is at least one, so a cap can never hide every
# failing pair while emitting only passing ones.
HiddenGroup = Literal["review"]
# Tiers with deterministic structural/token corroboration; the only ones that
# fail the default ``actionable`` policy.
ACTIONABLE_TIERS: frozenset[HybridTier] = frozenset(
    {"exact", "traditional_near", "hybrid_confirmed"}
)
# Tiers withheld from default reports; ``--include-review`` restores them.
WITHHELD_TIERS: frozenset[HybridTier] = frozenset({"semantic_review"})
# The CLI's default cap on the primary duplicate list, shared by terminal and
# JSON output. ``ReportPolicy()`` itself stays uncapped so library callers get
# the complete list unless they ask for the CLI's concise report.
DEFAULT_MAX_DUPLICATES = 20


@dataclass(frozen=True)
class ReportPolicy:
    """Visibility policy for one rendered report."""

    include_review: bool = False
    show_all: bool = False
    # Cap on emitted duplicate pairs, applied after the review filter to the
    # report ranking (actionable tiers first). ``None`` = no cap.
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


def _report_rank(pair: HybridDuplicate) -> int:
    """Return the primary-list group of a hybrid pair: actionable, advisory, review.

    :param pair: Hybrid duplicate to rank.
    :return: ``0`` for actionable tiers, ``2`` for withheld tiers, ``1`` otherwise.
    """
    if pair.tier in ACTIONABLE_TIERS:
        return 0
    if pair.tier in WITHHELD_TIERS:
        return 2
    return 1


def actionable_pairs(
    pairs: Iterable[HybridDuplicate | DuplicatePair], *, combined: bool
) -> list[HybridDuplicate | DuplicatePair]:
    """Return the pairs that fail the ``actionable`` policy.

    Combined mode carries tiers, so only :data:`ACTIONABLE_TIERS` count; the
    single-method modes have no tier classification, so every raw pair counts.

    :param pairs: Duplicate findings under consideration.
    :param combined: Whether ``pairs`` are hybrid pairs carrying tiers.
    :return: The actionable subset in input order.
    """
    if not combined:
        return list(pairs)
    return [
        pair
        for pair in pairs
        if isinstance(pair, HybridDuplicate) and pair.tier in ACTIONABLE_TIERS
    ]


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
        shown: list[HybridDuplicate] = []
        for pair in result.hybrid_duplicates:
            withheld = pair.tier in WITHHELD_TIERS and not policy.shows_review
            (omitted_review if withheld else shown).append(pair)
        # Report ranking: actionable tiers first, then semantic_high_confidence,
        # then any included semantic_review pairs. The stable sort keeps the
        # analyzer's confidence order inside each group and never touches
        # ``result.hybrid_duplicates``.
        duplicates = sorted(shown, key=_report_rank)
        if policy.show_all:
            traditional = list(result.traditional_duplicates)
            semantic = list(result.semantic_duplicates)
    else:
        # Traditional near pairs come out of the analyzer sorted by index pair,
        # not similarity, so the raw list is re-ranked here before capping;
        # Python's stable sort keeps ties (exact pairs at 1.0, equal-similarity
        # pairs) in analyzer order.
        duplicates = sorted(
            result.traditional_duplicates + result.semantic_duplicates,
            key=lambda pair: -pair.similarity,
        )

    # The cap keeps a prefix of the report ranking established above: tier
    # groups for combined mode, descending similarity for the raw modes. The
    # raw ``--show-all`` lists are diagnostic and stay complete.
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
    :raises ValueError: If ``policy`` is not a supported failure policy.
    """
    if policy not in ("actionable", "all", "none"):
        raise ValueError(f"Unknown failure policy {policy!r}; expected actionable, all, or none.")
    if policy == "none":
        return False
    failing = (
        actionable_pairs(duplicates, combined=combined)
        if policy == "actionable"
        else list(duplicates)
    )
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
    :raises ValueError: If ``policy`` is not a supported failure policy.
    """
    return _findings_fail(
        result.all_duplicates,
        result.potentially_unused,
        combined=result.analysis_mode == "combined",
        policy=policy,
        strict_unused=strict_unused,
    )


def hidden_only_failure(
    selection: ReportSelection,
    *,
    policy: FailOnPolicy,
    strict_unused: bool,
) -> frozenset[HiddenGroup]:
    """Return which omitted groups fail when the primary list and unused findings pass.

    The exit code is decided on the complete result, so a run can fail over
    ``semantic_review`` pairs the primary list withheld (``review``). Pairs cut
    by ``max_duplicates`` never qualify: the primary list ranks actionable tiers
    first and the cap is at least one, so whenever a truncated pair fails, an
    emitted pair already fails too.

    :param selection: Report selection derived from the complete result.
    :param policy: Selected finding policy.
    :param strict_unused: Whether unused findings are strict rather than heuristic.
    :return: Failing hidden groups; empty when the run passes or an emitted finding already fails it.
    :raises ValueError: If ``policy`` is not a supported failure policy.
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
    if _findings_fail(
        selection.omitted_review, (), combined=combined, policy=policy, strict_unused=strict_unused
    ):
        return frozenset({"review"})
    return frozenset()


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
