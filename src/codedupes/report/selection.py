"""Report visibility policy, finding selection, and finding exit policy.

Analysis results stay complete; this module decides which findings a report
emits and which findings make ``check`` fail. JSON, terminal, and exit-code
paths all read the same :class:`ReportSelection` so they cannot disagree.
Pairwise exact edges are grouped into :class:`ExactFamily` records here, so
the analysis layer keeps its complete edge list while reports show each
copy-paste family once.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from codedupes.models import (
    HYBRID_TIERS,
    AnalysisMode,
    AnalysisResult,
    CodeUnit,
    DuplicatePair,
    FocusSummary,
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
# Fingerprint methods whose raw pairs make up the ``exact`` tier.
ExactMethod = Literal["structural_hash", "token_hash"]
EXACT_METHODS: frozenset[str] = frozenset({"structural_hash", "token_hash"})
# The CLI's default caps on the primary duplicate list and the unused list,
# shared by terminal and JSON output. ``ReportPolicy()`` itself stays uncapped
# so library callers get the complete lists unless they ask for the CLI's
# concise report.
DEFAULT_MAX_DUPLICATES = 20
DEFAULT_MAX_UNUSED = 20


@dataclass(frozen=True)
class ReportPolicy:
    """Visibility policy for one rendered report."""

    include_review: bool = False
    show_all: bool = False
    # Cap on emitted duplicate findings, applied after the review filter to the
    # report ranking (families, then actionable tiers first). ``None`` = no cap.
    max_duplicates: int | None = None
    # Cap on emitted unused findings, applied to the size ranking. ``None`` = no cap.
    max_unused: int | None = None

    def __post_init__(self) -> None:
        """Reject a cap that would emit nothing.

        :raises ValueError: If ``max_duplicates`` or ``max_unused`` is below one.
        """
        for name in ("max_duplicates", "max_unused"):
            cap = getattr(self, name)
            if cap is not None and cap < 1:
                raise ValueError(f"{name} must be at least 1 or None.")

    @property
    def shows_review(self) -> bool:
        """Return whether withheld review pairs are listed.

        :return: ``True`` when ``include_review`` or ``show_all`` is set.
        """
        return self.include_review or self.show_all


@dataclass(frozen=True, eq=False)
class ExactFamily:
    """Units that are mutually exact duplicates, reported once instead of as C(n, 2) edges.

    ``method`` is the strongest fingerprint every member shares: ``token_hash``
    members are token-for-token identical (comments and whitespace aside),
    while ``structural_hash`` members match only after identifier and
    string-literal normalization.
    """

    members: tuple[CodeUnit, ...]
    method: ExactMethod

    @property
    def lines(self) -> int:
        """Return the line span of the largest member.

        :return: ``end_lineno - lineno + 1`` maximized over the members.
        """
        return max(unit.end_lineno - unit.lineno + 1 for unit in self.members)

    @property
    def redundant_lines(self) -> int:
        """Return the source lines removable by keeping one copy.

        :return: ``(len(members) - 1) * lines``; the family ranking key.
        """
        return (len(self.members) - 1) * self.lines

    @property
    def pair_count(self) -> int:
        """Return how many pairwise exact edges the family stands for.

        :return: ``n * (n - 1) / 2`` for ``n`` members.
        """
        count = len(self.members)
        return count * (count - 1) // 2


@dataclass(frozen=True)
class ReportSelection:
    """Findings chosen for one report, derived from a complete analysis result.

    Exact edges are grouped into ``exact_families``; every other duplicate stays
    a pair. ``exact_families`` + ``duplicates`` is the emitted primary list,
    ``truncated_exact_families`` + ``truncated`` is what the cap cut, and adding
    ``omitted_review`` gives every finding the analyzer produced for the mode.
    """

    result: AnalysisResult
    policy: ReportPolicy
    mode: AnalysisMode
    exact_families: list[ExactFamily]
    duplicates: list[HybridDuplicate] | list[DuplicatePair]
    omitted_review: list[HybridDuplicate]
    truncated_exact_families: list[ExactFamily]
    truncated: list[HybridDuplicate] | list[DuplicatePair]
    # Finding counts per tier over the complete result; ``exact`` counts families.
    duplicates_by_tier: dict[HybridTier, int]
    # Tier breakdown of what the cap cut: review pairs rank last, so a cap can
    # cut every included review pair while ``omitted_review`` stays empty.
    truncated_by_tier: dict[HybridTier, int]
    traditional_duplicates: list[DuplicatePair] | None
    semantic_duplicates: list[DuplicatePair] | None
    # Unused findings ranked largest first; ``truncated_unused`` is what the cap cut.
    potentially_unused: list[CodeUnit]
    truncated_unused: list[CodeUnit]
    units: list[CodeUnit]

    @property
    def all_exact_families(self) -> list[ExactFamily]:
        """Return every family in report order, kept first.

        :return: Emitted families followed by the ones the cap cut.
        """
        return self.exact_families + self.truncated_exact_families

    @property
    def exact_family_members(self) -> int:
        """Return the number of distinct units inside exact families.

        :return: Member count over kept and truncated families.
        """
        return len({unit.uid for family in self.all_exact_families for unit in family.members})

    @property
    def reported_findings(self) -> int:
        """Return the size of the emitted primary list.

        :return: Emitted families plus emitted pairs.
        """
        return len(self.exact_families) + len(self.duplicates)

    @property
    def truncated_findings(self) -> int:
        """Return how many primary-list items the cap cut.

        :return: Truncated families plus truncated pairs.
        """
        return len(self.truncated_exact_families) + len(self.truncated)

    @property
    def total_findings(self) -> int:
        """Return every duplicate finding the mode produced, families counted once.

        :return: ``reported_findings + len(omitted_review) + truncated_findings``.
        """
        return self.reported_findings + len(self.omitted_review) + self.truncated_findings

    @property
    def actionable_findings(self) -> int:
        """Return the findings in the complete result that fail ``actionable``.

        :return: Every family plus every actionable non-exact pair.
        """
        combined = self.mode == "combined"
        pairs = actionable_pairs(self.result.all_duplicates, combined=combined)
        return len(self.all_exact_families) + sum(not _is_exact_edge(pair) for pair in pairs)

    @property
    def reported_actionable_findings(self) -> int:
        """Return how many emitted findings fail ``actionable``.

        :return: Emitted families plus emitted actionable pairs.
        """
        combined = self.mode == "combined"
        return len(self.exact_families) + len(actionable_pairs(self.duplicates, combined=combined))


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


def unused_sort_key(unit: CodeUnit) -> tuple[int, int, tuple[str, int, str]]:
    """Return the unused-report ordering key: largest line span first.

    Reading cost scales with length, so the biggest dead definitions lead.
    Statement count breaks span ties and file position makes the order total.

    :param unit: Unit to order.
    :return: Sort key; ascending order lists the largest unit first.
    """
    return (
        -(unit.end_lineno - unit.lineno + 1),
        -(unit.statement_count or 0),
        unit_sort_key(unit),
    )


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


def _exact_edge_method(pair: HybridDuplicate | DuplicatePair) -> str | None:
    """Return the fingerprint behind an exact edge, or ``None`` for any other pair.

    :param pair: Hybrid or raw duplicate.
    :return: ``structural_hash`` or ``token_hash``; a hand-built exact hybrid without a recorded method counts as structural.
    """
    if isinstance(pair, HybridDuplicate):
        if pair.tier != "exact":
            return None
        return pair.exact_method or "structural_hash"
    return pair.method if pair.method in EXACT_METHODS else None


def _is_exact_edge(pair: HybridDuplicate | DuplicatePair) -> bool:
    """Return whether a pair belongs to the exact tier.

    :param pair: Hybrid or raw duplicate.
    :return: ``True`` for exact edges.
    """
    return _exact_edge_method(pair) is not None


def _connected_components(edges: Iterable[tuple[CodeUnit, CodeUnit]]) -> list[list[CodeUnit]]:
    """Union-find the endpoints of ``edges`` into connected components.

    :param edges: Unit pairs; every endpoint lands in exactly one component.
    :return: Components in first-seen order, members in first-seen order.
    """
    parent: dict[str, str] = {}
    units: dict[str, CodeUnit] = {}

    def find(uid: str) -> str:
        """Return the component root of ``uid``, halving the path on the way up.

        :param uid: Unit uid already registered in ``parent``.
        :return: Root uid.
        """
        while parent[uid] != uid:
            parent[uid] = parent[parent[uid]]
            uid = parent[uid]
        return uid

    for unit_a, unit_b in edges:
        for unit in (unit_a, unit_b):
            units.setdefault(unit.uid, unit)
            parent.setdefault(unit.uid, unit.uid)
        parent[find(unit_a.uid)] = find(unit_b.uid)

    components: dict[str, list[CodeUnit]] = {}
    for uid, unit in units.items():
        components.setdefault(find(uid), []).append(unit)
    return list(components.values())


def build_exact_families(
    edges: Iterable[HybridDuplicate | DuplicatePair],
) -> list[ExactFamily]:
    """Group exact edges into families, ranked by the lines a consolidation removes.

    Structural and token edges are unioned into components separately: token
    equality is normally finer than structural equality, but Python
    indentation can change the parse tree without changing the token stream,
    so a token edge does not always sit inside the structural component it
    would usually imply, and the two fingerprints cannot be assumed to nest.
    A structural component is labelled ``token_hash`` only when every one of
    its members also shares one token fingerprint (its uid set exactly
    matches a token component); otherwise it is ``structural_hash``, the
    strongest fingerprint every member shares — a token clique with a
    renamed relative folds into the larger structural family instead of
    keeping its own record. A token component already covered by a
    structural family (its uid set is a subset of one) contributes nothing
    further; any token component left over forms its own ``token_hash``
    family. Non-exact edges and self-edges are ignored.

    :param edges: Duplicate pairs; only exact edges contribute.
    :return: Families ordered by ``redundant_lines`` descending, then first member position.
    """
    by_method: dict[str, list[tuple[CodeUnit, CodeUnit]]] = {method: [] for method in EXACT_METHODS}
    for pair in edges:
        method = _exact_edge_method(pair)
        if method is None or pair.unit_a.uid == pair.unit_b.uid:
            continue
        by_method[method].append((pair.unit_a, pair.unit_b))

    structural_components = _connected_components(by_method["structural_hash"])
    token_components = _connected_components(by_method["token_hash"])
    structural_uid_sets = [frozenset(unit.uid for unit in c) for c in structural_components]
    token_uid_sets = [frozenset(unit.uid for unit in c) for c in token_components]

    families: list[ExactFamily] = []
    for component, uids in zip(structural_components, structural_uid_sets, strict=True):
        members = tuple(sorted(component, key=unit_sort_key))
        method: ExactMethod = "token_hash" if uids in token_uid_sets else "structural_hash"
        families.append(ExactFamily(members=members, method=method))
    for component, uids in zip(token_components, token_uid_sets, strict=True):
        if any(uids <= structural_uids for structural_uids in structural_uid_sets):
            continue
        members = tuple(sorted(component, key=unit_sort_key))
        families.append(ExactFamily(members=members, method="token_hash"))
    families.sort(key=lambda family: (-family.redundant_lines, unit_sort_key(family.members[0])))
    return families


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
    pairs: list[HybridDuplicate] | list[DuplicatePair]

    if mode == "combined":
        duplicates_by_tier.update(
            Counter(pair.tier for pair in result.hybrid_duplicates if pair.tier != "exact")
        )
        exact_edges: list[HybridDuplicate] | list[DuplicatePair] = [
            pair for pair in result.hybrid_duplicates if pair.tier == "exact"
        ]
        shown: list[HybridDuplicate] = []
        for pair in result.hybrid_duplicates:
            if pair.tier == "exact":
                continue
            withheld = pair.tier in WITHHELD_TIERS and not policy.shows_review
            (omitted_review if withheld else shown).append(pair)
        # Report ranking: actionable tiers first, then semantic_high_confidence,
        # then any included semantic_review pairs. The stable sort keeps the
        # analyzer's score order inside each group and never touches
        # ``result.hybrid_duplicates``.
        pairs = sorted(shown, key=_report_rank)
        if policy.show_all:
            traditional = list(result.traditional_duplicates)
            semantic = list(result.semantic_duplicates)
    else:
        raw = result.traditional_duplicates + result.semantic_duplicates
        exact_edges = [pair for pair in raw if _is_exact_edge(pair)]
        # Traditional near pairs come out of the analyzer sorted by index pair,
        # not similarity, so the raw list is re-ranked here before capping;
        # Python's stable sort keeps equal-similarity pairs in analyzer order.
        pairs = sorted(
            (pair for pair in raw if not _is_exact_edge(pair)),
            key=lambda pair: -pair.similarity,
        )

    # Exact edges collapse into families that lead the primary list; each
    # family is one item against the cap, ranked by the lines it would remove.
    families = build_exact_families(exact_edges)
    duplicates_by_tier["exact"] = len(families)
    ranked: list[ExactFamily | HybridDuplicate | DuplicatePair] = [*families, *pairs]

    # The cap keeps a prefix of the report ranking established above: families,
    # then tier groups for combined mode or descending similarity for the raw
    # modes. The raw ``--show-all`` lists are diagnostic and stay complete.
    cut: list[ExactFamily | HybridDuplicate | DuplicatePair] = []
    if policy.max_duplicates is not None:
        cut = ranked[policy.max_duplicates :]
        ranked = ranked[: policy.max_duplicates]
    exact_families = [item for item in ranked if isinstance(item, ExactFamily)]
    duplicates = [item for item in ranked if not isinstance(item, ExactFamily)]
    truncated_exact_families = [item for item in cut if isinstance(item, ExactFamily)]
    truncated = [item for item in cut if not isinstance(item, ExactFamily)]
    truncated_by_tier: dict[HybridTier, int] = dict.fromkeys(HYBRID_TIERS, 0)
    truncated_by_tier.update(
        Counter(pair.tier for pair in truncated if isinstance(pair, HybridDuplicate))
    )
    truncated_by_tier["exact"] = len(truncated_exact_families)

    # Unused findings rank by size and take their own cap; units referenced
    # only by cut findings leave the report with them.
    unused = sorted(result.potentially_unused, key=unused_sort_key)
    truncated_unused: list[CodeUnit] = []
    if policy.max_unused is not None:
        truncated_unused = unused[policy.max_unused :]
        unused = unused[: policy.max_unused]

    units = collect_units(
        (unit for family in exact_families for unit in family.members),
        _pair_units(duplicates),
        _pair_units(traditional or ()),
        _pair_units(semantic or ()),
        unused,
    )
    return ReportSelection(
        result=result,
        policy=policy,
        mode=mode,
        exact_families=exact_families,
        duplicates=duplicates,  # type: ignore[arg-type]
        omitted_review=omitted_review,
        truncated_exact_families=truncated_exact_families,
        truncated=truncated,  # type: ignore[arg-type]
        duplicates_by_tier=duplicates_by_tier,
        truncated_by_tier=truncated_by_tier,
        traditional_duplicates=traditional,
        semantic_duplicates=semantic,
        potentially_unused=unused,
        truncated_unused=truncated_unused,
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
    fail_on_incomplete: bool = False,
) -> bool:
    """Return whether the complete analysis result should make ``check`` exit one.

    The verdict is computed on every finding the analyzer produced, so report
    visibility never changes the exit status.

    :param result: Completed analysis result.
    :param policy: Selected finding policy.
    :param strict_unused: Whether unused findings are strict rather than heuristic.
    :param fail_on_incomplete: Whether an incomplete analysis also fails the run,
        independent of ``policy`` (applies under ``"none"`` too).
    :return: Whether findings, or an incomplete analysis, require exit code one.
    :raises ValueError: If ``policy`` is not a supported failure policy.
    """
    if fail_on_incomplete and result.analysis_status != "complete":
        return True
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
    ``semantic_review`` pairs the primary list withheld (``review``). Findings
    cut by ``max_duplicates`` never qualify: the primary list ranks exact
    families and then actionable tiers first and the cap is at least one, so
    whenever a truncated finding fails, an emitted finding already fails too.

    :param selection: Report selection derived from the complete result.
    :param policy: Selected finding policy.
    :param strict_unused: Whether unused findings are strict rather than heuristic.
    :return: Failing hidden groups; empty when the run passes or an emitted finding already fails it.
    :raises ValueError: If ``policy`` is not a supported failure policy.
    """
    combined = selection.mode == "combined"
    if policy not in ("actionable", "all", "none"):
        raise ValueError(f"Unknown failure policy {policy!r}; expected actionable, all, or none.")
    # An emitted family is an actionable finding under every failing policy.
    if policy != "none" and selection.exact_families:
        return frozenset()
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


def _in_focus(unit: CodeUnit, paths: tuple[Path, ...]) -> bool:
    """Return whether a unit's file falls under any focus path.

    :param unit: Unit to test.
    :param paths: Resolved focus paths (files or directories); a unit under
        a focus directory or matching a focus file counts as in focus.
    :return: Whether the unit is in focus.
    """
    return any(unit.file_path.is_relative_to(path) for path in paths)


def _finding_count(pairs: Sequence[HybridDuplicate | DuplicatePair]) -> int:
    """Count findings the way a report does: an exact family counts once.

    :param pairs: Duplicate pairs, raw or hybrid.
    :return: Exact families plus every non-exact pair.
    """
    families = build_exact_families(pairs)
    non_exact = sum(not _is_exact_edge(pair) for pair in pairs)
    return len(families) + non_exact


def _focus_pairs(
    pairs: Sequence[HybridDuplicate] | Sequence[DuplicatePair],
    *,
    kept_family_uids: frozenset[str],
    paths: tuple[Path, ...],
) -> list[HybridDuplicate] | list[DuplicatePair]:
    """Filter one duplicate list to the pairs a focused report keeps.

    An exact edge is one indivisible finding with the rest of its family, so
    it is kept only when both endpoints already belong to a family that has
    at least one in-focus member (``kept_family_uids``); a non-exact pair is
    its own finding, kept when either endpoint is in focus.

    :param pairs: Duplicate pairs to filter, raw or hybrid.
    :param kept_family_uids: Uids of every member of a family with an in-focus member.
    :param paths: Resolved focus paths.
    :return: The subset of ``pairs`` a focused report keeps, in input order.
    """
    kept: list[HybridDuplicate] | list[DuplicatePair] = []
    for pair in pairs:
        if _is_exact_edge(pair):
            if pair.unit_a.uid in kept_family_uids and pair.unit_b.uid in kept_family_uids:
                kept.append(pair)  # type: ignore[arg-type]
        elif _in_focus(pair.unit_a, paths) or _in_focus(pair.unit_b, paths):
            kept.append(pair)  # type: ignore[arg-type]
    return kept


def focus_result(result: AnalysisResult, paths: tuple[Path, ...]) -> AnalysisResult:
    """Scope one complete result's findings to a set of focus paths.

    The complete result stays corpus-wide: ``units``, every diagnostics list,
    ``unused_excluded_units``, ``embedding_stats``, and ``run`` are untouched.
    Only the duplicate and unused findings a focused report emits change, so
    exit codes and JSON/terminal reports built from the returned result cover
    exactly the focus scope. An exact-duplicate family is kept whole when any
    of its members is in focus, since consolidating it is one indivisible
    finding; every other duplicate pair is kept when either endpoint is in
    focus; a potentially-unused unit is kept when its file is in focus.

    :param result: Complete analysis result.
    :param paths: Resolved, deduplicated focus paths (files or directories); must be non-empty.
    :return: A new result scoped to ``paths``, with ``focus`` set.
    """
    families = build_exact_families(result.all_duplicates)
    kept_family_uids = frozenset(
        unit.uid
        for family in families
        if any(_in_focus(member, paths) for member in family.members)
        for unit in family.members
    )

    traditional = _focus_pairs(
        result.traditional_duplicates, kept_family_uids=kept_family_uids, paths=paths
    )
    semantic = _focus_pairs(
        result.semantic_duplicates, kept_family_uids=kept_family_uids, paths=paths
    )
    hybrid = _focus_pairs(result.hybrid_duplicates, kept_family_uids=kept_family_uids, paths=paths)
    unused = [unit for unit in result.potentially_unused if _in_focus(unit, paths)]

    focused = replace(
        result,
        traditional_duplicates=traditional,
        semantic_duplicates=semantic,
        hybrid_duplicates=hybrid,
        potentially_unused=unused,
    )
    out_of_focus_duplicates = _finding_count(result.all_duplicates) - _finding_count(
        focused.all_duplicates
    )
    out_of_focus_unused = len(result.potentially_unused) - len(unused)
    focus_units = sum(1 for unit in result.units if _in_focus(unit, paths))
    return replace(
        focused,
        focus=FocusSummary(
            paths=paths,
            units=focus_units,
            out_of_focus_duplicates=out_of_focus_duplicates,
            out_of_focus_unused=out_of_focus_unused,
        ),
    )
