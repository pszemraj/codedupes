"""Report visibility selection and finding exit policy."""

from __future__ import annotations

import random
from pathlib import Path
from typing import get_args

import pytest

from codedupes.models import (
    HYBRID_TIERS,
    AnalysisResult,
    CodeUnit,
    CodeUnitType,
    DuplicatePair,
    HybridDuplicate,
    HybridTier,
)
from codedupes.report.selection import (
    ACTIONABLE_TIERS,
    DEFAULT_MAX_DUPLICATES,
    DEFAULT_MAX_UNUSED,
    ExactFamily,
    ReportPolicy,
    actionable_pairs,
    assign_unit_ids,
    build_exact_families,
    collect_units,
    hidden_only_failure,
    run_should_fail,
    select_findings,
)


def _unit(
    tmp_path: Path,
    name: str,
    *,
    file: str = "a.py",
    start_byte: int = 0,
    lines: int = 2,
    statement_count: int | None = None,
    structural_hash: str | None = None,
    token_hash: str | None = None,
) -> CodeUnit:
    return CodeUnit(
        name=name,
        qualified_name=f"mod.{name}",
        unit_type=CodeUnitType.FUNCTION,
        file_path=tmp_path / file,
        lineno=1 + start_byte // 10,
        end_lineno=start_byte // 10 + lines,
        source=f"def {name}():\n    return 1\n",
        start_byte=start_byte,
        end_byte=start_byte + 20,
        statement_count=statement_count,
        structural_hash=structural_hash,
        token_hash=token_hash,
    )


def _hybrid(unit_a: CodeUnit, unit_b: CodeUnit, tier: HybridTier) -> HybridDuplicate:
    confidence = {
        "exact": 1.0,
        "traditional_near": 0.95,
        "hybrid_confirmed": 0.93,
        "semantic_high_confidence": 0.9,
        "semantic_review": 0.8,
    }[tier]
    return HybridDuplicate(unit_a=unit_a, unit_b=unit_b, tier=tier, confidence=confidence)


def _family_names(families: list[ExactFamily]) -> list[list[str]]:
    return [[unit.name for unit in family.members] for family in families]


def _result(tmp_path: Path, **overrides) -> AnalysisResult:
    a = _unit(tmp_path, "a", start_byte=0)
    b = _unit(tmp_path, "b", start_byte=40)
    c = _unit(tmp_path, "c", file="b.py", start_byte=0)
    fields = {
        "units": [a, b, c],
        "traditional_duplicates": [DuplicatePair(a, b, 1.0, "structural_hash")],
        "semantic_duplicates": [
            DuplicatePair(a, b, 0.95, "semantic"),
            DuplicatePair(b, c, 0.85, "semantic"),
        ],
        "hybrid_duplicates": [
            _hybrid(a, b, "exact"),
            _hybrid(b, c, "semantic_review"),
        ],
        "potentially_unused": [],
        "analysis_mode": "combined",
    }
    fields.update(overrides)
    return AnalysisResult(**fields)


def test_hybrid_tiers_matches_literal_and_contains_actionable_tiers():
    assert HYBRID_TIERS == get_args(HybridTier)
    assert ACTIONABLE_TIERS <= set(HYBRID_TIERS)


def test_select_findings_hides_review_by_default_without_touching_result(tmp_path):
    result = _result(tmp_path)

    selection = select_findings(result)

    # The exact edge is reported as a family, never as a pair.
    assert selection.duplicates == []
    assert _family_names(selection.exact_families) == [["a", "b"]]
    assert selection.exact_families[0].method == "structural_hash"
    assert selection.truncated_exact_families == []
    assert [pair.tier for pair in selection.omitted_review] == ["semantic_review"]
    assert selection.duplicates_by_tier == {
        "exact": 1,
        "traditional_near": 0,
        "hybrid_confirmed": 0,
        "semantic_high_confidence": 0,
        "semantic_review": 1,
    }
    assert selection.traditional_duplicates is None
    assert selection.semantic_duplicates is None
    assert len(result.hybrid_duplicates) == 2


def test_select_findings_include_review_preserves_analyzer_order(tmp_path):
    result = _result(tmp_path)

    selection = select_findings(result, ReportPolicy(include_review=True))

    assert selection.duplicates == result.hybrid_duplicates[1:]
    assert _family_names(selection.exact_families) == [["a", "b"]]
    assert selection.omitted_review == []
    assert selection.traditional_duplicates is None


def test_select_findings_show_all_implies_review_and_raw_lists(tmp_path):
    result = _result(tmp_path)

    selection = select_findings(result, ReportPolicy(show_all=True))

    assert selection.policy.shows_review
    assert selection.duplicates == result.hybrid_duplicates[1:]
    assert _family_names(selection.exact_families) == [["a", "b"]]
    assert selection.traditional_duplicates == result.traditional_duplicates
    assert selection.semantic_duplicates == result.semantic_duplicates


def test_select_findings_single_method_has_no_tier_filter(tmp_path):
    result = _result(tmp_path, analysis_mode="semantic", hybrid_duplicates=[])

    selection = select_findings(result, ReportPolicy(show_all=True))

    # Raw modes group exact edges too; every other pair is listed without a tier.
    assert selection.duplicates == result.semantic_duplicates
    assert _family_names(selection.exact_families) == [["a", "b"]]
    assert selection.omitted_review == []
    assert selection.duplicates_by_tier == dict.fromkeys(HYBRID_TIERS, 0) | {"exact": 1}
    assert selection.traditional_duplicates is None
    assert selection.semantic_duplicates is None


def test_select_findings_none_mode_is_empty(tmp_path):
    result = AnalysisResult(
        units=[],
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[],
        potentially_unused=[],
        analysis_mode="none",
    )

    selection = select_findings(result)

    assert selection.duplicates == []
    assert selection.units == []
    assert set(selection.duplicates_by_tier) == set(HYBRID_TIERS)


def test_collect_units_dedupes_and_sorts_by_file_then_offset(tmp_path):
    later = _unit(tmp_path, "later", file="a.py", start_byte=90)
    first = _unit(tmp_path, "first", file="a.py", start_byte=5)
    other = _unit(tmp_path, "other", file="b.py", start_byte=0)

    units = collect_units([later, other], [first, later], [first])

    assert [unit.name for unit in units] == ["first", "later", "other"]


def test_select_findings_omits_units_referenced_only_by_hidden_review(tmp_path):
    result = _result(tmp_path)

    hidden = select_findings(result)
    shown = select_findings(result, ReportPolicy(include_review=True))

    assert [unit.name for unit in hidden.units] == ["a", "b"]
    assert [unit.name for unit in shown.units] == ["a", "b", "c"]


def test_select_findings_includes_unused_units(tmp_path):
    orphan = _unit(tmp_path, "orphan", file="z.py")
    result = _result(tmp_path, potentially_unused=[orphan])

    selection = select_findings(result)

    assert selection.potentially_unused == [orphan]
    assert selection.truncated_unused == []
    assert selection.units[-1] is orphan


def test_select_findings_ranks_unused_by_span_then_statements_then_position(tmp_path):
    # Walk order is z.py, y.py, x.py, w.py; the report order is by size instead.
    long_thin = _unit(tmp_path, "long_thin", file="z.py", lines=40, statement_count=2)
    dense = _unit(tmp_path, "dense", file="y.py", lines=12, statement_count=11)
    sparse = _unit(tmp_path, "sparse", file="x.py", lines=12, statement_count=3)
    tie_later = _unit(
        tmp_path, "tie_later", file="w.py", start_byte=50, lines=12, statement_count=3
    )
    tie_first = _unit(tmp_path, "tie_first", file="w.py", start_byte=0, lines=12, statement_count=3)
    tiny = _unit(tmp_path, "tiny", file="v.py", lines=1)
    result = _result(
        tmp_path,
        potentially_unused=[long_thin, dense, sparse, tie_later, tie_first, tiny],
    )

    selection = select_findings(result)

    assert [unit.name for unit in selection.potentially_unused] == [
        "long_thin",  # 40 lines beats everything, two statements notwithstanding
        "dense",  # 12 lines, 11 statements
        "tie_first",  # 12 lines, 3 statements, w.py offset 0
        "tie_later",  # 12 lines, 3 statements, w.py offset 50
        "sparse",  # 12 lines, 3 statements, x.py
        "tiny",
    ]
    assert result.potentially_unused[0] is long_thin  # the result is untouched


def test_max_unused_caps_the_unused_list_and_drops_their_units(tmp_path):
    assert ReportPolicy().max_unused is None
    assert DEFAULT_MAX_UNUSED == 20
    unused = [_unit(tmp_path, f"dead{i}", file=f"d{i}.py", lines=30 - i) for i in range(25)]
    result = _result(tmp_path, potentially_unused=list(reversed(unused)))

    complete = select_findings(result)
    concise = select_findings(result, ReportPolicy(max_unused=DEFAULT_MAX_UNUSED))
    tight = select_findings(result, ReportPolicy(max_unused=3))

    assert [unit.name for unit in complete.potentially_unused] == [f"dead{i}" for i in range(25)]
    assert complete.truncated_unused == []
    assert [unit.name for unit in concise.potentially_unused] == [f"dead{i}" for i in range(20)]
    assert [unit.name for unit in concise.truncated_unused] == [f"dead{i}" for i in range(20, 25)]
    # Units referenced only by cut unused findings leave the report with them.
    assert {unit.name for unit in tight.units} == {"a", "b", "dead0", "dead1", "dead2"}
    assert len(tight.truncated_unused) == 22
    # The exit code still sees every unused finding.
    assert run_should_fail(result, policy="all", strict_unused=True) is True


def test_hidden_only_failure_is_all_or_nothing_under_an_unused_cap(tmp_path):
    a = _unit(tmp_path, "a")
    b = _unit(tmp_path, "b", start_byte=40)
    unused = [_unit(tmp_path, f"dead{i}", file=f"d{i}.py") for i in range(5)]
    result = _result(
        tmp_path,
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[_hybrid(a, b, "semantic_review")],
        potentially_unused=unused,
    )

    capped = select_findings(result, ReportPolicy(max_unused=1))

    assert len(capped.potentially_unused) == 1
    # One emitted unused finding already fails a policy that counts unused
    # (``all`` always does; ``actionable`` only when strict), so the withheld
    # review pair is never the only explanation.
    assert hidden_only_failure(capped, policy="all", strict_unused=True) == set()
    assert hidden_only_failure(capped, policy="all", strict_unused=False) == set()
    assert hidden_only_failure(capped, policy="actionable", strict_unused=True) == set()
    assert hidden_only_failure(capped, policy="actionable", strict_unused=False) == set()


def test_assign_unit_ids_ignores_edge_order(tmp_path):
    result = _result(tmp_path)
    shuffled = _result(tmp_path)
    random.Random(0).shuffle(shuffled.hybrid_duplicates)
    policy = ReportPolicy(include_review=True)

    ids = assign_unit_ids(select_findings(result, policy).units)
    shuffled_ids = assign_unit_ids(select_findings(shuffled, policy).units)

    assert ids == shuffled_ids
    assert list(ids.values()) == ["u0", "u1", "u2"]


def test_run_should_fail_uses_result_analysis_mode(tmp_path):
    a = _unit(tmp_path, "a")
    b = _unit(tmp_path, "b", start_byte=40)
    raw = [DuplicatePair(a, b, 0.9, "jaccard")]
    combined = _result(tmp_path, traditional_duplicates=raw, hybrid_duplicates=[])
    single = _result(
        tmp_path, traditional_duplicates=raw, hybrid_duplicates=[], analysis_mode="traditional"
    )

    assert run_should_fail(combined, policy="actionable", strict_unused=False) is False
    assert run_should_fail(single, policy="actionable", strict_unused=False) is True


@pytest.mark.parametrize("policy", ["actonable", "nonee"])
@pytest.mark.parametrize("has_findings", [False, True])
def test_failure_helpers_reject_unknown_policies(tmp_path, policy, has_findings):
    result = _result(tmp_path)
    if not has_findings:
        result.hybrid_duplicates.clear()

    with pytest.raises(ValueError, match="Unknown failure policy"):
        run_should_fail(result, policy=policy, strict_unused=False)
    with pytest.raises(ValueError, match="Unknown failure policy"):
        hidden_only_failure(select_findings(result), policy=policy, strict_unused=False)


def test_hidden_only_failure_names_withheld_review(tmp_path):
    a = _unit(tmp_path, "a")
    b = _unit(tmp_path, "b", start_byte=40)
    review_only = _result(
        tmp_path,
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[_hybrid(a, b, "semantic_review")],
    )
    selection = select_findings(review_only)

    assert hidden_only_failure(selection, policy="all", strict_unused=False) == {"review"}
    assert hidden_only_failure(selection, policy="actionable", strict_unused=False) == set()
    assert hidden_only_failure(selection, policy="none", strict_unused=False) == set()

    # An emitted failing pair explains the exit code on its own.
    with_visible = _result(tmp_path)
    assert (
        hidden_only_failure(select_findings(with_visible), policy="all", strict_unused=False)
        == set()
    )

    with_unused = _result(
        tmp_path,
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[_hybrid(a, b, "semantic_review")],
        potentially_unused=[a],
    )
    assert (
        hidden_only_failure(select_findings(with_unused), policy="all", strict_unused=True) == set()
    )

    shown = select_findings(review_only, ReportPolicy(include_review=True))
    assert hidden_only_failure(shown, policy="all", strict_unused=False) == set()


def _ranked_result(tmp_path: Path, tiers: list[HybridTier]) -> AnalysisResult:
    """Build a combined result whose hybrid list is ``tiers`` in analyzer order."""
    units = [_unit(tmp_path, f"f{i}", start_byte=i * 30) for i in range(len(tiers) + 1)]
    hybrid = [
        HybridDuplicate(units[i], units[i + 1], tier, confidence=1.0 - i * 0.01)
        for i, tier in enumerate(tiers)
    ]
    return _result(
        tmp_path,
        units=units,
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=hybrid,
    )


def test_max_duplicates_keeps_a_prefix_after_the_review_filter(tmp_path):
    result = _ranked_result(
        tmp_path,
        ["exact", "semantic_review", "hybrid_confirmed", "semantic_high_confidence"],
    )

    capped = select_findings(result, ReportPolicy(max_duplicates=2))

    assert _family_names(capped.exact_families) == [["f0", "f1"]]
    assert [pair.tier for pair in capped.duplicates] == ["hybrid_confirmed"]
    assert [pair.tier for pair in capped.truncated] == ["semantic_high_confidence"]
    assert [pair.tier for pair in capped.omitted_review] == ["semantic_review"]
    assert capped.reported_findings + len(capped.omitted_review) + capped.truncated_findings == 4
    assert capped.total_findings == 4
    # The tier breakdown still describes the complete result; the truncated
    # breakdown is zero-filled over every tier like it.
    assert capped.duplicates_by_tier["semantic_high_confidence"] == 1
    assert capped.truncated_by_tier == {
        "exact": 0,
        "traditional_near": 0,
        "hybrid_confirmed": 0,
        "semantic_high_confidence": 1,
        "semantic_review": 0,
    }
    # Units referenced only by truncated pairs drop out with them.
    assert [unit.name for unit in capped.units] == ["f0", "f1", "f2", "f3"]

    # Review pairs rank last, so including them changes the truncated set, not
    # the emitted prefix.
    with_review = select_findings(result, ReportPolicy(include_review=True, max_duplicates=2))
    assert _family_names(with_review.exact_families) == [["f0", "f1"]]
    assert [pair.tier for pair in with_review.duplicates] == ["hybrid_confirmed"]
    assert with_review.omitted_review == []
    assert [pair.tier for pair in with_review.truncated] == [
        "semantic_high_confidence",
        "semantic_review",
    ]
    # With nothing withheld, the truncated breakdown is the only place that
    # says the included review pair was cut rather than shown.
    assert with_review.truncated_by_tier["semantic_review"] == 1
    assert with_review.truncated_by_tier["semantic_high_confidence"] == 1


def test_select_findings_ranks_actionable_tiers_first_within_analyzer_order(tmp_path):
    tiers: list[HybridTier] = [
        "semantic_high_confidence",
        "hybrid_confirmed",
        "semantic_review",
        "traditional_near",
        "semantic_high_confidence",
    ]
    result = _ranked_result(tmp_path, tiers)

    selection = select_findings(result)

    assert [pair.tier for pair in selection.duplicates] == [
        "hybrid_confirmed",
        "traditional_near",
        "semantic_high_confidence",
        "semantic_high_confidence",
    ]
    # Analyzer (confidence) order survives inside each group.
    assert [pair.unit_a.name for pair in selection.duplicates] == ["f1", "f3", "f0", "f4"]
    assert [pair.tier for pair in selection.omitted_review] == ["semantic_review"]

    with_review = select_findings(result, ReportPolicy(include_review=True))
    assert [pair.tier for pair in with_review.duplicates] == [
        "hybrid_confirmed",
        "traditional_near",
        "semantic_high_confidence",
        "semantic_high_confidence",
        "semantic_review",
    ]
    # The complete result keeps the analyzer's ranking.
    assert [pair.tier for pair in result.hybrid_duplicates] == tiers


def test_actionable_pairs_filters_tiers_only_in_combined_mode(tmp_path):
    result = _ranked_result(
        tmp_path, ["semantic_high_confidence", "exact", "semantic_review", "traditional_near"]
    )

    combined = actionable_pairs(result.hybrid_duplicates, combined=True)
    assert [pair.tier for pair in combined] == ["exact", "traditional_near"]

    raw = [DuplicatePair(result.units[0], result.units[1], 0.9, "semantic")]
    assert actionable_pairs(raw, combined=False) == raw


def test_report_policy_default_is_uncapped_and_cli_default_is_twenty(tmp_path):
    assert ReportPolicy().max_duplicates is None
    assert DEFAULT_MAX_DUPLICATES == 20
    result = _ranked_result(tmp_path, ["hybrid_confirmed"] * 25)

    complete = select_findings(result)
    concise = select_findings(result, ReportPolicy(max_duplicates=DEFAULT_MAX_DUPLICATES))

    assert len(complete.duplicates) == 25
    assert len(concise.duplicates) == 20
    assert len(concise.truncated) == 5


def test_max_duplicates_is_a_no_op_at_or_above_the_admitted_count(tmp_path):
    result = _ranked_result(tmp_path, ["exact", "hybrid_confirmed"])

    exact = select_findings(result, ReportPolicy(max_duplicates=2))
    generous = select_findings(result, ReportPolicy(max_duplicates=50))

    assert exact.truncated == generous.truncated == []
    assert exact.truncated_exact_families == generous.truncated_exact_families == []
    assert exact.duplicates == generous.duplicates == result.hybrid_duplicates[1:]
    assert _family_names(exact.exact_families) == _family_names(generous.exact_families)
    assert _family_names(exact.exact_families) == [["f0", "f1"]]


def test_max_duplicates_leaves_show_all_raw_lists_complete(tmp_path):
    result = _result(tmp_path)

    selection = select_findings(result, ReportPolicy(show_all=True, max_duplicates=1))

    # The family takes the single slot; the review pair is cut, not the family.
    assert len(selection.exact_families) == 1
    assert selection.duplicates == []
    assert len(selection.truncated) == 1
    assert selection.reported_findings == 1
    assert selection.traditional_duplicates == result.traditional_duplicates
    assert selection.semantic_duplicates == result.semantic_duplicates


def test_max_duplicates_ranks_traditional_only_mode_by_similarity(tmp_path):
    # near_dupes come out of the analyzer sorted by index pair, not similarity,
    # so the raw list must be re-ranked before the cap is applied.
    a = _unit(tmp_path, "a", start_byte=0)
    b = _unit(tmp_path, "b", start_byte=20)
    c = _unit(tmp_path, "c", start_byte=40)
    d = _unit(tmp_path, "d", start_byte=60)
    e = _unit(tmp_path, "e", start_byte=80)
    f = _unit(tmp_path, "f", start_byte=100)
    exact = DuplicatePair(a, b, 1.0, "structural_hash")
    near_086 = DuplicatePair(b, c, 0.86, "jaccard")
    near_099 = DuplicatePair(c, d, 0.99, "jaccard")
    near_090_first = DuplicatePair(d, e, 0.90, "jaccard")
    near_090_second = DuplicatePair(e, f, 0.90, "jaccard")
    result = _result(
        tmp_path,
        traditional_duplicates=[exact, near_086, near_099, near_090_first, near_090_second],
        semantic_duplicates=[],
        hybrid_duplicates=[],
        analysis_mode="traditional",
    )

    capped = select_findings(result, ReportPolicy(max_duplicates=2))

    # The exact pair is a family and leads; the near pairs follow by similarity.
    assert _family_names(capped.exact_families) == [["a", "b"]]
    assert capped.duplicates == [near_099]
    # Truncated holds the rest, still ranked by descending similarity, and the
    # two 0.90 pairs keep their input order (stable sort over the tie).
    assert capped.truncated == [near_090_first, near_090_second, near_086]
    assert capped.duplicates_by_tier["exact"] == 1


def test_max_duplicates_applies_to_single_method_raw_lists(tmp_path):
    result = _result(tmp_path, analysis_mode="semantic", hybrid_duplicates=[])

    selection = select_findings(result, ReportPolicy(max_duplicates=1))

    assert _family_names(selection.exact_families) == [["a", "b"]]
    assert selection.duplicates == []
    assert selection.truncated == result.semantic_duplicates
    assert run_should_fail(result, policy="actionable", strict_unused=False) is True


@pytest.mark.parametrize("cap", [0, -1])
def test_report_policy_rejects_a_cap_that_emits_nothing(cap):
    with pytest.raises(ValueError, match="max_duplicates must be at least 1"):
        ReportPolicy(max_duplicates=cap)
    with pytest.raises(ValueError, match="max_unused must be at least 1"):
        ReportPolicy(max_unused=cap)


def test_build_exact_families_unions_per_method_and_labels_the_strongest_fingerprint(tmp_path):
    # Three structural-labelled edges over a copy-pasted trio (a clique from the
    # analyzer), one renamed pair, one token-only pair, and a self-edge.
    copies = [
        _unit(tmp_path, f"copy{i}", file=f"m{i}.py", structural_hash="s1", token_hash="t1")
        for i in range(3)
    ]
    renamed_a = _unit(tmp_path, "renamed_a", start_byte=100, structural_hash="s2", token_hash="t2")
    renamed_b = _unit(tmp_path, "renamed_b", start_byte=200, structural_hash="s2", token_hash="t3")
    indent_a = _unit(tmp_path, "indent_a", start_byte=300, structural_hash="s4", token_hash="t4")
    indent_b = _unit(tmp_path, "indent_b", start_byte=400, structural_hash="s5", token_hash="t4")
    edges = [
        DuplicatePair(copies[0], copies[1], 1.0, "structural_hash"),
        DuplicatePair(copies[1], copies[2], 1.0, "structural_hash"),
        DuplicatePair(copies[0], copies[2], 1.0, "structural_hash"),
        DuplicatePair(renamed_a, renamed_b, 1.0, "structural_hash"),
        DuplicatePair(indent_a, indent_b, 1.0, "token_hash"),
        DuplicatePair(renamed_a, renamed_a, 1.0, "structural_hash"),
        DuplicatePair(copies[0], renamed_a, 0.9, "jaccard"),
    ]

    families = build_exact_families(edges)

    assert _family_names(families) == [
        ["copy0", "copy1", "copy2"],
        ["renamed_a", "renamed_b"],
        ["indent_a", "indent_b"],
    ]
    assert [family.method for family in families] == [
        "token_hash",
        "structural_hash",
        "token_hash",
    ]
    assert [family.pair_count for family in families] == [3, 1, 1]
    assert [family.redundant_lines for family in families] == [4, 2, 2]

    # Hybrid exact edges group the same way, keyed on the recorded method; a
    # hand-built hybrid without one counts as structural.
    hybrid = build_exact_families(
        [
            HybridDuplicate(copies[0], copies[1], "exact", 1.0, exact_method="structural_hash"),
            HybridDuplicate(indent_a, indent_b, "exact", 1.0, exact_method="token_hash"),
            HybridDuplicate(renamed_a, renamed_b, "exact", 1.0),
            _hybrid(renamed_a, indent_a, "semantic_review"),
        ]
    )
    # Equal redundant lines fall back to the first member's position (a.py first).
    assert _family_names(hybrid) == [
        ["renamed_a", "renamed_b"],
        ["indent_a", "indent_b"],
        ["copy0", "copy1"],
    ]
    assert [family.method for family in hybrid] == ["structural_hash", "token_hash", "token_hash"]


def test_families_rank_first_by_redundant_lines_then_position(tmp_path):
    # Two copies of a 30-line function beat five copies of a 6-line helper
    # (30 vs 24 redundant lines); equal keys fall back to first-member position.
    big = [_unit(tmp_path, f"big{i}", file=f"b{i}.py", lines=30) for i in range(2)]
    small = [_unit(tmp_path, f"small{i}", file=f"s{i}.py", lines=6) for i in range(5)]
    tie_a = [_unit(tmp_path, f"tie_a{i}", file=f"a{i}.py", lines=20) for i in range(2)]
    tie_z = [_unit(tmp_path, f"tie_z{i}", file=f"z{i}.py", lines=20) for i in range(2)]
    hybrid = [
        HybridDuplicate(tie_z[0], tie_z[1], "exact", 1.0),
        HybridDuplicate(small[0], small[4], "exact", 1.0),
        HybridDuplicate(big[0], big[1], "exact", 1.0),
        HybridDuplicate(tie_a[0], tie_a[1], "exact", 1.0),
        _hybrid(big[0], small[0], "hybrid_confirmed"),
    ]
    for i in range(4):
        hybrid.append(HybridDuplicate(small[i], small[i + 1], "exact", 1.0))
    result = _result(
        tmp_path,
        units=big + small + tie_a + tie_z,
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=hybrid,
    )

    selection = select_findings(result)

    assert [len(family.members) for family in selection.exact_families] == [2, 5, 2, 2]
    assert [family.redundant_lines for family in selection.exact_families] == [30, 24, 20, 20]
    assert [family.members[0].name for family in selection.exact_families] == [
        "big0",
        "small0",
        "tie_a0",
        "tie_z0",
    ]
    assert [pair.tier for pair in selection.duplicates] == ["hybrid_confirmed"]


def test_max_duplicates_counts_a_family_as_one_finding(tmp_path):
    copies = [_unit(tmp_path, f"c{i}", file=f"c{i}.py") for i in range(5)]
    clique = [
        HybridDuplicate(copies[i], copies[j], "exact", 1.0)
        for i in range(5)
        for j in range(i + 1, 5)
    ]
    others = [_unit(tmp_path, f"o{i}", file=f"o{i}.py") for i in range(4)]
    pairs = [
        _hybrid(others[0], others[1], "hybrid_confirmed"),
        _hybrid(others[2], others[3], "semantic_high_confidence"),
    ]
    result = _result(
        tmp_path,
        units=copies + others,
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=clique + pairs,
    )

    capped = select_findings(result, ReportPolicy(max_duplicates=2))

    # Ten exact edges are one family, leaving one slot for the strongest pair.
    assert _family_names(capped.exact_families) == [[f"c{i}" for i in range(5)]]
    assert capped.exact_families[0].pair_count == 10
    assert [pair.tier for pair in capped.duplicates] == ["hybrid_confirmed"]
    assert [pair.tier for pair in capped.truncated] == ["semantic_high_confidence"]
    assert capped.duplicates_by_tier["exact"] == 1
    assert capped.truncated_by_tier["exact"] == 0
    assert capped.total_findings == 3
    assert capped.reported_findings == 2
    assert capped.actionable_findings == 2
    assert capped.reported_actionable_findings == 2
    assert capped.exact_family_members == 5
    # Family members are in the report units; the truncated pair's are not.
    assert [unit.name for unit in capped.units] == [f"c{i}" for i in range(5)] + ["o0", "o1"]

    # A cap of one keeps the family and cuts every pair; the cut family count
    # lands in truncated_by_tier.exact when a second family is squeezed out.
    tight = select_findings(result, ReportPolicy(max_duplicates=1))
    assert _family_names(tight.exact_families) == [[f"c{i}" for i in range(5)]]
    assert tight.duplicates == []
    assert tight.truncated_findings == 2
    two_families = _result(
        tmp_path,
        units=copies + others,
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=clique + [HybridDuplicate(others[0], others[1], "exact", 1.0)],
    )
    squeezed = select_findings(two_families, ReportPolicy(max_duplicates=1))
    assert _family_names(squeezed.truncated_exact_families) == [["o0", "o1"]]
    assert squeezed.truncated_by_tier["exact"] == 1
    assert squeezed.duplicates_by_tier["exact"] == 2
    assert squeezed.exact_family_members == 7


def test_hidden_only_failure_treats_a_kept_family_as_failing(tmp_path):
    a = _unit(tmp_path, "a")
    b = _unit(tmp_path, "b", start_byte=40)
    c = _unit(tmp_path, "c", start_byte=80)
    result = _result(
        tmp_path,
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[_hybrid(a, b, "exact"), _hybrid(b, c, "semantic_review")],
    )
    selection = select_findings(result)

    assert selection.duplicates == []
    assert len(selection.exact_families) == 1
    # The family, not the withheld review pair, explains the failure.
    assert hidden_only_failure(selection, policy="all", strict_unused=False) == set()
    assert hidden_only_failure(selection, policy="actionable", strict_unused=False) == set()
    assert hidden_only_failure(selection, policy="none", strict_unused=False) == set()


def test_truncation_never_hides_the_only_failing_pair(tmp_path):
    # Confidence is only tier-monotone at equal similarity, so a strong
    # semantic_high_confidence pair outranks an actionable hybrid_confirmed one
    # in the analyzer; the report ranks the actionable pair first so a cap of
    # one still emits the pair that fails the default policy.
    result = _ranked_result(
        tmp_path, ["semantic_high_confidence", "hybrid_confirmed", "semantic_review"]
    )
    capped = select_findings(result, ReportPolicy(max_duplicates=1))

    assert run_should_fail(result, policy="actionable", strict_unused=False) is True
    assert [pair.tier for pair in capped.duplicates] == ["hybrid_confirmed"]
    assert [pair.tier for pair in capped.truncated] == ["semantic_high_confidence"]
    assert hidden_only_failure(capped, policy="actionable", strict_unused=False) == set()
    assert hidden_only_failure(capped, policy="all", strict_unused=False) == set()

    # Truncating only advisory pairs hides nothing that fails.
    advisory = select_findings(
        _ranked_result(tmp_path, ["semantic_high_confidence", "semantic_high_confidence"]),
        ReportPolicy(max_duplicates=1),
    )
    assert run_should_fail(advisory.result, policy="actionable", strict_unused=False) is False
    assert hidden_only_failure(advisory, policy="actionable", strict_unused=False) == set()
