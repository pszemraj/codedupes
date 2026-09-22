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
    ExtractionDiagnostic,
    FocusSummary,
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
    focus_result,
    hidden_only_failure,
    run_should_fail,
    select_findings,
)
from tests.conftest import make_run_record


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
    score = {
        "exact": 1.0,
        "traditional_near": 0.95,
        "hybrid_confirmed": 0.93,
        "semantic_high_confidence": 0.9,
        "semantic_review": 0.8,
    }[tier]
    return HybridDuplicate(unit_a=unit_a, unit_b=unit_b, tier=tier, score=score)


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
        "run": make_run_record(tmp_path, mode="combined"),
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
    result = _result(tmp_path, run=make_run_record(tmp_path, mode="semantic"), hybrid_duplicates=[])

    selection = select_findings(result, ReportPolicy(show_all=True))

    # Raw modes group exact edges too; every other pair is listed without a tier.
    assert selection.duplicates == result.semantic_duplicates
    assert _family_names(selection.exact_families) == [["a", "b"]]
    assert selection.omitted_review == []
    assert selection.duplicates_by_tier == dict.fromkeys(HYBRID_TIERS, 0) | {"exact": 1}
    assert selection.traditional_duplicates is None
    assert selection.semantic_duplicates is None


def test_select_findings_unused_mode_has_no_duplicates(tmp_path):
    result = AnalysisResult(
        units=[],
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[],
        potentially_unused=[],
        run=make_run_record(tmp_path, mode="unused"),
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
        tmp_path,
        traditional_duplicates=raw,
        hybrid_duplicates=[],
        run=make_run_record(tmp_path, mode="traditional"),
    )

    assert run_should_fail(combined, policy="actionable", strict_unused=False) is False
    assert run_should_fail(single, policy="actionable", strict_unused=False) is True


def test_run_should_fail_fail_on_incomplete_ignores_policy(tmp_path):
    partial = _result(
        tmp_path,
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[],
        potentially_unused=[],
        semantic_fallback=True,
    )
    assert partial.analysis_status == "partial"

    for policy in ("actionable", "all", "none"):
        assert run_should_fail(partial, policy=policy, strict_unused=False) is False
        assert (
            run_should_fail(partial, policy=policy, strict_unused=False, fail_on_incomplete=True)
            is True
        )

    complete = _result(
        tmp_path,
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[],
        potentially_unused=[],
    )
    assert complete.analysis_status == "complete"
    assert (
        run_should_fail(complete, policy="actionable", strict_unused=False, fail_on_incomplete=True)
        is False
    )


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
        HybridDuplicate(units[i], units[i + 1], tier, score=1.0 - i * 0.01)
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
    # Analyzer (score) order survives inside each group.
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
        run=make_run_record(tmp_path, mode="traditional"),
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
    result = _result(tmp_path, run=make_run_record(tmp_path, mode="semantic"), hybrid_duplicates=[])

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


def test_build_exact_families_folds_token_cliques_into_their_structural_family(tmp_path):
    # A token clique with no structural edge among its members: its own
    # token_hash family.
    clique = [_unit(tmp_path, f"clique{i}", file=f"clique{i}.py") for i in range(3)]
    clique_edges = [
        DuplicatePair(clique[0], clique[1], 1.0, "token_hash"),
        DuplicatePair(clique[1], clique[2], 1.0, "token_hash"),
    ]

    # A renamed pair is structural-only: its own structural_hash family.
    renamed_a = _unit(tmp_path, "renamed_a", file="renamed.py", start_byte=0)
    renamed_b = _unit(tmp_path, "renamed_b", file="renamed.py", start_byte=100)
    renamed_edge = DuplicatePair(renamed_a, renamed_b, 1.0, "structural_hash")

    # An indentation pair is token-only with no structural counterpart at
    # all: its own token_hash family, same as an isolated token clique.
    indent_a = _unit(tmp_path, "indent_a", file="indent.py", start_byte=0)
    indent_b = _unit(tmp_path, "indent_b", file="indent.py", start_byte=100)
    indent_edge = DuplicatePair(indent_a, indent_b, 1.0, "token_hash")

    # Token-identical twins plus a structurally-equal (renamed) cousin: the
    # token clique's uid set is a subset of the larger structural component,
    # so it folds in instead of keeping its own record, and the family is
    # structural_hash because not every member shares a token fingerprint.
    twin_a = _unit(tmp_path, "twin_a", file="nested.py", start_byte=0)
    twin_b = _unit(tmp_path, "twin_b", file="nested.py", start_byte=100)
    cousin = _unit(tmp_path, "cousin", file="nested.py", start_byte=200)
    nested_edges = [
        DuplicatePair(twin_a, twin_b, 1.0, "token_hash"),
        DuplicatePair(twin_a, cousin, 1.0, "structural_hash"),
        DuplicatePair(twin_b, cousin, 1.0, "structural_hash"),
    ]

    # A token edge and a structural edge sharing one endpoint but not the
    # other stay two families: Python indentation can make X and Y
    # token-equal without making them structurally equal, so the token
    # component need not nest inside the structural component its shared
    # member also belongs to.
    chain_x = _unit(tmp_path, "chain_x", file="chain.py", start_byte=0)
    chain_y = _unit(tmp_path, "chain_y", file="chain.py", start_byte=100)
    chain_z = _unit(tmp_path, "chain_z", file="chain.py", start_byte=200)
    chain_edges = [
        DuplicatePair(chain_x, chain_y, 1.0, "token_hash"),
        DuplicatePair(chain_y, chain_z, 1.0, "structural_hash"),
    ]

    # A self-edge and a jaccard (non-exact) edge never contribute a family.
    ignored_edges = [
        DuplicatePair(renamed_a, renamed_a, 1.0, "structural_hash"),
        DuplicatePair(clique[0], indent_a, 0.9, "jaccard"),
    ]

    families = build_exact_families(
        [
            *clique_edges,
            renamed_edge,
            indent_edge,
            *nested_edges,
            *chain_edges,
            *ignored_edges,
        ]
    )

    by_members = {
        tuple(sorted(unit.name for unit in family.members)): family.method for family in families
    }
    assert by_members == {
        ("clique0", "clique1", "clique2"): "token_hash",
        ("renamed_a", "renamed_b"): "structural_hash",
        ("indent_a", "indent_b"): "token_hash",
        ("cousin", "twin_a", "twin_b"): "structural_hash",
        ("chain_x", "chain_y"): "token_hash",
        ("chain_y", "chain_z"): "structural_hash",
    }

    # A hand-built hybrid exact edge with no recorded exact_method counts as
    # structural, the strongest fingerprint it can be assumed to share.
    hybrid = build_exact_families([HybridDuplicate(renamed_a, renamed_b, "exact", 1.0)])
    assert [family.method for family in hybrid] == ["structural_hash"]


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


def test_focus_result_keeps_whole_families_touching_focus(tmp_path):
    # a, b live in a.py and form the exact family; c lives in b.py. Focusing
    # on b.py touches no member of the {a, b} family, so the whole family
    # drops even though it never overlaps the focus path directly.
    result = _result(tmp_path)
    focus_path = tmp_path / "b.py"

    focused = focus_result(result, (focus_path,))

    assert focused.traditional_duplicates == []
    assert [pair.tier for pair in focused.hybrid_duplicates] == ["semantic_review"]
    assert focused.semantic_duplicates == [result.semantic_duplicates[1]]
    assert focused.focus == FocusSummary(
        paths=(focus_path,), units=1, out_of_focus_duplicates=1, out_of_focus_unused=0
    )
    # The complete result stays corpus-wide.
    assert focused.units == result.units
    assert focused.run is result.run


def test_focus_result_filters_unused_and_leaves_units_and_diagnostics(tmp_path):
    diagnostic = ExtractionDiagnostic(
        file_path=tmp_path / "a.py", language="python", message="warn"
    )
    out_of_focus_unit = _unit(tmp_path, "unused_a", file="a.py", start_byte=200)
    in_focus_unit = _unit(tmp_path, "unused_d", file="b.py", start_byte=200)
    result = _result(
        tmp_path,
        potentially_unused=[out_of_focus_unit, in_focus_unit],
        extraction_diagnostics=[diagnostic],
        unused_excluded_units=2,
    )

    focused = focus_result(result, (tmp_path / "b.py",))

    assert focused.potentially_unused == [in_focus_unit]
    assert focused.focus.out_of_focus_unused == 1
    # Diagnostics, the corpus, and exclusion counts stay corpus-wide.
    assert focused.units == result.units
    assert focused.extraction_diagnostics == [diagnostic]
    assert focused.unused_excluded_units == 2
    assert focused.run is result.run


def test_focus_result_directory_focus_matches_descendants(tmp_path):
    nested = _unit(tmp_path, "nested", file="pkg/sub/nested.py")
    outside = _unit(tmp_path, "outside", file="other.py")
    result = _result(
        tmp_path,
        units=[nested, outside],
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[],
        potentially_unused=[nested, outside],
    )

    focused = focus_result(result, (tmp_path / "pkg",))

    assert focused.potentially_unused == [nested]
    assert focused.focus.units == 1


def test_focus_result_counts_reconcile_with_select_findings(tmp_path):
    result = _result(tmp_path)
    focused = focus_result(result, (tmp_path / "b.py",))

    complete_selection = select_findings(result)
    focused_selection = select_findings(focused)

    assert (
        complete_selection.total_findings
        == focused_selection.total_findings + focused.focus.out_of_focus_duplicates
    )
