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
    ReportPolicy,
    assign_unit_ids,
    collect_units,
    hidden_only_failure,
    run_should_fail,
    select_findings,
)


def _unit(tmp_path: Path, name: str, *, file: str = "a.py", start_byte: int = 0) -> CodeUnit:
    return CodeUnit(
        name=name,
        qualified_name=f"mod.{name}",
        unit_type=CodeUnitType.FUNCTION,
        file_path=tmp_path / file,
        lineno=1 + start_byte // 10,
        end_lineno=2 + start_byte // 10,
        source=f"def {name}():\n    return 1\n",
        start_byte=start_byte,
        end_byte=start_byte + 20,
    )


def _hybrid(unit_a: CodeUnit, unit_b: CodeUnit, tier: HybridTier) -> HybridDuplicate:
    confidence = {"exact": 1.0, "semantic_high_confidence": 0.9, "semantic_review": 0.8}[tier]
    return HybridDuplicate(unit_a=unit_a, unit_b=unit_b, tier=tier, confidence=confidence)


def _result(tmp_path: Path, **overrides) -> AnalysisResult:
    a = _unit(tmp_path, "a", start_byte=0)
    b = _unit(tmp_path, "b", start_byte=40)
    c = _unit(tmp_path, "c", file="b.py", start_byte=0)
    fields = {
        "units": [a, b, c],
        "traditional_duplicates": [DuplicatePair(a, b, 1.0, "ast_hash")],
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

    assert [pair.tier for pair in selection.duplicates] == ["exact"]
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

    assert selection.duplicates == result.hybrid_duplicates
    assert selection.omitted_review == []
    assert selection.traditional_duplicates is None


def test_select_findings_show_all_implies_review_and_raw_lists(tmp_path):
    result = _result(tmp_path)

    selection = select_findings(result, ReportPolicy(show_all=True))

    assert selection.policy.shows_review
    assert selection.duplicates == result.hybrid_duplicates
    assert selection.traditional_duplicates == result.traditional_duplicates
    assert selection.semantic_duplicates == result.semantic_duplicates


def test_select_findings_single_method_has_no_tier_filter(tmp_path):
    result = _result(tmp_path, analysis_mode="semantic", hybrid_duplicates=[])

    selection = select_findings(result, ReportPolicy(show_all=True))

    assert selection.duplicates == result.traditional_duplicates + result.semantic_duplicates
    assert selection.omitted_review == []
    assert set(selection.duplicates_by_tier.values()) == {0}
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
    assert selection.units[-1] is orphan


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

    assert [pair.tier for pair in capped.duplicates] == ["exact", "hybrid_confirmed"]
    assert [pair.tier for pair in capped.truncated] == ["semantic_high_confidence"]
    assert [pair.tier for pair in capped.omitted_review] == ["semantic_review"]
    assert len(capped.duplicates) + len(capped.omitted_review) + len(capped.truncated) == 4
    # The tier breakdown still describes the complete result.
    assert capped.duplicates_by_tier["semantic_high_confidence"] == 1
    # Units referenced only by truncated pairs drop out with them.
    assert [unit.name for unit in capped.units] == ["f0", "f1", "f2", "f3"]

    with_review = select_findings(result, ReportPolicy(include_review=True, max_duplicates=2))
    assert [pair.tier for pair in with_review.duplicates] == ["exact", "semantic_review"]
    assert with_review.omitted_review == []
    assert len(with_review.truncated) == 2


def test_max_duplicates_is_a_no_op_at_or_above_the_admitted_count(tmp_path):
    result = _ranked_result(tmp_path, ["exact", "hybrid_confirmed"])

    exact = select_findings(result, ReportPolicy(max_duplicates=2))
    generous = select_findings(result, ReportPolicy(max_duplicates=50))

    assert exact.truncated == generous.truncated == []
    assert exact.duplicates == generous.duplicates == result.hybrid_duplicates


def test_max_duplicates_leaves_show_all_raw_lists_complete(tmp_path):
    result = _result(tmp_path)

    selection = select_findings(result, ReportPolicy(show_all=True, max_duplicates=1))

    assert len(selection.duplicates) == 1
    assert len(selection.truncated) == 1
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
    exact = DuplicatePair(a, b, 1.0, "ast_hash")
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

    assert capped.duplicates == [exact, near_099]
    # Truncated holds the rest, still ranked by descending similarity, and the
    # two 0.90 pairs keep their input order (stable sort over the tie).
    assert capped.truncated == [near_090_first, near_090_second, near_086]


def test_max_duplicates_applies_to_single_method_raw_lists(tmp_path):
    result = _result(tmp_path, analysis_mode="semantic", hybrid_duplicates=[])

    selection = select_findings(result, ReportPolicy(max_duplicates=1))

    assert selection.duplicates == result.traditional_duplicates[:1]
    assert selection.truncated == result.semantic_duplicates
    assert run_should_fail(result, policy="actionable", strict_unused=False) is True


@pytest.mark.parametrize("cap", [0, -1])
def test_report_policy_rejects_a_cap_that_emits_nothing(cap):
    with pytest.raises(ValueError, match="at least 1"):
        ReportPolicy(max_duplicates=cap)


def test_hidden_only_failure_names_truncated_pairs(tmp_path):
    # Confidence is only tier-monotone at equal similarity, so a strong
    # semantic_high_confidence pair can outrank an actionable hybrid_confirmed
    # one; a cap of one then hides the pair that fails the default policy.
    result = _ranked_result(
        tmp_path, ["semantic_high_confidence", "hybrid_confirmed", "semantic_review"]
    )
    capped = select_findings(result, ReportPolicy(max_duplicates=1))

    assert run_should_fail(result, policy="actionable", strict_unused=False) is True
    assert hidden_only_failure(capped, policy="actionable", strict_unused=False) == {"truncated"}
    # Under --fail-on all the emitted pair fails by itself, so nothing hidden is named.
    assert hidden_only_failure(capped, policy="all", strict_unused=False) == set()

    # Truncating only advisory pairs hides nothing that fails.
    advisory = select_findings(
        _ranked_result(tmp_path, ["semantic_high_confidence", "semantic_high_confidence"]),
        ReportPolicy(max_duplicates=1),
    )
    assert run_should_fail(advisory.result, policy="actionable", strict_unused=False) is False
    assert hidden_only_failure(advisory, policy="actionable", strict_unused=False) == set()
