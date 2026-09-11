"""Report visibility selection and finding exit policy."""

from __future__ import annotations

import random
from pathlib import Path
from typing import get_args

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
    run_should_fail,
    select_findings,
    withheld_only_failure,
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


def test_withheld_only_failure_cases(tmp_path):
    a = _unit(tmp_path, "a")
    b = _unit(tmp_path, "b", start_byte=40)
    review_only = _result(
        tmp_path,
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[_hybrid(a, b, "semantic_review")],
    )
    selection = select_findings(review_only)

    assert withheld_only_failure(selection, policy="all", strict_unused=False) is True
    assert withheld_only_failure(selection, policy="actionable", strict_unused=False) is False
    assert withheld_only_failure(selection, policy="none", strict_unused=False) is False

    with_visible = _result(tmp_path)
    assert (
        withheld_only_failure(select_findings(with_visible), policy="all", strict_unused=False)
        is False
    )

    with_unused = _result(
        tmp_path,
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[_hybrid(a, b, "semantic_review")],
        potentially_unused=[a],
    )
    assert (
        withheld_only_failure(select_findings(with_unused), policy="all", strict_unused=True)
        is False
    )

    shown = select_findings(review_only, ReportPolicy(include_review=True))
    assert withheld_only_failure(shown, policy="all", strict_unused=False) is False
