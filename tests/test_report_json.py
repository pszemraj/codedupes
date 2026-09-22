"""Schema-v4 JSON serialization of check and search reports."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from codedupes.models import (
    HYBRID_TIERS,
    AnalysisResult,
    CodeUnit,
    CodeUnitType,
    DuplicatePair,
    HybridDuplicate,
)
from codedupes.report.json import (
    SCHEMA_VERSION,
    check_result_to_json,
    search_result_to_json,
    to_json_text,
)
from codedupes.report.selection import (
    DEFAULT_MAX_DUPLICATES,
    ReportPolicy,
    group_file_results,
    select_findings,
)

_ID = re.compile(r"^u\d+$")


def _unit(
    tmp_path: Path,
    name: str,
    *,
    file: str = "a.py",
    start_byte: int = 0,
    token_hash: str | None = None,
) -> CodeUnit:
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
        token_hash=token_hash,
    )


def _result(tmp_path: Path) -> AnalysisResult:
    a = _unit(tmp_path, "a", file="b.py", start_byte=0)
    b = _unit(tmp_path, "b", file="a.py", start_byte=40)
    c = _unit(tmp_path, "c", file="a.py", start_byte=0)
    orphan = _unit(tmp_path, "orphan", file="z.py")
    return AnalysisResult(
        units=[a, b, c, orphan],
        traditional_duplicates=[DuplicatePair(a, b, 1.0, "structural_hash")],
        semantic_duplicates=[DuplicatePair(b, c, 0.9, "semantic")],
        hybrid_duplicates=[
            HybridDuplicate(a, b, "exact", 1.0, has_exact=True, exact_method="structural_hash"),
            HybridDuplicate(b, c, "semantic_review", 0.8, semantic_similarity=0.9),
        ],
        potentially_unused=[orphan],
        analysis_mode="combined",
    )


def _payload(
    result: AnalysisResult,
    policy: ReportPolicy | None = None,
    *,
    fail_on: str = "actionable",
    exit_code: int = 1,
) -> dict:
    return check_result_to_json(
        select_findings(result, policy),
        fail_on=fail_on,
        exit_code=exit_code,
        strict_unused=False,
    )


def _referenced_ids(payload: dict) -> set[str]:
    ids = set(payload["potentially_unused"])
    for family in payload["exact_families"]:
        ids.update(family["members"])
    for key in ("duplicates", "traditional_duplicates", "semantic_duplicates"):
        for edge in payload.get(key, []):
            ids.update((edge["unit_a"], edge["unit_b"]))
    return ids


def test_check_json_v4_ids_resolve_and_have_no_orphans(tmp_path):
    result = _result(tmp_path)

    payload = _payload(result, ReportPolicy(show_all=True))

    assert payload["schema_version"] == SCHEMA_VERSION == 4
    assert _referenced_ids(payload) == set(payload["units"])
    assert all(_ID.match(key) for key in payload["units"])
    by_uid = {unit.uid: unit for unit in result.units}
    for key, record in payload["units"].items():
        assert by_uid[record["uid"]].name == record["name"], key


def test_check_json_v4_ids_follow_file_then_source_order(tmp_path):
    payload = _payload(_result(tmp_path), ReportPolicy(include_review=True))

    assert [payload["units"][f"u{i}"]["name"] for i in range(4)] == ["c", "b", "a", "orphan"]


def test_check_json_v4_hidden_review_units_are_absent(tmp_path):
    result = _result(tmp_path)

    default = _payload(result)
    with_review = _payload(result, ReportPolicy(include_review=True))

    assert {record["name"] for record in default["units"].values()} == {"a", "b", "orphan"}
    assert {record["name"] for record in with_review["units"].values()} == {"a", "b", "c", "orphan"}
    assert default["summary"]["reported_duplicates"] == 1
    assert default["summary"]["omitted_review_duplicates"] == 1
    assert with_review["summary"]["omitted_review_duplicates"] == 0


def test_check_json_round_trips_and_keeps_unit_ids_under_edge_shuffle(tmp_path):
    result = _result(tmp_path)
    shuffled = _result(tmp_path)
    random.Random(1).shuffle(shuffled.hybrid_duplicates)
    policy = ReportPolicy(include_review=True)

    payload = _payload(result, policy)
    assert json.loads(to_json_text(payload)) == payload
    assert _payload(shuffled, policy)["units"] == payload["units"]


def test_check_json_v4_summary_counts(tmp_path):
    payload = _payload(_result(tmp_path))
    summary = payload["summary"]

    assert set(summary["duplicates_by_tier"]) == set(HYBRID_TIERS)
    assert summary["duplicates_by_tier"] == {
        "exact": 1,
        "traditional_near": 0,
        "hybrid_confirmed": 0,
        "semantic_high_confidence": 0,
        "semantic_review": 1,
    }
    assert summary["hybrid_duplicates"] == 2
    assert summary["reported_duplicates"] + summary["omitted_review_duplicates"] == 2
    assert summary["truncated_duplicates"] == 0
    # The library policy is uncapped; only the CLI applies DEFAULT_MAX_DUPLICATES.
    assert summary["max_duplicates"] is None
    assert summary["actionable_duplicates"] == 1
    assert summary["reported_actionable_duplicates"] == 1
    assert summary["exact_family_members"] == 2
    assert summary["raw_traditional_duplicates"] == 1
    assert summary["raw_semantic_duplicates"] == 1
    assert summary["fail_on"] == "actionable"
    assert summary["strict_unused"] is False
    assert summary["exit_code"] == 1
    assert summary["hidden_only_failure"] == []


def test_check_json_hidden_only_failure_names_withheld_review(tmp_path):
    result = _result(tmp_path)
    result.hybrid_duplicates.pop(0)  # Leave only the semantic_review pair.
    result.traditional_duplicates.clear()
    result.potentially_unused.clear()

    withheld = _payload(result, fail_on="all", exit_code=1)["summary"]
    assert withheld["reported_duplicates"] == 0
    assert withheld["omitted_review_duplicates"] == 1
    assert withheld["actionable_duplicates"] == 0
    assert withheld["hidden_only_failure"] == ["review"]

    listed = _payload(result, ReportPolicy(include_review=True), fail_on="all", exit_code=1)
    assert listed["summary"]["hidden_only_failure"] == []

    # Review pairs never fail the default policy, so nothing hidden is named.
    passing = _payload(result, fail_on="actionable", exit_code=0)["summary"]
    assert passing["hidden_only_failure"] == []


def test_check_json_raw_modes_count_every_pair_as_actionable(tmp_path):
    result = _result(tmp_path)
    result.analysis_mode = "semantic"
    result.hybrid_duplicates.clear()

    summary = _payload(result, ReportPolicy(max_duplicates=1))["summary"]

    assert summary["actionable_duplicates"] == 2
    assert summary["reported_actionable_duplicates"] == 1
    assert summary["truncated_duplicates"] == 1
    # Raw pairs carry no tier, so only ``exact`` (families) is ever non-zero;
    # the family took the one slot and the semantic pair was cut.
    assert summary["duplicates_by_tier"] == dict.fromkeys(HYBRID_TIERS, 0) | {"exact": 1}
    assert summary["truncated_by_tier"] == dict.fromkeys(HYBRID_TIERS, 0)
    assert summary["hybrid_duplicates"] == 0


def test_check_json_truncated_by_tier_names_cut_review_pairs(tmp_path):
    result = _result(tmp_path)  # One exact pair, one semantic_review pair.

    withheld = _payload(result, ReportPolicy(max_duplicates=1))["summary"]
    assert withheld["omitted_review_duplicates"] == 1
    assert withheld["truncated_duplicates"] == 0
    assert withheld["truncated_by_tier"]["semantic_review"] == 0

    # Including review pairs ranks them last, so the same cap now cuts the
    # review pair instead of withholding it.
    cut = _payload(result, ReportPolicy(include_review=True, max_duplicates=1))["summary"]
    assert cut["reported_duplicates"] == 1
    assert cut["omitted_review_duplicates"] == 0
    assert cut["truncated_duplicates"] == 1
    assert cut["truncated_by_tier"] == {
        "exact": 0,
        "traditional_near": 0,
        "hybrid_confirmed": 0,
        "semantic_high_confidence": 0,
        "semantic_review": 1,
    }


def _chain_result(tmp_path: Path, pairs: int) -> AnalysisResult:
    """Build ``pairs`` hybrid_confirmed edges chaining ``pairs + 1`` units."""
    units = [_unit(tmp_path, f"f{i}", start_byte=i * 30) for i in range(pairs + 1)]
    hybrid = [
        HybridDuplicate(
            units[i],
            units[i + 1],
            "hybrid_confirmed",
            0.99 - i * 0.001,
            jaccard_similarity=0.9,
            semantic_similarity=0.9,
        )
        for i in range(pairs)
    ]
    return AnalysisResult(
        units=units,
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=hybrid,
        potentially_unused=[],
        analysis_mode="combined",
    )


def test_check_json_emits_every_selected_finding_untruncated(tmp_path):
    payload = _payload(_chain_result(tmp_path, 25))

    assert len(payload["duplicates"]) == 25
    assert len(payload["units"]) == 26
    assert payload["summary"]["truncated_duplicates"] == 0


def test_check_json_cli_default_policy_caps_at_twenty(tmp_path):
    payload = _payload(
        _chain_result(tmp_path, 25), ReportPolicy(max_duplicates=DEFAULT_MAX_DUPLICATES)
    )
    summary = payload["summary"]

    assert len(payload["duplicates"]) == 20
    assert len(payload["units"]) == 21
    assert summary["max_duplicates"] == 20
    assert summary["truncated_duplicates"] == 5
    assert summary["actionable_duplicates"] == 25
    assert summary["reported_actionable_duplicates"] == 20


def test_check_json_max_duplicates_caps_edges_and_units_but_not_counts(tmp_path):
    payload = _payload(_chain_result(tmp_path, 25), ReportPolicy(max_duplicates=10))
    summary = payload["summary"]

    assert len(payload["duplicates"]) == 10
    assert len(payload["units"]) == 11
    assert _referenced_ids(payload) == set(payload["units"])
    assert summary["hybrid_duplicates"] == 25
    assert summary["reported_duplicates"] == 10
    assert summary["truncated_duplicates"] == 15
    assert summary["max_duplicates"] == 10
    assert (
        summary["reported_duplicates"]
        + summary["omitted_review_duplicates"]
        + summary["truncated_duplicates"]
        == summary["hybrid_duplicates"]
    )
    assert summary["duplicates_by_tier"]["hybrid_confirmed"] == 25
    assert summary["exit_code"] == 1


def _family_result(
    tmp_path: Path, copies: int, *, method: str = "structural_hash"
) -> AnalysisResult:
    """Build one ``copies``-member exact clique plus one hybrid_confirmed pair."""
    members = [
        _unit(
            tmp_path,
            f"copy{i}",
            file=f"m{i}.py",
            token_hash="t" if method == "token_hash" else None,
        )
        for i in range(copies)
    ]
    near_a = _unit(tmp_path, "near_a", file="n.py", start_byte=0)
    near_b = _unit(tmp_path, "near_b", file="n.py", start_byte=40)
    hybrid = [
        HybridDuplicate(members[i], members[j], "exact", 1.0, has_exact=True, exact_method=method)
        for i in range(copies)
        for j in range(i + 1, copies)
    ]
    hybrid.append(
        HybridDuplicate(
            near_a, near_b, "hybrid_confirmed", 0.9, jaccard_similarity=0.9, semantic_similarity=0.9
        )
    )
    return AnalysisResult(
        units=members + [near_a, near_b],
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=hybrid,
        potentially_unused=[],
        analysis_mode="combined",
    )


def test_check_json_exact_family_record_replaces_pairwise_edges(tmp_path):
    payload = _payload(_family_result(tmp_path, 5, method="token_hash"))
    summary = payload["summary"]

    # Ten exact edges become one record; no exact edge remains in duplicates.
    assert payload["exact_families"] == [
        {
            "method": "token_hash",
            "members": ["u0", "u1", "u2", "u3", "u4"],
            "lines": 2,
            "redundant_lines": 8,
        }
    ]
    assert [edge["tier"] for edge in payload["duplicates"]] == ["hybrid_confirmed"]
    assert "has_exact" not in payload["duplicates"][0]
    assert _referenced_ids(payload) == set(payload["units"])
    assert summary["hybrid_duplicates"] == 2
    assert summary["reported_duplicates"] == 2
    assert summary["actionable_duplicates"] == 2
    assert summary["reported_actionable_duplicates"] == 2
    assert summary["duplicates_by_tier"]["exact"] == 1
    assert summary["exact_family_members"] == 5
    assert summary["hidden_only_failure"] == []


def test_check_json_family_cap_counts_and_truncated_by_tier_exact(tmp_path):
    result = _family_result(tmp_path, 5)
    # A second, smaller family ranks after the five-copy one.
    small_a = _unit(tmp_path, "small_a", file="s.py", start_byte=0)
    small_b = _unit(tmp_path, "small_b", file="s.py", start_byte=40)
    result.units.extend([small_a, small_b])
    result.hybrid_duplicates.append(HybridDuplicate(small_a, small_b, "exact", 1.0, has_exact=True))

    payload = _payload(result, ReportPolicy(max_duplicates=1))
    summary = payload["summary"]

    assert [family["members"] for family in payload["exact_families"]] == [
        ["u0", "u1", "u2", "u3", "u4"]
    ]
    assert payload["exact_families"][0]["method"] == "structural_hash"
    assert payload["duplicates"] == []
    assert len(payload["units"]) == 5
    assert summary["max_duplicates"] == 1
    assert summary["hybrid_duplicates"] == 3
    assert summary["reported_duplicates"] == 1
    assert summary["truncated_duplicates"] == 2
    assert summary["truncated_by_tier"] == {
        "exact": 1,
        "traditional_near": 0,
        "hybrid_confirmed": 1,
        "semantic_high_confidence": 0,
        "semantic_review": 0,
    }
    assert summary["duplicates_by_tier"]["exact"] == 2
    assert summary["exact_family_members"] == 7
    assert summary["actionable_duplicates"] == 3
    assert summary["reported_actionable_duplicates"] == 1
    assert (
        summary["reported_duplicates"]
        + summary["omitted_review_duplicates"]
        + summary["truncated_duplicates"]
        == summary["hybrid_duplicates"]
    )


def test_check_json_show_all_raw_edges_use_short_ids(tmp_path):
    payload = _payload(_result(tmp_path), ReportPolicy(show_all=True))

    assert payload["traditional_duplicates"] == [
        {"unit_a": "u2", "unit_b": "u1", "similarity": 1.0, "method": "structural_hash"}
    ]
    assert payload["semantic_duplicates"][0]["unit_a"] == "u1"
    assert payload["semantic_duplicates"][0]["unit_b"] == "u0"


def test_search_json_v4_unit_and_file_levels(tmp_path):
    a = _unit(tmp_path, "a", file="a.py", start_byte=0)
    b = _unit(tmp_path, "b", file="a.py", start_byte=40)
    c = _unit(tmp_path, "c", file="b.py", start_byte=0)
    hits = [(c, 0.95), (a, 0.9), (b, 0.8)]

    unit_level = search_result_to_json(
        "q", hits, 3, None, extraction_diagnostics=[], semantic_diagnostics=[]
    )
    assert unit_level["schema_version"] == 4
    assert [hit["unit"] for hit in unit_level["results"]] == ["u2", "u0", "u1"]
    assert unit_level["units"]["u2"]["uid"] == c.uid
    assert set(unit_level["units"]) == {"u0", "u1", "u2"}

    files = group_file_results(hits, top_k=1)
    file_level = search_result_to_json(
        "q", hits, 3, None, extraction_diagnostics=[], semantic_diagnostics=[], file_results=files
    )
    assert file_level["result_level"] == "file"
    assert file_level["results"][0]["matches"] == [{"unit": "u0", "score": 0.95}]
    assert set(file_level["units"]) == {"u0"}
    assert file_level["units"]["u0"]["uid"] == c.uid
