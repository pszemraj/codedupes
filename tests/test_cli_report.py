"""``codedupes check`` report selection: caps, families, review visibility, and exit codes."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
from click.testing import CliRunner

from codedupes import cli
from codedupes.models import (
    AnalysisResult,
    DuplicatePair,
    HybridDuplicate,
)
from tests.cli_helpers import build_copy, build_result_with_semantic_duplicate, build_unit
from tests.conftest import make_code_unit, patch_cli_analyzer


def test_cli_show_all_prints_raw_sections(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result_with_semantic_duplicate(tmp_path),
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--show-all"])
    assert result.exit_code == 1
    assert "Traditional Duplicates (Raw" in result.output
    assert "Semantic Duplicates (Raw" in result.output


def test_cli_never_reports_a_filtered_raw_duplicate_count(monkeypatch, tmp_path):
    # Every candidate pair now reaches a tier, so the old always-zero counter
    # and its surfaces are gone.
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result_with_semantic_duplicate(tmp_path),
    )
    runner = CliRunner()

    table_result = runner.invoke(cli.cli, ["check", str(path), "--show-all"])
    assert "Filtered raw duplicates" not in table_result.output
    assert "raw duplicate pairs" not in table_result.output

    json_result = runner.invoke(cli.cli, ["check", str(path), "--json"])
    assert "filtered_raw_duplicates" not in json.loads(json_result.output)["summary"]


def test_cli_traditional_panel_label_is_language_neutral(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = build_unit(tmp_path)
    copy = build_copy(tmp_path)
    near_a = make_code_unit(tmp_path, name="near_a", source="def near_a():\n    return 2", lineno=9)
    near_b = make_code_unit(
        tmp_path, name="near_b", source="def near_b():\n    return 3", lineno=13
    )
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=AnalysisResult(
            units=[unit, copy, near_a, near_b],
            traditional_duplicates=[
                DuplicatePair(unit_a=unit, unit_b=copy, similarity=1.0, method="token_hash"),
                DuplicatePair(unit_a=near_a, unit_b=near_b, similarity=0.9, method="jaccard"),
            ],
            semantic_duplicates=[],
            hybrid_duplicates=[],
            potentially_unused=[],
            analysis_mode="traditional",
        ),
    )

    result = CliRunner().invoke(cli.cli, ["check", str(path), "--traditional-only"])
    # Every language reports the shared structural fingerprint; no table names an AST.
    assert "Exact Duplicate Families (1 family)" in result.output
    assert "token_hash" in result.output
    assert "Exact duplicate families" in result.output
    assert "1 family (2 units)" in result.output
    assert "Near Duplicates (Jaccard)" in result.output
    assert "(1 pair)" in result.output
    assert "AST" not in result.output


def _build_capped_result_with_unused(tmp_path: Path, unused: int = 25) -> AnalysisResult:
    """Capped result plus ``unused`` dead units whose sizes descend with their index."""
    result = _build_capped_result(tmp_path)
    result.potentially_unused = [
        make_code_unit(
            tmp_path,
            name=f"dead_{i:02d}",
            source="def dead_{i:02d}():\n" + "    pass\n" * (unused - i),
            lineno=200 + 40 * i,
        )
        for i in range(unused)
    ]
    return result


def test_cli_full_table_lifts_the_pair_cap_and_the_unused_cap(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    patch_cli_analyzer(monkeypatch, cli, analyze_result=_build_capped_result_with_unused(tmp_path))

    runner = CliRunner()
    default_result = runner.invoke(cli.cli, ["check", str(path)])
    assert default_result.exit_code == 1
    # Both primary tables render every selected row; only the report caps bound them.
    assert "(20 pairs, 5 truncated)" in default_result.output
    assert "5 (5 semantic_high_confidence; use --max-duplicates all)" in default_result.output
    assert "Likely Dead Code (20 units, 5 truncated)" in default_result.output
    assert "Truncated dead code" in default_result.output
    assert "5 (use --max-unused all)" in default_result.output
    assert "--full-table" not in default_result.output

    full_result = runner.invoke(cli.cli, ["check", str(path), "--full-table"])
    assert full_result.exit_code == 1
    assert "(25 pairs)" in full_result.output
    assert "Likely Dead Code (25 units)" in full_result.output
    assert "Truncated duplicates" not in full_result.output
    assert "Truncated dead code" not in full_result.output

    # An explicit cap survives --full-table; the other cap still lifts.
    capped = runner.invoke(cli.cli, ["check", str(path), "--full-table", "--max-duplicates", "10"])
    assert "(10 pairs, 15 truncated)" in capped.output
    assert "Likely Dead Code (25 units)" in capped.output
    capped_unused = runner.invoke(
        cli.cli, ["check", str(path), "--full-table", "--max-unused", "10"]
    )
    assert "(25 pairs)" in capped_unused.output
    assert "Likely Dead Code (10 units, 15 truncated)" in capped_unused.output


def test_cli_unused_panel_ranks_by_line_span_and_shows_lines_column(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    patch_cli_analyzer(
        monkeypatch, cli, analyze_result=_build_capped_result_with_unused(tmp_path, unused=6)
    )
    runner = CliRunner()

    terminal = runner.invoke(
        cli.cli, ["check", str(path), "--max-unused", "3", "--output-width", "160"]
    )
    panel = terminal.output.split("Likely Dead Code")[1]
    assert "Lines" in panel
    assert re.findall(r"dead_\d{2}", panel) == ["dead_00", "dead_01", "dead_02"]
    # dead_00 spans seven lines (def plus six pass statements).
    assert re.search(r"dead_00\s*│\s*function\s*│\s*7\s*│", panel)

    payload = json.loads(
        runner.invoke(cli.cli, ["check", str(path), "--json", "--max-unused", "3"]).output
    )
    names = [payload["units"][key]["name"] for key in payload["potentially_unused"]]
    assert names == ["dead_00", "dead_01", "dead_02"]
    assert payload["summary"]["potentially_unused"] == 6
    assert payload["summary"]["reported_unused"] == 3
    assert payload["summary"]["truncated_unused"] == 3
    assert payload["summary"]["max_unused"] == 3
    assert not any(record["name"] == "dead_05" for record in payload["units"].values())


def test_cli_combined_exit_code_ignores_raw_filtered_findings(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = build_unit(tmp_path)
    duplicate = DuplicatePair(unit_a=unit, unit_b=unit, similarity=1.0, method="jaccard")
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=AnalysisResult(
            units=[unit],
            traditional_duplicates=[duplicate],
            semantic_duplicates=[],
            hybrid_duplicates=[],
            potentially_unused=[],
            analysis_mode="combined",
        ),
    )

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path)])
    assert result.exit_code == 0


@pytest.mark.parametrize(
    (
        "policy",
        "combined_mode",
        "strict_unused",
        "tier",
        "raw_duplicate",
        "include_unused",
        "expected",
    ),
    [
        ("actionable", True, False, "semantic_review", False, True, False),
        ("all", True, False, "semantic_review", False, True, True),
        ("none", True, True, "hybrid_confirmed", False, True, False),
        ("actionable", True, True, None, False, True, True),
        ("actionable", True, False, "hybrid_confirmed", False, False, True),
        ("actionable", True, False, "semantic_high_confidence", False, False, False),
        ("actionable", False, False, None, True, False, True),
    ],
)
def test_run_should_fail_policy(
    tmp_path,
    policy,
    combined_mode,
    strict_unused,
    tier,
    raw_duplicate,
    include_unused,
    expected,
):
    unit = build_unit(tmp_path)
    hybrid = (
        [
            HybridDuplicate(
                unit_a=unit,
                unit_b=unit,
                tier=tier,
                score=0.9,
            )
        ]
        if tier is not None
        else []
    )
    raw = (
        [DuplicatePair(unit_a=unit, unit_b=unit, similarity=0.9, method="semantic")]
        if raw_duplicate
        else []
    )
    result = AnalysisResult(
        units=[unit],
        traditional_duplicates=raw,
        semantic_duplicates=[],
        hybrid_duplicates=hybrid,
        potentially_unused=[unit] if include_unused else [],
        analysis_mode="combined" if combined_mode else "traditional",
    )

    assert cli.run_should_fail(result, policy=policy, strict_unused=strict_unused) is expected


def test_cli_fail_on_all_and_none(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = build_unit(tmp_path)
    result_obj = AnalysisResult(
        units=[unit],
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[
            HybridDuplicate(
                unit_a=unit,
                unit_b=unit,
                tier="semantic_review",
                score=0.8,
            )
        ],
        potentially_unused=[unit],
        analysis_mode="combined",
    )
    patch_cli_analyzer(monkeypatch, cli, analyze_result=result_obj)
    runner = CliRunner()

    default_result = runner.invoke(cli.cli, ["check", str(path)])
    all_result = runner.invoke(cli.cli, ["check", str(path), "--fail-on", "all", "--json"])
    all_terminal = runner.invoke(cli.cli, ["check", str(path), "--fail-on", "all"])
    none_result = runner.invoke(cli.cli, ["check", str(path), "--fail-on", "none"])

    assert default_result.exit_code == 0
    assert all_result.exit_code == 1
    assert all_terminal.exit_code == 1
    assert none_result.exit_code == 0
    assert "Failure policy" in default_result.output
    assert "actionable" in default_result.output
    assert "Finding status" in default_result.output
    assert "pass (exit 0)" in default_result.output
    # The unused unit fails --fail-on all on its own, so this is not a
    # withheld-only failure and the terminal must not claim it is.
    assert "fail (exit 1)" in all_terminal.output
    assert "only withheld" not in all_terminal.output
    summary = json.loads(all_result.output)["summary"]
    assert summary["fail_on"] == "all"
    assert summary["exit_code"] == 1
    assert summary["reported_duplicates"] == 0
    assert summary["omitted_review_duplicates"] == 1


def _build_tiered_result(tmp_path: Path) -> AnalysisResult:
    """Combined result with one confirmed pair, two review pairs, and an unused unit."""
    unit = build_unit(tmp_path)
    other = make_code_unit(tmp_path, name="other", source="def other():\n    return 2", lineno=5)
    review_only = make_code_unit(
        tmp_path, name="lonely", source="def lonely():\n    return 3", lineno=9
    )
    return AnalysisResult(
        units=[unit, other, review_only],
        traditional_duplicates=[
            DuplicatePair(unit_a=unit, unit_b=other, similarity=0.9, method="jaccard")
        ],
        semantic_duplicates=[
            DuplicatePair(unit_a=unit, unit_b=other, similarity=0.95, method="semantic"),
            DuplicatePair(unit_a=unit, unit_b=review_only, similarity=0.81, method="semantic"),
            DuplicatePair(unit_a=other, unit_b=review_only, similarity=0.80, method="semantic"),
        ],
        hybrid_duplicates=[
            HybridDuplicate(
                unit_a=unit,
                unit_b=other,
                tier="hybrid_confirmed",
                score=0.92,
                semantic_similarity=0.95,
                jaccard_similarity=0.9,
            ),
            HybridDuplicate(
                unit_a=unit,
                unit_b=review_only,
                tier="semantic_review",
                score=0.76,
                semantic_similarity=0.81,
            ),
            HybridDuplicate(
                unit_a=other,
                unit_b=review_only,
                tier="semantic_review",
                score=0.76,
                semantic_similarity=0.80,
            ),
        ],
        potentially_unused=[other],
        analysis_mode="combined",
    )


def _build_capped_result(tmp_path: Path, pairs: int = 25) -> AnalysisResult:
    """Combined result chaining ``pairs`` edges whose analyzer order is by score.

    Odd edges are ``hybrid_confirmed`` and even edges ``semantic_high_confidence``,
    so the report order (actionable first) differs from the analyzer order.
    """
    units = [
        make_code_unit(
            tmp_path,
            name=f"dup_{i:02d}",
            source=f"def dup_{i:02d}():\n    return {i}",
            lineno=1 + 3 * i,
        )
        for i in range(pairs + 1)
    ]
    hybrid = [
        HybridDuplicate(
            units[i],
            units[i + 1],
            "hybrid_confirmed" if i % 2 else "semantic_high_confidence",
            0.99 - i * 0.001,
            semantic_similarity=0.9,
            jaccard_similarity=0.9 if i % 2 else None,
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


def _terminal_pairs(output: str) -> list[tuple[str, str]]:
    """Return the (unit_a, unit_b) names rendered in the primary hybrid table."""
    names = re.findall(r"dup_\d{2}", output.split("Hybrid Duplicates")[1])
    return list(zip(names[::2], names[1::2], strict=True))


def _json_pairs(payload: dict) -> list[tuple[str, str]]:
    """Return the (unit_a, unit_b) names of the JSON primary edges in order."""
    units = payload["units"]
    return [
        (units[edge["unit_a"]]["name"], units[edge["unit_b"]]["name"])
        for edge in payload["duplicates"]
    ]


def _build_family_result(tmp_path: Path) -> AnalysisResult:
    """Combined result: one three-copy exact family, one confirmed pair, one review pair."""
    copies = [
        make_code_unit(
            tmp_path,
            name=f"copy_{i}",
            source=f"def copy_{i}(x):\n    y = x + 1\n    return y",
            lineno=1 + 4 * i,
        )
        for i in range(3)
    ]
    near_a = make_code_unit(
        tmp_path, name="near_a", source="def near_a():\n    return 2", lineno=20
    )
    near_b = make_code_unit(
        tmp_path, name="near_b", source="def near_b():\n    return 3", lineno=24
    )
    review = make_code_unit(
        tmp_path, name="review", source="def review():\n    return 4", lineno=28
    )
    hybrid = [
        HybridDuplicate(copies[i], copies[j], "exact", 1.0, exact_method="token_hash")
        for i in range(3)
        for j in range(i + 1, 3)
    ]
    hybrid.append(
        HybridDuplicate(
            near_a, near_b, "hybrid_confirmed", 0.9, jaccard_similarity=0.9, semantic_similarity=0.9
        )
    )
    hybrid.append(HybridDuplicate(near_b, review, "semantic_review", 0.8, semantic_similarity=0.88))
    return AnalysisResult(
        units=copies + [near_a, near_b, review],
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=hybrid,
        potentially_unused=[],
        analysis_mode="combined",
    )


@pytest.mark.grammar
def test_cli_exact_family_fixture_end_to_end():
    fixture = Path(__file__).resolve().parents[1] / "test_fixtures" / "exact_family"
    runner = CliRunner()
    base = ["check", str(fixture), "--traditional-only", "--no-unused"]

    result = runner.invoke(cli.cli, [*base, "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    units = payload["units"]

    # Five token-identical copies (ten raw edges) and one renamed pair collapse
    # into two family records; no exact edge remains in the pairwise list.
    assert payload["duplicates"] == []
    assert [
        (family["method"], sorted(units[m]["name"] for m in family["members"]), family["lines"])
        for family in payload["exact_families"]
    ] == [
        ("token_hash", ["render_receipt"] * 5, 16),
        ("structural_hash", ["sum_amounts", "sum_credits"], 11),
    ]
    assert payload["exact_families"][0]["redundant_lines"] == 4 * 16
    assert {
        units[m]["file"].rsplit("/", 1)[-1] for m in payload["exact_families"][0]["members"]
    } == {
        "exports.py",
        "invoices.py",
        "receipts.py",
        "refunds.py",
        "statements.py",
    }
    summary = payload["summary"]
    assert summary["duplicates_by_tier"]["exact"] == 2
    assert summary["exact_family_members"] == 7
    assert summary["reported_duplicates"] == 2
    assert summary["actionable_duplicates"] == 2
    assert summary["raw_traditional_duplicates"] == 11
    assert summary["exit_code"] == 1
    assert len(units) == 7

    capped = json.loads(runner.invoke(cli.cli, [*base, "--json", "--max-duplicates", "1"]).output)
    assert len(capped["exact_families"]) == 1
    assert len(capped["exact_families"][0]["members"]) == 5
    assert capped["summary"]["truncated_by_tier"]["exact"] == 1
    assert capped["summary"]["truncated_duplicates"] == 1
    assert capped["summary"]["duplicates_by_tier"]["exact"] == 2
    assert len(capped["units"]) == 5

    terminal = runner.invoke(cli.cli, [*base, "--output-width", "160"])
    assert terminal.exit_code == 1
    assert "Exact Duplicate Families (2 families)" in terminal.output
    assert "2 families (7 units)" in terminal.output
    assert "+1 more" in terminal.output
    assert "Traditional Duplicates" not in terminal.output


def test_cli_family_panel_leads_the_report_and_counts_as_one_finding(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    patch_cli_analyzer(monkeypatch, cli, analyze_result=_build_family_result(tmp_path))
    runner = CliRunner()

    wide = runner.invoke(cli.cli, ["check", str(path), "--output-width", "160"])
    assert wide.exit_code == 1
    families_at = wide.output.index("Exact Duplicate Families (1 family)")
    pairs_at = wide.output.index("Hybrid Duplicates (1 pair, 1 review withheld)")
    assert families_at < pairs_at
    panel = wide.output[families_at:pairs_at]
    assert "token_hash" in panel
    assert "copy_0" in panel
    # Three exact edges are one row; the family table lists the other members by location.
    assert panel.count("sample.py:") == 3
    assert "1 family (3 units)" in wide.output
    assert "2 (2 reported)" in wide.output  # Actionable duplicates
    assert "Reported duplicates" in wide.output

    compact = runner.invoke(cli.cli, ["check", str(path), "--output-width", "80"])
    assert compact.exit_code == 1
    assert "Members: 3" in compact.output
    assert "Method: token_hash" in compact.output

    capped = runner.invoke(cli.cli, ["check", str(path), "--max-duplicates", "1"])
    assert "Exact Duplicate Families (1 family)" in capped.output
    assert "no reported pairs; 1 semantic_review" in capped.output
    assert "1 (1 hybrid_confirmed; use --max-duplicates all)" in capped.output

    payload = json.loads(runner.invoke(cli.cli, ["check", str(path), "--json"]).output)
    assert len(payload["exact_families"]) == 1
    assert len(payload["exact_families"][0]["members"]) == 3
    assert payload["exact_families"][0]["method"] == "token_hash"
    assert [edge["tier"] for edge in payload["duplicates"]] == ["hybrid_confirmed"]
    assert payload["summary"]["hybrid_duplicates"] == 3
    assert payload["summary"]["reported_duplicates"] == 2
    assert payload["summary"]["omitted_review_duplicates"] == 1
    assert payload["summary"]["exact_family_members"] == 3


def test_cli_default_report_lists_the_same_twenty_pairs_in_terminal_and_json(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    patch_cli_analyzer(monkeypatch, cli, analyze_result=_build_capped_result(tmp_path))
    runner = CliRunner()

    terminal = runner.invoke(cli.cli, ["check", str(path)])
    as_json = runner.invoke(cli.cli, ["check", str(path), "--json"])
    assert terminal.exit_code == 1
    assert as_json.exit_code == 1
    payload = json.loads(as_json.output)
    summary = payload["summary"]

    pairs = _json_pairs(payload)
    assert len(pairs) == 20
    assert _terminal_pairs(terminal.output) == pairs
    # Actionable pairs lead even though the analyzer ranked them lower.
    assert [edge["tier"] for edge in payload["duplicates"][:12]] == ["hybrid_confirmed"] * 12
    assert [edge["tier"] for edge in payload["duplicates"][12:]] == ["semantic_high_confidence"] * 8
    assert summary["max_duplicates"] == 20
    assert summary["reported_duplicates"] == 20
    assert summary["truncated_duplicates"] == 5
    assert summary["truncated_by_tier"] == {
        "exact": 0,
        "traditional_near": 0,
        "hybrid_confirmed": 0,
        "semantic_high_confidence": 5,
        "semantic_review": 0,
    }
    assert summary["actionable_duplicates"] == 12
    assert summary["reported_actionable_duplicates"] == 12
    assert summary["strict_unused"] is False
    assert summary["hidden_only_failure"] == []
    assert "(20 pairs, 5 truncated)" in terminal.output
    assert "5 (5 semantic_high_confidence; use --max-duplicates all)" in terminal.output
    assert "Actionable duplicates" in terminal.output
    assert "12 (12 reported)" in terminal.output
    assert "more (use --full-table" not in terminal.output


@pytest.mark.parametrize(
    "expansion",
    [["--show-all"], ["--include-review"], ["--max-duplicates", "all"]],
    ids=lambda value: " ".join(value),
)
def test_cli_expansion_flags_lift_the_default_cap(monkeypatch, tmp_path, expansion):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    patch_cli_analyzer(monkeypatch, cli, analyze_result=_build_capped_result(tmp_path))
    runner = CliRunner()

    terminal = runner.invoke(cli.cli, ["check", str(path), *expansion])
    as_json = runner.invoke(cli.cli, ["check", str(path), "--json", *expansion])
    payload = json.loads(as_json.output)

    pairs = _json_pairs(payload)
    assert len(pairs) == 25
    assert _terminal_pairs(terminal.output) == pairs
    assert payload["summary"]["max_duplicates"] is None
    # Expansion flags lift both caps; an explicit --max-duplicates only lifts its own.
    lifts_unused = expansion != ["--max-duplicates", "all"]
    assert payload["summary"]["max_unused"] == (None if lifts_unused else 20)
    assert payload["summary"]["truncated_duplicates"] == 0
    assert "(25 pairs)" in terminal.output
    assert "Truncated duplicates" not in terminal.output


@pytest.mark.parametrize(
    "options",
    [
        ["--show-all", "--max-duplicates", "3"],
        ["--max-duplicates", "3", "--show-all"],
        ["--include-review", "--max-duplicates", "3"],
    ],
    ids=lambda value: " ".join(value),
)
def test_cli_explicit_max_duplicates_wins_over_expansion_flags(monkeypatch, tmp_path, options):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    patch_cli_analyzer(monkeypatch, cli, analyze_result=_build_capped_result(tmp_path))

    as_json = CliRunner().invoke(cli.cli, ["check", str(path), "--json", *options])
    payload = json.loads(as_json.output)

    assert len(payload["duplicates"]) == 3
    assert payload["summary"]["max_duplicates"] == 3
    assert payload["summary"]["truncated_duplicates"] == 22
    assert payload["summary"]["max_unused"] is None


@pytest.mark.parametrize(
    "options",
    [["--show-all", "--max-unused", "3"], ["--max-unused", "3", "--full-table"]],
    ids=lambda value: " ".join(value),
)
def test_cli_explicit_max_unused_wins_over_expansion_flags(monkeypatch, tmp_path, options):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    patch_cli_analyzer(monkeypatch, cli, analyze_result=_build_capped_result_with_unused(tmp_path))

    result = CliRunner().invoke(cli.cli, ["check", str(path), *options])

    assert result.exit_code == 1
    assert "(25 pairs)" in result.output
    assert "Likely Dead Code (3 units, 22 truncated)" in result.output


def test_cli_cap_keeps_the_actionable_pair_ahead_of_a_stronger_advisory_pair(monkeypatch, tmp_path):
    from codedupes import analyzer as analyzer_module

    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    kwargs = {"identifiers": frozenset({"x"}), "statement_count": 1}
    entry = make_code_unit(tmp_path, name="entry", source="def entry():\n    return 1", **kwargs)
    other = make_code_unit(
        tmp_path, name="other", source="def other():\n    return 2", lineno=5, **kwargs
    )
    third = make_code_unit(
        tmp_path, name="third", source="def third():\n    return 3", lineno=9, **kwargs
    )
    traditional = [DuplicatePair(entry, other, 0.86, "jaccard")]
    semantic = [
        DuplicatePair(entry, third, 0.95, "semantic"),
        DuplicatePair(entry, other, 0.90, "semantic"),
    ]
    hybrid = analyzer_module._synthesize_hybrid_duplicates(
        traditional, semantic, jaccard_threshold=0.85
    )
    # Production synthesis ranks the uncorroborated pair first by score.
    assert [pair.tier for pair in hybrid] == ["semantic_high_confidence", "hybrid_confirmed"]
    result = AnalysisResult(
        units=[entry, other, third],
        traditional_duplicates=traditional,
        semantic_duplicates=semantic,
        hybrid_duplicates=hybrid,
        potentially_unused=[],
        analysis_mode="combined",
    )
    patch_cli_analyzer(monkeypatch, cli, analyze_result=result)
    runner = CliRunner()

    capped = runner.invoke(cli.cli, ["check", str(path), "--max-duplicates", "1"])
    assert capped.exit_code == 1
    table = capped.output.split("Hybrid Duplicates")[1]
    assert "hybrid_confirmed" in table
    assert "other" in table
    assert "third" not in table
    assert "fail (exit 1)" in capped.output
    assert "to list them in the primary report" not in capped.output
    assert "1 (1 semantic_high_confidence; use --max-duplicates all)" in capped.output

    as_json = runner.invoke(cli.cli, ["check", str(path), "--json", "--max-duplicates", "1"])
    payload = json.loads(as_json.output)
    assert [edge["tier"] for edge in payload["duplicates"]] == ["hybrid_confirmed"]
    assert payload["summary"]["truncated_duplicates"] == 1
    assert payload["summary"]["actionable_duplicates"] == 1
    assert payload["summary"]["reported_actionable_duplicates"] == 1
    assert payload["summary"]["hidden_only_failure"] == []

    raw = runner.invoke(
        cli.cli, ["check", str(path), "--json", "--show-all", "--max-duplicates", "1"]
    )
    raw_payload = json.loads(raw.output)
    assert len(raw_payload["duplicates"]) == 1
    assert len(raw_payload["traditional_duplicates"]) == 1
    assert len(raw_payload["semantic_duplicates"]) == 2

    # Report ranking never reorders the analyzer's result.
    assert [pair.tier for pair in result.hybrid_duplicates] == [
        "semantic_high_confidence",
        "hybrid_confirmed",
    ]


def test_cli_max_duplicates_ranks_traditional_only_by_similarity(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = build_unit(tmp_path)
    other = make_code_unit(tmp_path, name="other", source="def other():\n    return 2", lineno=5)
    result = replace(
        _build_tiered_result(tmp_path),
        # analyzer order is index-pair order, not similarity: 0.86 precedes 0.99.
        traditional_duplicates=[
            DuplicatePair(unit_a=unit, unit_b=other, similarity=0.86, method="jaccard"),
            DuplicatePair(unit_a=other, unit_b=unit, similarity=0.99, method="jaccard"),
        ],
        semantic_duplicates=[],
        hybrid_duplicates=[],
        analysis_mode="traditional",
    )
    patch_cli_analyzer(monkeypatch, cli, analyze_result=result)
    runner = CliRunner()

    as_json = runner.invoke(
        cli.cli,
        [
            "check",
            str(path),
            "--traditional-only",
            "--no-unused",
            "--json",
            "--max-duplicates",
            "1",
        ],
    )

    assert as_json.exit_code == 1
    output = json.loads(as_json.output)
    assert len(output["duplicates"]) == 1
    assert output["duplicates"][0]["similarity"] == 0.99
    assert output["summary"]["truncated_duplicates"] == 1


@pytest.mark.parametrize("include_review", [False, True], ids=["default", "include-review"])
def test_cli_review_visibility(monkeypatch, tmp_path, include_review):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    patch_cli_analyzer(monkeypatch, cli, analyze_result=lambda: _build_tiered_result(tmp_path))
    runner = CliRunner()
    options = ["--include-review"] if include_review else []

    json_result = runner.invoke(
        cli.cli,
        ["check", str(path), "--json", "--no-unused", *options],
    )
    assert json_result.exit_code == 1
    output = json.loads(json_result.output)
    assert output["summary"]["hybrid_duplicates"] == 3
    assert output["summary"]["duplicates_by_tier"] == {
        "exact": 0,
        "traditional_near": 0,
        "hybrid_confirmed": 1,
        "semantic_high_confidence": 0,
        "semantic_review": 2,
    }
    assert "traditional_duplicates" not in output

    terminal = runner.invoke(cli.cli, ["check", str(path), "--no-unused", *options])
    assert terminal.exit_code == 1
    if include_review:
        assert output["summary"]["reported_duplicates"] == 3
        assert output["summary"]["omitted_review_duplicates"] == 0
        assert [edge["tier"] for edge in output["duplicates"]] == [
            "hybrid_confirmed",
            "semantic_review",
            "semantic_review",
        ]
        assert {record["name"] for record in output["units"].values()} == {
            "entry",
            "other",
            "lonely",
        }
        assert "Withheld review candidates" not in terminal.output
        assert "(3 pairs)" in terminal.output
        assert "lonely" in terminal.output
    else:
        assert output["summary"]["reported_duplicates"] == 1
        assert output["summary"]["omitted_review_duplicates"] == 2
        assert [edge["tier"] for edge in output["duplicates"]] == ["hybrid_confirmed"]
        assert {record["name"] for record in output["units"].values()} == {"entry", "other"}
        assert "Reported duplicates" in terminal.output
        assert "Withheld review candidates" in terminal.output
        assert "2 (use --include-review)" in terminal.output
        assert "semantic_review" in terminal.output
        assert "(1 pair, 2 review withheld)" in terminal.output
        assert "lonely" not in terminal.output


def test_cli_show_all_implies_include_review(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    patch_cli_analyzer(monkeypatch, cli, analyze_result=lambda: _build_tiered_result(tmp_path))
    runner = CliRunner()

    result = runner.invoke(cli.cli, ["check", str(path), "--json", "--no-unused", "--show-all"])
    assert result.exit_code == 1
    output = json.loads(result.output)
    assert output["summary"]["reported_duplicates"] == 3
    assert output["summary"]["omitted_review_duplicates"] == 0
    assert output["summary"]["max_duplicates"] is None
    assert len(output["traditional_duplicates"]) == 1
    assert len(output["semantic_duplicates"]) == 3


def test_cli_max_duplicates_caps_the_report_but_not_the_exit_code(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    patch_cli_analyzer(monkeypatch, cli, analyze_result=lambda: _build_tiered_result(tmp_path))
    runner = CliRunner()

    json_result = runner.invoke(
        cli.cli,
        ["check", str(path), "--json", "--no-unused", "--include-review", "--max-duplicates", "2"],
    )
    assert json_result.exit_code == 1
    output = json.loads(json_result.output)
    summary = output["summary"]
    assert summary["hybrid_duplicates"] == 3
    assert summary["reported_duplicates"] == 2
    assert summary["omitted_review_duplicates"] == 0
    assert summary["truncated_duplicates"] == 1
    assert summary["max_duplicates"] == 2
    assert [edge["tier"] for edge in output["duplicates"]] == [
        "hybrid_confirmed",
        "semantic_review",
    ]
    # Ids stay resolvable; units referenced only by the cut pair are dropped.
    referenced = {edge[key] for edge in output["duplicates"] for key in ("unit_a", "unit_b")}
    assert referenced == set(output["units"])

    terminal = runner.invoke(cli.cli, ["check", str(path), "--no-unused", "--max-duplicates", "1"])
    assert terminal.exit_code == 1
    # Default policy withholds both review pairs first, so the cap cuts nothing
    # and the summary does not mention it.
    assert "Truncated duplicates" not in terminal.output
    assert "(1 pair, 2 review withheld)" in terminal.output

    capped = runner.invoke(
        cli.cli,
        ["check", str(path), "--no-unused", "--include-review", "--max-duplicates", "1"],
    )
    assert capped.exit_code == 1
    # Review pairs rank last, so the cap cut both of them; the note says so
    # because the withheld row is absent under --include-review.
    assert "2 (2 semantic_review; use --max-duplicates all)" in capped.output
    assert "Withheld review candidates" not in capped.output
    assert "(1 pair, 2 truncated)" in capped.output
    assert "lonely" not in capped.output

    cut_review = runner.invoke(
        cli.cli,
        ["check", str(path), "--json", "--no-unused", "--include-review", "--max-duplicates", "1"],
    )
    cut_summary = json.loads(cut_review.output)["summary"]
    assert cut_summary["omitted_review_duplicates"] == 0
    assert cut_summary["truncated_duplicates"] == 2
    assert cut_summary["truncated_by_tier"]["semantic_review"] == 2
    assert cut_summary["duplicates_by_tier"]["semantic_review"] == 2


def test_cli_max_duplicates_applies_to_single_method_modes(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    result = replace(
        _build_tiered_result(tmp_path),
        traditional_duplicates=[],
        hybrid_duplicates=[],
        analysis_mode="semantic",
    )
    patch_cli_analyzer(monkeypatch, cli, analyze_result=result)
    runner = CliRunner()

    as_json = runner.invoke(
        cli.cli,
        ["check", str(path), "--semantic-only", "--no-unused", "--json", "--max-duplicates", "1"],
    )
    assert as_json.exit_code == 1
    output = json.loads(as_json.output)
    assert output["summary"]["raw_semantic_duplicates"] == 3
    assert output["summary"]["reported_duplicates"] == 1
    assert output["summary"]["truncated_duplicates"] == 2
    assert len(output["duplicates"]) == 1

    terminal = runner.invoke(
        cli.cli, ["check", str(path), "--semantic-only", "--no-unused", "--max-duplicates", "1"]
    )
    assert terminal.exit_code == 1
    assert "Reported duplicates" in terminal.output
    assert "2 (use --max-duplicates all)" in terminal.output
    assert "(1 pair, 2 truncated)" in terminal.output


def test_cli_withheld_only_result_prints_a_placeholder_instead_of_nothing(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = build_unit(tmp_path)
    review_only = AnalysisResult(
        units=[unit],
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[
            HybridDuplicate(unit_a=unit, unit_b=unit, tier="semantic_review", score=0.8)
        ],
        potentially_unused=[],
        analysis_mode="combined",
    )
    patch_cli_analyzer(monkeypatch, cli, analyze_result=review_only)
    runner = CliRunner()

    result = runner.invoke(cli.cli, ["check", str(path)])
    assert result.exit_code == 0
    assert "Hybrid Duplicates: no reported pairs; 1 semantic_review" in result.output
    assert "Withheld review candidates" in result.output
    assert "pass (exit 0)" in result.output
    # --fail-on actionable ignores withheld review pairs, so no withheld-only note.
    assert "only withheld" not in result.output

    # Under --fail-on all the withheld pair decides the exit code, and the
    # terminal says so instead of failing over an invisible finding.
    strict = runner.invoke(cli.cli, ["check", str(path), "--fail-on", "all"])
    assert strict.exit_code == 1
    assert "only withheld semantic_review candidates fail --fail-on all" in strict.output
    assert "use --include-review to list them" in strict.output
    listed = runner.invoke(cli.cli, ["check", str(path), "--fail-on", "all", "--include-review"])
    assert listed.exit_code == 1
    assert "fail (exit 1)" in listed.output
    assert "only withheld" not in listed.output

    # JSON names the same hidden group so consumers can explain the exit code.
    as_json = runner.invoke(cli.cli, ["check", str(path), "--json", "--fail-on", "all"])
    summary = json.loads(as_json.output)["summary"]
    assert summary["exit_code"] == 1
    assert summary["reported_duplicates"] == 0
    assert summary["hidden_only_failure"] == ["review"]


def test_cli_semantic_only_uses_raw_findings_for_exit(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = build_unit(tmp_path)
    duplicate = DuplicatePair(unit_a=unit, unit_b=unit, similarity=0.95, method="semantic")
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=AnalysisResult(
            units=[unit],
            traditional_duplicates=[],
            semantic_duplicates=[duplicate],
            hybrid_duplicates=[],
            potentially_unused=[],
            analysis_mode="semantic",
        ),
    )

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--semantic-only"])
    assert result.exit_code == 1
    assert "Semantic Duplicates (Embedding)" in result.output
