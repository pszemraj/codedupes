"""Rich terminal rendering: tables, panels, widths, and literal markup."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from codedupes import cli
from codedupes.models import (
    AnalysisResult,
    DuplicatePair,
    ExtractionDiagnostic,
    HybridDuplicate,
)
from tests.cli_helpers import build_result, build_result_with_semantic_duplicate, build_unit
from tests.conftest import make_code_unit, make_run_record, patch_cli_analyzer


def test_cli_table_output_uses_auto_progress(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result(tmp_path),
        captured_configs=captured,
    )

    result = CliRunner().invoke(cli.cli, ["check", str(path)])

    assert captured[0].progress == "auto"
    assert "Embeddings" in result.output
    assert "model not loaded" in result.output


def test_cli_check_prints_run_panel(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result(tmp_path),
    )

    result = CliRunner().invoke(cli.cli, ["check", str(path)])

    run_pos = result.output.index("Run")
    summary_pos = result.output.index("Analysis Summary")
    assert run_pos < summary_pos
    assert "Hybrid Duplicates" not in result.output[:summary_pos]
    assert "Scope" in result.output
    assert "combined" in result.output
    assert "extraction=completed" in result.output


def test_cli_reports_semantic_diagnostics(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = build_unit(tmp_path)
    result_obj = AnalysisResult(
        units=[unit],
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[],
        potentially_unused=[],
        run=make_run_record(tmp_path, mode="combined"),
        semantic_diagnostics=[
            ExtractionDiagnostic(
                file_path=unit.file_path,
                language="python",
                code="semantic-warning",
                message="sample.entry has a semantic warning",
                lineno=1,
                end_lineno=2,
            )
        ],
    )
    patch_cli_analyzer(monkeypatch, cli, analyze_result=result_obj)
    runner = CliRunner()

    table_result = runner.invoke(cli.cli, ["check", str(path)])
    assert "Semantic diagnostics" in table_result.output
    assert "semantic warning" in table_result.output

    json_result = runner.invoke(cli.cli, ["check", str(path), "--json"])
    payload = json.loads(json_result.output)
    assert payload["run"]["checks"]["semantic"]["diagnostics"] == 1
    assert payload["semantic_diagnostics"][0]["code"] == "semantic-warning"


def test_cli_reports_unused_diagnostics(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = build_unit(tmp_path)
    result_obj = AnalysisResult(
        units=[unit],
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[],
        potentially_unused=[],
        run=make_run_record(tmp_path, mode="combined"),
        unused_diagnostics=[
            ExtractionDiagnostic(
                file_path=unit.file_path,
                language="python",
                code="unused-parse-error",
                message="SyntaxError: invalid syntax",
                lineno=7,
            )
        ],
    )
    patch_cli_analyzer(monkeypatch, cli, analyze_result=result_obj)
    runner = CliRunner()

    table_result = runner.invoke(cli.cli, ["check", str(path)])
    assert "Unused diagnostics" in table_result.output
    assert "invalid syntax" in table_result.output

    json_result = runner.invoke(cli.cli, ["check", str(path), "--json"])
    payload = json.loads(json_result.output)
    assert "unused_diagnostics" not in payload["summary"]
    assert payload["unused_diagnostics"][0]["code"] == "unused-parse-error"


def test_cli_output_width_option(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result_with_semantic_duplicate(tmp_path),
    )

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--output-width", "200"])
    assert result.exit_code == 1
    assert cli.console.width == 200


def test_cli_table_locations_disambiguate_same_named_files(monkeypatch, tmp_path):
    unit_a = make_code_unit(
        tmp_path, name="helper", source="def helper():\n    return 1", lineno=12
    )
    unit_b = make_code_unit(
        tmp_path, name="helper", source="def helper():\n    return 1", lineno=12
    )
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    unit_a.file_path = tmp_path / "alpha" / "utils.py"
    unit_b.file_path = tmp_path / "beta" / "utils.py"
    monkeypatch.chdir(tmp_path)

    assert cli.format_location(unit_a) == os.path.join("alpha", "utils.py") + ":12"
    assert cli.format_location(unit_b) == os.path.join("beta", "utils.py") + ":12"
    assert cli.format_location(unit_a) != cli.format_location(unit_b)


def test_cli_source_panel_titles_preserve_bracketed_module_names(tmp_path: Path) -> None:
    source = "def duplicate(value):\n    result = value + 1\n    return result\n"
    (tmp_path / "[bold].py").write_text(source, encoding="utf-8")
    (tmp_path / "other.py").write_text(source, encoding="utf-8")

    result = CliRunner().invoke(
        cli.cli,
        [
            "check",
            str(tmp_path),
            "--traditional-only",
            "--no-unused",
            "--no-tiny-filter",
            "--show-source",
            "--full-table",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "[bold].duplicate" in result.stdout
    assert "other.duplicate" in result.stdout

    table_only = CliRunner().invoke(
        cli.cli,
        [
            "check",
            str(tmp_path),
            "--traditional-only",
            "--no-unused",
            "--no-tiny-filter",
            "--full-table",
        ],
    )

    assert table_only.exit_code == 1, table_only.output
    assert "[bold].duplicate" in table_only.stdout


def test_cli_source_lines_bounds_source_panels(tmp_path: Path) -> None:
    body = "\n".join(f"    x{i} = {i}" for i in range(5))
    source = f"def duplicate(value):\n{body}\n    return value\n"
    (tmp_path / "a.py").write_text(source, encoding="utf-8")
    (tmp_path / "b.py").write_text(source, encoding="utf-8")

    result = CliRunner().invoke(
        cli.cli,
        [
            "check",
            str(tmp_path),
            "--traditional-only",
            "--no-unused",
            "--no-tiny-filter",
            "--show-source",
            "--source-lines",
            "2",
            "--full-table",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "def duplicate(value):" in result.stdout
    assert "x0 = 0" in result.stdout
    assert "x1 = 1" not in result.stdout
    assert "more line" in result.stdout


def test_cli_show_diff_prints_the_differing_operator(tmp_path: Path) -> None:
    source = "def add(a, b):\n    return a + b\n\n\ndef add_alt(a, b):\n    return a - b\n"
    (tmp_path / "sample.py").write_text(source, encoding="utf-8")

    result = CliRunner().invoke(
        cli.cli,
        [
            "check",
            str(tmp_path),
            "--traditional-only",
            "--no-unused",
            "--no-tiny-filter",
            "--traditional-threshold",
            "0.5",
            "--show-diff",
        ],
    )

    assert result.exit_code == 1, result.output
    # A single-statement body dedents flush left, so only the operator differs.
    assert "-return a + b" in result.stdout
    assert "+return a - b" in result.stdout
    assert "sample.add" in result.stdout
    assert "sample.py:1" in result.stdout


def test_cli_show_diff_skips_token_families_and_diffs_structural_families(
    tmp_path: Path,
) -> None:
    (tmp_path / "file1.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (tmp_path / "file2.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (tmp_path / "file3.py").write_text(
        "def calc(value):\n    result = value + 1\n    return result\n", encoding="utf-8"
    )
    (tmp_path / "file4.py").write_text(
        "def calc(value):\n    total = value + 1\n    return total\n", encoding="utf-8"
    )

    result = CliRunner().invoke(
        cli.cli,
        [
            "check",
            str(tmp_path),
            "--traditional-only",
            "--no-unused",
            "--no-tiny-filter",
            "--show-diff",
            "--full-table",
        ],
    )

    assert result.exit_code == 1, result.output
    # structural_hash family (renamed local): diffed against the first member.
    assert "file3.calc vs file4.calc" in result.stdout
    assert "-result = value + 1" in result.stdout
    assert "+total = value + 1" in result.stdout
    # token_hash family (byte-identical copies): no diff panel, nothing to show.
    assert "file1.helper vs file2.helper" not in result.stdout


def test_cli_show_diff_respects_source_lines_budget(tmp_path: Path) -> None:
    increments_a = "\n".join(["    result = result + 1"] * 5)
    increments_b = "\n".join(["    total = total + 1"] * 5)
    (tmp_path / "file_a.py").write_text(
        f"def calc(value):\n    result = value + 1\n{increments_a}\n    return result\n",
        encoding="utf-8",
    )
    (tmp_path / "file_b.py").write_text(
        f"def calc(value):\n    total = value + 1\n{increments_b}\n    return total\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli.cli,
        [
            "check",
            str(tmp_path),
            "--traditional-only",
            "--no-unused",
            "--no-tiny-filter",
            "--show-diff",
            "--source-lines",
            "6",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "-result = value + 1" in result.stdout
    assert "+total = value + 1" not in result.stdout
    assert "more diff line" in result.stdout


@pytest.mark.parametrize("result_level", ["unit", "file"])
def test_cli_table_locations_preserve_bracketed_path_segments(monkeypatch, tmp_path, result_level):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = build_unit(tmp_path)
    unit.file_path = tmp_path / "corpus" / "pages" / "[id].ts"
    monkeypatch.chdir(tmp_path)
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result(tmp_path),
        search_results=[(unit, 0.99)],
    )

    result = CliRunner().invoke(
        cli.cli, ["search", str(path), "entry", "--result-level", result_level]
    )

    assert result.exit_code == 0
    expected_path = os.path.join("corpus", "pages", "[id].ts")
    assert expected_path + (":1" if result_level == "unit" else "") in result.stdout


def test_cli_diagnostics_preserve_bracketed_fields(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    result_obj = build_result(tmp_path)
    result_obj.extraction_diagnostics = [
        ExtractionDiagnostic(
            file_path=tmp_path / "pages" / "[id].ts",
            language="typescript",
            code="partial-parse",
            message="unexpected [token]",
            lineno=1,
            end_lineno=1,
        )
    ]
    patch_cli_analyzer(monkeypatch, cli, analyze_result=result_obj)

    # A wide console keeps the diagnostic on one line: the default width wraps
    # it at a point that depends on the pytest tmp-path length, splitting the
    # asserted message on some machines.
    result = CliRunner().invoke(cli.cli, ["check", str(path), "--output-width", "400"])

    assert result.exit_code == 1
    assert "[typescript]" in result.stdout
    assert "[id].ts" in result.stdout
    assert "unexpected [token]" in result.stdout


def test_cli_table_location_uses_absolute_path_when_relative_path_is_longer(monkeypatch, tmp_path):
    deep_cwd = tmp_path.joinpath(*(f"level-{index}" for index in range(60)))
    deep_cwd.mkdir(parents=True)
    unit = build_unit(tmp_path)
    unit.file_path = tmp_path / "corpus" / "algorithm.py"
    monkeypatch.chdir(deep_cwd)

    assert cli.format_location(unit) == f"{unit.file_path}:1"


@pytest.mark.parametrize("command", [["info", "--verbose"], ["cache", "info"]])
@pytest.mark.parametrize("width", [80, 160])
def test_cli_diagnostic_tables_respect_width(command, width, monkeypatch, tmp_path):
    cache_path = tmp_path / "[red]literal[/red]" / ("long-cache-path-" * 8)
    monkeypatch.setenv("CODEDUPES_CACHE_DIR", str(cache_path))
    result = CliRunner().invoke(cli.cli, [*command, "--output-width", str(width)])

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert "╭" in result.stdout and "│" in result.stdout
    assert max(map(len, result.stdout.splitlines())) <= width
    # Reassemble wrapped value cells to verify paths are neither markup nor truncated.
    values = "".join(
        line.split("│")[-2].strip() for line in result.stdout.splitlines() if "│" in line
    )
    assert str(cache_path) in values
    assert "\x1b[" not in result.stdout


@pytest.mark.parametrize("command", [["info"], ["cache", "info"], ["cache", "clear"]])
def test_cli_diagnostic_width_validation(command):
    result = CliRunner().invoke(cli.cli, [*command, "--output-width", "79"])
    assert result.exit_code == 2
    assert "must be >= 80" in result.output


@pytest.mark.parametrize(
    "command",
    [[], ["check"], ["search"], ["info"], ["cache"], ["cache", "info"], ["cache", "clear"]],
)
def test_cli_all_command_help_is_formatted(command):
    result = CliRunner().invoke(cli.cli, [*command, "--help"])
    short = CliRunner().invoke(cli.cli, [*command, "-h"])
    assert result.exit_code == short.exit_code == 0
    assert result.stdout == short.stdout
    assert "Usage:" in result.stdout
    assert "╭" in result.stdout
    if command in (["check"], ["search"], ["info"], ["cache", "info"], ["cache", "clear"]):
        assert "--output-width" in result.stdout
        assert "160" in result.stdout


@pytest.mark.parametrize("width", [80, 120])
@pytest.mark.parametrize("command", ["check", "search"])
def test_cli_long_results_keep_scores_and_headers(monkeypatch, tmp_path, width, command):
    monkeypatch.chdir(tmp_path)
    unit = build_unit(tmp_path)
    unit.name = "calculate_normalized_customer_score"
    unit.qualified_name = "customer_scoring.calculate_normalized_customer_score"
    unit.file_path = tmp_path / "deeply" / "nested" / "customer_scoring.py"
    path = tmp_path / "sample.py"
    path.write_text(unit.source)
    duplicate = DuplicatePair(unit, unit, 0.91, "jaccard")
    hybrid = HybridDuplicate(
        unit,
        unit,
        tier="hybrid_confirmed",
        score=0.94,
        semantic_similarity=0.96,
        jaccard_similarity=0.91,
    )
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=AnalysisResult(
            units=[unit],
            traditional_duplicates=[duplicate],
            semantic_duplicates=[],
            hybrid_duplicates=[hybrid],
            potentially_unused=[unit],
            run=make_run_record(tmp_path, mode="combined"),
        ),
        search_results=[(unit, 0.99)],
    )
    args = [command, str(path), "--output-width", str(width)]
    args += ["--show-all"] if command == "check" else ["entry"]
    result = CliRunner().invoke(cli.cli, args)

    assert result.exit_code == (1 if command == "check" else 0), result.output
    assert max(map(len, result.stdout.splitlines())) <= width
    assert "…" not in result.stdout
    if command == "check":
        for field in (
            "Score",
            "Semantic",
            "Jaccard",
            "94.00%",
            "96.00%",
            "91.00%",
            "Similarity",
            "Name",
            "Type",
            "Location",
            "function",
        ):
            assert field in result.stdout
        if width < 120:
            assert "Evidence" in result.stdout and "Code units" in result.stdout
    else:
        for field in ("Rank", "Score", "Name", "Location", "99.00%"):
            assert field in result.stdout
