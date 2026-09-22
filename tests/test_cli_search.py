"""``codedupes search``: indexing, ranking, result levels, and query validation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from click.testing import CliRunner

from codedupes import cli
from codedupes.models import (
    AnalysisResult,
    ExtractionDiagnostic,
)
from tests.cli_helpers import build_result, build_unit
from tests.conftest import make_code_unit, patch_cli_analyzer
from tests.embedding_cache_helpers import CountingModel, patch_get_model


def test_cli_search_json_surfaces_semantic_diagnostics(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = build_unit(tmp_path)
    result_obj = AnalysisResult(
        units=[unit],
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[],
        potentially_unused=[],
        analysis_mode="semantic",
    )
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=result_obj,
        search_results=[(unit, 0.91)],
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
    runner = CliRunner()

    result = runner.invoke(cli.cli, ["search", str(path), "entry", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    result_uid = payload["results"][0]["unit"]
    assert payload["units"][result_uid]["name"] == "entry"
    assert payload["semantic_diagnostics"][0]["code"] == "semantic-warning"


@pytest.mark.parametrize(
    ("filename", "source", "diagnostic_code"),
    [
        ("broken.py", "def broken(\n", "partial-parse"),
        ("broken.js", "function broken( {", "partial-parse"),
    ],
)
def test_cli_search_surfaces_extraction_failures(
    tmp_path: Path, filename: str, source: str, diagnostic_code: str
) -> None:
    """Preserve real Tree-sitter failures in both search report formats."""
    path = tmp_path / filename
    path.write_text(source, encoding="utf-8")
    runner = CliRunner()
    args = ["search", str(path), "entry", "--no-cache", "--min-statements", "99"]
    result = runner.invoke(cli.cli, [*args, "--json"])

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["summary"]["indexed_units"] == 0
    assert payload["results"] == []
    assert payload["extraction_diagnostics"][0]["code"] == diagnostic_code
    assert payload["extraction_diagnostics"][0]["file"] == str(path)

    terminal = runner.invoke(cli.cli, args)
    assert terminal.exit_code == 0, terminal.output
    assert "Extraction diagnostics" in terminal.stdout
    assert " ".join(payload["extraction_diagnostics"][0]["message"].split()) in " ".join(
        terminal.stdout.split()
    )
    assert filename in terminal.stdout


def test_cli_search_indexes_without_running_full_analysis(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    class IndexOnlyAnalyzer:
        def __init__(self, config):
            del config
            self.extraction_diagnostics = []
            self.semantic_diagnostics = []
            self.embedding_stats = None

        def analyze(self, _path):
            raise AssertionError("search must build its corpus via index(), not analyze()")

        def index(self, _path):
            return 1

        def search(self, query, top_k=10):
            del query, top_k
            return [(build_unit(tmp_path), 0.99)]

    monkeypatch.setattr(cli, "CodeAnalyzer", IndexOnlyAnalyzer)
    runner = CliRunner()

    result = runner.invoke(cli.cli, ["search", str(path), "entry", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    result_uid = payload["results"][0]["unit"]
    assert payload["units"][result_uid]["name"] == "entry"


@pytest.mark.parametrize("result_level", [None, "file"])
@pytest.mark.parametrize("as_json", [False, True])
def test_cli_search_file_ranking_groups_before_top_k(
    monkeypatch, tmp_path: Path, result_level: str | None, as_json: bool
) -> None:
    """Return distinct files without letting one file's unit hits fill top-k."""
    hits = [
        (
            replace(
                make_code_unit(tmp_path, name=name, source=f"def {name}():\n    return 1"),
                file_path=tmp_path / directory / "shared.py",
                lineno=line,
            ),
            score,
        )
        for directory, name, line, score in [
            ("first", "alpha", 10, 0.99),
            ("first", "beta", 20, 0.98),
            ("first", "gamma", 30, 0.97),
            ("first", "delta", 40, 0.96),
            ("second", "epsilon", 50, 0.95),
            ("third", "zeta", 60, 0.90),
        ]
    ]
    requested_limits = []

    class RankedAnalyzer:
        def __init__(self, config):
            self.extraction_diagnostics = []
            self.semantic_diagnostics = []
            self.embedding_stats = None

        def index(self, path):
            return len(hits)

        def search(self, query, top_k=10):
            requested_limits.append(top_k)
            return hits[:top_k]

    monkeypatch.setattr(cli, "CodeAnalyzer", RankedAnalyzer)
    args = ["search", str(tmp_path), "find helpers", "--top-k", "2"]
    if result_level is not None:
        args += ["--result-level", result_level]
    if as_json:
        args += ["--json"]
    result = CliRunner().invoke(cli.cli, args)

    assert result.exit_code == 0, result.output
    assert requested_limits == [6 if result_level == "file" else 2]
    if as_json:
        payload = json.loads(result.output)
        assert payload["summary"]["results"] == 2
        assert payload["summary"]["indexed_units"] == 6
        if result_level == "file":
            assert payload["result_level"] == "file"
            first, second = payload["results"]
            assert first["file"] == str(tmp_path / "first" / "shared.py")
            assert second["file"] == str(tmp_path / "second" / "shared.py")
            assert first["score"] == 0.99
            assert second["score"] == 0.95
            assert first["matching_units"] == 4
            assert second["matching_units"] == 1
            assert [payload["units"][hit["unit"]]["name"] for hit in first["matches"]] == [
                "alpha",
                "beta",
                "gamma",
            ]
            assert [hit["score"] for hit in first["matches"]] == [0.99, 0.98, 0.97]
            assert len(payload["units"]) == 4
        else:
            assert "result_level" not in payload
            assert [payload["units"][hit["unit"]]["name"] for hit in payload["results"]] == [
                "alpha",
                "beta",
            ]
    elif result_level == "file":
        assert "Matching code units" in result.output
        assert "sample.alpha:10 (99.00%)" in result.output
        assert "sample.gamma:30 (97.00%)" in result.output
        assert "sample.epsilon:50 (95.00%)" in result.output
        assert "+1 more matching units" in result.output
        assert "delta" not in result.output
        assert "zeta" not in result.output
    else:
        assert "alpha" in result.output
        assert "beta" in result.output
        assert "gamma" not in result.output


@pytest.mark.parametrize("indexed_units", [0, 4])
@pytest.mark.parametrize("as_json", [False, True])
def test_cli_file_search_without_matches(monkeypatch, tmp_path, indexed_units, as_json):
    if indexed_units:
        _patch_search_analyzer(monkeypatch, indexed_units=indexed_units)
    args = ["search", str(tmp_path), "nothing", "--result-level", "file"]
    if as_json:
        args.append("--json")
    result = CliRunner().invoke(cli.cli, args)

    assert result.exit_code == 0, result.output
    if as_json:
        payload = json.loads(result.output)
        assert payload["result_level"] == "file"
        assert payload["results"] == []
        assert payload["units"] == {}
        assert payload["summary"]["results"] == 0
        assert payload["summary"]["indexed_units"] == indexed_units
    else:
        assert "No matches found" in result.output


def test_cli_search_rejects_unknown_result_level_before_indexing(monkeypatch, tmp_path):
    def unexpected_analyzer(config):
        raise AssertionError("Invalid result levels must fail before indexing")

    monkeypatch.setattr(cli, "CodeAnalyzer", unexpected_analyzer)
    result = CliRunner().invoke(
        cli.cli, ["search", str(tmp_path), "entry", "--result-level", "directory"]
    )
    assert result.exit_code == 2
    assert "Invalid value" in result.output


@pytest.mark.parametrize("query", ["", " \t"])
def test_cli_search_rejects_blank_query_before_indexing(monkeypatch, tmp_path, query):
    def unexpected_analyzer(config):
        raise AssertionError("Blank queries must fail before indexing")

    monkeypatch.setattr(cli, "CodeAnalyzer", unexpected_analyzer)
    result = CliRunner().invoke(cli.cli, ["search", str(tmp_path), query])

    assert result.exit_code == 2
    assert "query must be a non-empty string" in result.output


def _patch_search_analyzer(
    monkeypatch,
    *,
    indexed_units: int = 0,
    extracted_unit_count: int = 1,
    results: list | None = None,
    index_error: Exception | None = None,
    semantic_diagnostics: list[ExtractionDiagnostic] | None = None,
) -> None:
    """Patch the CLI analyzer with a search double that controls the index size."""

    class StubSearchAnalyzer:
        def __init__(self, config):
            del config
            self.extracted_unit_count = extracted_unit_count
            self.extraction_diagnostics = []
            self.semantic_diagnostics = list(semantic_diagnostics or [])
            self.embedding_stats = None

        def index(self, _path):
            if index_error is not None:
                raise index_error
            return indexed_units

        def search(self, query, top_k=10):
            del query, top_k
            return list(results or [])

    monkeypatch.setattr(cli, "CodeAnalyzer", StubSearchAnalyzer)


def test_cli_search_warns_when_candidate_filters_emptied_the_index(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    _patch_search_analyzer(monkeypatch, indexed_units=0)

    result = CliRunner().invoke(cli.cli, ["search", str(path), "entry", "--min-statements", "3"])

    assert result.exit_code == 0
    assert "search index is empty" in result.stderr
    assert "--min-statements" in result.stderr
    # The zero-hit table still renders, but no longer alone.
    assert "No matches found" in result.stdout


def test_cli_search_empty_extraction_warning_does_not_blame_candidate_filters(
    monkeypatch, tmp_path
):
    path = tmp_path / "empty"
    path.mkdir()
    _patch_search_analyzer(monkeypatch, indexed_units=0, extracted_unit_count=0)

    result = CliRunner().invoke(cli.cli, ["search", str(path), "entry"])

    assert result.exit_code == 0
    assert "extraction produced no code units" in result.stderr
    assert "--min-statements" not in result.stderr
    assert "--semantic-unit-type" not in result.stderr


def test_cli_search_empty_index_uses_eligibility_reason_with_semantic_diagnostics(
    monkeypatch, tmp_path
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    diagnostic = ExtractionDiagnostic(
        file_path=path,
        language="python",
        code="semantic-warning",
        message="sample.entry has a semantic warning",
        lineno=1,
        end_lineno=2,
    )
    _patch_search_analyzer(
        monkeypatch,
        indexed_units=0,
        extracted_unit_count=1,
        semantic_diagnostics=[diagnostic],
    )

    # A wide console keeps the diagnostic on one line: the default width wraps
    # it at a point that depends on the pytest tmp-path length, splitting the
    # asserted message on some machines.
    result = CliRunner().invoke(cli.cli, ["search", str(path), "entry", "--output-width", "400"])

    assert result.exit_code == 0
    assert "semantic eligibility filtering removed all" in result.stderr
    assert "Semantic diagnostics" in result.stdout
    assert "semantic warning" in result.stdout


def test_cli_search_does_not_warn_when_the_index_has_units(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    _patch_search_analyzer(monkeypatch, indexed_units=4)

    result = CliRunner().invoke(cli.cli, ["search", str(path), "entry"])

    assert result.exit_code == 0
    assert "search index is empty" not in result.output
    assert "No matches found" in result.stdout


@pytest.mark.parametrize("indexed_units", [0, 7])
def test_cli_search_json_reports_indexed_unit_count(monkeypatch, tmp_path, indexed_units):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    _patch_search_analyzer(monkeypatch, indexed_units=indexed_units)

    result = CliRunner().invoke(cli.cli, ["search", str(path), "entry", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 4
    assert payload["summary"]["indexed_units"] == indexed_units
    assert payload["results"] == []
    assert result.stderr == ""


def test_cli_search_reports_path_deleted_after_validation(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    _patch_search_analyzer(
        monkeypatch,
        index_error=FileNotFoundError("Path does not exist"),
    )

    result = CliRunner().invoke(cli.cli, ["search", str(path), "entry"])

    assert result.exit_code == 1
    assert not isinstance(result.exception, FileNotFoundError)
    assert "Error: Path does not exist" in result.stderr


@pytest.mark.parametrize("phase", ["construction", "query"])
@pytest.mark.parametrize("as_json", [False, True])
def test_cli_search_reports_runtime_failures(monkeypatch, tmp_path, phase, as_json):
    def fail(*args, **kwargs):
        raise FileNotFoundError("model asset disappeared")

    if phase == "construction":
        monkeypatch.setattr(cli, "CodeAnalyzer", fail)
    else:
        patch_cli_analyzer(
            monkeypatch, cli, analyze_result=build_result(tmp_path), search_results=fail
        )
    args = ["search", str(tmp_path), "entry"] + (["--json"] if as_json else [])
    result = CliRunner().invoke(cli.cli, args)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "model asset disappeared" in result.stderr
    assert "Traceback" not in result.stderr
    assert not isinstance(result.exception, FileNotFoundError)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_cli_search_rejects_nonfinite_threshold_before_analysis(monkeypatch, tmp_path, value):
    def unexpected(*args, **kwargs):
        pytest.fail("Non-finite threshold reached analysis")

    monkeypatch.setattr(cli, "CodeAnalyzer", unexpected)
    result = CliRunner().invoke(
        cli.cli,
        ["search", str(tmp_path), "entry", "--threshold", value, "--json"],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "semantic_threshold must be finite" in result.stderr


def test_cli_search_builds_search_mode_config(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result(tmp_path),
        search_results=[],
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            "search",
            str(path),
            "find entry",
            "--instruction-prefix",
            "custom: ",
            "--threshold",
            "-0.5",
        ],
    )

    assert result.exit_code == 0
    assert captured[0].mode == "search"
    assert captured[0].instruction_prefix == "custom: "
    assert captured[0].semantic_threshold == -0.5


@pytest.mark.parametrize("as_json", [False, True])
def test_cli_search_threshold_profile_output(monkeypatch, tmp_path, as_json):
    """Check search-specific threshold-profile rendering in both output modes."""
    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result(tmp_path),
        captured_configs=captured,
        search_results=[],
    )
    args = ["search", str(tmp_path), "query", "--threshold-profile", "generic"]
    if as_json:
        args.append("--json")
    result = CliRunner().invoke(cli.cli, args)
    assert result.exit_code == 0, result.output
    if as_json:
        json.loads(result.stdout)
        assert "Effective search threshold:" not in result.output
    else:
        assert result.output.count("Effective search threshold:") == 1
        assert "threshold-profile=generic" in result.output
    assert "Use --threshold-profile generic" not in result.output
    assert captured[-1].threshold_profile == "generic"
    assert captured[-1].semantic_threshold is None
    result = CliRunner().invoke(
        cli.cli, args + ["--threshold", "0.67", "--semantic-threshold", "0.91"]
    )
    assert result.exit_code == 0, result.output
    assert captured[-1].semantic_threshold == 0.91
    if not as_json:
        assert "Effective search threshold: 0.91 (explicit numeric override)" in result.output


def test_cli_search_defaults_to_code_retrieval_task(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result(tmp_path),
        search_results=[(build_unit(tmp_path), 0.99)],
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["search", str(path), "entry"])
    assert result.exit_code == 0
    assert captured[0].semantic_task == "code-retrieval"
    assert captured[0].semantic_unit_types == ("function", "method")
    assert captured[0].search_document == "source"


@pytest.mark.parametrize(
    "options, reason",
    [
        (["--instruction-prefix", "custom"], "instruction prefix"),
        (["--model-revision", "other"], "model revision"),
        (["--trust-remote-code"], "trust_remote_code"),
        (["--model", "embeddinggemma", "--semantic-task", "classification"], "semantic task"),
    ],
)
def test_cli_search_rejects_uncalibrated_context_before_indexing(
    monkeypatch, tmp_path: Path, options: list[str], reason: str
) -> None:
    """Reject known-invalid options before analyzer construction or corpus work."""
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n", encoding="utf-8")
    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result(tmp_path),
        captured_configs=captured,
    )
    args = ["search", str(path), "entry", *options]
    runner = CliRunner()
    result = runner.invoke(cli.cli, args)
    assert result.exit_code == 2, result.output
    assert reason in " ".join(result.output.replace("│", "").split())
    assert "explicit threshold" in result.output
    assert captured == []

    for option in ("--threshold", "--semantic-threshold"):
        result = runner.invoke(cli.cli, [*args, option, "0.0"])
        assert result.exit_code == 0, result.output
    assert len(captured) == 2


def test_cli_contextual_search_requires_explicit_threshold(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    model = CountingModel()
    patch_get_model(monkeypatch, model)
    args = [
        "search",
        str(path),
        "entry",
        "--search-document",
        "contextual",
        "--min-statements",
        "0",
        "--device",
        "cpu",
        "--json",
    ]
    runner = CliRunner()
    missing = runner.invoke(cli.cli, args)

    assert missing.exit_code == 2
    assert "contextual" in missing.output
    assert "explicit threshold" in missing.output
    assert "--semantic-threshold" in missing.output
    assert model.encode_calls == []

    for option in ("--semantic-threshold", "--threshold"):
        result = runner.invoke(cli.cli, [*args, option, "0.0"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["summary"]["results"] == 1
    assert len(model.encode_calls) == 2
    assert model.encode_calls[-1] == ["entry"]


def test_cli_search_semantic_unit_type_pass_through(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result(tmp_path),
        search_results=[(build_unit(tmp_path), 0.99)],
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["search", str(path), "entry", "--semantic-unit-type", "class"],
    )
    assert result.exit_code == 0
    assert captured[0].semantic_unit_types == ("class",)


def test_cli_search_threshold_precedence(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result(tmp_path),
        search_results=[(build_unit(tmp_path), 0.99)],
        captured_configs=captured,
    )
    runner = CliRunner()

    result_default = runner.invoke(cli.cli, ["search", str(path), "entry"])
    assert result_default.exit_code == 0
    assert captured[-1].semantic_threshold is None

    result_shared = runner.invoke(cli.cli, ["search", str(path), "entry", "--threshold", "0.4"])
    assert result_shared.exit_code == 0
    assert captured[-1].semantic_threshold == 0.4

    result_override = runner.invoke(
        cli.cli,
        ["search", str(path), "entry", "--threshold", "0.4", "--semantic-threshold", "0.6"],
    )
    assert result_override.exit_code == 0
    assert captured[-1].semantic_threshold == 0.6


def test_cli_search_help_is_search_specific() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["search", "--help"])

    assert result.exit_code == 0
    assert "also narrows traditional duplicate scope in combined mode" not in result.output
    assert "Default test exclusions" in result.output
    assert "scan root are always excluded." in result.output


@pytest.mark.parametrize("result_level", ["unit", "file"])
@pytest.mark.parametrize("query", ["parse [/] markup", "render [bold]text[/bold]"])
def test_cli_search_preserves_literal_query_markup(tmp_path, result_level, query):
    result = CliRunner().invoke(
        cli.cli, ["search", str(tmp_path), query, "--result-level", result_level]
    )

    assert result.exit_code == 0, result.output
    assert repr(query) in result.stdout
