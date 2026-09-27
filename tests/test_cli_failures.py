"""Failure paths: semantic backend errors, fallback, stderr routing, and process exit."""

from __future__ import annotations

import json
import logging
import sys

import pytest
from click.testing import CliRunner

from codedupes import cli
from codedupes.languages import GrammarUnavailableError
from codedupes.logging_utils import NOISY_EXTERNAL_LOGGERS
from codedupes.models import (
    AnalysisResult,
)
from codedupes.semantic import SemanticBackendError
from tests.cli_helpers import build_result
from tests.conftest import patch_cli_analyzer


def _raise_semantic_backend_error(*_args, **_kwargs):
    raise SemanticBackendError("semantic backend mismatch")


@pytest.mark.parametrize(
    ("command", "tail_args"),
    [("check", []), ("search", ["entry"])],
)
def test_cli_surfaces_analyzer_config_validation_error(monkeypatch, tmp_path, command, tail_args):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    def _raise_config_error(**_kwargs):
        raise ValueError("invalid config")

    monkeypatch.setattr(cli, "AnalyzerConfig", _raise_config_error)

    runner = CliRunner()
    result = runner.invoke(cli.cli, [command, str(path), *tail_args])
    assert result.exit_code == 2
    assert "invalid config" in result.output


def test_cli_check_fails_on_semantic_backend_error_without_fallback(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def _dead():\n    return 1\n\ndef keep(y):\n    return y + 1\n")

    from codedupes import analyzer as analyzer_module

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", _raise_semantic_backend_error)

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--min-statements", "0"])
    assert result.exit_code == 3
    assert "Error during analysis" in result.output
    assert "--allow-semantic-fallback" in result.output
    # The wrapper must carry the root cause: --verbose is the only other route
    # to it and it is rejected with --json.
    assert "semantic backend mismatch" in result.output


def test_cli_check_degrades_on_semantic_backend_error_with_fallback(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def _dead():\n    return 1\n\ndef keep(y):\n    return y + 1\n")

    from codedupes import analyzer as analyzer_module

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", _raise_semantic_backend_error)

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--min-statements", "0", "--allow-semantic-fallback"],
    )
    assert result.exit_code == 0
    assert "Semantic analysis unavailable" in result.output


def test_cli_check_degrades_on_semantic_backend_error_in_json(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def _dead():\n    return 1\n\ndef keep(y):\n    return y + 1\n")

    from codedupes import analyzer as analyzer_module

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", _raise_semantic_backend_error)

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--min-statements", "0", "--allow-semantic-fallback", "--json"],
    )
    assert result.exit_code == 0

    assert result.output.lstrip().startswith("{"), (
        f"Expected pure JSON output, got: {result.output!r}"
    )
    payload = json.loads(result.output)
    assert payload["summary"]["exit_code"] == 0
    assert payload["summary"]["semantic_fallback"] is True
    assert payload["summary"]["semantic_fallback_reason"] is not None
    assert "Semantic analysis unavailable" in payload["summary"]["semantic_fallback_reason"]
    assert payload["analysis_status"] == "partial"
    assert payload["run"]["checks"]["semantic"]["status"] == "fallback"


@pytest.mark.parametrize(
    ("error", "expected_text"),
    [
        (RuntimeError("analysis exploded"), "Error during analysis: analysis exploded"),
        (GrammarUnavailableError("grammar missing"), "Parser unavailable: grammar missing"),
        (FileNotFoundError("Path does not exist"), "Error: Path does not exist"),
    ],
    ids=["runtime", "grammar", "missing-path"],
)
def test_cli_check_json_keeps_errors_off_stdout(monkeypatch, tmp_path, error, expected_text):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    def _raise() -> AnalysisResult:
        raise error

    patch_cli_analyzer(monkeypatch, cli, analyze_result=_raise)

    result = CliRunner().invoke(cli.cli, ["check", str(path), "--json"])

    assert result.exit_code == 3
    # --json promises machine-parseable JSON only on stdout.
    assert result.stdout == ""
    assert expected_text in result.stderr


def test_cli_check_log_output_goes_to_stderr(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    result = CliRunner().invoke(cli.cli, ["check", str(path), "--traditional-only"])

    assert "Extracting code units" in result.stderr
    assert "Extracting code units" not in result.stdout


def test_cli_check_verbose_traceback_goes_to_stderr(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    def _raise() -> AnalysisResult:
        raise RuntimeError("analysis exploded")

    patch_cli_analyzer(monkeypatch, cli, analyze_result=_raise)

    result = CliRunner().invoke(cli.cli, ["check", str(path), "--verbose"])

    assert result.exit_code == 3
    assert "Traceback" in result.stderr
    assert "Traceback" not in result.stdout


@pytest.mark.parametrize(
    ("args", "expected_message"),
    [
        (["check", "--semantic-only", "--min-statements", "0"], "Error during analysis"),
        (["search", "entry"], "Error during search"),
    ],
)
def test_cli_semantic_required_modes_fail_on_semantic_backend_error(
    monkeypatch, tmp_path, args, expected_message
):
    path = tmp_path / "sample.py"
    path.write_text("def entry(x):\n    return x + 1\n")

    from codedupes import analyzer as analyzer_module

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", _raise_semantic_backend_error)
    # `search` builds its corpus through index()/compute_embeddings, not the
    # duplicate-mining entry point.
    monkeypatch.setattr(analyzer_module, "compute_embeddings", _raise_semantic_backend_error)

    runner = CliRunner()
    result = runner.invoke(cli.cli, [args[0], str(path), *args[1:]])
    assert result.exit_code == 3
    assert expected_message in result.output


def test_setup_logging_quiets_external_loggers() -> None:
    prior = {name: logging.getLogger(name).level for name in NOISY_EXTERNAL_LOGGERS}
    try:
        cli.setup_logging(verbose=False)
        for logger_name in NOISY_EXTERNAL_LOGGERS:
            assert logging.getLogger(logger_name).level == logging.WARNING
    finally:
        for name, level in prior.items():
            logging.getLogger(name).setLevel(level)


def test_main_propagates_check_exit_code(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    patch_cli_analyzer(monkeypatch, cli, analyze_result=lambda: build_result(tmp_path))
    monkeypatch.setattr(sys, "argv", ["codedupes", "check", str(path), "--json"])

    assert cli.main() == 1


def test_main_renders_usage_errors_with_rich(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["codedupes", "info", "--unknown-option"])

    assert cli.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "╭" in captured.err and "╯" in captured.err
    assert "No such option" in captured.err
    assert "--unknown-option" in captured.err
