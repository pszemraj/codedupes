from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from codedupes import cli
from codedupes.devices import DeviceDiagnostics
from codedupes.embedding_cache import EmbeddingCache
from codedupes.models import (
    AnalysisResult,
    CodeUnit,
    CodeUnitType,
    DuplicatePair,
    HybridDuplicate,
)
from codedupes.semantic import SemanticBackendError
from tests.conftest import patch_cli_analyzer


def _build_unit(tmp_path: Path) -> CodeUnit:
    return CodeUnit(
        name="entry",
        qualified_name="sample.entry",
        unit_type=CodeUnitType.FUNCTION,
        file_path=tmp_path / "sample.py",
        lineno=1,
        end_lineno=2,
        source="def entry():\n    return 1",
        is_public=True,
        is_exported=False,
    )


def _build_result(tmp_path: Path) -> AnalysisResult:
    unit = _build_unit(tmp_path)
    duplicate = DuplicatePair(
        unit_a=unit,
        unit_b=unit,
        similarity=1.0,
        method="ast_hash",
    )
    hybrid = HybridDuplicate(
        unit_a=unit,
        unit_b=unit,
        tier="exact",
        confidence=1.0,
        has_exact=True,
    )

    return AnalysisResult(
        units=[unit],
        traditional_duplicates=[duplicate],
        semantic_duplicates=[],
        hybrid_duplicates=[hybrid],
        potentially_unused=[unit],
        analysis_mode="combined",
        filtered_raw_duplicates=0,
    )


def _raise_semantic_backend_error(*_args, **_kwargs):
    raise SemanticBackendError("semantic backend mismatch")


def _build_result_with_semantic_duplicate(tmp_path: Path) -> AnalysisResult:
    result = _build_result(tmp_path)
    unit = _build_unit(tmp_path)
    result.semantic_duplicates = [
        DuplicatePair(unit_a=unit, unit_b=unit, similarity=0.95, method="semantic")
    ]
    return result


def test_cli_json_output_hybrid_default(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        search_results=[(_build_unit(tmp_path), 0.99)],
        captured_configs=captured,
    )
    runner = CliRunner()

    result = runner.invoke(cli.cli, ["check", str(path), "--json"])
    assert result.exit_code == 1
    output = json.loads(result.output)

    assert "summary" in output
    assert output["summary"]["hybrid_duplicates"] == 1
    assert output["summary"]["potentially_unused"] == 1
    assert "hybrid_duplicates" in output
    assert "traditional_duplicates" not in output
    assert "semantic_duplicates" not in output
    assert captured[0].include_private is True

    result = runner.invoke(cli.cli, ["search", str(path), "entry", "--json", "--top-k", "1"])
    assert result.exit_code == 0
    search_output = json.loads(result.output)
    assert search_output["query"] == "entry"
    assert search_output["results"][0]["name"] == "entry"


def test_cli_json_show_all_includes_raw_sections(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result_with_semantic_duplicate(tmp_path),
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--json", "--show-all"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert "traditional_duplicates" in payload
    assert "semantic_duplicates" in payload


def test_cli_no_private_option_check(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--no-private"])
    assert result.exit_code == 1
    assert captured[0].include_private is False


def test_cli_model_semantic_flags_pass_through(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            "check",
            str(path),
            "--instruction-prefix",
            "Represent this code: ",
            "--model-revision",
            "test-rev",
            "--semantic-task",
            "classification",
            "--no-trust-remote-code",
            "--suppress-test-semantic",
            "--semantic-unit-type",
            "class",
            "--no-tiny-filter",
            "--tiny-cutoff",
            "4",
            "--tiny-near-jaccard-min",
            "0.95",
            "--show-all",
        ],
    )

    assert result.exit_code == 1
    assert captured[0].instruction_prefix == "Represent this code: "
    assert captured[0].model_revision == "test-rev"
    assert captured[0].trust_remote_code is False
    assert captured[0].suppress_test_semantic_matches is True
    assert captured[0].semantic_task == "classification"
    assert captured[0].semantic_unit_types == ("class",)
    assert captured[0].filter_tiny_traditional is False
    assert captured[0].tiny_unit_statement_cutoff == 4
    assert captured[0].tiny_near_jaccard_min == 0.95


def test_cli_allow_semantic_fallback_pass_through(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--allow-semantic-fallback"])

    assert result.exit_code == 1
    assert captured[0].allow_semantic_fallback is True


def test_cli_model_revision_defaults_to_auto_none(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            "check",
            str(path),
            "--model",
            "sentence-transformers/all-MiniLM-L6-v2",
        ],
    )

    assert result.exit_code == 1
    assert captured[0].model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert captured[0].model_revision is None


@pytest.mark.parametrize(
    ("command", "tail_args", "expected_exit"),
    [
        ("check", [], 1),
        ("search", ["entry"], 0),
    ],
)
def test_cli_local_model_path_pass_through(
    monkeypatch,
    tmp_path,
    command,
    tail_args,
    expected_exit,
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    model_dir = tmp_path / "saved-model"
    model_dir.mkdir()
    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        search_results=[(_build_unit(tmp_path), 0.99)],
        captured_configs=captured,
    )

    result = CliRunner().invoke(
        cli.cli,
        [command, str(path), *tail_args, "--model", str(model_dir)],
    )

    assert result.exit_code == expected_exit
    assert captured[0].model_name == str(model_dir)


def test_cli_threshold_precedence(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    monkeypatch.setattr(
        cli,
        "get_default_semantic_threshold",
        lambda model_name: 0.73 if model_name == "gte-modernbert-base" else 0.82,
    )

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()

    result_default = runner.invoke(cli.cli, ["check", str(path)])
    assert result_default.exit_code == 1
    assert captured[-1].semantic_threshold == 0.73
    assert captured[-1].jaccard_threshold == cli.DEFAULT_TRADITIONAL_THRESHOLD
    assert captured[-1].semantic_unit_types == ("function", "method")
    assert captured[-1].filter_tiny_traditional is True
    assert captured[-1].tiny_unit_statement_cutoff == 3
    assert captured[-1].tiny_near_jaccard_min == 0.93

    result_shared = runner.invoke(cli.cli, ["check", str(path), "--threshold", "0.67"])
    assert result_shared.exit_code == 1
    assert captured[-1].semantic_threshold == 0.67
    assert captured[-1].jaccard_threshold == 0.67

    result_override = runner.invoke(
        cli.cli,
        [
            "check",
            str(path),
            "--threshold",
            "0.67",
            "--semantic-threshold",
            "0.91",
            "--traditional-threshold",
            "0.44",
        ],
    )
    assert result_override.exit_code == 1
    assert captured[-1].semantic_threshold == 0.91
    assert captured[-1].jaccard_threshold == 0.44


def test_cli_semantic_only_shared_threshold_does_not_set_traditional_threshold(
    monkeypatch, tmp_path
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--semantic-only", "--threshold", "0.7"],
    )
    assert result.exit_code == 1
    assert captured[-1].semantic_threshold == 0.7
    assert captured[-1].jaccard_threshold == cli.DEFAULT_TRADITIONAL_THRESHOLD


def test_cli_traditional_only_omits_semantic_defaults(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--traditional-only"])
    assert result.exit_code == 1
    assert captured[-1].run_semantic is False
    assert captured[-1].semantic_threshold is None
    assert captured[-1].semantic_task is None


def test_cli_traditional_only_shared_threshold_sets_only_traditional_threshold(
    monkeypatch, tmp_path
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--traditional-only", "--threshold", "0.9"],
    )
    assert result.exit_code == 1
    assert captured[-1].jaccard_threshold == 0.9
    assert captured[-1].semantic_threshold is None
    assert captured[-1].semantic_task is None


def test_cli_search_defaults_to_code_retrieval_task(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        search_results=[(_build_unit(tmp_path), 0.99)],
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["search", str(path), "entry"])
    assert result.exit_code == 0
    assert captured[0].semantic_task == "code-retrieval"
    assert captured[0].semantic_unit_types == ("function", "method")


def test_cli_search_semantic_unit_type_pass_through(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        search_results=[(_build_unit(tmp_path), 0.99)],
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
        analyze_result=lambda: _build_result(tmp_path),
        search_results=[(_build_unit(tmp_path), 0.99)],
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


def test_cli_requires_explicit_command(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(cli.cli, [str(path), "--no-private"])
    assert result.exit_code == 2


@pytest.mark.parametrize(
    ("command", "tail_args"),
    [("check", []), ("search", ["entry"])],
)
def test_cli_rejects_missing_path(tmp_path, command, tail_args):
    missing = tmp_path / "missing.py"
    runner = CliRunner()
    result = runner.invoke(cli.cli, [command, str(missing), *tail_args])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_cli_invalid_threshold(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--threshold", "1.2"])
    assert result.exit_code == 2
    assert "must be in [0.0, 1.0]" in result.output


def test_cli_rejects_conflicting_single_method_flags(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--semantic-only", "--traditional-only"],
    )
    assert result.exit_code == 2


@pytest.mark.parametrize(
    ("flag", "expected_message"),
    [
        ("--show-all", "--show-all is only valid in default combined mode."),
        (
            "--allow-semantic-fallback",
            "--allow-semantic-fallback is only valid in default combined mode.",
        ),
    ],
)
def test_cli_rejects_combined_only_flags_in_single_method_modes(tmp_path, flag, expected_message):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    semantic_result = runner.invoke(
        cli.cli,
        ["check", str(path), "--semantic-only", flag],
    )
    assert semantic_result.exit_code == 2
    assert expected_message in semantic_result.output

    traditional_result = runner.invoke(
        cli.cli,
        ["check", str(path), "--traditional-only", flag],
    )
    assert traditional_result.exit_code == 2
    assert expected_message in traditional_result.output


@pytest.mark.parametrize(
    ("command", "tail_args", "rich_args", "expected_option"),
    [
        ("check", [], ["--show-source"], "--show-source"),
        ("search", ["entry"], ["--verbose"], "--verbose"),
        ("check", [], ["--output-width", "160"], "--output-width"),
    ],
)
def test_cli_rejects_json_with_rich_only_flags(
    tmp_path,
    command,
    tail_args,
    rich_args,
    expected_option,
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [command, str(path), *tail_args, "--json", *rich_args],
    )
    assert result.exit_code == 2
    assert f"Cannot use {expected_option} with --json." in result.output


@pytest.mark.parametrize(
    ("command", "tail_args", "enabled_flag", "disabled_flag"),
    [
        ("check", [], "--trust-remote-code", "--no-trust-remote-code"),
        ("search", ["entry"], "--trust-remote-code", "--no-trust-remote-code"),
        ("check", [], "--mps-fallback", "--no-mps-fallback"),
        ("search", ["entry"], "--mps-fallback", "--no-mps-fallback"),
    ],
)
def test_cli_rejects_conflicting_paired_flags(
    tmp_path,
    command,
    tail_args,
    enabled_flag,
    disabled_flag,
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [command, str(path), *tail_args, enabled_flag, disabled_flag],
    )
    assert result.exit_code == 2
    assert f"Cannot combine {enabled_flag} and {disabled_flag}." in result.output


@pytest.mark.parametrize(
    ("extra_args", "expected_option"),
    [
        (["--semantic-threshold", "0.9"], "--semantic-threshold"),
        (["--semantic-task", "classification"], "--semantic-task"),
        (["--instruction-prefix", "prefix"], "--instruction-prefix"),
        (["--model", "sentence-transformers/all-MiniLM-L6-v2"], "--model"),
        (["--model-revision", "rev1"], "--model-revision"),
        (["--trust-remote-code"], "--trust-remote-code"),
        (["--no-trust-remote-code"], "--no-trust-remote-code"),
        (["--batch-size", "4"], "--batch-size"),
        (["--min-lines", "1"], "--min-lines"),
        (["--semantic-unit-type", "class"], "--semantic-unit-type"),
        (["--suppress-test-semantic"], "--suppress-test-semantic"),
    ],
)
def test_cli_rejects_all_semantic_mode_flags_with_traditional_only(
    tmp_path, extra_args, expected_option
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--traditional-only", *extra_args],
    )

    assert result.exit_code == 2
    assert f"Cannot use {expected_option}" in result.output


@pytest.mark.parametrize(
    ("extra_args", "expected_option"),
    [
        (["--traditional-threshold", "0.8"], "--traditional-threshold"),
        (["--no-tiny-filter"], "--no-tiny-filter"),
        (["--tiny-cutoff", "4"], "--tiny-cutoff"),
        (["--tiny-near-jaccard-min", "0.95"], "--tiny-near-jaccard-min"),
    ],
)
def test_cli_rejects_all_traditional_mode_flags_with_semantic_only(
    tmp_path, extra_args, expected_option
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--semantic-only", *extra_args],
    )

    assert result.exit_code == 2
    assert f"Cannot use {expected_option}" in result.output


def test_cli_rejects_strict_unused_with_no_unused(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--no-unused", "--strict-unused"],
    )

    assert result.exit_code == 2
    assert "Cannot combine --no-unused and --strict-unused" in result.output


def test_cli_info_exit_zero():
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["info"])
    assert result.exit_code == 0
    assert "codedupes" in result.output.lower()
    assert "pytorch:" in result.output.lower()
    assert "mps built/available:" in result.output.lower()
    assert "mlx loaded in process:" in result.output.lower()
    assert "built-in semantic model aliases" in result.output.lower()
    assert "semantic_threshold=0.96 search_threshold=0.5" in result.output


def test_cli_info_configures_mps_environment_before_diagnostics(monkeypatch):
    order: list[str] = []

    def _record_configure(requested_device, *, fallback):
        order.append(f"configure:{requested_device}:{fallback}")

    def _record_diagnostics(requested_device):
        order.append(f"diagnostics:{requested_device}")
        return DeviceDiagnostics(
            requested=requested_device,
            resolved="cpu",
            torch_available=True,
            cuda_available=False,
            mps_built=False,
            mps_available=False,
            mps_fallback_env="1",
            mlx_loaded=False,
        )

    monkeypatch.setattr(cli, "configure_mps_environment", _record_configure)
    monkeypatch.setattr(cli, "get_device_diagnostics", _record_diagnostics)

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["info"])

    assert result.exit_code == 0
    assert order == [
        f"configure:{cli.DEFAULT_SEMANTIC_DEVICE}:None",
        f"diagnostics:{cli.DEFAULT_SEMANTIC_DEVICE}",
    ]


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


def test_cli_help_and_version():
    runner = CliRunner()

    help_result = runner.invoke(cli.cli, ["--help"])
    assert help_result.exit_code == 0
    assert "Commands:" in help_result.output
    assert "check" in help_result.output
    assert "search" in help_result.output

    version_result = runner.invoke(cli.cli, ["--version"])
    assert version_result.exit_code == 0
    assert version_result.output.lower().startswith("codedupes")


def test_cli_search_help_is_search_specific() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["search", "--help"])

    assert result.exit_code == 0
    assert "also narrows traditional duplicate scope in combined mode" not in result.output
    assert "Built-in" in result.output
    assert "always apply." in result.output


@pytest.mark.parametrize("command", ["check", "search"])
def test_cli_help_advertises_local_model_directories(command: str) -> None:
    result = CliRunner().invoke(cli.cli, [command, "--help"])

    assert result.exit_code == 0
    assert "complete local model directory" in result.output


def test_cli_output_width_option(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result_with_semantic_duplicate(tmp_path),
    )

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--output-width", "200"])
    assert result.exit_code == 1
    assert cli.console.width == 200


def test_cli_show_all_prints_raw_sections(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result_with_semantic_duplicate(tmp_path),
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--show-all"])
    assert result.exit_code == 1
    assert "Traditional Duplicates (Raw" in result.output
    assert "Semantic Duplicates (Raw" in result.output


def test_cli_full_table_disables_truncation(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = _build_unit(tmp_path)
    hybrid = HybridDuplicate(
        unit_a=unit,
        unit_b=unit,
        tier="exact",
        confidence=1.0,
        has_exact=True,
    )
    result_obj = AnalysisResult(
        units=[unit],
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[hybrid for _ in range(25)],
        potentially_unused=[],
        analysis_mode="combined",
        filtered_raw_duplicates=0,
    )
    patch_cli_analyzer(monkeypatch, cli, analyze_result=result_obj)

    runner = CliRunner()
    default_result = runner.invoke(cli.cli, ["check", str(path)])
    assert default_result.exit_code == 1
    assert "... and 5 more" in default_result.output

    full_result = runner.invoke(cli.cli, ["check", str(path), "--full-table"])
    assert full_result.exit_code == 1
    assert "... and 5 more" not in full_result.output


def test_cli_invalid_output_width(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--output-width", "60"])
    assert result.exit_code == 2
    assert "must be >= 80" in result.output


def test_cli_check_fails_on_semantic_backend_error_without_fallback(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def _dead():\n    return 1\n\ndef keep(y):\n    return y + 1\n")

    from codedupes import analyzer as analyzer_module

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", _raise_semantic_backend_error)

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--min-lines", "0"])
    assert result.exit_code == 1
    assert "Error during analysis" in result.output
    assert "--allow-semantic-fallback" in result.output


def test_cli_check_degrades_on_semantic_backend_error_with_fallback(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def _dead():\n    return 1\n\ndef keep(y):\n    return y + 1\n")

    from codedupes import analyzer as analyzer_module

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", _raise_semantic_backend_error)

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--min-lines", "0", "--allow-semantic-fallback"],
    )
    assert result.exit_code == 1
    assert "Semantic analysis unavailable" in result.output


def test_cli_check_degrades_on_semantic_backend_error_in_json(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def _dead():\n    return 1\n\ndef keep(y):\n    return y + 1\n")

    from codedupes import analyzer as analyzer_module

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", _raise_semantic_backend_error)

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--min-lines", "0", "--allow-semantic-fallback", "--json"],
    )
    assert result.exit_code == 1

    assert result.output.lstrip().startswith("{"), (
        f"Expected pure JSON output, got: {result.output!r}"
    )
    payload = json.loads(result.output)
    assert payload["summary"]["semantic_fallback"] is True
    assert payload["summary"]["semantic_fallback_reason"] is not None
    assert "Semantic analysis unavailable" in payload["summary"]["semantic_fallback_reason"]


@pytest.mark.parametrize(
    ("args", "expected_message"),
    [
        (["check", "--semantic-only", "--min-lines", "0"], "Error during analysis"),
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

    runner = CliRunner()
    result = runner.invoke(cli.cli, [args[0], str(path), *args[1:]])
    assert result.exit_code == 1
    assert expected_message in result.output


def test_cli_combined_exit_code_ignores_raw_filtered_findings(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = _build_unit(tmp_path)
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
            analysis_mode="traditional",
            filtered_raw_duplicates=1,
        ),
    )

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path)])
    assert result.exit_code == 0


def test_cli_semantic_only_uses_raw_findings_for_exit(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = _build_unit(tmp_path)
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
            filtered_raw_duplicates=0,
        ),
    )

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--semantic-only"])
    assert result.exit_code == 1
    assert "Semantic Duplicates (Embedding)" in result.output


def test_setup_logging_quiets_external_loggers() -> None:
    cli.setup_logging(verbose=False)
    for logger_name in cli._NOISY_EXTERNAL_LOGGERS:
        assert logging.getLogger(logger_name).level == logging.WARNING


def test_main_propagates_check_exit_code(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    patch_cli_analyzer(monkeypatch, cli, analyze_result=lambda: _build_result(tmp_path))
    monkeypatch.setattr(sys, "argv", ["codedupes", "check", str(path), "--json"])

    assert cli.main() == 1


@pytest.mark.parametrize(
    ("command", "tail_args"),
    [("check", []), ("search", ["entry"])],
)
def test_cli_device_controls_pass_through(
    monkeypatch,
    tmp_path,
    command,
    tail_args,
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        search_results=[(_build_unit(tmp_path), 0.99)],
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            command,
            str(path),
            *tail_args,
            "--device",
            "mps",
            "--no-mps-fallback",
            "--mps-memory-fraction",
            "0.8",
        ],
    )

    expected_exit = 1 if command == "check" else 0
    assert result.exit_code == expected_exit, result.output
    assert captured[0].device == "mps"
    assert captured[0].mps_fallback is False
    assert captured[0].mps_memory_fraction == 0.8


def test_cli_rejects_unsafe_mps_memory_fraction(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--mps-memory-fraction", "0"],
    )

    assert result.exit_code == 2
    assert "must be finite and in the interval (0.0, 2.0]" in result.output


def test_cli_rejects_mps_memory_fraction_with_cpu_device(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            "check",
            str(path),
            "--device",
            "cpu",
            "--mps-memory-fraction",
            "0.8",
        ],
    )

    assert result.exit_code == 2
    assert "mps_memory_fraction requires device='mps' or device='auto'" in result.output


@pytest.mark.parametrize(
    ("extra_args", "expected_option"),
    [
        (["--device", "mps"], "--device"),
        (["--mps-fallback"], "--mps-fallback"),
        (["--no-mps-fallback"], "--no-mps-fallback"),
        (["--mps-memory-fraction", "0.8"], "--mps-memory-fraction"),
    ],
)
def test_cli_rejects_device_controls_with_traditional_only(
    tmp_path,
    extra_args,
    expected_option,
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--traditional-only", *extra_args],
    )

    assert result.exit_code == 2
    assert f"Cannot use {expected_option}" in result.output


@pytest.mark.parametrize(
    ("command", "tail_args", "expected_exit_code"),
    [("check", [], 1), ("search", ["entry"], 0)],
)
def test_cli_no_cache_flag_disables_embedding_cache(
    monkeypatch,
    tmp_path,
    command,
    tail_args,
    expected_exit_code,
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        search_results=[(_build_unit(tmp_path), 0.9)],
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, [command, str(path), *tail_args, "--no-cache"])

    assert result.exit_code == expected_exit_code
    assert captured[0].embedding_cache is False


def test_cli_traditional_only_accepts_no_cache_as_noop(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )

    result = CliRunner().invoke(
        cli.cli,
        ["check", str(path), "--traditional-only", "--no-cache"],
    )

    assert result.exit_code == 1
    assert captured[0].run_semantic is False
    assert captured[0].embedding_cache is False


def test_cli_check_defaults_to_embedding_cache_enabled(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path)])

    assert result.exit_code == 1
    assert captured[0].embedding_cache is True


def test_cli_cache_info_reports_empty_cache():
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cache", "info"])

    assert result.exit_code == 0
    assert "Cache path:" in result.output
    assert "Entries: 0" in result.output


def test_cli_cache_info_reports_populated_cache(tmp_path):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    cache.put_many(scope, "some/model", "rev1", [("k1", np.array([1.0, 2.0], dtype=np.float32))])

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cache", "info"])

    assert result.exit_code == 0
    assert "Entries: 1" in result.output
    assert "some/model: 1" in result.output


def test_cli_cache_clear_removes_all_entries(tmp_path):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    cache.put_many(scope, "some/model", "rev1", [("k1", np.array([1.0, 2.0], dtype=np.float32))])

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cache", "clear"])

    assert result.exit_code == 0
    assert "Cleared 1 cached embedding" in result.output
    assert cache.stats()["entries"] == 0


def test_cli_cache_clear_scoped_to_model(tmp_path):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    cache.put_many(
        scope,
        "Alibaba-NLP/gte-modernbert-base",
        "rev1",
        [("k1", np.array([1.0, 2.0], dtype=np.float32))],
    )
    cache.put_many(scope, "other/model", "rev1", [("k2", np.array([3.0, 4.0], dtype=np.float32))])

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cache", "clear", "--model", "gte-modernbert-base"])

    assert result.exit_code == 0
    assert "Cleared 1 cached embedding" in result.output
    remaining = cache.stats()
    assert remaining["entries"] == 1
    assert remaining["models"] == {"other/model": 1}
