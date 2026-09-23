"""Option parsing and validation shared by ``check`` and ``search``."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from codedupes import cli
from tests.cli_helpers import build_result, build_unit, run_cli_subprocess
from tests.conftest import patch_cli_analyzer
from tests.semantic_helpers import fail_if_called


def _unwrapped(output: str) -> str:
    """Rejoin a rendered error panel's message across its line wrapping.

    :param output: Captured CLI output.
    :return: Output with panel borders dropped and every whitespace run reduced to one space.
    """
    return " ".join(output.replace("│", " ").split())


@pytest.mark.parametrize("value", ["nan", "1.1"])
def test_cli_check_rejects_invalid_duplicate_threshold_before_analysis(
    monkeypatch, tmp_path, value
):
    monkeypatch.setattr(cli, "CodeAnalyzer", fail_if_called)
    result = CliRunner().invoke(cli.cli, ["check", str(tmp_path), "--threshold", value, "--json"])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "[0.0, 1.0]" in result.stderr


def test_cli_no_private_option_check(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result(tmp_path),
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
        analyze_result=lambda: build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            "check",
            str(path),
            "--semantic-threshold",
            "0.9",
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


def test_cli_check_rejects_uncalibrated_context_as_usage_error(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    patch_cli_analyzer(monkeypatch, cli, analyze_result=lambda: build_result(tmp_path))
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--instruction-prefix", "Represent this code: "],
    )

    assert result.exit_code == 2
    assert "provide semantic_threshold explicitly" in result.output


def test_cli_allow_semantic_fallback_pass_through(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result(tmp_path),
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
        analyze_result=lambda: build_result(tmp_path),
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
        analyze_result=lambda: build_result(tmp_path),
        search_results=[(build_unit(tmp_path), 0.99)],
        captured_configs=captured,
    )

    result = CliRunner().invoke(
        cli.cli,
        [command, str(path), *tail_args, "--model", str(model_dir)],
    )

    assert result.exit_code == expected_exit
    assert captured[0].model_name == str(model_dir)


@pytest.mark.parametrize("command", ["check", "search"])
def test_cli_invalid_threshold_profile_and_help(tmp_path, command):
    args = [command, str(tmp_path)] + (["query"] if command == "search" else [])
    result = CliRunner().invoke(cli.cli, args + ["--threshold-profile", "invalid"])
    assert result.exit_code == 2
    assert "Invalid value for '--threshold-profile'" in result.output
    help_result = CliRunner().invoke(cli.cli, [command, "--help"])
    assert help_result.exit_code == 0
    assert "--threshold-profile" in help_result.output


def test_cli_threshold_precedence(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()

    result_default = runner.invoke(cli.cli, ["check", str(path)])
    assert result_default.exit_code == 1
    # No override: the analyzer applies the profile's per-language gates.
    assert captured[-1].semantic_threshold is None
    assert captured[-1].jaccard_threshold == cli.DEFAULT_TRADITIONAL_THRESHOLD
    assert captured[-1].semantic_unit_types == ("function", "method")
    assert captured[-1].filter_tiny_traditional is True
    # The CLI defaults must be the library defaults, not a second hardcoded copy.
    assert captured[-1].tiny_unit_statement_cutoff == cli.DEFAULT_TINY_UNIT_STATEMENT_CUTOFF
    assert captured[-1].tiny_unit_statement_cutoff == 3

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
        analyze_result=lambda: build_result(tmp_path),
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
        analyze_result=lambda: build_result(tmp_path),
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
        analyze_result=lambda: build_result(tmp_path),
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


@pytest.mark.parametrize(
    "choice", ["auto", "generic", "gte-modernbert-base", "embeddinggemma-300m"]
)
def test_cli_cross_language_flag_passes_through(monkeypatch, tmp_path, choice):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()

    result_default = runner.invoke(cli.cli, ["check", str(path)])
    assert result_default.exit_code == 1
    assert captured[-1].cross_language is False

    result_flag = runner.invoke(
        cli.cli, ["check", str(path), "--cross-language", "--threshold-profile", choice]
    )
    assert result_flag.exit_code == 1
    assert captured[-1].cross_language is True
    assert captured[-1].threshold_profile == choice


@pytest.mark.parametrize("command_tail", [[], ["entry"]], ids=["check", "search"])
@pytest.mark.parametrize("explicit_options", [False, True])
def test_cli_options_ignore_automatic_environment_variables(
    monkeypatch, tmp_path, command_tail, explicit_options
) -> None:
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
    command = "search" if command_tail else "check"
    runner = CliRunner(
        env={
            "CODEDUPES_DEVICE": "cpu",
            "CODEDUPES_MODEL": "test-model",
            "CODEDUPES_NO_CACHE": "1",
            "CODEDUPES_LANGUAGES": "invalid-language",
            "CODEDUPES_THRESHOLD": "not-a-number",
            "CODEDUPES_NO_PRIVATE": "1",
            "CODEDUPES_EXCLUDE": "*.py",
            f"CODEDUPES_{command.upper()}_DEVICE": "cpu",
            f"CODEDUPES_{command.upper()}_MODEL": "test-model",
            f"CODEDUPES_{command.upper()}_THRESHOLD": "not-a-number",
        }
    )

    args = [command, str(path), *command_tail]
    if explicit_options:
        args.extend(["--device", "cpu", "--model", "explicit-model", "--no-cache"])
    result = runner.invoke(cli.cli, args)

    assert result.exit_code == (0 if command_tail else 1), result.output
    assert captured[0].device == ("cpu" if explicit_options else cli.DEFAULT_SEMANTIC_DEVICE)
    assert captured[0].model_name == ("explicit-model" if explicit_options else cli.DEFAULT_MODEL)
    # The cache library's explicit process control is independent of CLI parsing.
    assert captured[0].embedding_cache is (not explicit_options)
    assert captured[0].languages is None
    assert captured[0].include_private is True
    assert captured[0].exclude_patterns is None

    help_result = runner.invoke(cli.cli, [command, "--help"])
    assert help_result.exit_code == 0
    assert "CODEDUPES_" not in help_result.output


def test_cli_requires_explicit_command(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(cli.cli, [str(path), "--no-private"])
    assert result.exit_code == 2


@pytest.mark.parametrize(
    "token",
    [".", "./src", "/tmp", "~/whatever", "srcish"],
    ids=["dot", "dot-relative", "absolute", "tilde", "bare-name"],
)
def test_cli_no_subcommand_token_exits_usage_error(token):
    """A sole path-like or bare-name token with no subcommand must be a usage error.

    Click's ``resolve_command`` re-parses unmatched command tokens whose first
    character is non-alphanumeric (``.``, ``/``, ``~``, ...); older click releases
    used to re-run this with an emptied ``ctx.args``, which spuriously hit the
    group's no-args help path and exited 0 instead of raising a usage error.
    """
    runner = CliRunner()
    result = runner.invoke(cli.cli, [token])
    assert result.exit_code == 2
    assert f"No such command {token!r}." in result.output
    assert "Commands:" not in result.output


def test_cli_no_args_prints_help_and_exits_usage_error():
    result = run_cli_subprocess([])
    assert result.returncode == 2
    assert "Commands" in result.stdout
    assert "check" in result.stdout
    assert "search" in result.stdout


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


@pytest.mark.parametrize(
    ("options", "expected_message"),
    [
        (["--threshold", "1.2"], "must be in [0.0, 1.0]"),
        (["--output-width", "60"], "must be >= 80"),
        (["--max-duplicates", "0"], "must be a positive integer or 'all'"),
        (["--max-duplicates", "-1"], "must be a positive integer or 'all'"),
        (["--max-duplicates", "1.5"], "must be a positive integer or 'all'"),
        (["--max-duplicates", "many"], "must be a positive integer or 'all'"),
        (["--max-unused", "0"], "must be a positive integer or 'all'"),
        (["--max-unused", "many"], "must be a positive integer or 'all'"),
        (["--mps-memory-fraction", "0"], "must be finite and in the interval (0.0, 2.0]"),
        (["--no-unused", "--strict-unused"], "Cannot combine --no-unused and --strict-unused"),
    ],
    ids=lambda value: " ".join(value) if isinstance(value, list) else None,
)
def test_cli_check_rejects_invalid_option_values(tmp_path, options, expected_message):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    result = CliRunner().invoke(cli.cli, ["check", str(path), *options])

    assert result.exit_code == 2
    assert expected_message in result.output


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("--semantic-only", "--traditional-only"),
        ("--semantic-only", "--unused-only"),
        ("--traditional-only", "--unused-only"),
        ("--unused-only", "--no-unused"),
    ],
)
def test_cli_rejects_conflicting_single_method_flags(tmp_path, first, second):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), first, second],
    )
    assert result.exit_code == 2


@pytest.mark.parametrize(
    ("flag", "expected_message"),
    [
        ("--show-all", "--show-all is only valid in default combined mode."),
        ("--include-review", "--include-review is only valid in default combined mode."),
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
    for mode_flag in ("--semantic-only", "--traditional-only", "--unused-only"):
        result = runner.invoke(cli.cli, ["check", str(path), mode_flag, flag])
        assert result.exit_code == 2
        assert expected_message in result.output


@pytest.mark.parametrize(
    ("command", "tail_args", "rich_args", "expected_option"),
    [
        ("check", [], ["--full-table"], "--full-table"),
        ("check", [], ["--show-diff"], "--show-diff"),
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
        (["--cross-language"], "--cross-language"),
        (["--semantic-task", "classification"], "--semantic-task"),
        (["--instruction-prefix", "prefix"], "--instruction-prefix"),
        (["--model", "sentence-transformers/all-MiniLM-L6-v2"], "--model"),
        (["--model-revision", "rev1"], "--model-revision"),
        (["--trust-remote-code"], "--trust-remote-code"),
        (["--no-trust-remote-code"], "--no-trust-remote-code"),
        (["--batch-size", "4"], "--batch-size"),
        (["--min-statements", "1"], "--min-statements"),
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


def test_cli_unused_only_builds_an_unused_only_config(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--unused-only", "--strict-unused"])

    assert result.exit_code == 1
    assert captured[-1].run_traditional is False
    assert captured[-1].run_semantic is False
    assert captured[-1].run_unused is True
    assert captured[-1].strict_unused is True
    assert captured[-1].semantic_threshold is None
    assert captured[-1].semantic_task is None
    assert captured[-1].jaccard_threshold == cli.DEFAULT_TRADITIONAL_THRESHOLD


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--threshold", "0.8"],
        ["--traditional-threshold", "0.8"],
        ["--max-duplicates", "5"],
        ["--show-diff"],
        ["--include-review"],
        ["--show-all"],
        ["--model", "sentence-transformers/all-MiniLM-L6-v2"],
        ["--device", "cpu"],
    ],
)
def test_cli_rejects_duplicate_controls_with_unused_only(tmp_path, extra_args):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--unused-only", *extra_args])

    assert result.exit_code == 2
    # --include-review/--show-all are rejected by the combined-mode-only check
    # (which fires for every exclusive mode) before the unused-only-specific
    # rejection; every other option is rejected by name with --unused-only.
    assert extra_args[0] in result.output


def test_cli_unused_only_can_show_source_in_json_and_terminal(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def _helper():\n    value = 1\n    return value\n", encoding="utf-8")
    runner = CliRunner()
    args = ["check", str(path), "--unused-only", "--source-lines", "1"]

    json_result = runner.invoke(cli.cli, [*args, "--json"])
    assert json_result.exit_code == 0, json_result.output
    payload = json.loads(json_result.output)
    assert payload["units"]["u0"]["source"] == "def _helper():"
    assert payload["units"]["u0"]["source_lines_omitted"] == 2

    terminal_result = runner.invoke(cli.cli, args)
    assert terminal_result.exit_code == 0, terminal_result.output
    assert "Potentially Unused" in terminal_result.output
    assert "def _helper():" in terminal_result.output


def test_cli_rejects_max_unused_with_no_unused(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--no-unused", "--max-unused", "5"])

    assert result.exit_code == 2
    assert "Cannot use --max-unused with --no-unused" in result.output


def test_cli_help_and_version():
    help_result = run_cli_subprocess(["--help"])
    assert help_result.returncode == 0
    assert "Commands" in help_result.stdout
    assert "check" in help_result.stdout
    assert "search" in help_result.stdout

    version_result = run_cli_subprocess(["--version"])
    assert version_result.returncode == 0
    assert version_result.stdout.lower().startswith("codedupes")


@pytest.mark.parametrize("command", ["check", "search"])
def test_cli_help_advertises_local_model_directories(command: str) -> None:
    result = CliRunner(env={"COLUMNS": "200"}).invoke(cli.cli, [command, "--help"])

    assert result.exit_code == 0
    assert "complete local model directory" in result.output


@pytest.mark.parametrize(
    ("command", "tail_args", "mps_fallback_flag", "expected_mps_fallback"),
    [
        ("check", [], "--no-mps-fallback", False),
        ("search", ["entry"], "--no-mps-fallback", False),
        ("check", [], "--mps-fallback", True),
        ("search", ["entry"], "--mps-fallback", True),
    ],
)
def test_cli_device_controls_pass_through(
    monkeypatch,
    tmp_path,
    command,
    tail_args,
    mps_fallback_flag,
    expected_mps_fallback,
):
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
        [
            command,
            str(path),
            *tail_args,
            "--device",
            "mps",
            mps_fallback_flag,
            "--mps-memory-fraction",
            "0.8",
        ],
    )

    expected_exit = 1 if command == "check" else 0
    assert result.exit_code == expected_exit, result.output
    assert captured[0].device == "mps"
    assert captured[0].mps_fallback is expected_mps_fallback
    assert captured[0].mps_memory_fraction == 0.8


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
        (["--strict-revision-cache"], "--strict-revision-cache"),
        (["--loose-revision-cache"], "--loose-revision-cache"),
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


def test_cli_focus_rejects_file_targets_and_out_of_root_paths(tmp_path):
    file_target = tmp_path / "sample.py"
    file_target.write_text("def entry():\n    return 1\n")

    result_file_target = CliRunner().invoke(
        cli.cli, ["check", str(file_target), "--focus", str(file_target)]
    )
    assert result_file_target.exit_code == 2
    assert "--focus requires a directory target" in _unwrapped(result_file_target.output)

    root = tmp_path / "root"
    root.mkdir()
    (root / "sample.py").write_text("def entry():\n    return 1\n")
    outside = tmp_path / "outside.py"
    outside.write_text("def other():\n    return 2\n")

    result_outside = CliRunner().invoke(cli.cli, ["check", str(root), "--focus", str(outside)])
    assert result_outside.exit_code == 2
    assert "is not inside the scan root" in _unwrapped(result_outside.output)


def test_cli_focus_missing_path_is_a_click_error(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    missing = tmp_path / "missing.py"

    result = CliRunner().invoke(cli.cli, ["check", str(root), "--focus", str(missing)])

    assert result.exit_code == 2
    assert "does not exist" in result.output
