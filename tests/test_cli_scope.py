"""Extraction scope options: exclusions and symlink handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from codedupes import cli
from tests.cli_helpers import build_result
from tests.conftest import patch_cli_analyzer
from tests.embedding_cache_helpers import CountingModel, patch_get_model


def test_cli_focus_displays_brackets_in_path(tmp_path: Path) -> None:
    root = tmp_path / "[red]"
    root.mkdir()
    source = root / "sample.py"
    source.write_text("def entry():\n    return 1\n", encoding="utf-8")

    result = CliRunner().invoke(
        cli.cli,
        ["check", str(root), "--traditional-only", "--no-unused", "--focus", str(source)],
    )

    assert result.exit_code == 0, result.output
    assert "Focus" in result.output
    assert "[red]" in result.output


@pytest.mark.parametrize(("command", "expected_exit_code"), [("check", 1), ("search", 0)])
@pytest.mark.parametrize("include_tests", [False, True])
def test_cli_exclusions_extend_defaults(
    monkeypatch, tmp_path, command, expected_exit_code, include_tests
):
    from codedupes.extractor import CodeExtractor

    for relative in ["keep.py", "test_entry.py", "pkg/examples/deep.py", "node_modules/mod.py"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def entry():\n    return 1\n", encoding="utf-8")
    captured = []
    patch_cli_analyzer(
        monkeypatch, cli, analyze_result=build_result(tmp_path), captured_configs=captured
    )
    args = [command, str(tmp_path)] + (["entry"] if command == "search" else ["--traditional-only"])
    args += ["--exclude", "examples", "--json"]
    if include_tests:
        args.append("--no-default-excludes")
    result = CliRunner().invoke(cli.cli, args)
    assert result.exit_code == expected_exit_code, result.output
    units = CodeExtractor(tmp_path, exclude_patterns=captured[0].exclude_patterns).extract_all()
    assert {unit.file_path.name for unit in units} == (
        {"keep.py", "test_entry.py"} if include_tests else {"keep.py"}
    )


@pytest.mark.parametrize(("command", "expected_exit_code"), [("check", 1), ("search", 0)])
def test_cli_implicit_default_exclusions_preserve_analyzer_default(
    monkeypatch, tmp_path, command, expected_exit_code
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n", encoding="utf-8")
    captured = []
    patch_cli_analyzer(
        monkeypatch, cli, analyze_result=build_result(tmp_path), captured_configs=captured
    )

    args = [command, str(path)]
    if command == "check":
        args.append("--traditional-only")
    else:
        args.append("entry")
    result = CliRunner().invoke(cli.cli, args)

    assert result.exit_code == expected_exit_code, result.output
    assert captured[0].exclude_patterns is None


@pytest.mark.parametrize(("command", "expected_exit_code"), [("check", 1), ("search", 0)])
@pytest.mark.parametrize("scan_ignored", [False, True])
def test_cli_no_gitignore_threads_through_to_the_analyzer(
    monkeypatch, tmp_path, command, expected_exit_code, scan_ignored
):
    (tmp_path / "sample.py").write_text("def entry():\n    return 1\n", encoding="utf-8")
    captured = []
    patch_cli_analyzer(
        monkeypatch, cli, analyze_result=build_result(tmp_path), captured_configs=captured
    )

    args = [command, str(tmp_path), "--traditional-only" if command == "check" else "entry"]
    if scan_ignored:
        args.append("--no-gitignore")
    result = CliRunner().invoke(cli.cli, args)

    assert result.exit_code == expected_exit_code, result.output
    assert captured[0].respect_gitignore is (not scan_ignored)


@pytest.mark.parametrize("command", ["check", "search"])
@pytest.mark.parametrize("outside_root", [False, True])
@pytest.mark.parametrize("symlinked_parent", [False, True])
def test_cli_explicit_symlink_exclusions(
    monkeypatch, tmp_path, command, outside_root, symlinked_parent
):
    root = tmp_path / "project"
    root.mkdir()
    target = (tmp_path if outside_root else root) / "target.py"
    target.write_text("def entry():\n    return 1\n", encoding="utf-8")
    alias = root / "test_alias.py"
    alias.symlink_to(target)
    if symlinked_parent:
        parent_alias = tmp_path / "project_link"
        parent_alias.symlink_to(root, target_is_directory=True)
        alias = parent_alias / alias.name
    model = CountingModel()
    patch_get_model(monkeypatch, model)

    args = [command, str(alias), "--json"]
    if command == "check":
        args += ["--traditional-only", "--no-unused"]
        count_key = "total_units"
    else:
        args += ["entry", "--device", "cpu", "--min-statements", "0", "--threshold", "0"]
        count_key = "indexed_units"

    for options, expected_count in (
        ([], 1),
        (["--no-default-excludes"], 1),
        (["--exclude", "unrelated.py"], 1),
        (["--exclude", alias.name], 0),
        (["--no-default-excludes", "--exclude", alias.name], 0),
    ):
        result = CliRunner().invoke(cli.cli, [*args, *options])
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["summary"][count_key] == expected_count
        assert result.stderr == ""


@pytest.mark.grammar
@pytest.mark.parametrize("exclude", ["ignored.cpp", "examples"])
def test_cli_header_detection_ignores_excluded_symlink_targets(tmp_path, exclude):
    (tmp_path / "main.c").write_text("int main(void) { return 1; }", encoding="utf-8")
    (tmp_path / "header.h").write_text(
        "static inline int header(void) { return 2; }", encoding="utf-8"
    )
    target = tmp_path / "examples" / "ignored.cpp"
    target.parent.mkdir()
    target.write_text("", encoding="utf-8")
    (tmp_path / "alias.cpp").symlink_to(target)

    result = CliRunner().invoke(
        cli.cli,
        [
            "check",
            str(tmp_path),
            "--exclude",
            exclude,
            "--traditional-only",
            "--no-unused",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["summary"]["units_by_language"] == {"c": 2}
    assert payload["extraction_diagnostics"] == []
