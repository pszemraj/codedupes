from __future__ import annotations

import shlex
from collections.abc import Iterator
from pathlib import Path

from click.testing import CliRunner

from codedupes import cli
from codedupes.models import AnalysisResult
from tests.conftest import patch_cli_analyzer


def _empty_result() -> AnalysisResult:
    """Return a successful analysis result for documentation CLI parsing.

    :return: Empty analysis result with no findings.
    """
    return AnalysisResult(
        units=[],
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[],
        potentially_unused=[],
        analysis_mode="none",
    )


def _iter_bash_commands(path: Path) -> Iterator[tuple[int, str]]:
    """Yield logical shell commands from fenced Bash blocks in one Markdown file.

    :param path: Markdown file to scan.
    :return: ``(line_number, command)`` pairs with backslash continuations joined.
    """
    in_bash = False
    command_parts: list[str] = []
    command_line = 0
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            if in_bash and command_parts:
                yield command_line, " ".join(command_parts)
                command_parts = []
            in_bash = stripped in {"```bash", "```sh", "```shell"} if not in_bash else False
            continue
        if not in_bash or not stripped or stripped.startswith("#"):
            continue

        if not command_parts:
            command_line = line_number
        continued = stripped.endswith("\\")
        command_parts.append(stripped[:-1].rstrip() if continued else stripped)
        if not continued:
            yield command_line, " ".join(command_parts)
            command_parts = []


def test_readme_and_docs_codedupes_examples_are_parseable(monkeypatch, tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text("def entry():\n    return 1\n")
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=_empty_result,
        search_results=[],
    )

    repo_root = Path(__file__).parents[1]
    markdown_paths = [repo_root / "README.md", *sorted((repo_root / "docs").glob("*.md"))]
    examples: list[tuple[Path, int, list[str]]] = []
    for markdown_path in markdown_paths:
        for line_number, command in _iter_bash_commands(markdown_path):
            argv = shlex.split(command, comments=True)
            if argv and argv[0] == "codedupes":
                examples.append((markdown_path, line_number, argv[1:]))

    assert examples, "Expected at least one documented codedupes command"
    runner = CliRunner()
    for markdown_path, line_number, argv in examples:
        invocation = list(argv)
        if invocation[0] in {"check", "search"}:
            invocation[1] = str(sample)
        result = runner.invoke(cli.cli, invocation)
        assert result.exit_code in {0, 1}, (
            f"Unparseable command at {markdown_path.relative_to(repo_root)}:{line_number}: "
            f"codedupes {shlex.join(argv)}\n{result.output}"
        )
