from __future__ import annotations

import shlex
from pathlib import Path

from click.testing import CliRunner

from codedupes import cli
from codedupes.models import AnalysisResult
from tests.conftest import patch_cli_analyzer


def _empty_result() -> AnalysisResult:
    return AnalysisResult(
        units=[],
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[],
        potentially_unused=[],
        analysis_mode="none",
    )


def test_readme_and_docs_cli_examples_are_parseable(monkeypatch, tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text("def entry():\n    return 1\n")

    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=_empty_result,
        search_results=[],
    )

    commands = [
        "codedupes check ./src",
        "codedupes check ./src --json --threshold 0.82",
        "codedupes check ./src --semantic-only",
        "codedupes check ./src --traditional-only --no-unused",
        "codedupes check ./src --show-all",
        "codedupes check ./src --full-table",
        "codedupes check ./src --output-width 200",
        'codedupes search ./src "sum values in a list" --top-k 5',
        'codedupes search ./src "normalize request payload" --json',
        "codedupes info",
    ]

    runner = CliRunner()
    for command in commands:
        argv = shlex.split(command)[1:]
        argv = [str(sample) if token == "./src" else token for token in argv]
        result = runner.invoke(cli.cli, argv)
        assert result.exit_code in {0, 1}, (
            f"Expected parseable invocation for {command!r}, "
            f"got exit_code={result.exit_code} output={result.output!r}"
        )
