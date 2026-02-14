from __future__ import annotations

import json
import sys
from pathlib import Path

from codedupes import cli
from codedupes.models import AnalysisResult, CodeUnit, CodeUnitType, DuplicatePair


def _build_result(tmp_path: Path) -> AnalysisResult:
    unit = CodeUnit(
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

    dup = DuplicatePair(
        unit_a=unit,
        unit_b=unit,
        similarity=1.0,
        method="ast_hash",
    )

    return AnalysisResult(
        units=[unit],
        exact_duplicates=[dup],
        semantic_duplicates=[],
        potentially_unused=[unit],
    )


def test_cli_json_output(monkeypatch, tmp_path, capsys):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []

    class DummyAnalyzer:
        def __init__(self, config):
            captured.append(config)

        def analyze(self, _path):
            return _build_result(tmp_path)

    monkeypatch.setattr(cli, "CodeAnalyzer", DummyAnalyzer)
    monkeypatch.setattr(sys, "argv", ["codedupes", str(path), "--json"])

    assert cli.main() == 1
    output = json.loads(capsys.readouterr().out)

    assert "summary" in output
    assert output["summary"]["potentially_unused"] == 1
    assert captured[0].include_private


def test_cli_no_private_option(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []

    class DummyAnalyzer:
        def __init__(self, config):
            captured.append(config)

        def analyze(self, _path):
            return _build_result(tmp_path)

    monkeypatch.setattr(cli, "CodeAnalyzer", DummyAnalyzer)
    monkeypatch.setattr(sys, "argv", ["codedupes", str(path), "--no-private"])

    assert cli.main() == 1
    assert captured[0].include_private is False


def test_cli_invalid_threshold(monkeypatch, tmp_path, capsys):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    monkeypatch.setattr(sys, "argv", ["codedupes", str(path), "--threshold", "1.2"])
    assert cli.main() == 1
    assert "threshold must be in [0.0, 1.0]" in capsys.readouterr().out
