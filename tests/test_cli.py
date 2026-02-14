from __future__ import annotations

import json
import re
from pathlib import Path

from click.testing import CliRunner

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

    duplicate = DuplicatePair(
        unit_a=unit,
        unit_b=unit,
        similarity=1.0,
        method="ast_hash",
    )

    return AnalysisResult(
        units=[unit],
        exact_duplicates=[duplicate],
        semantic_duplicates=[],
        potentially_unused=[unit],
    )


def _build_units(tmp_path: Path) -> list[CodeUnit]:
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
    return [unit]


def test_cli_json_output(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []

    class DummyAnalyzer:
        def __init__(self, config):
            captured.append(config)

        def analyze(self, _path):
            return _build_result(tmp_path)

        def search(self, query, top_k=10):
            return [(_build_units(tmp_path)[0], 0.99)]

    monkeypatch.setattr(cli, "CodeAnalyzer", DummyAnalyzer)
    runner = CliRunner()

    result = runner.invoke(cli.cli, ["check", str(path), "--json"])
    assert result.exit_code == 1
    output = json.loads(result.output)

    assert "summary" in output
    assert output["summary"]["potentially_unused"] == 1
    assert captured[0].include_private is True

    result = runner.invoke(cli.cli, ["search", str(path), "entry", "--json", "--top-k", "1"])
    assert result.exit_code == 0
    search_output = json.loads(result.output)
    assert search_output["query"] == "entry"
    assert search_output["results"][0]["name"] == "entry"


def test_cli_no_private_option_check(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []

    class DummyAnalyzer:
        def __init__(self, config):
            captured.append(config)

        def analyze(self, _path):
            return _build_result(tmp_path)

        def search(self, query, top_k=10):
            return []

    monkeypatch.setattr(cli, "CodeAnalyzer", DummyAnalyzer)
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--no-private"])
    assert result.exit_code == 1
    assert captured[0].include_private is False


def test_cli_requires_explicit_command(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(cli.cli, [str(path), "--no-private"])
    assert result.exit_code == 2


def test_cli_invalid_threshold(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--threshold", "1.2"])
    assert result.exit_code == 2
    assert "must be in [0.0, 1.0]" in result.output


def test_cli_info_exit_zero():
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["info"])
    assert result.exit_code == 0
    assert "codedupes" in result.output.lower()


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


def test_cli_output_width_option(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    class DummyAnalyzer:
        def __init__(self, _config):
            pass

        def analyze(self, _path):
            return _build_result(tmp_path)

        def search(self, query, top_k=10):
            return []

    monkeypatch.setattr(cli, "CodeAnalyzer", DummyAnalyzer)

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--output-width", "200"])
    assert result.exit_code == 1


def test_cli_invalid_output_width(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--output-width", "60"])
    assert result.exit_code == 2
    assert "must be >= 80" in result.output


def test_no_banned_runtime_practice() -> None:
    root = Path(__file__).resolve().parents[1]
    scanned_files = [root / "README.md"]
    scanned_files.extend(root.joinpath("src").rglob("*.py"))
    scanned_files.extend(root.joinpath("tests").rglob("*.py"))

    python_m_pattern = " ".join(["python", "-m", "codedupes"])
    sys_path_pattern = ".".join(["sys", "path"])
    subprocess_import_pattern = " ".join(["import", "subprocess"])
    subprocess_attr_pattern = "subprocess" + "."

    violations: list[str] = []
    for file in scanned_files:
        if file == Path(__file__):
            continue
        text = file.read_text(encoding="utf-8", errors="ignore")
        if python_m_pattern in text:
            violations.append(f"{file}: python -m codedupes usage")

        if sys_path_pattern in text:
            violations.append(f"{file}: sys.path usage")

        if subprocess_import_pattern in text or subprocess_attr_pattern in text:
            violations.append(f"{file}: subprocess usage")

    assert violations == []


def test_no_relative_imports_outside_init() -> None:
    package_root = Path(__file__).resolve().parents[1] / "src" / "codedupes"
    offenders: list[str] = []
    pattern = re.compile(r"^\\s*from \\.\\w")

    for file_path in package_root.rglob("*.py"):
        if file_path.name == "__init__.py":
            continue
        for line_no, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.match(line):
                offenders.append(f"{file_path}:{line_no}:{line.strip()}")

    assert offenders == []
