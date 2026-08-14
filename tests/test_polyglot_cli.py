"""CLI contracts for language selection and polyglot result metadata."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from codedupes import cli
from codedupes.models import AnalysisResult, CodeUnit, CodeUnitType, ExtractionDiagnostic
from tests.conftest import patch_cli_analyzer


def _rust_result(tmp_path: Path) -> AnalysisResult:
    unit = CodeUnit(
        name="run",
        qualified_name="sample.run",
        unit_type=CodeUnitType.FUNCTION,
        file_path=tmp_path / "sample.rs",
        lineno=1,
        end_lineno=3,
        source="pub fn run() -> i32 { 1 }",
        language="rust",
        dialect="rust",
        native_kind="function_item",
        start_byte=0,
        end_byte=24,
        start_column=0,
        end_column=24,
        statement_count=1,
        is_public=True,
        is_exported=True,
    )
    return AnalysisResult(
        units=[unit],
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[],
        potentially_unused=[],
        analysis_mode="traditional",
        extraction_diagnostics=[
            ExtractionDiagnostic(
                file_path=unit.file_path,
                language="rust",
                code="partial-parse",
                message="fixture diagnostic",
                lineno=1,
                end_lineno=1,
            )
        ],
        unused_excluded_units=1,
    )


def test_check_language_aliases_reach_analyzer_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "sample.rs"
    source.write_text("pub fn run() -> i32 { 1 }\n")
    captured: list[object] = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=_rust_result(tmp_path),
        captured_configs=captured,
    )

    result = CliRunner().invoke(
        cli.cli,
        [
            "check",
            str(source),
            "--language",
            "rs",
            "--traditional-only",
            "--no-unused",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured
    assert captured[-1].languages == ("rust",)  # type: ignore[attr-defined]


def test_check_json_adds_language_ranges_and_diagnostics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "sample.rs"
    source.write_text("pub fn run() -> i32 { 1 }\n")
    patch_cli_analyzer(monkeypatch, cli, analyze_result=_rust_result(tmp_path))

    result = CliRunner().invoke(
        cli.cli,
        [
            "check",
            str(source),
            "--language",
            "rust",
            "--traditional-only",
            "--no-unused",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"]["units_by_language"] == {"rust": 1}
    assert payload["summary"]["unused_supported_languages"] == ["python"]
    assert payload["summary"]["unused_excluded_units"] == 1
    assert payload["extraction_diagnostics"][0]["language"] == "rust"

    # Unit records are additive in duplicate/unused/search output. Exercise the
    # serializer directly because an empty-finding run has no unit array.
    unit_payload = cli._unit_to_dict(_rust_result(tmp_path).units[0])
    assert unit_payload["language"] == "rust"
    assert unit_payload["dialect"] == "rust"
    assert unit_payload["native_kind"] == "function_item"
    assert unit_payload["start_byte"] == 0
    assert unit_payload["end_byte"] == 24
    assert unit_payload["statement_count"] == 1


def test_source_highlighting_uses_unit_language(tmp_path: Path) -> None:
    unit = _rust_result(tmp_path).units[0]

    assert cli._syntax_lexer(unit) == "rust"
    unit.language = "typescript"
    unit.dialect = "tsx"
    assert cli._syntax_lexer(unit) == "typescript"
