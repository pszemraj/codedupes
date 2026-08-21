"""Calibration-script contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tree_sitter")
pytest.importorskip("tree_sitter_rust")

from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from scripts.validate_calibration_corpus import _rejected_extraction_diagnostics

pytestmark = pytest.mark.grammar


def test_validator_rejects_partial_parse_with_usable_units(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.rs"
    source_path.write_text(
        "pub fn valid() -> i32 { 1 }\nlet = ;\n",
        encoding="utf-8",
    )
    result = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=False,
            run_unused=False,
            languages=("rust",),
        )
    ).analyze(tmp_path)

    assert [unit.name for unit in result.units] == ["valid"]
    rejected = _rejected_extraction_diagnostics(result.extraction_diagnostics)
    assert [diagnostic.code for diagnostic in rejected] == ["partial-parse"]
