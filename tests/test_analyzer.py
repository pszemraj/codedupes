from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from tests.conftest import build_two_function_source, create_project


@pytest.mark.parametrize(
    "analyzer_config, expected_unused",
    [
        (
            AnalyzerConfig(run_traditional=False, run_semantic=False, run_unused=True),
            {"used", "unused"},
        ),
        (
            AnalyzerConfig(
                run_traditional=True,
                run_semantic=False,
                run_unused=False,
                jaccard_threshold=0.5,
            ),
            set(),
        ),
    ],
)
def test_unused_detection_config_variants(tmp_path: Path, analyzer_config, expected_unused) -> None:
    project = create_project(tmp_path, build_two_function_source())
    analyzer = CodeAnalyzer(analyzer_config)

    result = analyzer.analyze(project)

    assert {unit.name for unit in result.potentially_unused} == expected_unused


def test_integration_on_mixed_project(tmp_path: Path) -> None:
    src_root = tmp_path / "project"
    src_root.mkdir()

    (src_root / "bad.py").write_text("def bad(:\n    pass")
    (src_root / "tests").mkdir()
    (src_root / "tests" / "test_skip.py").write_text("def test_case():\n    return 1")
    (src_root / "util.py").write_text(
        dedent(
            """
            def add(a, b):
                return a + b

            def sum_values(x, y):
                return x + y

            def helper():
                return 2

            def caller():
                return helper()

            class Engine:
                def run(self):
                    return helper()

                def _internal(self):
                    return 0

            def _private_entry():
                return helper()

            def get_value():
                return 3

            def set_value(value):
                return value
            """
        ).strip()
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_semantic=False,
            run_traditional=True,
            run_unused=True,
            include_private=False,
        )
    )
    result = analyzer.analyze(src_root)

    assert len(result.exact_duplicates) >= 1
    assert not any("tests" in str(unit.file_path) for unit in result.units)
    names = {unit.name for unit in result.potentially_unused}
    assert "caller" in names
    assert "_private_entry" not in names


def test_search_requires_embeddings(tmp_path: Path) -> None:
    source = "def entry():\n    return 1\n"
    create_project(tmp_path, source)
    project = tmp_path / "src"
    analyzer = CodeAnalyzer(
        AnalyzerConfig(run_semantic=False, run_traditional=False, run_unused=False)
    )

    analyzer.analyze(project)
    with pytest.raises(RuntimeError, match="run_semantic=True"):
        analyzer.search("entry")


def test_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError, match="jaccard_threshold"):
        AnalyzerConfig(jaccard_threshold=1.5)

    with pytest.raises(ValueError, match="semantic_threshold"):
        AnalyzerConfig(semantic_threshold=-0.1)


def test_empty_directory_analysis(tmp_path: Path) -> None:
    analyzer = CodeAnalyzer()
    result = analyzer.analyze(tmp_path)

    assert result.units == []
    assert result.exact_duplicates == []
    assert result.semantic_duplicates == []
    assert result.potentially_unused == []
