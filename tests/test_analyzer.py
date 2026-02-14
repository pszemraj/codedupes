from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer


def _write_project(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "src"
    path.mkdir()
    (path / "__init__.py").write_text("")
    (path / "mod.py").write_text(dedent(source).strip())
    return path


def test_run_unused_with_semantic_only(tmp_path: Path) -> None:
    source = """
    def used():
        return 1

    def unused():
        return 2
    """

    project = _write_project(tmp_path, source)
    analyzer = CodeAnalyzer(
        AnalyzerConfig(run_traditional=False, run_semantic=False, run_unused=True)
    )

    result = analyzer.analyze(project)

    unused = {unit.name for unit in result.potentially_unused}
    assert unused == {"used", "unused"}


def test_traditional_without_unused_detection(tmp_path: Path) -> None:
    source = """
    def used():
        return 1

    def unused():
        return 2
    """

    project = _write_project(tmp_path, source)
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=False,
            run_unused=False,
            jaccard_threshold=0.5,
        )
    )

    result = analyzer.analyze(project)
    assert result.potentially_unused == []


def test_search_requires_embeddings(tmp_path: Path) -> None:
    source = "def entry():\n    return 1\n"
    _write_project(tmp_path, source)
    project = tmp_path / "src"
    analyzer = CodeAnalyzer(
        AnalyzerConfig(run_semantic=False, run_traditional=False, run_unused=False)
    )

    analyzer.analyze(project)
    with pytest.raises(RuntimeError, match="run_semantic=True"):
        analyzer.search("entry")


def test_invalid_threshold_raises(tmp_path: Path) -> None:
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
