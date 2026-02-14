from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any, Callable

from codedupes.extractor import CodeExtractor
from codedupes.models import AnalysisResult, CodeUnit


def write_source_file(tmp_path: Path, source: str, filename: str = "sample.py") -> Path:
    path = tmp_path / filename
    path.write_text(dedent(source).strip() + "\n")
    return path


def extract_units(
    tmp_path: Path,
    source: str,
    *,
    filename: str = "sample.py",
    include_private: bool = False,
    exclude_patterns: list[str] | None = None,
) -> list[CodeUnit]:
    path = write_source_file(tmp_path, source, filename)
    extractor = CodeExtractor(
        tmp_path, exclude_patterns=exclude_patterns, include_private=include_private
    )
    return list(extractor.extract_from_file(path))


def create_project(tmp_path: Path, source: str, *, module: str = "mod.py") -> Path:
    project_root = tmp_path / "src"
    project_root.mkdir()
    (project_root / "__init__.py").write_text("")
    write_source_file(project_root, source, module)
    return project_root


def build_two_function_source() -> str:
    """Small fixture source containing a used/unused function pair."""
    return dedent(
        """
        def used():
            return 1

        def unused():
            return 2
        """
    ).strip()


def extract_arithmetic_units(
    tmp_path: Path,
    *,
    include_private: bool = False,
    exclude_patterns: list[str] | None = None,
) -> list[CodeUnit]:
    """Extract a tiny deterministic two-function module."""
    source = dedent(
        """
        def first(x):
            return x + 1

        def second(x):
            return x + 2
        """
    ).strip()
    return extract_units(
        tmp_path,
        source,
        include_private=include_private,
        exclude_patterns=exclude_patterns,
    )


def patch_cli_analyzer(
    monkeypatch: Any,
    cli_module: Any,
    *,
    analyze_result: AnalysisResult | Callable[[], AnalysisResult],
    search_results: (
        list[tuple[CodeUnit, float]] | Callable[[str, int], list[tuple[CodeUnit, float]]] | None
    ) = None,
    captured_configs: list[Any] | None = None,
) -> None:
    """Patch CLI analyzer construction with a configurable test double."""

    class DummyAnalyzer:
        def __init__(self, config: Any) -> None:
            if captured_configs is not None:
                captured_configs.append(config)

        def analyze(self, _path: Path) -> AnalysisResult:
            return analyze_result() if callable(analyze_result) else analyze_result

        def search(self, query: str, top_k: int = 10) -> list[tuple[CodeUnit, float]]:
            if callable(search_results):
                return search_results(query, top_k)
            return [] if search_results is None else search_results

    monkeypatch.setattr(cli_module, "CodeAnalyzer", DummyAnalyzer)
