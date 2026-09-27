"""Shared builders and runners for the test_polyglot_tree_sitter test modules."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from codedupes.extractor import CodeExtractor
from codedupes.languages.base import BackendResult
from codedupes.models import CodeUnit


def extract(
    tmp_path: Path,
    filename: str,
    source: str,
    *,
    include_private: bool = True,
) -> list[CodeUnit]:
    path = tmp_path / filename
    path.write_text(dedent(source).strip() + "\n", encoding="utf-8")
    language = {
        ".py": "python",
        ".c": "c",
        ".h": "c",
        ".rs": "rust",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
    }[path.suffix]
    extractor = CodeExtractor(
        tmp_path,
        include_private=include_private,
        languages=(language,),
    )
    return list(extractor.extract_from_file(path))


# ---------------------------------------------------------------------------
# Python backend.
# ---------------------------------------------------------------------------


def python_result(
    tmp_path: Path,
    source: str,
    *,
    filename: str = "sample.py",
    include_private: bool = True,
) -> BackendResult:
    """Extract one Python file through the extractor, keeping its diagnostics."""
    path = tmp_path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(source).strip() + "\n", encoding="utf-8")
    extractor = CodeExtractor(
        tmp_path,
        include_private=include_private,
        include_stubs=True,
        languages=("python",),
    )
    units = tuple(extractor.extract_from_file(path))
    return BackendResult(units, tuple(extractor.diagnostics))


def python_units(
    tmp_path: Path,
    source: str,
    *,
    filename: str = "sample.py",
    include_private: bool = True,
) -> dict[str, CodeUnit]:
    result = python_result(tmp_path, source, filename=filename, include_private=include_private)
    return {unit.qualified_name: unit for unit in result.units}
