from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from codedupes.extractor import CodeExtractor
from codedupes.models import CodeUnit


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
