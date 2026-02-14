from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from codedupes.traditional import (
    build_reference_graph,
    find_potentially_unused,
)
from codedupes.traditional import run_traditional_analysis
from tests.conftest import extract_units


def test_exact_duplicates_via_ast_hash(tmp_path: Path) -> None:
    source = dedent(
        """
        def foo(a, b):
            return a + b

        def bar(x, y):
            return x + y
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)

    exact, near, _ = run_traditional_analysis(units, jaccard_threshold=0.85)

    assert len(exact) == 1
    assert len(near) == 0
    methods = {pair.method for pair in exact}
    assert methods == {"ast_hash"}


def test_near_duplicates_threshold_boundary(tmp_path: Path) -> None:
    source = dedent(
        """
        def first(a, b):
            return a + b + a

        def second(a, c):
            return a + c + c

        def third(a, b):
            return b + 2
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)

    exact_low, near_low, _ = run_traditional_analysis(units, jaccard_threshold=0.3)
    exact_high, near_high, _ = run_traditional_analysis(units, jaccard_threshold=0.95)

    assert len(near_low) >= 1
    assert len(near_high) == 0
    assert len(exact_low) == 0


def test_alias_aware_reference_graph(tmp_path: Path) -> None:
    source = dedent(
        """
        def helper(value):
            return value

        alias = helper

        def caller(value):
            return alias(value)

        def dead():
            return 0
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=False)
    build_reference_graph(units)

    unused = find_potentially_unused(units, strict_unused=True)
    names = {unit.name for unit in unused}

    assert "helper" not in names
    assert "caller" in names
    assert "dead" in names


def test_public_function_is_skipped_by_default(tmp_path: Path) -> None:
    source = dedent(
        """
        def public_function():
            return 1

        def _private_function():
            return 2

        def _unused_private():
            return _private_function() + public_function()
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    unused = find_potentially_unused(units, strict_unused=False)

    names = {unit.name for unit in unused}
    assert "public_function" not in names
    assert "_private_function" in names


def test_noqa_and_main_block_mark_as_used(tmp_path: Path) -> None:
    source = dedent(
        """
        def ignored_unused():  # noqa: codedupes
            return 42

        def used_by_main():
            return 7

        if __name__ == "__main__":
            used_by_main()
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    build_reference_graph(units, project_root=tmp_path)
    unused = find_potentially_unused(units, strict_unused=True)
    names = {unit.name for unit in unused}

    assert "ignored_unused" not in names
    assert "used_by_main" not in names


def test_pyproject_entry_points_mark_as_used(tmp_path: Path) -> None:
    source = dedent(
        """
        def cli_entry():
            return 1

        def helper():
            return 2
        """
    ).strip()
    (tmp_path / "pyproject.toml").write_text(
        dedent(
            """
            [project]
            name = "sample"
            scripts = { sample-cli = "sample_module:cli_entry" }
            """
        ).strip()
    )
    project = tmp_path / "src"
    project.mkdir()
    (project / "__init__.py").write_text("")
    (project / "sample_module.py").write_text(source)
    extractor_file = project / "sample_module.py"

    from codedupes.extractor import CodeExtractor

    units = list(CodeExtractor(project).extract_from_file(extractor_file))
    assert len(units) == 2
    build_reference_graph(units, project_root=tmp_path)
    unused = find_potentially_unused(units, strict_unused=True)
    names = {unit.name for unit in unused}
    assert "cli_entry" not in names
    assert "helper" in names
