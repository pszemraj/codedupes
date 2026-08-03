from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from codedupes import traditional as traditional_module
from codedupes.traditional import (
    build_reference_graph,
    extract_identifiers,
    find_potentially_unused,
    run_traditional_analysis,
)
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

    exact, near = run_traditional_analysis(units, jaccard_threshold=0.85)

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

    exact_low, near_low = run_traditional_analysis(units, jaccard_threshold=0.3)
    _exact_high, near_high = run_traditional_analysis(units, jaccard_threshold=0.95)

    assert len(near_low) >= 1
    assert len(near_high) == 0
    assert len(exact_low) == 0


def test_identifier_extraction_ignores_python_builtins() -> None:
    identifiers = extract_identifiers("def count(values):\n    return len(list(values))\n")

    assert identifiers == {"count", "values"}


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


def test_main_block_references_are_resolved_once_per_file(tmp_path: Path, monkeypatch) -> None:
    source = dedent(
        """
        def first():
            return 1

        def second():
            return 2

        if __name__ == "__main__":
            first()
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    extractions: list[Path] = []
    resolutions: list[str] = []

    def fake_extract_main_block_references(path: Path) -> set[str]:
        extractions.append(path)
        return {"first"}

    def fake_resolve_reference_targets(reference: str, _aliases: dict[str, str]) -> set[str]:
        resolutions.append(reference)
        return {reference}

    monkeypatch.setattr(
        traditional_module,
        "_extract_main_block_references",
        fake_extract_main_block_references,
    )
    monkeypatch.setattr(
        traditional_module,
        "_resolve_reference_targets",
        fake_resolve_reference_targets,
    )

    build_reference_graph(units)

    assert len(extractions) == 1
    assert resolutions == ["first"]


def test_non_call_references_count_as_usage(tmp_path: Path) -> None:
    """Callback-style, property, and annotation references must mark units as used."""
    source = dedent(
        '''
        class Marker:
            """Annotation-only class."""

        class Config:
            """Holds a property accessed without a call."""

            @property
            def cached_value(self):
                return 1

        def validate(value):
            return value

        def register(callback):
            return callback

        def annotate(value: Marker) -> Marker:
            return value

        def wire():
            register(callback=validate)
            return Config().cached_value

        def orphan():
            return None
        '''
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    build_reference_graph(units)
    unused_names = {unit.name for unit in find_potentially_unused(units, strict_unused=True)}

    # validate is only a keyword-argument reference, cached_value only a
    # property access, Marker only an annotation — none are calls.
    assert "validate" not in unused_names
    assert "cached_value" not in unused_names
    assert "Marker" not in unused_names
    # A genuinely unreferenced unit is still flagged.
    assert "orphan" in unused_names


def test_unused_analysis_skips_only_proven_ast_visitor_hooks(tmp_path: Path) -> None:
    source = dedent(
        """
        from ast import NodeTransformer as AstTransformer
        from framework import NodeVisitor

        class Visitor(AstTransformer):
            def visit_Name(self, node):
                return node

            def unused_helper(self):
                return 1

        class ImportedWorker(NodeVisitor):
            def visit_Name(self, node):
                return node

        class NodeVisitor:
            pass

        class LocalWorker(NodeVisitor):
            def visit_Name(self, node):
                return node

        class Ordinary:
            def visit_Name(self, node):
                return node
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)

    unused = find_potentially_unused(units, strict_unused=True)
    qualified_names = {unit.qualified_name for unit in unused}

    assert "sample.Visitor.visit_Name" not in qualified_names
    assert "sample.Visitor.unused_helper" in qualified_names
    assert "sample.LocalWorker.visit_Name" in qualified_names
    assert "sample.ImportedWorker.visit_Name" in qualified_names
    assert "sample.Ordinary.visit_Name" in qualified_names


def test_cross_file_ast_visitor_hooks_are_not_flagged_as_unused(tmp_path: Path) -> None:
    (tmp_path / "base.py").write_text(
        dedent(
            """
            import ast

            class Base(ast.NodeVisitor):
                def visit_Name(self, node):
                    return self.generic_visit(node)
            """
        ).strip()
        + "\n"
    )
    (tmp_path / "concrete.py").write_text(
        dedent(
            """
            from base import Base

            class Concrete(Base):
                def visit_Call(self, node):
                    return self.generic_visit(node)
            """
        ).strip()
        + "\n"
    )

    from codedupes.extractor import CodeExtractor

    units = CodeExtractor(tmp_path, include_private=True).extract_all()
    build_reference_graph(units)
    unused = find_potentially_unused(units, strict_unused=True)
    qualified_names = {unit.qualified_name for unit in unused}

    assert "concrete.Concrete.visit_Call" not in qualified_names
