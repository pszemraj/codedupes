from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent

from codedupes.extractor import CodeExtractor, compute_ast_hash, compute_token_hash
from codedupes.models import CodeUnitType
from tests.conftest import extract_units


def test_nested_scope_extraction_and_private_filtering(tmp_path: Path) -> None:
    code = dedent(
        """
        def top_level(value):
            def nested(value):
                return value * 2

            return nested(value)

        class Container:
            def method(self, value):
                return value

            class Inner:
                def inner_method(self):
                    return 1

            def _private(self):
                return 2

        class _PrivateClass:
            pass
        """
    ).strip()

    units = extract_units(tmp_path, code, include_private=False)
    names = {unit.qualified_name: unit.unit_type for unit in units}

    assert names["sample.top_level"] == CodeUnitType.FUNCTION
    assert names["sample.top_level.nested"] == CodeUnitType.FUNCTION
    assert names["sample.Container"] == CodeUnitType.CLASS
    assert names["sample.Container.method"] == CodeUnitType.METHOD
    assert names["sample.Container.Inner"] == CodeUnitType.CLASS
    assert names["sample.Container.Inner.inner_method"] == CodeUnitType.METHOD
    assert "sample.Container._private" not in names
    assert "sample._PrivateClass" not in names


def test_compute_ast_hash_normalizes_variable_names() -> None:
    first = ast.parse("def add(a, b):\n    return a + b").body[0]
    second = ast.parse("def total(x, y):\n    return x + y").body[0]

    assert compute_ast_hash(first) == compute_ast_hash(second)


def test_compute_token_hash_ignores_formatting() -> None:
    assert compute_token_hash("def f(x):\n    return x + 1") == compute_token_hash(
        "def f( x ):\n\treturn x+1"
    )


def test_parse_error_is_skipped(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    root.joinpath("__init__.py").write_text("")
    bad = root / "bad.py"
    bad.write_text("def broken(:\n    pass\n")
    extractor = CodeExtractor(root, include_private=False)

    assert list(extractor.extract_from_file(bad)) == []


def test_extract_all_deduplicates_symlinked_paths(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("")

    source = dedent(
        """
        def sample():
            return 1
        """
    ).strip()
    real = package / "real.py"
    real.write_text(source)
    alias = package / "alias.py"
    alias.symlink_to(real)

    extractor = CodeExtractor(package, include_private=False)
    units = extractor.extract_all()
    assert len(units) == 1


def test_get_module_name_handles_stub_suffix(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("")
    stub = package / "typed_mod.pyi"
    stub.write_text("def entry() -> int: ...\n")

    extractor = CodeExtractor(package, include_private=True, include_stubs=True)
    units = list(extractor.extract_from_file(stub))

    assert len(units) == 1
    assert units[0].qualified_name == "typed_mod.entry"


def test_extract_all_skips_common_artifact_directories(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(
        dedent(
            """
            def keep():
                return 1
            """
        ).strip()
        + "\n"
    )

    target_dir = root / "target"
    target_dir.mkdir()
    (target_dir / "generated.py").write_text(
        dedent(
            """
            def ignore_me():
                return 2
            """
        ).strip()
        + "\n"
    )

    node_modules_dir = root / "node_modules"
    node_modules_dir.mkdir()
    (node_modules_dir / "lib.py").write_text(
        dedent(
            """
            def ignore_me_too():
                return 3
            """
        ).strip()
        + "\n"
    )

    extractor = CodeExtractor(root, include_private=True)
    units = extractor.extract_all()
    qualified_names = {unit.qualified_name for unit in units}

    assert "pkg.main.keep" in qualified_names
    assert all("ignore_me" not in name for name in qualified_names)


def test_extract_from_file_respects_exclude_patterns(tmp_path: Path) -> None:
    source = "def entry():\n    return 1\n"
    file_path = tmp_path / "sample.py"
    file_path.write_text(source)

    extractor = CodeExtractor(tmp_path, exclude_patterns=["sample.py"], include_private=True)
    units = list(extractor.extract_from_file(file_path))
    assert units == []


def test_extract_all_double_star_pattern_matches_root_level_files(tmp_path: Path) -> None:
    source = "def entry():\n    return 1\n"
    file_path = tmp_path / "sample.py"
    file_path.write_text(source)

    extractor = CodeExtractor(tmp_path, exclude_patterns=["**/sample.py"], include_private=True)
    units = extractor.extract_all()
    assert units == []


def test_ast_visitor_methods_are_marked_as_dynamic_dispatch_hooks(tmp_path: Path) -> None:
    source = dedent(
        """
        import ast

        class DirectVisitor(ast.NodeVisitor):
            def visit_Name(self, node):
                return self.generic_visit(node)

            def helper(self):
                return 1

        class DerivedVisitor(DirectVisitor):
            def visit_Call(self, node):
                return self.generic_visit(node)

        class OrdinaryWalker:
            def visit_Name(self, node):
                return node
        """
    ).strip()
    file_path = tmp_path / "visitors.py"
    file_path.write_text(source)

    units = list(CodeExtractor(tmp_path, include_private=True).extract_from_file(file_path))
    by_qualified_name = {unit.qualified_name: unit for unit in units}

    assert by_qualified_name["visitors.DirectVisitor.visit_Name"].is_dynamic_dispatch_hook is True
    assert by_qualified_name["visitors.DerivedVisitor.visit_Call"].is_dynamic_dispatch_hook is True
    assert by_qualified_name["visitors.DirectVisitor.helper"].is_dynamic_dispatch_hook is False
    assert by_qualified_name["visitors.OrdinaryWalker.visit_Name"].is_dynamic_dispatch_hook is False


def _base_visitor_source() -> str:
    """Source for a module defining a class that directly inherits ``ast.NodeVisitor``."""
    return dedent(
        """
        import ast

        class Base(ast.NodeVisitor):
            def visit_Name(self, node):
                return self.generic_visit(node)
        """
    ).strip()


def test_cross_file_direct_import_marks_visitor_hook(tmp_path: Path) -> None:
    (tmp_path / "base.py").write_text(_base_visitor_source() + "\n")
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

    units = CodeExtractor(tmp_path, include_private=True).extract_all()
    by_qualified_name = {unit.qualified_name: unit for unit in units}

    assert by_qualified_name["concrete.Concrete.visit_Call"].is_dynamic_dispatch_hook is True


def test_cross_file_package_relative_import_marks_visitor_hook(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "base.py").write_text(_base_visitor_source() + "\n")
    (pkg / "concrete.py").write_text(
        dedent(
            """
            from .base import Base

            class Concrete(Base):
                def visit_Call(self, node):
                    return self.generic_visit(node)
            """
        ).strip()
        + "\n"
    )

    units = CodeExtractor(tmp_path, include_private=True).extract_all()
    by_qualified_name = {unit.qualified_name: unit for unit in units}

    assert by_qualified_name["pkg.concrete.Concrete.visit_Call"].is_dynamic_dispatch_hook is True


def test_cross_file_transitive_import_chain_marks_visitor_hook(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(_base_visitor_source() + "\n")
    (tmp_path / "b.py").write_text(
        dedent(
            """
            from a import Base

            class Mid(Base):
                def helper(self):
                    return 1
            """
        ).strip()
        + "\n"
    )
    (tmp_path / "c.py").write_text(
        dedent(
            """
            from b import Mid

            class Concrete(Mid):
                def visit_Call(self, node):
                    return self.generic_visit(node)
            """
        ).strip()
        + "\n"
    )

    units = CodeExtractor(tmp_path, include_private=True).extract_all()
    by_qualified_name = {unit.qualified_name: unit for unit in units}

    assert by_qualified_name["c.Concrete.visit_Call"].is_dynamic_dispatch_hook is True


def test_cross_file_import_alias_marks_visitor_hook(tmp_path: Path) -> None:
    (tmp_path / "base.py").write_text(_base_visitor_source() + "\n")
    (tmp_path / "concrete.py").write_text(
        dedent(
            """
            from base import Base as Mixin

            class Concrete(Mixin):
                def visit_Call(self, node):
                    return self.generic_visit(node)
            """
        ).strip()
        + "\n"
    )

    units = CodeExtractor(tmp_path, include_private=True).extract_all()
    by_qualified_name = {unit.qualified_name: unit for unit in units}

    assert by_qualified_name["concrete.Concrete.visit_Call"].is_dynamic_dispatch_hook is True


def test_cross_file_unresolvable_third_party_base_is_not_marked(tmp_path: Path) -> None:
    (tmp_path / "concrete.py").write_text(
        dedent(
            """
            from totally_external_package import SomethingElse

            class Concrete(SomethingElse):
                def visit_Call(self, node):
                    return node
            """
        ).strip()
        + "\n"
    )

    units = CodeExtractor(tmp_path, include_private=True).extract_all()
    by_qualified_name = {unit.qualified_name: unit for unit in units}

    assert by_qualified_name["concrete.Concrete.visit_Call"].is_dynamic_dispatch_hook is False


def test_cross_file_unrelated_class_named_node_visitor_is_not_marked(tmp_path: Path) -> None:
    (tmp_path / "base.py").write_text(
        dedent(
            """
            class NodeVisitor:
                def visit_Name(self, node):
                    return node
            """
        ).strip()
        + "\n"
    )
    (tmp_path / "concrete.py").write_text(
        dedent(
            """
            from base import NodeVisitor

            class Concrete(NodeVisitor):
                def visit_Call(self, node):
                    return node
            """
        ).strip()
        + "\n"
    )

    units = CodeExtractor(tmp_path, include_private=True).extract_all()
    by_qualified_name = {unit.qualified_name: unit for unit in units}

    assert by_qualified_name["base.NodeVisitor.visit_Name"].is_dynamic_dispatch_hook is False
    assert by_qualified_name["concrete.Concrete.visit_Call"].is_dynamic_dispatch_hook is False
