from __future__ import annotations

import ast
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from codedupes.extractor import CodeExtractor, compute_ast_hash, compute_token_hash
from codedupes.models import CodeUnitType
from tests.conftest import extract_units


def test_nested_scope_extraction_and_private_filtering(tmp_path: Path) -> None:
    code = dedent(
        """
        def top_level(value):
            def nested(value):
                return value * 2

            class Local:
                def method(self):
                    return value

            return nested(value), Local

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
    assert names["sample.top_level.Local"] == CodeUnitType.CLASS
    assert names["sample.top_level.Local.method"] == CodeUnitType.METHOD
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


def test_extracted_source_includes_the_full_decorator_block(tmp_path: Path) -> None:
    code = dedent(
        """
        @first
        @configured(mode="strict")
        def decorated(value):
            return value
        """
    ).strip()

    (unit,) = extract_units(tmp_path, code, include_private=True)

    assert unit.lineno == 3
    assert unit.source == code + "\n"
    assert {"first", "configured"} <= unit.referenced_names


def test_reference_extraction_separates_function_headers_from_local_bindings(
    tmp_path: Path,
) -> None:
    code = dedent(
        """
        def decorate(function):
            return function

        class Annotation:
            pass

        def make_default():
            return 0

        def parameter():
            return "module parameter"

        def local():
            return "module local"

        def item():
            return "module item"

        @decorate
        def target(parameter: Annotation = make_default()):
            local = 1
            values = (1, 2)
            return parameter, local, [item for item in values]
        """
    ).strip()

    units = extract_units(tmp_path, code, include_private=True)
    target = next(unit for unit in units if unit.qualified_name == "sample.target")

    assert {"sample.decorate", "sample.Annotation", "sample.make_default"} <= (
        target.referenced_names
    )
    assert {"parameter", "local", "item", "values"}.isdisjoint(target.referenced_names)


def test_reference_extraction_preserves_qualified_lexical_targets(tmp_path: Path) -> None:
    code = dedent(
        """
        from remote import imported as module_alias

        def target():
            return "module"

        def late():
            return "module"

        def outer():
            def target():
                return "enclosing"

            def use_nonlocal():
                nonlocal target
                return target()

            def use_global():
                global target
                return target()

            def use_late_binding():
                return late()

            def late():
                return "nested"

            def use_imports():
                from other import imported as local_alias

                return module_alias(), local_alias()

            def rebound():
                return "nested"

            rebound = 2

            def use_rebound():
                return rebound

            replacement = 2

            def replacement():
                return "nested replacement"

            def use_replacement():
                return replacement

            def use_conditional_import(flag):
                if flag:
                    from first import imported as selected
                else:
                    from second import imported as selected

                return selected()

            def use_try_import():
                try:
                    from first import imported as selected
                    risky()
                    from second import imported as selected
                except RuntimeError:
                    pass

                return selected()

            def walrus_rebound():
                return "nested walrus"

            [walrus_rebound := 2 for _ in [0]]

            def use_walrus_rebound():
                return walrus_rebound

            def maybe_kept():
                return "nested maybe"

            [maybe_kept := 2 for _ in []]

            def use_maybe_kept():
                return maybe_kept

            def nested_rebound():
                return "nested comprehension"

            [[(nested_rebound := 2) for _ in [0]] for _ in [0]]

            def use_nested_rebound():
                return nested_rebound

            def short_circuit_kept():
                return "nested bool"

            flag and (short_circuit_kept := 2)

            def use_short_circuit_kept():
                return short_circuit_kept

            def assert_kept():
                return "nested assert"

            assert flag, (assert_kept := 2)

            def use_assert_kept():
                return assert_kept

            def comparison_kept():
                return "nested comparison"

            flag < 0 < (comparison_kept := 2)

            def use_comparison_kept():
                return comparison_kept

            return (
                use_nonlocal,
                use_global,
                use_late_binding,
                use_imports,
                use_rebound,
                use_replacement,
                use_conditional_import,
                use_try_import,
                use_walrus_rebound,
                use_maybe_kept,
                use_nested_rebound,
                use_short_circuit_kept,
                use_assert_kept,
                use_comparison_kept,
            )
        """
    ).strip()

    units = extract_units(tmp_path, code, include_private=True)
    by_name = {unit.qualified_name: unit for unit in units}

    assert by_name["sample.outer.use_nonlocal"].referenced_names == {"sample.outer.target"}
    assert by_name["sample.outer.use_global"].referenced_names == {"target"}
    assert by_name["sample.outer.use_late_binding"].referenced_names == {"sample.outer.late"}
    assert by_name["sample.outer.use_imports"].referenced_names == {
        "module_alias",
        "other.imported",
    }
    assert "sample.outer.rebound" not in by_name["sample.outer.use_rebound"].referenced_names
    assert by_name["sample.outer.use_replacement"].referenced_names == {"sample.outer.replacement"}
    assert by_name["sample.outer.use_conditional_import"].referenced_names == {
        "first.imported",
        "second.imported",
    }
    assert {"first.imported", "second.imported"} <= by_name[
        "sample.outer.use_try_import"
    ].referenced_names
    assert (
        "sample.outer.walrus_rebound"
        not in by_name["sample.outer.use_walrus_rebound"].referenced_names
    )
    assert by_name["sample.outer.use_maybe_kept"].referenced_names == {"sample.outer.maybe_kept"}
    assert (
        "sample.outer.nested_rebound"
        not in by_name["sample.outer.use_nested_rebound"].referenced_names
    )
    assert (
        "sample.outer.short_circuit_kept"
        in by_name["sample.outer.use_short_circuit_kept"].referenced_names
    )
    assert "sample.outer.assert_kept" in by_name["sample.outer.use_assert_kept"].referenced_names
    assert (
        "sample.outer.comparison_kept"
        in by_name["sample.outer.use_comparison_kept"].referenced_names
    )


def test_nested_definition_headers_use_the_parent_runtime_state(tmp_path: Path) -> None:
    code = dedent(
        """
        def target():
            return "module"

        def outer():
            @target
            def target(value=target):
                return value

            class Local(Local):
                pass
        """
    ).strip()

    units = extract_units(tmp_path, code, include_private=True)
    outer = next(unit for unit in units if unit.qualified_name == "sample.outer")

    assert "target" not in outer.referenced_names
    assert "sample.outer.target" not in outer.referenced_names
    assert "Local" not in outer.referenced_names


def test_reference_extraction_uses_python_class_lookup_rules(tmp_path: Path) -> None:
    code = dedent(
        """
        def hook():
            return "module"

        class Early:
            before = hook()

            def hook(self):
                return "early"

        class Late:
            def hook(self):
                return "late"

            after = hook

            def caller(self):
                return hook()

        class Iterates:
            for item in hook():
                pass

        class Finalizes:
            try:
                risky()
                hook = 0
            finally:
                seen = hook

        """
    ).strip()

    units = extract_units(tmp_path, code, include_private=True)
    by_name = {unit.qualified_name: unit for unit in units}

    assert by_name["sample.Early"].referenced_names == {"sample.hook"}
    assert {"hook", "sample.Late.hook"} <= by_name["sample.Late"].referenced_names
    assert by_name["sample.Late.caller"].referenced_names == {"hook"}
    assert "sample.hook" in by_name["sample.Iterates"].referenced_names
    assert "sample.hook" in by_name["sample.Finalizes"].referenced_names


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_type_parameters_shadow_same_named_module_definitions(tmp_path: Path) -> None:
    code = dedent(
        """
        class _T:
            pass

        def generic[_T](value: _T) -> _T:
            return _T

        def Target():
            return "module annotation"

        def annotated[T](value: Target):
            Target = 2
            return value

        class Generic[_T]:
            value: _T

            def method(self) -> _T:
                return _T
        """
    ).strip()

    units = extract_units(tmp_path, code, include_private=True)
    by_name = {unit.qualified_name: unit for unit in units}

    assert "_T" not in by_name["sample.generic"].referenced_names
    assert "_T" not in by_name["sample.Generic"].referenced_names
    assert "_T" not in by_name["sample.Generic.method"].referenced_names
    assert "sample.Target" in by_name["sample.annotated"].referenced_names


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
        import ast as renamed_ast
        from ast import *

        class DirectVisitor(ast.NodeVisitor):
            def visit_Name(self, node):
                return self.generic_visit(node)

            def helper(self):
                return 1

        class DerivedVisitor(DirectVisitor):
            def visit_Call(self, node):
                return self.generic_visit(node)

        class DirectTransformer(ast.NodeTransformer):
            def visit_BinOp(self, node):
                return self.generic_visit(node)

        class AliasedVisitor(renamed_ast.NodeVisitor):
            def visit_Constant(self, node):
                return self.generic_visit(node)

        class StarImportedTransformer(NodeTransformer):
            def visit_UnaryOp(self, node):
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
    assert (
        by_qualified_name["visitors.DirectTransformer.visit_BinOp"].is_dynamic_dispatch_hook is True
    )
    assert (
        by_qualified_name["visitors.AliasedVisitor.visit_Constant"].is_dynamic_dispatch_hook is True
    )
    assert (
        by_qualified_name["visitors.StarImportedTransformer.visit_UnaryOp"].is_dynamic_dispatch_hook
        is True
    )
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


def test_cross_file_src_layout_import_marks_visitor_hook(tmp_path: Path) -> None:
    project = tmp_path / "project"
    pkg = project / "src" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "base.py").write_text(_base_visitor_source() + "\n")
    (pkg / "concrete.py").write_text(
        dedent(
            """
            from pkg.base import Base

            class Concrete(Base):
                def visit_Call(self, node):
                    return self.generic_visit(node)
            """
        ).strip()
        + "\n"
    )

    units = CodeExtractor(project, include_private=True).extract_all()
    by_qualified_name = {unit.qualified_name: unit for unit in units}

    assert (
        by_qualified_name["src.pkg.concrete.Concrete.visit_Call"].is_dynamic_dispatch_hook is True
    )


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


def test_cross_file_diamond_inheritance_marks_visitor_hook(tmp_path: Path) -> None:
    (tmp_path / "base.py").write_text(_base_visitor_source() + "\n")
    (tmp_path / "left.py").write_text("from base import Base\n\nclass Left(Base):\n    pass\n")
    (tmp_path / "right.py").write_text("from base import Base\n\nclass Right(Base):\n    pass\n")
    (tmp_path / "concrete.py").write_text(
        dedent(
            """
            from left import Left
            from right import Right

            class Concrete(Left, Right):
                def visit_Call(self, node):
                    return self.generic_visit(node)
            """
        ).strip()
        + "\n"
    )

    units = CodeExtractor(tmp_path, include_private=True).extract_all()
    by_qualified_name = {unit.qualified_name: unit for unit in units}

    assert by_qualified_name["concrete.Concrete.visit_Call"].is_dynamic_dispatch_hook is True


def test_cross_file_identity_collision_does_not_share_visitor_proof(tmp_path: Path, caplog) -> None:
    # A regular top-level package and a conventional ``src`` package can both
    # import as ``pkg.mod``. Their classes remain separate proof nodes even when
    # the user-facing source-qualified names differ.
    top_package = tmp_path / "pkg"
    top_package.mkdir()
    (top_package / "__init__.py").write_text("")
    (top_package / "mod.py").write_text(
        "class Helper:\n    def visit_thing(self, node):\n        return node\n"
    )

    src_package = tmp_path / "src" / "pkg"
    src_package.mkdir(parents=True)
    (src_package / "__init__.py").write_text("")
    (src_package / "mod.py").write_text(
        "import ast\n\nclass Helper(ast.NodeVisitor):\n"
        "    def visit_thing(self, node):\n        return self.generic_visit(node)\n"
    )

    with caplog.at_level("WARNING", logger="codedupes.extractor"):
        units = CodeExtractor(tmp_path, include_private=True).extract_all()
    by_qualified_name = {unit.qualified_name: unit for unit in units}

    assert by_qualified_name["pkg.mod.Helper.visit_thing"].is_dynamic_dispatch_hook is False
    assert by_qualified_name["src.pkg.mod.Helper.visit_thing"].is_dynamic_dispatch_hook is True
    assert "ambiguous import identity pkg.mod.Helper" in caplog.text


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


def test_visitor_base_bindings_follow_scope_and_document_order(tmp_path: Path) -> None:
    (tmp_path / "bindings.py").write_text(
        dedent(
            """
            import ast
            from ast import NodeVisitor as ImportedBase

            class AssignmentBase(ast.NodeVisitor):
                pass

            AssignmentBase = object

            class AssignmentWorker(AssignmentBase):
                def visit_Name(self, node):
                    return node

            class ImportedBase:
                pass

            class ImportWorker(ImportedBase):
                def visit_Name(self, node):
                    return node

            class GlobalBase:
                pass

            class Outer:
                class GlobalBase(ast.NodeVisitor):
                    pass

            class ScopeWorker(GlobalBase):
                def visit_Name(self, node):
                    return node
            """
        ).strip()
        + "\n"
    )
    (tmp_path / "base.py").write_text(_base_visitor_source() + "\n")
    (tmp_path / "concrete.py").write_text(
        dedent(
            """
            from base import Base

            class Base:
                pass

            class CrossFileWorker(Base):
                def visit_Name(self, node):
                    return node
            """
        ).strip()
        + "\n"
    )

    units = CodeExtractor(tmp_path, include_private=True).extract_all()
    by_qualified_name = {unit.qualified_name: unit for unit in units}

    unrelated_workers = {
        "bindings.AssignmentWorker.visit_Name",
        "bindings.ImportWorker.visit_Name",
        "bindings.ScopeWorker.visit_Name",
        "concrete.CrossFileWorker.visit_Name",
    }
    assert all(
        by_qualified_name[qualified_name].is_dynamic_dispatch_hook is False
        for qualified_name in unrelated_workers
    )


def test_visitor_proof_joins_across_control_flow_paths(tmp_path: Path) -> None:
    file_path = tmp_path / "flow.py"
    file_path.write_text(
        dedent(
            """
            FLAG = False

            class Plain:
                pass

            if FLAG:
                from ast import NodeVisitor as Plain

            class BranchWorker(Plain):
                def visit_dead(self, node):
                    return node

            if FLAG:
                from ast import NodeVisitor as Agreed
            else:
                from ast import NodeTransformer as Agreed

            class AgreedWorker(Agreed):
                def visit_kept(self, node):
                    return node

            try:
                from ast import NodeVisitor as Guarded
            except ImportError:
                Guarded = object

            class GuardedWorker(Guarded):
                def visit_dead(self, node):
                    return node

            try:
                from ast import NodeVisitor as Retried
            except ImportError:
                from ast import NodeTransformer as Retried

            class RetriedWorker(Retried):
                def visit_kept(self, node):
                    return node

            from ast import NodeVisitor as Looped

            for Looped in [object]:
                pass

            class LoopWorker(Looped):
                def visit_dead(self, node):
                    return node

            from ast import NodeVisitor as Captured

            match object():
                case Captured:
                    pass

            class MatchWorker(Captured):
                def visit_dead(self, node):
                    return node
            """
        ).strip()
        + "\n"
    )

    units = list(CodeExtractor(tmp_path, include_private=True).extract_from_file(file_path))
    by_qualified_name = {unit.qualified_name: unit for unit in units}

    # A binding is proof only when every reachable path establishes it.
    assert by_qualified_name["flow.AgreedWorker.visit_kept"].is_dynamic_dispatch_hook is True
    assert by_qualified_name["flow.RetriedWorker.visit_kept"].is_dynamic_dispatch_hook is True
    revoked_workers = {
        "flow.BranchWorker.visit_dead",
        "flow.GuardedWorker.visit_dead",
        "flow.LoopWorker.visit_dead",
        "flow.MatchWorker.visit_dead",
    }
    assert all(
        by_qualified_name[qualified_name].is_dynamic_dispatch_hook is False
        for qualified_name in revoked_workers
    )


def test_control_flow_dependent_class_does_not_confer_cross_file_proof(tmp_path: Path) -> None:
    (tmp_path / "base.py").write_text(_base_visitor_source() + "\n")
    (tmp_path / "maybe.py").write_text(
        dedent(
            """
            import ast

            FLAG = False

            if FLAG:
                class Exported(ast.NodeVisitor):
                    pass
            else:
                class Exported:
                    pass
            """
        ).strip()
        + "\n"
    )
    (tmp_path / "factory.py").write_text(
        dedent(
            """
            import ast

            class Local:
                pass

            def build():
                class Local(ast.NodeVisitor):
                    pass

                return Local
            """
        ).strip()
        + "\n"
    )
    (tmp_path / "platform_visitor.py").write_text(
        dedent(
            """
            import sys

            from base import Base

            if sys.platform == "darwin":
                class PlatformVisitor(Base):
                    def visit_Call(self, node):
                        return self.generic_visit(node)
            """
        ).strip()
        + "\n"
    )
    (tmp_path / "user.py").write_text(
        dedent(
            """
            from factory import Local
            from maybe import Exported

            class MaybeWorker(Exported):
                def visit_dead(self, node):
                    return node

            class FactoryWorker(Local):
                def visit_dead(self, node):
                    return node
            """
        ).strip()
        + "\n"
    )

    units = CodeExtractor(tmp_path, include_private=True).extract_all()
    by_qualified_name = {unit.qualified_name: unit for unit in units}

    # Identities whose definition depends on control flow (or lives inside a
    # function) never confer proof to importers...
    assert by_qualified_name["user.MaybeWorker.visit_dead"].is_dynamic_dispatch_hook is False
    assert by_qualified_name["user.FactoryWorker.visit_dead"].is_dynamic_dispatch_hook is False
    # ...but a conditionally defined class with certain bases still receives
    # proof for its own methods.
    assert (
        by_qualified_name["platform_visitor.PlatformVisitor.visit_Call"].is_dynamic_dispatch_hook
        is True
    )


def test_all_exports_come_only_from_module_scope(tmp_path: Path) -> None:
    source = dedent(
        """
        import sys

        if sys.platform == "darwin":
            __all__ = ["platform_api"]

        def platform_api():
            return 1

        def build_exports():
            __all__ = ["local_only"]
            return __all__

        def local_only():
            return 2

        class Config:
            __all__ = ["config_attr"]

        def config_attr():
            return 3
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    by_name = {unit.name: unit for unit in units}

    # Module-level exports count even inside module-level control flow...
    assert by_name["platform_api"].is_exported is True
    # ...but an ``__all__`` local to a function or class body never exempts
    # module names from unused analysis.
    assert by_name["local_only"].is_exported is False
    assert by_name["config_attr"].is_exported is False
