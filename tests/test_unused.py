from __future__ import annotations

import ast
import logging
from pathlib import Path
from textwrap import dedent

import pytest

from codedupes import unused as unused_module
from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from codedupes.extractor import CodeExtractor
from codedupes.models import CodeUnit
from codedupes.unused import (
    build_reference_graph,
    collect_module_references,
    find_potentially_unused,
)
from tests.conftest import extract_units


def _unit(units: list[CodeUnit], qualified_name: str) -> CodeUnit:
    """Return the single unit with this qualified name.

    :param units: Extracted units.
    :param qualified_name: Fully qualified unit name.
    :return: The matching unit.
    """
    matches = [unit for unit in units if unit.qualified_name == qualified_name]
    assert len(matches) == 1, [unit.qualified_name for unit in units]
    return matches[0]


def _module_ref(units: list[CodeUnit]) -> str:
    """Return the synthetic module-scope referrer id for the units' file.

    :param units: Extracted units sharing one file.
    :return: Module referrer id.
    """
    return f"__module__::{units[0].file_path}"


def _referenced_graph(tmp_path: Path, source: str) -> tuple[list[CodeUnit], set[str]]:
    """Extract every symbol, build the graph, and report strict-mode unused names.

    :param tmp_path: Test directory.
    :param source: Module source.
    :return: Units and the names strict mode reports as unused.
    """
    units = extract_units(tmp_path, source, include_private=True)
    build_reference_graph(units)
    unused = find_potentially_unused(units, strict_unused=True)
    return units, {unit.name for unit in unused}


def _package_graph(
    tmp_path: Path, modules: dict[str, str]
) -> tuple[Path, list[CodeUnit], set[str]]:
    """Write a package, extract every file, build the graph, and report strict-mode unused names.

    :param tmp_path: Test directory.
    :param modules: Module sources keyed by file name under the package.
    :return: Package root, its units, and the names strict mode reports.
    """
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "__init__.py").write_text("")
    for name, source in modules.items():
        (root / name).write_text(dedent(source).strip() + "\n")
    extractor = CodeExtractor(tmp_path, include_private=True)
    units = extractor.extract_all()
    build_reference_graph(units, source_files=extractor.extracted_files["python"])
    unused = find_potentially_unused(units, strict_unused=True)
    return root, units, {unit.name for unit in unused}


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


def test_main_block_references_survive_a_bom(tmp_path: Path) -> None:
    source = dedent(
        """
        def used_by_main():
            return 7

        if __name__ == "__main__":
            used_by_main()
        """
    ).strip()
    path = tmp_path / "bom_sample.py"
    path.write_bytes(b"\xef\xbb\xbf" + source.encode("utf-8"))

    units = list(CodeExtractor(tmp_path, include_private=True).extract_from_file(path))
    build_reference_graph(units, project_root=tmp_path)
    unused = find_potentially_unused(units, strict_unused=True)

    assert "used_by_main" not in {unit.name for unit in unused}


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

    units = list(CodeExtractor(project).extract_from_file(extractor_file))
    assert len(units) == 2
    build_reference_graph(units, project_root=tmp_path)
    unused = find_potentially_unused(units, strict_unused=True)
    names = {unit.name for unit in unused}
    assert "cli_entry" not in names
    assert "helper" in names


def test_pyproject_entry_point_groups_mark_as_used(tmp_path: Path) -> None:
    """``[project.entry-points."group"]`` tables seed entry points like ``scripts`` does."""
    source = dedent(
        """
        def plugin_entry():
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

            [project.entry-points."sample.plugins"]
            default = "sample_module:plugin_entry"
            """
        ).strip()
    )
    project = tmp_path / "src"
    project.mkdir()
    (project / "__init__.py").write_text("")
    (project / "sample_module.py").write_text(source)

    units = list(CodeExtractor(project).extract_from_file(project / "sample_module.py"))
    build_reference_graph(units, project_root=tmp_path)
    unused = find_potentially_unused(units, strict_unused=True)

    assert _unit(units, "sample_module.plugin_entry").references == {"project.entrypoint"}
    assert {unit.name for unit in unused} == {"helper"}


def test_reference_graph_parses_each_file_once(tmp_path: Path, monkeypatch) -> None:
    source = dedent(
        """
        def first():
            return 1

        def second():
            return first()

        if __name__ == "__main__":
            first()
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    parsed: list[Path] = []
    real_parse = unused_module._parse_module

    def counting_parse(path: Path) -> ast.Module | None:
        parsed.append(path)
        return real_parse(path)

    monkeypatch.setattr(unused_module, "_parse_module", counting_parse)

    build_reference_graph(units)

    assert parsed == [units[0].file_path]
    assert _unit(units, "sample.first").references == {
        _unit(units, "sample.second").uid,
        _module_ref(units),
    }


def test_property_read_is_a_reference(tmp_path: Path) -> None:
    source = dedent(
        """
        class _Config:
            @property
            def width(self):
                return 1

        def _measure(config):
            return config.width
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    assert "width" not in unused
    assert _unit(units, "sample._Config.width").references == {_unit(units, "sample._measure").uid}


def test_bound_method_callback_is_a_reference(tmp_path: Path) -> None:
    source = dedent(
        """
        import shutil

        class _Cleaner:
            def _on_error(self, func, path, exc_info):
                pass

            def run(self, path):
                shutil.rmtree(path, onerror=self._on_error)
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    assert "_on_error" not in unused
    # The method body is attributed to the method and to its class.
    assert _unit(units, "sample._Cleaner._on_error").references == {
        _unit(units, "sample._Cleaner.run").uid,
        _unit(units, "sample._Cleaner").uid,
    }


def test_annotations_are_references(tmp_path: Path) -> None:
    source = dedent(
        """
        class _Node:
            pass

        class _Leaf:
            pass

        class _Edge:
            pass

        def _walk(node: _Node, edges: "list[_Edge]") -> "_Leaf | None":
            found: "_Leaf | None" = None
            return found
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    walker = _unit(units, "sample._walk").uid
    module = _module_ref(units)
    # Signature annotations evaluate in the enclosing (module) namespace; the
    # annotated assignment inside the body belongs to the function.
    assert _unit(units, "sample._Node").references == {module}
    assert _unit(units, "sample._Edge").references == {module}
    assert _unit(units, "sample._Leaf").references == {module, walker}
    assert unused == {"_walk"}


def test_base_class_is_a_reference(tmp_path: Path) -> None:
    source = dedent(
        """
        class _Base:
            pass

        class _Derived(_Base):
            pass
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    assert _unit(units, "sample._Base").references == {_module_ref(units)}
    assert unused == {"_Derived"}


def test_module_level_registration_is_a_reference(tmp_path: Path) -> None:
    source = dedent(
        """
        def _handler():
            return 1

        def register(fn):
            return fn

        register(_handler)
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    assert _unit(units, "sample._handler").references == {_module_ref(units)}
    assert _unit(units, "sample.register").references == {_module_ref(units)}
    assert unused == set()


def test_attribute_store_through_property_setter_is_a_reference(tmp_path: Path) -> None:
    source = dedent(
        """
        class _Box:
            @property
            def _value(self):
                return self.__dict__.get("v")

            @_value.setter
            def _value(self, value):
                self.__dict__["v"] = value

        def _fill(box):
            box._value = 3
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    filler = _unit(units, "sample._fill").uid
    setters = [unit for unit in units if unit.name == "_value"]
    assert len(setters) == 2
    for unit in setters:
        assert filler in unit.references
    assert "_value" not in unused


def test_self_recursion_is_not_a_reference(tmp_path: Path) -> None:
    source = dedent(
        """
        def _factorial(n):
            return 1 if n <= 1 else n * _factorial(n - 1)
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    assert _unit(units, "sample._factorial").references == set()
    assert unused == {"_factorial"}


def test_self_recursive_method_is_not_a_reference(tmp_path: Path) -> None:
    """A method's own body must not reach it through the enclosing class scope."""
    source = dedent(
        """
        class _Node:
            def _walk(self):
                return self._walk()
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    assert _unit(units, "sample._Node._walk").references == set()
    assert unused == {"_Node", "_walk"}


def test_decorator_and_default_argument_names_are_references(tmp_path: Path) -> None:
    source = dedent(
        """
        def _decorate(fn):
            return fn

        def _default():
            return 0

        @_decorate
        def run(value=_default()):
            return value
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    # Decorators and defaults evaluate in the enclosing (module) namespace.
    assert _unit(units, "sample._decorate").references == {_module_ref(units)}
    assert _unit(units, "sample._default").references == {_module_ref(units)}
    assert unused == {"run"}


def test_class_body_alias_is_a_reference(tmp_path: Path) -> None:
    source = dedent(
        """
        def _visit_impl(self, node):
            return node

        class _Visitor:
            visit_Name = _visit_impl
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    assert _unit(units, "sample._visit_impl").references == {_unit(units, "sample._Visitor").uid}
    assert unused == {"_Visitor"}


def test_nested_definition_references_count_for_the_enclosing_unit(tmp_path: Path) -> None:
    source = dedent(
        """
        def _helper():
            return 1

        def _outer():
            def _inner():
                return _helper()

            return _inner
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    outer = _unit(units, "sample._outer")
    inner = _unit(units, "sample._outer._inner")
    assert _unit(units, "sample._helper").references == {outer.uid, inner.uid}
    assert inner.references == {outer.uid}
    assert unused == {"_outer"}


def test_decorated_definition_references_are_attributed_to_its_unit(tmp_path: Path) -> None:
    source = dedent(
        """
        def _helper():
            return 1

        def _decorate(fn):
            return fn

        @_decorate
        def _target():
            return _helper()
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    target = _unit(units, "sample._target")
    # The backend starts a decorated unit at its first decorator; the
    # definition key also carries the def line so any other unit builder resolves.
    definitions = {
        definition.name: definition.linenos
        for definition in collect_module_references(target.file_path).definitions
    }
    assert definitions["_target"] == (7, 8)
    assert target.lineno == 7
    assert _unit(units, "sample._helper").references == {target.uid}
    assert unused == {"_target"}


def test_callback_passed_as_a_value_is_a_reference(tmp_path: Path) -> None:
    source = dedent(
        """
        def _increment(value):
            return value + 1

        def _apply(values):
            return list(map(_increment, values))
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    assert _unit(units, "sample._increment").references == {_unit(units, "sample._apply").uid}
    assert unused == {"_apply"}


def test_public_methods_of_node_visitor_subclass_are_framework_referenced(tmp_path: Path) -> None:
    source = dedent(
        """
        import ast

        class _Walker(ast.NodeVisitor):
            def visit_Name(self, node):
                return node

            def _helper(self, node):
                return node
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    assert _unit(units, "sample._Walker.visit_Name").references == {"framework::ast.NodeVisitor"}
    assert _unit(units, "sample._Walker._helper").references == set()
    assert unused == {"_Walker", "_helper"}


def test_logging_filter_subclass_method_is_framework_referenced(tmp_path: Path) -> None:
    source = dedent(
        """
        import logging

        class _Quiet(logging.Filter):
            def filter(self, record):
                return True
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    assert _unit(units, "sample._Quiet.filter").references == {"framework::logging.Filter"}
    assert "filter" not in unused


def test_framework_derivation_is_transitive(tmp_path: Path) -> None:
    source = dedent(
        """
        from ast import NodeVisitor

        class _Base(NodeVisitor):
            pass

        class _Mid(_Base):
            pass

        class _Leaf(_Mid):
            def visit_Call(self, node):
                return node
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    assert _unit(units, "sample._Leaf.visit_Call").references == {"framework::_Mid"}
    assert unused == {"_Leaf"}


def test_project_only_bases_are_not_framework_derived(tmp_path: Path) -> None:
    source = dedent(
        """
        class _Root:
            def run(self):
                return 1

        class _Child(_Root):
            def step(self):
                return 2

        class _Plain(object):
            def go(self):
                return 3
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    for qualified_name in ("sample._Root.run", "sample._Child.step", "sample._Plain.go"):
        assert _unit(units, qualified_name).references == set()
    assert unused == {"run", "step", "go", "_Child", "_Plain"}


def test_public_method_of_public_class_is_skipped_by_default(tmp_path: Path) -> None:
    source = dedent(
        """
        class Service:
            def run(self):
                return 1

            def _helper(self):
                return 2
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    build_reference_graph(units)

    default_names = {unit.name for unit in find_potentially_unused(units, strict_unused=False)}
    strict_names = {unit.name for unit in find_potentially_unused(units, strict_unused=True)}

    assert default_names == {"_helper"}
    assert strict_names == {"run", "_helper"}


def test_public_method_of_private_class_is_reported_by_default(tmp_path: Path) -> None:
    source = dedent(
        """
        class _Service:
            def run(self):
                return 1
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    build_reference_graph(units)

    default_names = {unit.name for unit in find_potentially_unused(units, strict_unused=False)}

    assert default_names == {"run", "_Service"}


def test_main_module_reports_unreferenced_public_functions_by_default(tmp_path: Path) -> None:
    """An entry-point module is not public API; its main-block call still counts as a use."""
    units = extract_units(
        tmp_path,
        """
        def run():
            return 1

        def unused_helper():
            return 2

        if __name__ == "__main__":
            run()
        """,
        filename="__main__.py",
    )
    build_reference_graph(units)

    unused = find_potentially_unused(units, strict_unused=False)

    assert _unit(units, "__main__.run").references == {_module_ref(units)}
    assert [unit.qualified_name for unit in unused] == ["__main__.unused_helper"]


def test_public_definitions_nested_in_a_private_function_are_reported_by_default(
    tmp_path: Path,
) -> None:
    """Every segment of the qualified name must be public for the surface rule to apply."""
    source = dedent(
        """
        def _factory():
            def nested_public():
                return 1

            class Local:
                def local_method(self):
                    return 2

            return Local
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    build_reference_graph(units)

    default_names = {
        unit.qualified_name for unit in find_potentially_unused(units, strict_unused=False)
    }

    assert default_names == {
        "sample._factory",
        "sample._factory.nested_public",
        "sample._factory.Local.local_method",
    }


def test_framework_rule_resolves_bases_through_module_aliases(tmp_path: Path) -> None:
    """An imported-as or assigned alias of a project class is still a project base."""
    _root, units, unused = _package_graph(
        tmp_path,
        {
            "mod.py": """
                class _Props:
                    pass
                """,
            "other.py": """
                from .mod import _Props as _Base

                class _FromAlias(_Base):
                    def dead_public(self):
                        return 1

                class _Later:
                    pass

                _Alias = _Later

                class _Via(_Alias):
                    def also_dead(self):
                        return 2
                """,
        },
    )

    assert _unit(units, "pkg.other._FromAlias.dead_public").references == set()
    assert _unit(units, "pkg.other._Via.also_dead").references == set()
    assert unused == {"_FromAlias", "dead_public", "_Via", "also_dead"}


def test_literal_annotation_values_are_not_references(tmp_path: Path) -> None:
    """``Literal["run"]`` names a value, not the unit ``run``; other subscripts still count."""
    source = dedent(
        """
        from collections.abc import Sequence
        from typing import Literal

        class _Node:
            pass

        def run():
            return 1

        def _dispatch(mode: Literal["run", "stop"], nodes: Sequence["_Node"]) -> None:
            return None
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    assert _unit(units, "sample.run").references == set()
    assert _unit(units, "sample._Node").references == {_module_ref(units)}
    assert unused == {"run", "_dispatch"}


def test_import_in_a_module_without_units_is_a_reference(tmp_path: Path) -> None:
    """A re-export module yields no units but is still parsed, and its import counts."""
    root, units, unused = _package_graph(
        tmp_path,
        {
            "impl.py": """
                def reexported():
                    return 1
                """,
            "api.py": """
                from .impl import reexported

                __all__ = ["reexported"]
                """,
        },
    )

    assert [unit.qualified_name for unit in units] == ["pkg.impl.reexported"]
    assert _unit(units, "pkg.impl.reexported").references == {
        f"__module__::{(root / 'api.py').resolve()}"
    }
    assert unused == set()


def test_analyzer_hands_every_visited_python_file_to_the_unused_analysis(
    tmp_path: Path,
) -> None:
    """``CodeAnalyzer`` forwards the extractor's file list so unit-less modules count."""
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "impl.py").write_text("def reexported():\n    return 1\n\ndef _dead():\n    return 2\n")
    (root / "api.py").write_text("from .impl import reexported\n")
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False, run_semantic=False, run_unused=True, strict_unused=True
        )
    )

    result = analyzer.analyze(root)

    assert [unit.name for unit in result.potentially_unused] == ["_dead"]


def test_non_utf8_module_still_contributes_references(tmp_path: Path) -> None:
    """The graph decodes lossily like the extractor instead of dropping the file."""
    path = tmp_path / "legacy.py"
    path.write_bytes(
        "def _latin():\n    return 'café'\n\ndef _latin_user():\n    return _latin()\n".encode(
            "latin-1"
        )
    )

    units = list(CodeExtractor(tmp_path, include_private=True).extract_from_file(path))
    build_reference_graph(units)
    unused = find_potentially_unused(units, strict_unused=True)

    assert _unit(units, "legacy._latin").references == {_unit(units, "legacy._latin_user").uid}
    assert {unit.name for unit in unused} == {"_latin_user"}


def test_module_the_stdlib_parser_rejects_warns_and_contributes_no_references(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """tree-sitter recovers the intact units, but ``ast`` sees no references at all."""
    source = dedent(
        """
        def _intact():
            return 1

        def _caller():
            return _intact()

        def _oops(:
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    assert [unit.qualified_name for unit in units] == ["sample._intact", "sample._caller"]

    with caplog.at_level(logging.WARNING, logger="codedupes.unused"):
        build_reference_graph(units)
    unused = find_potentially_unused(units, strict_unused=True)

    [record] = [record for record in caplog.records if record.name == "codedupes.unused"]
    assert record.levelno == logging.WARNING
    assert record.getMessage().startswith(
        f"Unused analysis collected no references from {units[0].file_path}: SyntaxError"
    )
    assert _unit(units, "sample._intact").references == set()
    assert {unit.name for unit in unused} == {"_intact", "_caller"}


def test_deep_elif_chain_warns_instead_of_aborting_the_analysis(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``ast.NodeVisitor`` recurses per node; a generated chain must not crash ``analyze()``."""
    branches = "\n".join(f"    elif x == {i}:\n        return {i}" for i in range(1, 600))
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "chain.py").write_text(
        f"def _big(x):\n    if x == 0:\n        return 0\n{branches}\n\n"
        "def _user():\n    return _big(1)\n"
    )
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False, run_semantic=False, run_unused=True, strict_unused=True
        )
    )

    with caplog.at_level(logging.WARNING, logger="codedupes.unused"):
        result = analyzer.analyze(root)

    [record] = [record for record in caplog.records if record.name == "codedupes.unused"]
    assert record.getMessage().startswith(
        f"Unused analysis collected no references from {(root / 'chain.py').resolve()}: "
    )
    assert "recursion" in record.getMessage()
    assert {unit.name for unit in result.potentially_unused} == {"_big", "_user"}


def test_abstractmethod_exemption_reads_only_the_units_own_decorators(tmp_path: Path) -> None:
    """The decorated method is exempt; its class and a body mentioning the text are not."""
    source = dedent(
        """
        import abc
        from abc import abstractmethod

        class _Holder(abc.ABC):
            @abc.abstractmethod
            def _do(self):
                return 1

            @abstractmethod
            def _step(self):
                return 2

        def _fake():
            return "@abstractmethod"
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    assert _unit(units, "sample._Holder._do").references == set()
    assert unused == {"_Holder", "_fake"}
