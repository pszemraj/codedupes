from __future__ import annotations

import ast
import logging
import shutil
import subprocess
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

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


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


def test_public_referenced_only_from_a_filtered_private_definition_is_not_reported(
    tmp_path: Path,
) -> None:
    """A private caller the extractor dropped must still credit what it calls."""
    source = dedent(
        """
        def public():
            return 1

        def _private_caller():
            return public()
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=False)
    build_reference_graph(units)

    assert [unit.name for unit in units] == ["public"]
    unused = find_potentially_unused(units, strict_unused=True)

    assert "public" not in {unit.name for unit in unused}
    file_path = units[0].file_path
    assert _unit(units, "sample.public").references == {
        f"__definition__::{file_path}::_private_caller::4"
    }


def test_public_method_inside_a_private_class_still_credits_what_it_calls(
    tmp_path: Path,
) -> None:
    """A public method whose private container was dropped still credits its calls."""
    source = dedent(
        """
        def public_helper():
            return 1

        class _Service:
            def run(self):
                return public_helper()
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=False)
    build_reference_graph(units)

    assert [unit.name for unit in units] == ["public_helper"]
    unused = find_potentially_unused(units, strict_unused=True)

    assert "public_helper" not in {unit.name for unit in unused}


def test_ignore_directive_marks_the_unit_as_used(tmp_path: Path) -> None:
    source = dedent(
        """
        def ignored_unused():  # codedupes: ignore
            return 42
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    build_reference_graph(units, project_root=tmp_path)
    unused = find_potentially_unused(units, strict_unused=True)

    assert "ignored_unused" not in {unit.name for unit in unused}


def test_noqa_marker_no_longer_suppresses(tmp_path: Path) -> None:
    """The retired ``noqa: codedupes`` marker no longer suppresses anything."""
    source = dedent(
        """
        def ignored_unused():  # noqa: codedupes
            return 42
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    build_reference_graph(units, project_root=tmp_path)
    unused = find_potentially_unused(units, strict_unused=True)

    assert "ignored_unused" in {unit.name for unit in unused}


def test_directive_in_a_string_or_docstring_does_not_suppress(tmp_path: Path) -> None:
    """A ``codedupes: ignore`` marker inside a string or docstring is not a directive."""
    source = dedent(
        '''
        def unused_with_docstring():
            """codedupes: ignore"""
            return 1

        def unused_with_string():
            return "codedupes: ignore"
        '''
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    build_reference_graph(units, project_root=tmp_path)
    unused = find_potentially_unused(units, strict_unused=True)
    names = {unit.name for unit in unused}

    assert "unused_with_docstring" in names
    assert "unused_with_string" in names


def test_directive_propagates_down_not_up(tmp_path: Path) -> None:
    """A directive marks its own unit and everything nested in it, but never its container.

    ``_Service`` is private, so absent its own directive both it and ``run``
    would be reported (see ``test_public_method_of_private_class_is_reported_by_default``).
    """
    source = dedent(
        """
        def outer():
            def inner():  # codedupes: ignore
                return 1

            return 2

        class _Service:  # codedupes: ignore
            def run(self):
                return 1
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    build_reference_graph(units, project_root=tmp_path)
    unused = find_potentially_unused(units, strict_unused=True)
    names = {unit.name for unit in unused}

    assert "outer" in names
    assert "inner" not in names
    assert "_Service" not in names
    assert "run" not in names


def test_ignore_unused_alone_leaves_duplicates_reported(tmp_path: Path) -> None:
    """``codedupes: ignore[unused]`` suppresses only the unused finding, not duplicates."""
    source = dedent(
        """
        def unused_helper():  # codedupes: ignore[unused]
            return 1
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    build_reference_graph(units, project_root=tmp_path)
    unused = find_potentially_unused(units, strict_unused=True)

    assert "unused_helper" not in {unit.name for unit in unused}
    assert _unit(units, "sample.unused_helper").suppressions == frozenset({"unused"})


def test_suppressed_unused_counts_only_would_be_findings(tmp_path: Path) -> None:
    """``run_unused_analysis`` counts only directive-suppressed units that would else be findings."""
    source = dedent(
        """
        def public_exempt():  # codedupes: ignore
            return 1

        def _private_unused():  # codedupes: ignore
            return 2
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    report = unused_module.run_unused_analysis(units, project_root=tmp_path, strict_unused=False)

    # public_exempt would not be a finding even absent the directive (public
    # surface exemption under the default, non-strict policy), so it must not
    # be counted as suppressed; only _private_unused would-be-reported.
    assert report.suppressed == 1


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


_ENTRY_POINT_ANALYZER_CONFIG = AnalyzerConfig(
    run_traditional=False, run_semantic=False, run_unused=True, strict_unused=True
)


def _entry_point_project(tmp_path: Path) -> Path:
    """Write a project whose pyproject.toml names entry points inside a src package.

    :param tmp_path: Test directory.
    :return: Project root, containing ``pyproject.toml`` and ``src/pkg/``.
    """
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        dedent(
            """
            [project]
            name = "proj"
            version = "0.1"

            [project.scripts]
            app = "pkg.cli:_main"
            cls = "pkg.cli:App.run"
            """
        ).strip()
        + "\n"
    )
    pkg = root / "src" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "cli.py").write_text(
        dedent(
            """
            def _main():
                return 1


            class App:
                def run(self):
                    return 2
            """
        ).strip()
        + "\n"
    )
    (pkg / "other.py").write_text(
        dedent(
            """
            def _main():
                return 3
            """
        ).strip()
        + "\n"
    )
    return root


def test_entry_points_credit_only_the_named_module(tmp_path: Path) -> None:
    """Only the named module receives entry-point credit, even with matching basenames."""
    root = _entry_point_project(tmp_path)
    cli = root / "src" / "pkg" / "cli.py"
    cli.write_text(
        cli.read_text()
        + dedent(
            """

            def factory():
                def _main():
                    return 4
                return 5


            class Outer:
                class App:
                    def run(self):
                        return 6
            """
        )
    )
    other_pkg = root / "src" / "other"
    other_pkg.mkdir()
    (other_pkg / "__init__.py").write_text("def _main():\n    return 0\n")
    (other_pkg / "cli.py").write_text(
        "def _main():\n    return 0\n\nclass App:\n    def run(self):\n        return 0\n"
    )
    analyzer = CodeAnalyzer(_ENTRY_POINT_ANALYZER_CONFIG)

    result = analyzer.analyze(root)

    unused_by_file = {
        (unit.file_path.relative_to(root).as_posix(), unit.name)
        for unit in result.potentially_unused
    }
    unused_qualified = {unit.qualified_name for unit in result.potentially_unused}
    # The object path must match exactly, not as a suffix of a nested definition.
    assert "src.pkg.cli.factory._main" in unused_qualified
    assert "src.pkg.cli.Outer.App.run" in unused_qualified
    assert ("src/pkg/other.py", "_main") in unused_by_file
    assert "src.pkg.cli._main" not in unused_qualified
    assert "src.pkg.cli.App.run" not in unused_qualified
    assert ("src/other/cli.py", "_main") in unused_by_file
    assert ("src/other/cli.py", "run") in unused_by_file
    assert ("src/other/__init__.py", "_main") in unused_by_file


@pytest.mark.parametrize(
    "levels_above",
    [
        pytest.param(("src",), id="one-level-above-scan-root"),
        pytest.param(("src", "pkg"), id="two-levels-above-scan-root"),
    ],
)
def test_entry_points_resolve_above_the_scan_root(
    tmp_path: Path, levels_above: tuple[str, ...]
) -> None:
    """``pyproject.toml`` one or two levels above the scan root is still found."""
    root = _entry_point_project(tmp_path)
    analyzer = CodeAnalyzer(_ENTRY_POINT_ANALYZER_CONFIG)

    result = analyzer.analyze(root.joinpath(*levels_above))

    unused_by_file = {(unit.file_path.name, unit.name) for unit in result.potentially_unused}
    assert ("other.py", "_main") in unused_by_file
    assert ("cli.py", "_main") not in unused_by_file


def test_entry_points_resolve_for_a_single_file_scan(tmp_path: Path) -> None:
    """A single-file target resolves the project root from the file's directory."""
    root = _entry_point_project(tmp_path)
    analyzer = CodeAnalyzer(_ENTRY_POINT_ANALYZER_CONFIG)

    result = analyzer.analyze(root / "src" / "pkg" / "cli.py")

    assert {unit.name for unit in result.potentially_unused} == set()


@pytest.mark.parametrize("scan_root", [".", "src", "src/pkg"])
def test_entry_point_in_a_package_init_resolves_from_any_scan_root(
    tmp_path: Path, scan_root: str
) -> None:
    """``pkg:_main`` names the ``pkg/__init__.py`` definition whatever its qualified name."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        dedent(
            """
            [project]
            name = "proj"
            version = "0.1"

            [project.scripts]
            app = "pkg:_main"
            """
        ).strip()
        + "\n"
    )
    pkg = root / "src" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("def _main():\n    return 1\n")
    analyzer = CodeAnalyzer(_ENTRY_POINT_ANALYZER_CONFIG)

    result = analyzer.analyze(root / scan_root)

    assert {unit.name for unit in result.potentially_unused} == set()


@requires_git
def test_pyproject_above_the_git_root_is_ignored(tmp_path: Path) -> None:
    """A ``pyproject.toml`` outside the git work tree is not treated as the project root."""
    (tmp_path / "pyproject.toml").write_text(
        dedent(
            """
            [project]
            name = "outer"
            version = "0.1"

            [project.scripts]
            app = "pkg.cli:_main"
            """
        ).strip()
        + "\n"
    )
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    pkg = root / "src" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "cli.py").write_text("def _main():\n    return 1\n")
    analyzer = CodeAnalyzer(_ENTRY_POINT_ANALYZER_CONFIG)

    result = analyzer.analyze(root / "src")

    assert {unit.name for unit in result.potentially_unused} == {"_main"}


def test_entry_point_without_an_object_credits_nothing(tmp_path: Path) -> None:
    """A malformed target with no ``:`` separator is skipped, not credited by last segment."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        dedent(
            """
            [project]
            name = "proj"
            version = "0.1"

            [project.scripts]
            app = "pkg.cli.run"
            """
        ).strip()
        + "\n"
    )
    pkg = root / "src" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "cli.py").write_text("def run():\n    return 1\n")
    analyzer = CodeAnalyzer(_ENTRY_POINT_ANALYZER_CONFIG)

    result = analyzer.analyze(root)

    assert {unit.name for unit in result.potentially_unused} == {"run"}


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
    # A project decorator may register what it wraps, so ``run`` counts as reached.
    assert _unit(units, "sample.run").references == {"decorator::_decorate"}
    assert unused == set()


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


@pytest.mark.parametrize("module_homonym", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_class_body_alias_of_a_method_credits_the_method(
    tmp_path: Path, module_homonym: bool, strict: bool
) -> None:
    """A class-body load sees the class's own definitions before the module's."""
    homonym = "def _parse(text):\n    return int(text)\n\n" if module_homonym else ""
    source = (
        f"{homonym}"
        "class Decoder:\n"
        "    def _parse(self, text):\n        return text[::-1]\n"
        "    parse = _parse\n"
    )
    units = extract_units(tmp_path, source, include_private=True)
    build_reference_graph(units)
    unused = {unit.qualified_name for unit in find_potentially_unused(units, strict_unused=strict)}

    decoder = _unit(units, "sample.Decoder")
    assert decoder.uid in _unit(units, "sample.Decoder._parse").references
    assert "sample.Decoder._parse" not in unused
    if module_homonym:
        # Statement order is ignored, so the module function stays a candidate.
        assert decoder.uid in _unit(units, "sample._parse").references


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "def _helper():\n    return 'old'\n"
            "def _replace():\n"
            "    global _helper\n"
            "    def _helper():\n        return 'new'\n"
            "def run():\n"
            "    _replace()\n"
            "    return _helper()\n",
            id="global",
        ),
        pytest.param(
            "def outer():\n"
            "    def _helper():\n        return 'old'\n"
            "    def _replace():\n"
            "        nonlocal _helper\n"
            "        def _helper():\n            return 'new'\n"
            "    _replace()\n"
            "    return _helper()\n",
            id="nonlocal",
        ),
    ],
)
def test_definitions_rebound_through_global_or_nonlocal_are_references(
    tmp_path: Path, source: str
) -> None:
    """A ``def`` under ``global``/``nonlocal`` binds outside its lexical parent, so every
    same-named definition stays a candidate for loads of that name."""
    units, unused = _referenced_graph(tmp_path, source)

    assert [unit for unit in units if unit.name == "_helper"]
    assert all(unit.references for unit in units if unit.name == "_helper")
    assert "_helper" not in unused


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
    assert target.references == {"decorator::_decorate"}
    assert unused == set()


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


def test_registration_decorators_reach_the_definition_and_wrappers_do_not(tmp_path: Path) -> None:
    """Any decorator but a standard-library wrapper may register what it decorates."""
    source = dedent(
        """
        import functools
        import functools as ft
        from contextlib import contextmanager

        from django.dispatch import receiver
        from flask import Flask

        app = Flask(__name__)

        @app.route("/")
        def index():
            return "hi"

        @receiver("post_save")
        def on_saved(sender):
            return sender

        @functools.lru_cache
        def dead_cached():
            return 1

        @ft.cache
        def dead_aliased():
            return 2

        @contextmanager
        def dead_context():
            yield

        class Settings:
            @staticmethod
            def dead_static():
                return 3
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    assert _unit(units, "sample.index").references == {"decorator::app.route"}
    assert _unit(units, "sample.on_saved").references == {"decorator::receiver"}
    assert unused == {"dead_cached", "dead_aliased", "dead_context", "dead_static"}


def test_pytest_hooks_are_exempt_outside_conftest(tmp_path: Path) -> None:
    source = "def pytest_addoption(parser):\n    parser.addoption('--x')\n\n\ndef dead():\n    return 1\n"
    _units, unused = _referenced_graph(tmp_path, source)

    assert unused == {"dead"}


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


def test_production_function_referenced_only_from_tests_is_not_reported(tmp_path: Path) -> None:
    """A production function called only from a default-excluded test file is not unused.

    ``test_impl.py`` is a reference-only file: it is never extracted into a
    ``CodeUnit``, so crediting ``helper`` depends on the definition-to-referrer
    fallback for a referrer with no matching unit (see A1).
    """
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "impl.py").write_text("def helper():\n    return 1\n")
    tests_dir = root / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_impl.py").write_text(
        "from pkg.impl import helper\n\n\ndef test_helper():\n    assert helper() == 1\n"
    )
    config = AnalyzerConfig(
        run_traditional=False, run_semantic=False, run_unused=True, strict_unused=True
    )

    default_result = CodeAnalyzer(config).analyze(root)
    assert "helper" not in {unit.name for unit in default_result.potentially_unused}

    from codedupes.extractor import DEFAULT_EXCLUDE_PATTERNS

    explicit_defaults = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=False,
            run_unused=True,
            strict_unused=True,
            exclude_patterns=DEFAULT_EXCLUDE_PATTERNS.copy(),
        )
    ).analyze(root)
    assert "helper" not in {unit.name for unit in explicit_defaults.potentially_unused}

    excluded_config = AnalyzerConfig(
        run_traditional=False,
        run_semantic=False,
        run_unused=True,
        strict_unused=True,
        exclude_patterns=[*DEFAULT_EXCLUDE_PATTERNS, "tests"],
    )
    excluded_result = CodeAnalyzer(excluded_config).analyze(root)
    assert "helper" in {unit.name for unit in excluded_result.potentially_unused}

    same_shape_config = AnalyzerConfig(
        run_traditional=False,
        run_semantic=False,
        run_unused=True,
        strict_unused=True,
        exclude_patterns=[*DEFAULT_EXCLUDE_PATTERNS, "**/tests/**"],
    )
    same_shape_result = CodeAnalyzer(same_shape_config).analyze(root)
    assert "helper" in {unit.name for unit in same_shape_result.potentially_unused}


@pytest.mark.parametrize(
    "test_body",
    [
        "def _loop():\n    return _loop() + impl.helper()\n",
        (
            "def outer():\n"
            "    def _loop():\n"
            "        return _loop()\n"
            "    return _loop() + impl.helper()\n"
        ),
        (
            "def outer():\n"
            "    def branch():\n"
            "        def _loop():\n"
            "            return 1\n"
            "        return _loop()\n"
            "    return branch() + impl.helper()\n"
        ),
    ],
)
def test_excluded_definition_recursion_does_not_credit_production_names(
    tmp_path: Path,
    test_body: str,
) -> None:
    """A reference-only definition credits its calls, but not its own name."""
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "impl.py").write_text("def _loop():\n    return 1\n\n\ndef helper():\n    return 2\n")
    tests_dir = root / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_impl.py").write_text("import pkg.impl as impl\n\n\n" + test_body)

    result = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=False,
            run_unused=True,
            strict_unused=True,
        )
    ).analyze(root)

    assert [unit.name for unit in result.potentially_unused] == ["_loop"]


@pytest.mark.parametrize(
    ("production", "test_body"),
    [
        pytest.param(
            "def helper():\n    return 1\n",
            (
                "from pkg.impl import *\n\nclass Test:\n"
                "    def helper(self):\n        return helper()\n"
            ),
            id="method-homonym-is-not-in-scope",
        ),
        pytest.param(
            "def helper():\n    return 1\n",
            (
                "from pkg.impl import *\n\nclass Test:\n"
                "    def helper():\n        return 0\n    value = helper()\n"
            ),
            id="class-body-falls-back-to-name-matching",
        ),
        pytest.param(
            "class Obj:\n    def helper(self):\n        return 1\n",
            (
                "from pkg.impl import Obj\n\ndef test_call():\n"
                "    def helper():\n        return 0\n"
                "    return Obj().helper()\n"
            ),
            id="attribute-load-is-name-matched",
        ),
    ],
)
def test_class_scopes_and_attribute_loads_do_not_shadow_production_names(
    tmp_path: Path, production: str, test_body: str
) -> None:
    """Only enclosing function scopes and the module top level shadow a bare name."""
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "impl.py").write_text(production)
    tests_dir = root / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_impl.py").write_text(test_body)

    result = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=False,
            run_unused=True,
            strict_unused=True,
        )
    ).analyze(root)

    assert "helper" not in {unit.name for unit in result.potentially_unused}


@pytest.mark.parametrize("sibling_call", [False, True])
def test_filtered_nested_bindings_do_not_mask_sibling_references(
    tmp_path: Path, sibling_call: bool
) -> None:
    """A nested binding shadows its own loads, not loads in sibling scopes."""
    sibling = "    def _sibling():\n        return helper()\n" if sibling_call else ""
    source = (
        "def helper():\n    return 1\n\n"
        "def public():\n"
        "    def _branch():\n"
        "        def helper():\n            return 2\n"
        "        return helper()\n"
        f"{sibling}"
        "    return _branch()\n"
    )
    units = extract_units(tmp_path, source, include_private=False)
    build_reference_graph(units)

    assert bool(_unit(units, "sample.helper").references) is sibling_call


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            "    if flag:\n        def _helper():\n            return 1\n"
            "    else:\n        def _helper():\n            return 2\n"
            "    return _helper()\n",
            id="conditional-branches",
        ),
        pytest.param(
            "    def _helper(fn):\n        return fn\n"
            "    @_helper\n    def _helper():\n        return 1\n"
            "    return _helper()\n",
            id="decorator-rebinds-the-name",
        ),
        pytest.param(
            "    for step in range(2):\n        if step:\n            _helper()\n"
            "        def _helper():\n            return 1\n",
            id="loop-use-before-definition",
        ),
    ],
)
def test_every_same_named_local_definition_is_a_candidate(tmp_path: Path, body: str) -> None:
    """Resolution ignores statement order, so no binding the load can reach is reported."""
    units, _unused = _referenced_graph(tmp_path, f"def outer(flag):\n{body}")
    outer = _unit(units, "sample.outer")
    helpers = [unit for unit in units if unit.name == "_helper"]

    assert helpers
    assert all(outer.uid in helper.references for helper in helpers)


def test_global_declaration_bypasses_nested_definition(tmp_path: Path) -> None:
    """A bare global load credits the module definition, not a nested homonym."""
    source = (
        "def _helper():\n    return 1\n"
        "def outer():\n"
        "    def _helper():\n        return 2\n"
        "    def caller():\n"
        "        global _helper\n"
        "        return _helper()\n"
        "    return caller()\n"
    )
    units, _unused = _referenced_graph(tmp_path, source)
    module_helper, nested_helper = sorted(
        (unit for unit in units if unit.name == "_helper"), key=lambda u: u.lineno
    )

    assert module_helper.references
    assert nested_helper.references == set()


def test_default_excluded_symlink_directory_does_not_import_external_references(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "test_impl.py").write_text(
        "from pkg.impl import helper\n\nhelper()\n", encoding="utf-8"
    )
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    (root / "impl.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (root / "tests").symlink_to(outside, target_is_directory=True)

    result = CodeAnalyzer(
        AnalyzerConfig(run_traditional=False, run_semantic=False, strict_unused=True)
    ).analyze(root)

    assert "helper" in {unit.name for unit in result.potentially_unused}


@pytest.mark.parametrize("target_name", ["source.py", "test_alias.py", None])
def test_reference_graph_counts_in_tree_file_symlink_once(tmp_path: Path, target_name: str | None):
    source = tmp_path / "source.py"
    source.write_text("def _loop():\n    return _loop()\n", encoding="utf-8")
    alias = tmp_path / "test_alias.py"
    alias.symlink_to(source)
    target = tmp_path / target_name if target_name is not None else tmp_path

    result = CodeAnalyzer(
        AnalyzerConfig(run_traditional=False, run_semantic=False, strict_unused=True)
    ).analyze(target)

    assert [unit.name for unit in result.potentially_unused] == ["_loop"]
    assert result.potentially_unused[0].references == set()


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
        diagnostics = build_reference_graph(units)
    unused = find_potentially_unused(units, strict_unused=True)

    [record] = [record for record in caplog.records if record.name == "codedupes.unused"]
    assert record.levelno == logging.WARNING
    assert record.getMessage().startswith(
        f"Unused analysis collected no references from {units[0].file_path}: SyntaxError"
    )
    assert [d.code for d in diagnostics] == ["unused-parse-error"]
    assert diagnostics[0].lineno == 7
    assert "SyntaxError" in diagnostics[0].message
    assert _unit(units, "sample._intact").references == set()
    assert {unit.name for unit in unused} == {"_intact", "_caller"}


def _write_elif_chain(root: Path, branch_count: int) -> None:
    """Write a package with a generated ``elif`` chain and its lone caller.

    :param root: Package directory to write into (must already exist).
    :param branch_count: Number of generated ``elif`` branches.
    :return: ``None``.
    """
    branches = "\n".join(f"    elif x == {i}:\n        return {i}" for i in range(1, branch_count))
    (root / "__init__.py").write_text("")
    (root / "chain.py").write_text(
        f"def _big(x):\n    if x == 0:\n        return 0\n{branches}\n\n"
        "def _user():\n    return _big(1)\n"
    )


def test_deep_elif_chain_is_analyzed_without_a_recursion_bailout(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The raised visitor recursion limit must absorb a few hundred nested branches."""
    root = tmp_path / "pkg"
    root.mkdir()
    _write_elif_chain(root, 600)
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False, run_semantic=False, run_unused=True, strict_unused=True
        )
    )

    with caplog.at_level(logging.WARNING, logger="codedupes.unused"):
        result = analyzer.analyze(root)

    assert [record for record in caplog.records if record.name == "codedupes.unused"] == []
    assert result.unused_diagnostics == []
    # _user calls _big, so _big is referenced and drops out; _user stays.
    assert {unit.name for unit in result.potentially_unused} == {"_user"}


def test_visitor_recursion_bailout_is_reported_per_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A chain deeper than even the raised limit becomes a diagnostic, not a crash."""
    monkeypatch.setattr(unused_module, "_VISIT_RECURSION_LIMIT", 100)
    root = tmp_path / "pkg"
    root.mkdir()
    _write_elif_chain(root, 600)
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False, run_semantic=False, run_unused=True, strict_unused=True
        )
    )

    result = analyzer.analyze(root)

    assert [d.code for d in result.unused_diagnostics] == ["unused-recursion-limit"]
    assert {unit.name for unit in result.potentially_unused} == {"_big", "_user"}


def test_parser_stack_overflow_is_a_diagnostic_not_a_crash(tmp_path: Path) -> None:
    """``ast.parse`` itself overflows its C stack on pathological nesting; survive it."""
    path = tmp_path / "deep.py"
    branches = "\n".join(f"    elif x == {i}:\n        return {i}" for i in range(1, 6000))
    path.write_text(f"def _big(x):\n    if x == 0:\n        return 0\n{branches}\n")

    units = list(CodeExtractor(tmp_path, include_private=True).extract_from_file(path))
    diagnostics = build_reference_graph(units)

    assert [d.code for d in diagnostics] == ["unused-recursion-limit"]


def test_abstractmethod_exemption_reads_only_the_units_own_decorators(tmp_path: Path) -> None:
    """The decorated method is exempt; its class, a body mentioning the text, and a
    same-prefix decorator name are not (that one is reached as a registration instead)."""
    source = dedent(
        """
        import abc
        from abc import abstractmethod

        def abstractmethodish(func):
            return func

        class _Holder(abc.ABC):
            @abc.abstractmethod
            def _do(self):
                return 1

            @abstractmethod
            def _step(self):
                return 2

            @abstractmethodish
            def _lookalike(self):
                return 3

        def _fake():
            return "@abstractmethod"
        """
    ).strip()
    units, unused = _referenced_graph(tmp_path, source)

    assert _unit(units, "sample._Holder._do").references == set()
    lookalike = _unit(units, "sample._Holder._lookalike")
    assert not unused_module._is_abstract(lookalike)
    assert lookalike.references == {"decorator::abstractmethodish"}
    assert unused == {"_Holder", "_fake"}


def test_test_file_exemption_matches_the_default_exclude_shapes(tmp_path: Path) -> None:
    """The test-file exemption covers ``conftest.py`` and the default exclude shapes, not any ``_test`` substring."""
    source = "def _dead():\n    return 1\n"
    expect_reported = {
        "conftest.py": False,
        "legacy_testament.py": True,
        "probe_test.py": False,
        "probe_tests.py": False,
        "test_probe.py": False,
    }
    for filename, reported in expect_reported.items():
        units = extract_units(tmp_path, source, filename=filename, include_private=True)
        build_reference_graph(units)
        unused = find_potentially_unused(units, strict_unused=True)
        assert ("_dead" in {unit.name for unit in unused}) is reported, filename
