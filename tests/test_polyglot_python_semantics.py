"""Python backend: identifiers, dunder-all filtering, privacy rules, and error recovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedupes.languages.tree_sitter_backend import _PYTHON_BUILTINS
from tests.polyglot_helpers import python_result, python_units

pytestmark = pytest.mark.grammar


def test_python_identifiers_include_api_names_and_exclude_builtins(tmp_path: Path) -> None:
    units = python_units(
        tmp_path,
        """
        class Bag:
            def add(self, item):
                cls = type(self)
                return self.items.append(item, key=len(item))
        """,
    )
    identifiers = units["sample.Bag.add"].identifiers

    assert {"add", "item", "items", "append", "key"} <= identifiers
    assert not ({"self", "cls", "len", "type", "return", "def"} & identifiers)


def test_python_private_filter_keeps_dunder_and_mangled_names(tmp_path: Path) -> None:
    """A single underscore is private; ``__x`` and ``__x__`` are not public but still
    extract, and a filtered private class takes its methods with it."""
    source = """
    def _private():
        return 1

    def __mangled():
        return 2

    def __dunder__():
        return 3

    class _Hidden:
        def visible(self):
            return 4

    class Shown:
        def _helper(self):
            return 5

        def __init__(self):
            pass
    """
    everything = python_units(tmp_path, source)
    public_only = python_units(tmp_path, source, include_private=False)

    assert len(everything) == 8
    assert set(public_only) == {
        "sample.__mangled",
        "sample.__dunder__",
        "sample.Shown",
        "sample.Shown.__init__",
    }
    init = public_only["sample.Shown.__init__"]
    assert not init.is_public
    assert init.is_dunder


def test_python_dunder_all_unions_assignment_and_augmented_assignment(tmp_path: Path) -> None:
    units = python_units(
        tmp_path,
        """
        __all__ = ["alpha"]
        __all__ += ("beta",)

        def alpha():
            return 1

        def beta():
            return 2

        def gamma():
            return 3
        """,
    )

    assert {name: unit.is_exported for name, unit in units.items()} == {
        "sample.alpha": True,
        "sample.beta": True,
        "sample.gamma": False,
    }


def test_python_dunder_all_accepts_bare_tuples_and_module_level_containers(
    tmp_path: Path,
) -> None:
    """``__all__`` inside a module-level ``if``/``try`` runs at import; one inside a
    function body does not."""
    units = python_units(
        tmp_path,
        """
        import sys

        __all__ = "alpha", "beta"

        if sys.version_info >= (3, 12):
            __all__ += ["gamma"]
        else:
            __all__ += ["gamma"]

        try:
            from ._fast import delta
            __all__ += ("delta",)
        except ImportError:
            pass

        def _register():
            __all__ = ["epsilon"]

        def alpha():
            return 1

        def beta():
            return 2

        def gamma():
            return 3

        def delta():
            return 4

        def epsilon():
            return 5
        """,
    )

    assert {name: unit.is_exported for name, unit in units.items()} == {
        "sample._register": False,
        "sample.alpha": True,
        "sample.beta": True,
        "sample.gamma": True,
        "sample.delta": True,
        "sample.epsilon": False,
    }


def test_python_bodiless_definitions_yield_no_units_and_no_diagnostics(tmp_path: Path) -> None:
    """``def f():`` with nothing under it is a CPython syntax error that tree-sitter
    accepts as an empty block, so there is no body to fingerprint and no error node
    to report."""
    result = python_result(tmp_path, "def a():\n\ndef b():\n\nclass C:\n\nx = 1\n")

    assert result.units == ()
    assert result.diagnostics == ()


def test_python_builtins_exclude_the_site_injected_names(tmp_path: Path) -> None:
    """``exit``/``quit``/``help`` come from ``site``, not the language, so they stay identifiers."""
    units = python_units(
        tmp_path,
        """
        def bail(items):
            exit(1)
            return len(items)
        """,
    )
    identifiers = units["sample.bail"].identifiers

    assert "exit" in identifiers
    assert "len" not in identifiers
    assert "exit" not in _PYTHON_BUILTINS
    assert {"len", "print", "self", "cls", "def"} <= _PYTHON_BUILTINS


def test_python_private_function_filter_drops_its_nested_definitions(tmp_path: Path) -> None:
    """A filtered private container of any kind takes what it nests with it."""
    units = python_units(
        tmp_path,
        """
        def _outer():
            def inner():
                return 1

            class Local:
                def run(self):
                    return 2

            return inner, Local

        def outer():
            def inner():
                return 3

            return inner
        """,
        include_private=False,
    )

    assert set(units) == {"sample.outer", "sample.outer.inner"}


@pytest.mark.parametrize(
    ("filename", "expected_qualified_name"),
    [
        ("pkg/__init__.py", "pkg.func"),
        ("__init__.py", "func"),
        ("pkg/mod.py", "pkg.mod.func"),
        ("pkg/mod.pyi", "pkg.mod.func"),
    ],
)
def test_python_module_prefix_follows_package_layout(
    tmp_path: Path, filename: str, expected_qualified_name: str
) -> None:
    units = python_units(tmp_path, "def func():\n    return 1", filename=filename)

    assert list(units) == [expected_qualified_name]


def test_python_error_recovery_reports_partial_parse_and_skips_the_broken_unit(
    tmp_path: Path,
) -> None:
    result = python_result(
        tmp_path,
        """
        def broken(:
            pass

        def ok():
            return 1
        """,
    )

    assert [unit.qualified_name for unit in result.units] == ["sample.ok"]
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "partial-parse",
        "unit-parse-error",
    ]
    assert all(diagnostic.language == "python" for diagnostic in result.diagnostics)
