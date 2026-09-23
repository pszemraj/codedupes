"""Python backend: nesting, decorators, docstring pruning, formatting-only rewrites, and structural fingerprints."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedupes.models import CodeUnit, CodeUnitType
from tests.polyglot_helpers import python_result, python_units

pytestmark = pytest.mark.grammar


def _python_unit(tmp_path: Path, filename: str, source: str) -> CodeUnit:
    """Return the outermost unit of a single-definition source."""
    return min(python_result(tmp_path, source, filename=filename).units, key=lambda u: u.start_byte)


def test_python_nested_scopes_methods_and_decorated_spans(tmp_path: Path) -> None:
    """Decorators belong to the unit; qualified names follow lexical nesting in order."""
    source = '''
    import functools

    @functools.lru_cache
    def top(value):
        def inner(v):
            return v + 1
        return inner(value)

    class Widget:
        """Doc."""

        @property
        def _hidden(self):
            return 1

        def shown(self):
            class Local:
                def run(self):
                    return 2
            return Local

    def outer():
        class C:
            def m(self):
                return 1
        return C
    '''
    units = python_units(tmp_path, source)
    raw = (tmp_path / "sample.py").read_bytes()

    assert {name: unit.unit_type for name, unit in units.items()} == {
        "sample.top": CodeUnitType.FUNCTION,
        "sample.top.inner": CodeUnitType.FUNCTION,
        "sample.Widget": CodeUnitType.CLASS,
        "sample.Widget._hidden": CodeUnitType.METHOD,
        "sample.Widget.shown": CodeUnitType.METHOD,
        "sample.Widget.shown.Local": CodeUnitType.CLASS,
        "sample.Widget.shown.Local.run": CodeUnitType.METHOD,
        "sample.outer": CodeUnitType.FUNCTION,
        "sample.outer.C": CodeUnitType.CLASS,
        "sample.outer.C.m": CodeUnitType.METHOD,
    }
    assert {unit.native_kind for unit in units.values()} == {
        "function_definition",
        "class_definition",
    }
    assert all(unit.language == "python" and unit.dialect == "python" for unit in units.values())
    for unit in units.values():
        assert raw[unit.start_byte : unit.end_byte].decode("utf-8") == unit.source
        assert not unit.source.endswith("\n")

    top = units["sample.top"]
    assert top.source.startswith("@functools.lru_cache\ndef top")
    assert (top.lineno, top.end_lineno, top.start_column) == (3, 7, 0)
    hidden = units["sample.Widget._hidden"]
    assert hidden.source.startswith("@property\n    def _hidden")
    assert (hidden.lineno, hidden.end_lineno, hidden.start_column) == (12, 14, 4)
    assert units["sample.Widget.shown.Local.run"].start_column == 12


def test_python_renamed_locals_hash_structurally_equal_but_api_shape_does_not(
    tmp_path: Path,
) -> None:
    """Local names normalize; attribute, keyword, and import names are API shape."""
    units = python_units(
        tmp_path,
        """
        def add(a, b):
            return a + b

        def total(x, y):
            return x + y

        def attr_a(o):
            return o.alpha

        def attr_b(o):
            return o.beta

        def kw_a():
            return call(key=1)

        def kw_b():
            return call(name=1)

        def imp_a():
            import os
            return os

        def imp_b():
            import sys
            return sys

        def sync_f():
            return 1

        async def async_f():
            return 1
        """,
    )

    assert units["sample.add"].structural_hash == units["sample.total"].structural_hash
    assert units["sample.add"].token_hash != units["sample.total"].token_hash
    assert units["sample.attr_a"].structural_hash != units["sample.attr_b"].structural_hash
    assert units["sample.kw_a"].structural_hash != units["sample.kw_b"].structural_hash
    assert units["sample.imp_a"].structural_hash != units["sample.imp_b"].structural_hash
    assert units["sample.sync_f"].structural_hash != units["sample.async_f"].structural_hash


def test_python_docstrings_are_pruned_structurally_but_kept_in_tokens(tmp_path: Path) -> None:
    """Docstrings drop positionally at every nesting level; comments never count."""
    documented = _python_unit(
        tmp_path,
        "documented.py",
        '''
        def helper(a):
            """Explain."""
            def inner():
                """Inner doc."""
                return a
            return inner()
        ''',
    )
    bare = _python_unit(
        tmp_path,
        "bare.py",
        """
        def helper(a):
            def inner():
                return a
            return inner()
        """,
    )
    commented = _python_unit(
        tmp_path,
        "commented.py",
        """
        def helper(a):
            # explain
            def inner():
                # inner note
                return a
            return inner()
        """,
    )

    assert documented.structural_hash == bare.structural_hash
    assert documented.token_hash != bare.token_hash
    assert commented.structural_hash == bare.structural_hash
    assert commented.token_hash == bare.token_hash


def test_python_only_a_leading_plain_string_is_a_docstring(tmp_path: Path) -> None:
    """f-strings and bytes are not ``ast.Constant(str)``, and a string inside an
    ``if`` block is a statement, so none of them prune."""
    units = python_units(
        tmp_path,
        """
        def formatted(a):
            f"not a docstring"
            return a

        def raw_bytes(a):
            b"not a docstring"
            return a

        def plain(a):
            return a

        def conditional(x):
            if x:
                "note"
            return x

        def unconditional(x):
            if x:
                pass
            return x
        """,
    )

    assert units["sample.formatted"].structural_hash != units["sample.plain"].structural_hash
    assert units["sample.raw_bytes"].structural_hash != units["sample.plain"].structural_hash
    assert (
        units["sample.conditional"].structural_hash != units["sample.unconditional"].structural_hash
    )


def test_python_escape_sequence_literal_text_survives_the_token_hash(tmp_path: Path) -> None:
    """tree-sitter-python only exposes the escape as a child; the surrounding
    literal text must still reach the token stream."""
    first = _python_unit(tmp_path, "first.py", 'def text():\n    return "a\\nb"\n')
    second = _python_unit(tmp_path, "second.py", 'def text():\n    return "a\\nc"\n')

    assert first.token_hash != second.token_hash
    assert first.structural_hash == second.structural_hash


def test_python_backslash_continuation_is_formatting(tmp_path: Path) -> None:
    continued = _python_unit(
        tmp_path,
        "continued.py",
        "def add(a):\n    return a + \\\n        1\n",
    )
    joined = _python_unit(tmp_path, "joined.py", "def add(a):\n    return a + 1\n")

    assert continued.structural_hash == joined.structural_hash
    assert continued.token_hash == joined.token_hash


@pytest.mark.parametrize(
    ("formatted", "plain"),
    [
        pytest.param(
            "def f():\n    x = 1; y = 2\n    return x + y\n",
            "def f():\n    x = 1\n    y = 2\n    return x + y\n",
            id="semicolon-separator",
        ),
        pytest.param(
            "def f(a, b):\n    x = (\n        a + b\n    )\n    return x\n",
            "def f(a, b):\n    x = a + b\n    return x\n",
            id="grouping-parentheses",
        ),
        pytest.param(
            'def f():\n    raise ValueError("long "\n        "continued")\n',
            'def f():\n    raise ValueError("long continued")\n',
            id="implicit-concatenation",
        ),
        pytest.param(
            "def f(a, b):\n    return g(\n        a,\n        b,\n    )\n",
            "def f(a, b):\n    return g(a, b)\n",
            id="call-trailing-comma",
        ),
        pytest.param(
            "def f(a, b):\n    return {\n        a,\n        b,\n    }\n",
            "def f(a, b):\n    return {a, b}\n",
            id="set-trailing-comma",
        ),
        pytest.param(
            "def f(data, a, b):\n    return data[a, b,]\n",
            "def f(data, a, b):\n    return data[a, b]\n",
            id="multiple-indices-trailing-comma",
        ),
        pytest.param(
            "def f(data, keys):\n    return data[*keys,]\n",
            "def f(data, keys):\n    return data[*keys]\n",
            id="starred-index-trailing-comma",
        ),
        pytest.param(
            'def f():\n    ("doc")\n    return 1\n',
            'def f():\n    "doc"\n    return 1\n',
            id="parenthesized-docstring",
        ),
    ],
)
def test_python_formatting_only_rewrites_keep_the_structural_hash(
    tmp_path: Path, formatted: str, plain: str
) -> None:
    """Separators, magic trailing commas, grouping parentheses, and split literals
    are what a formatter adds; ``ast`` has no node for any of them."""
    first = _python_unit(tmp_path, "formatted.py", formatted)
    second = _python_unit(tmp_path, "plain.py", plain)

    assert first.structural_hash == second.structural_hash
    assert first.token_hash != second.token_hash
    assert first.statement_count == second.statement_count


def test_python_parenthesized_docstring_is_pruned(tmp_path: Path) -> None:
    """``("doc")`` is ``Constant(str)`` to ``ast``, so it prunes like a bare docstring."""
    wrapped = _python_unit(tmp_path, "wrapped.py", 'def f():\n    ("doc")\n    return 1\n')
    bare = _python_unit(tmp_path, "bare.py", "def f():\n    return 1\n")

    assert wrapped.structural_hash == bare.structural_hash
    assert (wrapped.statement_count, bare.statement_count) == (1, 1)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(
            "def f(a, b, c):\n    return (a + b) * c\n",
            "def f(a, b, c):\n    return a + b * c\n",
            id="parentheses-change-precedence",
        ),
        pytest.param(
            "def f(a):\n    return (a,)\n",
            "def f(a):\n    return (a)\n",
            id="tuple-versus-grouped-name",
        ),
        pytest.param(
            'def f(x):\n    return "a" f"{x.y}"\n',
            'def f(x):\n    return "a" f"{x}"\n',
            id="concatenation-with-an-fstring-part",
        ),
    ],
)
def test_python_grouping_that_changes_the_tree_stays_structural(
    tmp_path: Path, first: str, second: str
) -> None:
    """Dropping the parentheses node keeps the nesting it expressed, a one-tuple is
    not a grouped name, and a concatenation with an interpolation is still walked."""
    left = _python_unit(tmp_path, "first.py", first)
    right = _python_unit(tmp_path, "second.py", second)

    assert left.structural_hash != right.structural_hash


def test_python_decorators_are_part_of_the_unit_fingerprints(tmp_path: Path) -> None:
    """A decorated unit starts at its decorator, so the decorator reaches both
    hashes and the identifier set."""
    units = python_units(
        tmp_path,
        """
        import functools

        @functools.lru_cache
        def cached(a):
            return a

        def plain(a):
            return a
        """,
    )
    cached = units["sample.cached"]
    plain = units["sample.plain"]

    assert cached.structural_hash != plain.structural_hash
    assert cached.token_hash != plain.token_hash
    assert {"functools", "lru_cache"} <= cached.identifiers
    assert not ({"functools", "lru_cache"} & plain.identifiers)


def test_python_fstring_interpolation_is_structural(tmp_path: Path) -> None:
    """Interpolated expressions are code; the literal text around them is not."""
    units = python_units(
        tmp_path,
        """
        def name(x):
            return f"v{x}w"

        def member(x):
            return f"v{x.y}w"

        def relabeled(x):
            return f"v{x}z"
        """,
    )

    assert units["sample.name"].structural_hash != units["sample.member"].structural_hash
    assert units["sample.name"].structural_hash == units["sample.relabeled"].structural_hash
    assert units["sample.name"].token_hash != units["sample.relabeled"].token_hash


@pytest.mark.parametrize(
    ("source", "expected_count"),
    [
        pytest.param(
            """
            def f(x):
                if x:
                    a = 1
                elif x is None:
                    a = 2
                else:
                    a = 3
                return a
            """,
            6,
            id="if-elif-else-return",
        ),
        pytest.param(
            """
            def f(items):
                for item in items:
                    use(item)
                else:
                    done()
            """,
            3,
            id="for-else",
        ),
        pytest.param(
            """
            def f(ready):
                while ready:
                    ready = step()
                else:
                    done()
            """,
            3,
            id="while-else",
        ),
        pytest.param(
            """
            def f():
                try:
                    risky()
                except ValueError as error:
                    handle(error)
                else:
                    celebrate()
                finally:
                    cleanup()
            """,
            5,
            id="try-except-else-finally",
        ),
        pytest.param(
            """
            def f(path):
                with open(path) as handle:
                    data = handle.read()
                return data
            """,
            3,
            id="with",
        ),
        pytest.param(
            """
            def f(value):
                match value:
                    case 0:
                        return "zero"
                    case _:
                        return "other"
            """,
            3,
            id="match-two-cases",
        ),
        pytest.param(
            """
            def outer():
                def inner():
                    return 1
                return inner
            """,
            2,
            id="nested-def",
        ),
        pytest.param(
            """
            def outer():
                @staticmethod
                def inner():
                    return 1
                return inner
            """,
            2,
            id="decorated-nested-def",
        ),
        pytest.param(
            '''
            def f():
                """Only a docstring."""
            ''',
            0,
            id="docstring-only",
        ),
        pytest.param(
            """
            def f():
                ...
            """,
            1,
            id="ellipsis",
        ),
        pytest.param(
            '''
            def f():
                """Doc."""
                ...
            ''',
            1,
            id="docstring-ellipsis",
        ),
        pytest.param(
            '''
            class Widget:
                """Doc."""

                size: int = 1

                @staticmethod
                def build():
                    return Widget()

                def run(self):
                    return self.size
            ''',
            3,
            id="class-body",
        ),
        pytest.param(
            """
            async def f(items, lock):
                async for item in items:
                    await item
                async with lock as held:
                    held.touch()
            """,
            4,
            id="async-for-await-async-with",
        ),
        pytest.param(
            """
            def f():
                x = 1; y = 2
            """,
            2,
            id="semicolon-separated",
        ),
        pytest.param(
            '''
            def sample(a, b):
                """doc"""
                x = 1
                return a + b + x
            ''',
            2,
            id="docstring-then-statements",
        ),
        pytest.param(
            """
            def guarded():
                try:
                    a = 1
                    b = 2
                    c = 3
                    return a + b + c
                except ValueError:
                    return 0
            """,
            # try + 4 body statements + handler return; the except clause itself
            # is an ``ast.excepthandler``, not a statement.
            6,
            id="try-with-multi-statement-body",
        ),
        pytest.param(
            """
            def managed(path):
                with open(path) as handle:
                    first = handle.readline()
                    second = handle.readline()
                    return first + second
            """,
            4,
            id="with-multi-statement-body",
        ),
        pytest.param(
            """
            def looped(items):
                for item in items:
                    if item:
                        yield item
                    else:
                        continue
            """,
            4,
            id="loop-if-else",
        ),
        pytest.param(
            """
            def outer():
                def inner():
                    a = 1
                    b = 2
                    return a + b

                class Helper:
                    def method(self):
                        return 1

                return inner
            """,
            # inner (1) + Helper (1) + return (1); nested bodies belong to their own units.
            3,
            id="nested-def-and-class",
        ),
    ],
)
def test_python_statement_counts_follow_ast_stmt_semantics(
    tmp_path: Path, source: str, expected_count: int
) -> None:
    """Counts recurse through control-flow bodies, count nested definitions once,
    and never count a docstring. ``elif`` is a nested ``ast.If``, so it counts."""
    result = python_result(tmp_path, source)
    outer = min(result.units, key=lambda unit: unit.start_byte)

    assert outer.statement_count == expected_count


def test_python_directive_attachment(tmp_path: Path) -> None:
    """``codedupes: ignore`` attaches by grammar position, never by textual proximity."""
    trailing_on_def = python_units(
        tmp_path,
        """
        def f():  # codedupes: ignore
            return 1
        """,
        filename="trailing_on_def.py",
    )
    assert trailing_on_def["trailing_on_def.f"].suppressions == {"unused", "duplicates"}

    on_decorator_line = python_units(
        tmp_path,
        """
        def _decorate(fn):
            return fn

        @_decorate  # codedupes: ignore
        def f():
            return 1
        """,
        filename="on_decorator_line.py",
    )
    assert on_decorator_line["on_decorator_line.f"].suppressions == {"unused", "duplicates"}

    between_decorators = python_units(
        tmp_path,
        """
        def _decorate(fn):
            return fn

        @_decorate
        # codedupes: ignore
        @_decorate
        def f():
            return 1
        """,
        filename="between_decorators.py",
    )
    assert between_decorators["between_decorators.f"].suppressions == {"unused", "duplicates"}

    block_above_decorators = python_units(
        tmp_path,
        """
        def _decorate(fn):
            return fn

        # codedupes: ignore
        @_decorate
        def f():
            return 1
        """,
        filename="block_above_decorators.py",
    )
    assert block_above_decorators["block_above_decorators.f"].suppressions == {
        "unused",
        "duplicates",
    }

    above_first_method = python_units(
        tmp_path,
        """
        class C:
            # codedupes: ignore
            def m(self):
                return 1
        """,
        filename="above_first_method.py",
    )
    assert above_first_method["above_first_method.C"].suppressions == set()
    assert above_first_method["above_first_method.C.m"].suppressions == {"unused", "duplicates"}

    nested_inner = python_units(
        tmp_path,
        """
        def outer():
            # codedupes: ignore
            def inner():
                return 1
            return inner
        """,
        filename="nested_inner.py",
    )
    assert nested_inner["nested_inner.outer"].suppressions == set()
    assert nested_inner["nested_inner.outer.inner"].suppressions == {"unused", "duplicates"}

    class_directive = python_units(
        tmp_path,
        """
        # codedupes: ignore
        class C:
            def m(self):
                return 1
        """,
        filename="class_directive.py",
    )
    assert class_directive["class_directive.C"].suppressions == {"unused", "duplicates"}
    assert class_directive["class_directive.C.m"].suppressions == {"unused", "duplicates"}

    string_marker = python_units(
        tmp_path,
        """
        def f():
            "# codedupes: ignore"
            return 1
        """,
        filename="string_marker.py",
    )
    assert string_marker["string_marker.f"].suppressions == set()

    docstring_marker = python_units(
        tmp_path,
        '''
        def f():
            """codedupes: ignore"""
            return 1
        ''',
        filename="docstring_marker.py",
    )
    assert docstring_marker["docstring_marker.f"].suppressions == set()

    trailing_assignment = python_units(
        tmp_path,
        """
        x = 1  # codedupes: ignore
        def f():
            return 1
        """,
        filename="trailing_assignment.py",
    )
    assert trailing_assignment["trailing_assignment.f"].suppressions == set()

    blank_line_breaks = python_units(
        tmp_path,
        """
        # codedupes: ignore

        def f():
            return 1
        """,
        filename="blank_line_breaks.py",
    )
    assert blank_line_breaks["blank_line_breaks.f"].suppressions == set()

    unknown_kind_result = python_result(
        tmp_path,
        """
        def f():  # codedupes: ignore[bogus]
            return 1
        """,
        filename="unknown_kind.py",
    )
    assert unknown_kind_result.units[0].suppressions == set()
    assert [d.code for d in unknown_kind_result.diagnostics] == ["suppression-syntax"]

    unclosed_result = python_result(
        tmp_path,
        """
        def f():  # codedupes: ignore[unused
            return 1
        """,
        filename="unclosed_kind.py",
    )
    assert unclosed_result.units[0].suppressions == set()
    assert [d.code for d in unclosed_result.diagnostics] == ["suppression-syntax"]

    single_line_body = python_result(
        tmp_path,
        "def f(): return 1  # codedupes: ignore[duplicates]\n",
        filename="single_line_body.py",
    )
    assert single_line_body.units[0].suppressions == {"duplicates"}
