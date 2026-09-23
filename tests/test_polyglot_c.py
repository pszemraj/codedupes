"""C backend: definition extraction, static detection, and structural-hash normalization."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedupes.models import CodeUnitType
from tests.polyglot_helpers import extract

pytestmark = pytest.mark.grammar


def test_c_extracts_definitions_and_ignores_prototypes(tmp_path: Path) -> None:
    units = extract(
        tmp_path,
        "sample.c",
        """
        int declared(int value);

        static int private_helper(int value) {
            return value + 1;
        }

        int public_helper(int value) {
            return private_helper(value);
        }
        """,
    )

    assert {unit.name for unit in units} == {"private_helper", "public_helper"}
    assert all(unit.unit_type == CodeUnitType.FUNCTION for unit in units)
    assert all(unit.language == "c" and unit.dialect == "c" for unit in units)
    assert next(unit for unit in units if unit.name == "private_helper").is_public is False


def test_c_static_detection_ignores_array_parameters_and_comments(tmp_path: Path) -> None:
    """C99 ``[static n]`` parameters and prose both contain the word ``static``."""
    units = extract(
        tmp_path,
        "sample.c",
        """
        int copy_row(int destination[static 4]) { return destination[0]; }

        int /* keeps a static cache */ cached(void) { return 1; }

        static int hidden(void) { return 2; }
        """,
    )

    assert {unit.name: unit.is_public for unit in units} == {
        "copy_row": True,
        "cached": True,
        "hidden": False,
    }


def test_c_structural_hash_normalizes_names_and_keeps_operator_semantics(tmp_path: Path) -> None:
    units = extract(
        tmp_path,
        "sample.c",
        """
        int add(int a, int b) { return a + b; }
        int total(int x, int y) { return x + y; }
        int subtract(int x, int y) { return x - y; }
        """,
    )
    by_name = {unit.name: unit for unit in units}

    assert by_name["add"].structural_hash == by_name["total"].structural_hash
    assert by_name["total"].structural_hash != by_name["subtract"].structural_hash


def test_c_suppression_directive_attachment(tmp_path: Path) -> None:
    """``codedupes: ignore`` attaches through C's comment forms."""
    [line_comment] = extract(
        tmp_path,
        "line_comment.c",
        """
        // codedupes: ignore
        int foo(void) {
            return 1;
        }
        """,
    )
    assert line_comment.suppressions == {"unused", "duplicates"}

    [block_comment] = extract(
        tmp_path,
        "block_comment.c",
        """
        /* codedupes: ignore */
        int foo(void) {
            return 1;
        }
        """,
    )
    assert block_comment.suppressions == {"unused", "duplicates"}

    [trailing_brace] = extract(
        tmp_path,
        "trailing_brace.c",
        """
        int foo(void) {  // codedupes: ignore
            return 1;
        }
        """,
    )
    assert trailing_brace.suppressions == {"unused", "duplicates"}

    # tree-sitter-c parses a comment after a one-line unit as its next sibling.
    first, last = extract(
        tmp_path,
        "one_line.c",
        """
        int foo(void) { return 1; } int bar(void) { return 2; } // codedupes: ignore
        """,
    )
    assert (first.suppressions, last.suppressions) == (set(), {"unused", "duplicates"})
