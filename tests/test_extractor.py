from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent

from codedupes.extractor import compute_ast_hash
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
