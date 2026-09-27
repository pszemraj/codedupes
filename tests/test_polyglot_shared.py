"""Cross-language extraction guarantees: grammar readiness, byte-range fidelity, the fingerprint schema, recursion limits, and non-UTF-8 diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedupes.extractor import CodeExtractor
from codedupes.languages.registry import get_grammar_statuses
from codedupes.languages.tree_sitter_backend import FINGERPRINT_SCHEMA_VERSION
from tests.polyglot_helpers import extract

pytestmark = pytest.mark.grammar


def test_every_pinned_grammar_probes_ready_on_this_interpreter() -> None:
    """The live probe must construct a real parser for all six dialects."""
    statuses = get_grammar_statuses()

    assert len(statuses) == 6
    assert all(status.available and status.error is None for status in statuses)


@pytest.mark.parametrize(
    ("filename", "source", "expected_qualified_name"),
    [
        ("sample.py", "def add(left, right):\n    return left + right\n", "sample.add"),
        ("sample.c", "int add(int left, int right) { return left + right; }\n", "sample.add"),
        ("sample.rs", "pub fn add(left: i32, right: i32) -> i32 { left + right }\n", "sample.add"),
        ("sample.js", "export const add = (left, right) => left + right;\n", "sample.add"),
        (
            "sample.ts",
            "export function add(left: number, right: number): number { return left + right; }\n",
            "sample.add",
        ),
        (
            "component.tsx",
            "export const Card = (props: { title: string }) => <h1>{props.title}</h1>;\n",
            "component.Card",
        ),
    ],
)
def test_every_dialect_reproduces_unit_source_from_byte_ranges(
    tmp_path: Path,
    filename: str,
    source: str,
    expected_qualified_name: str,
) -> None:
    units = extract(tmp_path, filename, source)
    source_bytes = (tmp_path / filename).read_bytes()

    assert expected_qualified_name in {unit.qualified_name for unit in units}
    for unit in units:
        assert source_bytes[unit.start_byte : unit.end_byte].decode("utf-8") == unit.source


@pytest.mark.parametrize(
    ("filename", "source", "expected_hash"),
    [
        ("sample.py", "def add(a, b):\n    return a + b", "d5e992360de8b5f7"),
        ("sample.c", "int add(int a, int b) { return a + b; }", "055dad2cb951cd16"),
        ("sample.rs", "pub fn add(a: i32, b: i32) -> i32 { a + b }", "f0e9ce5598395030"),
        ("sample.js", "function add(a, b) { return a + b; }", "06dc5b63c208ce88"),
        (
            "sample.ts",
            "function add(a: number, b: number): number { return a + b; }",
            "80a8dc13209a88a4",
        ),
    ],
)
def test_structural_hash_golden_values_pin_the_fingerprint_schema(
    tmp_path: Path,
    filename: str,
    source: str,
    expected_hash: str,
) -> None:
    """Nothing else persists these hashes, so canonical-stream drift would
    otherwise silently rename every fingerprint."""
    assert FINGERPRINT_SCHEMA_VERSION == 2, "bump the goldens below alongside this constant"
    units = extract(tmp_path, filename, source)

    assert [unit.structural_hash for unit in units] == [expected_hash]


def test_deeply_nested_source_does_not_hit_the_recursion_limit(tmp_path: Path) -> None:
    depth = 5000
    source = f"int deep(int value) {{ return {'(' * depth}value{')' * depth}; }}"

    units = extract(tmp_path, "sample.c", source)

    assert [unit.name for unit in units] == ["deep"]
    assert units[0].structural_hash


def test_deeply_nested_c_declarator_does_not_hit_the_recursion_limit(tmp_path: Path) -> None:
    depth = 2000
    source = f"int {'(' * depth}deep{')' * depth}(int value) {{ return value; }}"

    units = extract(tmp_path, "sample.c", source)

    assert [unit.name for unit in units] == ["deep"]


def test_deeply_nested_object_binding_does_not_hit_the_recursion_limit(tmp_path: Path) -> None:
    depth = 2000
    source = (
        "const root = "
        + "{ nested: " * depth
        + "{ leaf: function () { return 1; } }"
        + " }" * depth
        + ";"
    )

    units = extract(tmp_path, "sample.js", source)

    assert any(unit.qualified_name.endswith(".leaf") for unit in units)


@pytest.mark.parametrize(
    ("filename", "source", "language"),
    [
        ("legacy.js", "function greet() { return 'café'; }\n", "javascript"),
        ("legacy.py", "def greet():\n    return 'café'\n", "python"),
    ],
)
def test_non_utf8_source_is_analyzed_but_reported(
    tmp_path: Path, filename: str, source: str, language: str
) -> None:
    """Recall-first decoding keeps the unit, but replacement characters reach the
    fingerprints and embeddings, so the corruption must not stay silent."""
    path = tmp_path / filename
    path.write_bytes(source.encode("latin-1"))

    extractor = CodeExtractor(tmp_path, include_private=True, languages=(language,))
    units = list(extractor.extract_from_file(path))

    assert [unit.qualified_name for unit in units] == ["legacy.greet"]
    assert [diagnostic.code for diagnostic in extractor.diagnostics] == ["invalid-utf8"]
    assert "�" in units[0].source
