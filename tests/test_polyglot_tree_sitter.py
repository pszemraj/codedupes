"""End-to-end extraction tests against the exact pinned grammar wheels."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

pytest.importorskip("tree_sitter")
pytest.importorskip("tree_sitter_c")
pytest.importorskip("tree_sitter_rust")
pytest.importorskip("tree_sitter_javascript")
pytest.importorskip("tree_sitter_typescript")

from codedupes.extractor import CodeExtractor
from codedupes.models import CodeUnit, CodeUnitType

pytestmark = pytest.mark.grammar


def _extract(
    tmp_path: Path,
    filename: str,
    source: str,
    *,
    include_private: bool = True,
) -> list[CodeUnit]:
    path = tmp_path / filename
    path.write_text(dedent(source).strip() + "\n", encoding="utf-8")
    language = {
        ".c": "c",
        ".h": "c",
        ".rs": "rust",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
    }[path.suffix]
    extractor = CodeExtractor(
        tmp_path,
        include_private=include_private,
        languages=(language,),
    )
    return list(extractor.extract_from_file(path))


@pytest.mark.parametrize(
    ("filename", "source", "expected_qualified_name"),
    [
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
    units = _extract(tmp_path, filename, source)
    source_bytes = (tmp_path / filename).read_bytes()

    assert expected_qualified_name in {unit.qualified_name for unit in units}
    for unit in units:
        assert source_bytes[unit.start_byte : unit.end_byte].decode("utf-8") == unit.source


def test_c_extracts_definitions_and_ignores_prototypes(tmp_path: Path) -> None:
    units = _extract(
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


def test_c_structural_hash_normalizes_names_and_keeps_operator_semantics(tmp_path: Path) -> None:
    units = _extract(
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


def test_rust_extracts_free_impl_trait_and_nested_functions(tmp_path: Path) -> None:
    units = _extract(
        tmp_path,
        "sample.rs",
        """
        pub fn top(value: i32) -> i32 {
            fn nested(value: i32) -> i32 { value + 1 }
            nested(value)
        }

        struct Widget;
        impl Widget {
            fn private_method(&self) -> i32 { 1 }
            pub fn public_method(&self) -> i32 {
                fn local_helper() -> i32 { 2 }
                local_helper()
            }
        }

        trait Service {
            fn default_method(&self) -> i32 { 3 }
            fn required(&self) -> i32;
        }
        """,
    )
    names = {unit.qualified_name: unit.unit_type for unit in units}

    assert names["sample.top"] == CodeUnitType.FUNCTION
    assert names["sample.top.nested"] == CodeUnitType.FUNCTION
    assert names["sample.Widget.private_method"] == CodeUnitType.METHOD
    assert names["sample.Widget.public_method"] == CodeUnitType.METHOD
    assert names["sample.Widget.public_method.local_helper"] == CodeUnitType.FUNCTION
    assert names["sample.Service.default_method"] == CodeUnitType.METHOD
    assert all(not name.endswith("required") for name in names)


def test_javascript_extracts_modern_stable_unit_forms(tmp_path: Path) -> None:
    units = _extract(
        tmp_path,
        "sample.js",
        """
        export function top(value) { return value + 1; }
        const arrow = (value) => value + 2;

        class Worker {
            run(value) { return value + 3; }
            handle = (value) => value + 4;
        }

        const service = {
            load(value) { return value + 5; },
            save: (value) => value + 6,
        };

        const Factory = class {
            make() {
                const local = (value) => value + 7;
                return local;
            }
        };

        function outer() {
            const inner = (value) => value + 8;
            const nestedService = {
                load(value) { return value + 9; },
            };
            return inner(nestedService.load(1));
        }

        export default function () { return 10; }
        """,
    )
    names = {unit.qualified_name: unit.unit_type for unit in units}

    assert names["sample.top"] == CodeUnitType.FUNCTION
    assert names["sample.arrow"] == CodeUnitType.FUNCTION
    assert names["sample.Worker"] == CodeUnitType.CLASS
    assert names["sample.Worker.run"] == CodeUnitType.METHOD
    assert names["sample.Worker.handle"] == CodeUnitType.METHOD
    assert names["sample.service.load"] == CodeUnitType.METHOD
    assert names["sample.service.save"] == CodeUnitType.FUNCTION
    assert names["sample.Factory"] == CodeUnitType.CLASS
    assert names["sample.Factory.make"] == CodeUnitType.METHOD
    assert names["sample.Factory.make.local"] == CodeUnitType.FUNCTION
    assert names["sample.outer"] == CodeUnitType.FUNCTION
    assert names["sample.outer.inner"] == CodeUnitType.FUNCTION
    assert names["sample.outer.nestedService.load"] == CodeUnitType.METHOD
    assert names["sample.default"] == CodeUnitType.FUNCTION


def test_javascript_export_marking_stops_at_function_boundaries(tmp_path: Path) -> None:
    units = _extract(
        tmp_path,
        "sample.js",
        """
        export function outer() {
            const inner = (value) => value + 1;
            function nested(value) { return value + 2; }
            return nested(inner(0));
        }
        export class Api {
            run(value) { return value + 3; }
            handle = (value) => value + 4;
        }
        const helper = (value) => value + 5;
        exports.legacy = (value) => value + 6;
        """,
    )
    exported = {unit.qualified_name: unit.is_exported for unit in units}

    assert exported["sample.outer"] is True
    assert exported["sample.outer.inner"] is False
    assert exported["sample.outer.nested"] is False
    assert exported["sample.Api"] is True
    assert exported["sample.Api.run"] is True
    assert exported["sample.Api.handle"] is True
    assert exported["sample.helper"] is False
    assert exported["sample.exports.legacy"] is True


def test_typescript_excludes_signatures_and_ambient_declarations(tmp_path: Path) -> None:
    units = _extract(
        tmp_path,
        "sample.ts",
        """
        function parse(value: string): number;
        function parse(value: Uint8Array): number;
        function parse(value: string | Uint8Array): number { return value.length; }

        declare function ambient(value: string): number;
        declare class Ambient { run(): void; }

        abstract class Base {
            abstract required(): void;
            concrete(value: number): number { return value + 1; }
            handler = (value: number): number => value + 2;
        }

        namespace Utilities {
            export function normalize(value: string): string { return value.trim(); }
        }
        """,
    )
    names = [unit.qualified_name for unit in units]

    assert names.count("sample.parse") == 1
    assert "sample.ambient" not in names
    assert "sample.Ambient" not in names
    assert "sample.Base" in names
    assert "sample.Base.concrete" in names
    assert "sample.Base.handler" in names
    assert "sample.Utilities.normalize" in names
    assert all(not name.endswith("required") for name in names)


def test_typescript_accessibility_and_naming_rules_gate_private_extraction(
    tmp_path: Path,
) -> None:
    source = """
    class Widget {
        private handle = (v: number): number => v + 1;
        #secret = (v: number): number => v + 2;
        _conventional = (v: number): number => v + 3;
        private hidden(v: number): number { return v + 4; }
        protected guarded(v: number): number { return v + 5; }
        public shown(v: number): number { return v + 6; }
    }
    function _privateFn(v: number): number { return v; }
    """

    with_private = {unit.name for unit in _extract(tmp_path, "sample.ts", source)}
    public_only = {
        unit.name for unit in _extract(tmp_path, "sample.ts", source, include_private=False)
    }

    assert with_private == {
        "Widget",
        "handle",
        "#secret",
        "_conventional",
        "hidden",
        "guarded",
        "shown",
        "_privateFn",
    }
    assert public_only == {"Widget", "shown"}


def test_tsx_and_unicode_use_exact_byte_slices(tmp_path: Path) -> None:
    units = _extract(
        tmp_path,
        "component.tsx",
        """
        // café before the unit forces byte and character offsets to diverge.
        export const Card = (props: { title: string }) => (
            <section><h1>{props.title}</h1></section>
        );
        """,
    )

    card = next(unit for unit in units if unit.name == "Card")
    source_bytes = (tmp_path / "component.tsx").read_bytes()

    assert card.dialect == "tsx"
    assert source_bytes[card.start_byte : card.end_byte].decode("utf-8") == card.source
    assert "<section>" in card.source
