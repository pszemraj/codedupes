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
from codedupes.languages.registry import get_grammar_statuses
from codedupes.models import CodeUnit, CodeUnitType

pytestmark = pytest.mark.grammar


def test_every_pinned_grammar_probes_ready_on_this_interpreter() -> None:
    """The live probe must construct a real parser for all five dialects."""
    statuses = get_grammar_statuses()

    assert len(statuses) == 5
    assert all(status.available and status.error is None for status in statuses)


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


@pytest.mark.parametrize(
    ("filename", "source", "expected_hash"),
    [
        ("sample.c", "int add(int a, int b) { return a + b; }", "d4c1889345f6cbb2"),
        ("sample.rs", "pub fn add(a: i32, b: i32) -> i32 { a + b }", "ebe1b5fca595a210"),
        ("sample.js", "function add(a, b) { return a + b; }", "cb3daad8bab7b59a"),
        (
            "sample.ts",
            "function add(a: number, b: number): number { return a + b; }",
            "da777fa591d4b571",
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
    otherwise silently rename every non-Python fingerprint."""
    units = _extract(tmp_path, filename, source)

    assert [unit.structural_hash for unit in units] == [expected_hash]


def test_deeply_nested_source_does_not_hit_the_recursion_limit(tmp_path: Path) -> None:
    depth = 5000
    source = f"int deep(int value) {{ return {'(' * depth}value{')' * depth}; }}"

    units = _extract(tmp_path, "sample.c", source)

    assert [unit.name for unit in units] == ["deep"]
    assert units[0].structural_hash


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


def test_rust_statement_counts_include_tail_expressions_once(tmp_path: Path) -> None:
    units = _extract(
        tmp_path,
        "sample.rs",
        """
        fn tail_only() -> i32 { 1 }

        fn threshold_boundary() {
            first();
            second();
            third()
        }

        fn control_flow(ready: bool) {
            if ready { run(); }
        }
        """,
    )
    counts = {unit.name: unit.statement_count for unit in units}

    assert counts == {
        "tail_only": 1,
        "threshold_boundary": 3,
        "control_flow": 2,
    }


def test_rust_trailing_comment_is_not_a_tail_expression(tmp_path: Path) -> None:
    """A trailing comment must not inflate the count past the semantic gate."""
    units = _extract(
        tmp_path,
        "sample.rs",
        """
        fn commented() {
            first();
            second();
            // trailing note
        }

        fn block_commented() {
            first();
            second();
            /* trailing note */
        }
        """,
    )

    assert {unit.name: unit.statement_count for unit in units} == {
        "commented": 2,
        "block_commented": 2,
    }


def test_rust_skips_cfg_test_modules_and_test_functions(tmp_path: Path) -> None:
    units = _extract(
        tmp_path,
        "sample.rs",
        """
        pub fn real(value: i32) -> i32 { value + 1 }

        #[cfg(test)]
        mod tests {
            use super::*;

            #[test]
            fn checks_real() { assert_eq!(real(1), 2); }

            fn helper() -> i32 { 1 }
        }

        #[test]
        fn free_standing_check() { assert!(true); }

        #[cfg(all(test, feature = "slow"))]
        mod slow_tests {
            fn slow_helper() -> i32 { 2 }
        }

        #[cfg(all(feature = "slow", test))]
        mod reordered_tests {
            fn reordered_helper() -> i32 { 3 }
        }

        #[cfg(not(test))]
        fn production_only(value: i32) -> i32 { value }

        struct Marker;
        impl Marker {
            #[inline]
            pub fn tagged(&self) -> i32 { 3 }
        }
        """,
    )

    assert {unit.qualified_name for unit in units} == {
        "sample.real",
        "sample.production_only",
        "sample.Marker.tagged",
    }


def test_rust_trait_methods_inherit_trait_visibility(tmp_path: Path) -> None:
    units = _extract(
        tmp_path,
        "sample.rs",
        """
        trait Hidden {
            fn hidden(&self) -> i32 { 1 }
        }

        pub trait Shown {
            fn shown(&self) -> i32 { 2 }
        }
        """,
        include_private=False,
    )

    assert {unit.qualified_name for unit in units} == {"sample.Shown.shown"}


def test_rust_test_attribute_survives_an_intervening_comment(tmp_path: Path) -> None:
    """Attributes and their item are often separated by a documentation comment."""
    units = _extract(
        tmp_path,
        "sample.rs",
        """
        pub fn real(value: i32) -> i32 { value + 1 }

        #[cfg(test)]
        // Unit tests for the module above.
        mod tests {
            fn helper() -> i32 { 1 }
        }

        #[test]
        /* Checks the happy path. */
        fn free_standing_check() { assert!(true); }
        """,
    )

    assert {unit.qualified_name for unit in units} == {"sample.real"}


def test_rust_trait_impl_methods_survive_the_public_filter(tmp_path: Path) -> None:
    """``impl Trait for Type`` methods cannot carry ``pub``, yet they are the trait's API."""
    units = _extract(
        tmp_path,
        "sample.rs",
        """
        pub struct Widget;

        impl std::fmt::Display for Widget {
            fn fmt(&self, formatter: &mut Formatter) -> Result {
                formatter.write_str("widget")
            }
        }

        impl Widget {
            fn inherent_private(&self) -> i32 { 1 }
            pub fn inherent_public(&self) -> i32 { 2 }
        }
        """,
        include_private=False,
    )

    assert {unit.qualified_name for unit in units} == {
        "sample.Widget.fmt",
        "sample.Widget.inherent_public",
    }


def test_rust_token_hash_ignores_in_body_comments(tmp_path: Path) -> None:
    """tree-sitter-rust comments carry delimiter children; pruning must catch them."""
    [plain] = _extract(
        tmp_path,
        "plain.rs",
        """
        fn double_plus(value: i32) -> i32 {
            let doubled = value * 2;
            doubled + 1
        }
        """,
    )
    [commented] = _extract(
        tmp_path,
        "commented.rs",
        """
        fn double_plus(value: i32) -> i32 {
            let doubled = value * 2; // inline note
            /* block note */
            doubled + 1
        }
        """,
    )

    assert plain.token_hash == commented.token_hash
    assert plain.structural_hash == commented.structural_hash


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


def test_javascript_unicode_identifiers_are_extracted(tmp_path: Path) -> None:
    """Unicode identifiers are legal ES2015+, so ASCII-only naming would drop units."""
    units = _extract(
        tmp_path,
        "sample.js",
        """
        export function café(value) { return value + 1; }
        const naïve = (value) => café(value) + 2;

        class Größe {
            länge(value) { return value + 3; }
        }
        """,
    )
    names = {unit.qualified_name for unit in units}

    assert names == {
        "sample.café",
        "sample.naïve",
        "sample.Größe",
        "sample.Größe.länge",
    }
    assert "café" in next(unit for unit in units if unit.name == "naïve").identifiers


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
