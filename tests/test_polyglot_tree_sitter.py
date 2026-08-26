"""End-to-end extraction tests against the exact pinned grammar wheels."""

from __future__ import annotations

import time
from pathlib import Path
from textwrap import dedent

import pytest

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
    otherwise silently rename every non-Python fingerprint."""
    units = _extract(tmp_path, filename, source)

    assert [unit.structural_hash for unit in units] == [expected_hash]


def test_renamed_declarations_hash_structurally_equal(tmp_path: Path) -> None:
    """A declaration's own name normalizes like Python def/class names do."""
    units = _extract(
        tmp_path,
        "store.ts",
        """
        class Store {
          load(key: string): string {
            const raw = this.backend.get(key);
            return JSON.parse(raw);
          }
          fetch(key: string): string {
            const raw = this.backend.get(key);
            return JSON.parse(raw);
          }
        }
        class Alpha {
          run(x: number): number { return x + 1; }
        }
        class Beta {
          run(x: number): number { return x + 1; }
        }
        class Gamma {
          go(x: number): number { return x * 2; }
        }
        class Delta {
          walk(x: number): number { return x * 2; }
        }
        """,
    )
    by_name = {unit.qualified_name: unit for unit in units}

    assert (
        by_name["store.Store.load"].structural_hash == by_name["store.Store.fetch"].structural_hash
    )
    assert by_name["store.Store.load"].token_hash != by_name["store.Store.fetch"].token_hash
    assert by_name["store.Alpha"].structural_hash == by_name["store.Beta"].structural_hash
    assert by_name["store.Gamma"].structural_hash == by_name["store.Delta"].structural_hash


def test_object_literal_method_units_normalize_their_own_name(tmp_path: Path) -> None:
    units = _extract(
        tmp_path,
        "registry.js",
        """
        const handlers = {
          alpha() { const value = this.compute(); return value; },
          beta() { const value = this.compute(); return value; },
        };
        """,
    )
    by_name = {unit.name: unit for unit in units if unit.unit_type == CodeUnitType.METHOD}

    assert by_name["alpha"].structural_hash == by_name["beta"].structural_hash
    assert by_name["alpha"].token_hash != by_name["beta"].token_hash


def test_object_literal_member_names_stay_structural_shape_inside_units(tmp_path: Path) -> None:
    """Object keys are data shape, like Python dict keys: renaming one changes
    the containing unit's structure even when the member is a method."""
    units = _extract(
        tmp_path,
        "shape.js",
        """
        function first() { return { alpha: 1, beta: 2 }; }
        function second() { return { alpha: 1, gamma: 2 }; }
        function third() { return { alpha: 1, beta: 2 }; }
        function make() { return { run() { return 1; } }; }
        function build() { return { exec() { return 1; } }; }
        """,
    )
    by_name = {unit.name: unit for unit in units if unit.unit_type == CodeUnitType.FUNCTION}

    assert by_name["first"].structural_hash != by_name["second"].structural_hash
    assert by_name["first"].structural_hash == by_name["third"].structural_hash
    assert by_name["make"].structural_hash != by_name["build"].structural_hash


def test_deeply_nested_source_does_not_hit_the_recursion_limit(tmp_path: Path) -> None:
    depth = 5000
    source = f"int deep(int value) {{ return {'(' * depth}value{')' * depth}; }}"

    units = _extract(tmp_path, "sample.c", source)

    assert [unit.name for unit in units] == ["deep"]
    assert units[0].structural_hash


def test_deeply_nested_c_declarator_does_not_hit_the_recursion_limit(tmp_path: Path) -> None:
    depth = 2000
    source = f"int {'(' * depth}deep{')' * depth}(int value) {{ return value; }}"

    units = _extract(tmp_path, "sample.c", source)

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

    units = _extract(tmp_path, "sample.js", source)

    assert any(unit.qualified_name.endswith(".leaf") for unit in units)


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


def test_c_static_detection_ignores_array_parameters_and_comments(tmp_path: Path) -> None:
    """C99 ``[static n]`` parameters and prose both contain the word ``static``."""
    units = _extract(
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
        "sample.Widget.std::fmt::Display.fmt",
        "sample.Widget.inherent_public",
    }


def test_rust_local_trait_visibility_gates_its_impl_methods(tmp_path: Path) -> None:
    """``impl LocalTrait for Type`` methods are only as visible as the trait.

    Path-qualified traits stay public: cross-file resolution is out of scope,
    so unresolved traits err on the recall-first side.
    """
    units = _extract(
        tmp_path,
        "sample.rs",
        """
        pub struct Widget;

        trait Sealed {
            fn seal(&self) -> i32;
        }

        trait Convert<T> {
            fn convert(&self) -> T;
        }

        pub trait Open {
            fn open(&self) -> i32;
        }

        impl Sealed for Widget {
            fn seal(&self) -> i32 { 1 }
        }

        impl Convert<u32> for Widget {
            fn convert(&self) -> u32 { 3 }
        }

        impl Open for Widget {
            fn open(&self) -> i32 { 2 }
        }

        impl std::fmt::Display for Widget {
            fn fmt(&self, formatter: &mut Formatter) -> Result {
                formatter.write_str("widget")
            }
        }
        """,
        include_private=False,
    )

    assert {unit.qualified_name for unit in units} == {
        "sample.Widget.Open.open",
        "sample.Widget.std::fmt::Display.fmt",
    }


def test_rust_trait_impls_of_one_method_name_get_distinct_qualified_names(
    tmp_path: Path,
) -> None:
    """Two traits routinely require the same method name on one type."""
    units = _extract(
        tmp_path,
        "sample.rs",
        """
        pub struct Widget;

        impl Display for Widget {
            fn fmt(&self, formatter: &mut Formatter) -> Result {
                formatter.write_str("shown")
            }
        }

        impl Debug for Widget {
            fn fmt(&self, formatter: &mut Formatter) -> Result {
                formatter.write_str("debug")
            }
        }

        impl Widget {
            pub fn render(&self) -> i32 { 1 }
        }
        """,
    )
    names = [unit.qualified_name for unit in units]

    assert sorted(names) == [
        "sample.Widget.Debug.fmt",
        "sample.Widget.Display.fmt",
        "sample.Widget.render",
    ]
    assert len(set(names)) == len(names)
    assert {unit.name for unit in units} == {"fmt", "render"}


def test_rust_generic_and_nested_module_trait_impls_keep_clean_segments(
    tmp_path: Path,
) -> None:
    """Generic trait arguments and module nesting both belong in the impl path."""
    units = _extract(
        tmp_path,
        "sample.rs",
        """
        pub struct Widget;

        impl From<u32> for Widget {
            fn from(value: u32) -> Self { Widget }
        }

        mod inner {
            pub struct Gadget;

            impl Display for Gadget {
                fn fmt(&self, formatter: &mut Formatter) -> Result {
                    formatter.write_str("gadget")
                }
            }
        }
        """,
    )

    assert {unit.qualified_name for unit in units} == {
        "sample.Widget.From<u32>.from",
        "sample.inner.Gadget.Display.fmt",
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


def test_named_class_expressions_use_their_external_bindings(tmp_path: Path) -> None:
    units = _extract(
        tmp_path,
        "sample.js",
        """
        const Public = class Internal { run() { return 1; } };
        const Other = class Internal { run() { return 2; } };
        export { Public };
        """,
    )
    exported = {unit.qualified_name: unit.is_exported for unit in units}

    assert exported == {
        "sample.Public": True,
        "sample.Public.run": True,
        "sample.Other": False,
        "sample.Other.run": False,
    }


def test_javascript_object_literal_class_values_keep_their_class_segment(
    tmp_path: Path,
) -> None:
    """Registry literals of anonymous classes routinely repeat one method name."""
    units = _extract(
        tmp_path,
        "sample.js",
        """
        const registry = {
            Alpha: class { run(value) { return value; } },
            Beta: class { run(value) { return value + 1; } },
        };
        """,
    )
    names = [unit.qualified_name for unit in units]

    assert sorted(names) == [
        "sample.registry.Alpha",
        "sample.registry.Alpha.run",
        "sample.registry.Beta",
        "sample.registry.Beta.run",
    ]
    assert len(set(names)) == len(names)
    assert {unit.name for unit in units if unit.unit_type == CodeUnitType.METHOD} == {"run"}


def test_javascript_object_literal_and_lexical_scopes_nest_in_order(tmp_path: Path) -> None:
    """Object-literal and lexical containers must resolve under one rule."""
    units = _extract(
        tmp_path,
        "sample.js",
        """
        const api = {
            build() {
                class Inner { run(value) { return value; } }
                return Inner;
            },
        };
        """,
    )

    assert {unit.qualified_name: unit.unit_type for unit in units} == {
        "sample.api.build": CodeUnitType.METHOD,
        "sample.api.build.Inner": CodeUnitType.CLASS,
        "sample.api.build.Inner.run": CodeUnitType.METHOD,
    }


def test_javascript_export_clause_marks_nested_object_literal_methods(tmp_path: Path) -> None:
    """Deferred export lists name the base binding, not the dotted container path."""
    units = _extract(
        tmp_path,
        "sample.js",
        """
        const registry = {
            Alpha: class { run(value) { return value; } },
        };
        export const api = { list() { return 1; } };
        const hidden = { Gamma: class { run(value) { return value; } } };

        export { registry };
        """,
    )
    exported = {unit.qualified_name: unit.is_exported for unit in units}

    assert exported["sample.registry.Alpha"] is True
    assert exported["sample.registry.Alpha.run"] is True
    assert exported["sample.api.list"] is True
    assert exported["sample.hidden.Gamma"] is False
    assert exported["sample.hidden.Gamma.run"] is False


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


def test_javascript_export_clauses_mark_referenced_top_level_units(tmp_path: Path) -> None:
    """Deferred export lists are the idiomatic barrel-file shape and must count."""
    units = _extract(
        tmp_path,
        "sample.js",
        """
        function alpha(value) { return value + 1; }
        const beta = (value) => value + 2;
        class Gamma { run(value) { return value + 3; } }
        function delta(value) { return value + 4; }
        function omega(value) { return value + 5; }

        export { alpha, beta as renamed, Gamma };
        export default delta;
        """,
    )
    exported = {unit.qualified_name: unit.is_exported for unit in units}

    assert exported["sample.alpha"] is True
    assert exported["sample.beta"] is True
    assert exported["sample.Gamma"] is True
    assert exported["sample.Gamma.run"] is True
    assert exported["sample.delta"] is True
    assert exported["sample.omega"] is False


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


def test_javascript_class_member_count_includes_static_initializer_blocks(
    tmp_path: Path,
) -> None:
    """The tree-sitter-javascript node is ``class_static_block``, not ``static_block``."""
    units = _extract(
        tmp_path,
        "sample.js",
        """
        class Worker {
            static { Worker.ready = true; }
            run(value) { return value + 1; }
        }
        """,
    )
    worker = next(unit for unit in units if unit.unit_type == CodeUnitType.CLASS)

    assert worker.statement_count == 2


def test_typescript_nested_abstract_class_counts_as_one_nested_scope(tmp_path: Path) -> None:
    """A nested abstract class must not leak its members into the outer count."""
    units = _extract(
        tmp_path,
        "sample.ts",
        """
        function factory(): unknown {
            abstract class Base {
                run(): number { return 1; }
                other(): number { return 2; }
            }
            return Base;
        }
        """,
    )
    counts = {unit.qualified_name: unit.statement_count for unit in units}

    assert counts["sample.factory"] == 2


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


def test_private_container_members_are_dropped_with_their_container(tmp_path: Path) -> None:
    """The Python extractor skips descendants of a filtered class; so must this one."""
    units = _extract(
        tmp_path,
        "sample.ts",
        """
        class _Internal {
            run(value: number): number { return value + 1; }
            nested = (value: number): number => value + 2;
        }

        export class Public {
            run(value: number): number { return value + 3; }
        }
        """,
        include_private=False,
    )

    assert {unit.qualified_name for unit in units} == {"sample.Public", "sample.Public.run"}


def test_private_named_class_expression_drops_with_its_field_binding(tmp_path: Path) -> None:
    units = _extract(
        tmp_path,
        "sample.ts",
        """
        class Holder {
            private hidden = class Visible { run(): number { return 1; } };
            shown = class Internal { run(): number { return 2; } };
        }
        """,
        include_private=False,
    )

    assert {unit.qualified_name for unit in units} == {
        "sample.Holder",
        "sample.Holder.shown",
        "sample.Holder.shown.run",
    }


def test_jsx_display_copy_is_normalized_in_structural_fingerprints(tmp_path: Path) -> None:
    """JSX text is display copy, exactly like the string literals already normalized."""
    [first] = _extract(tmp_path, "first.tsx", "export const Card = () => <h1>Hello</h1>;\n")
    [second] = _extract(
        tmp_path,
        "second.tsx",
        "export const Card = () => <h1>Goodbye, friend</h1>;\n",
    )
    [structural] = _extract(
        tmp_path,
        "third.tsx",
        "export const Card = () => <h1>Hello<br /></h1>;\n",
    )

    assert first.structural_hash == second.structural_hash
    assert first.structural_hash != structural.structural_hash


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


def test_non_utf8_source_is_analyzed_but_reported(tmp_path: Path) -> None:
    """Recall-first decoding keeps the unit, but replacement characters reach the
    fingerprints and embeddings, so the corruption must not stay silent."""
    path = tmp_path / "legacy.js"
    path.write_bytes("function greet() { return 'café'; }\n".encode("latin-1"))

    extractor = CodeExtractor(tmp_path, include_private=True, languages=("javascript",))
    units = list(extractor.extract_from_file(path))

    assert [unit.qualified_name for unit in units] == ["legacy.greet"]
    assert [diagnostic.code for diagnostic in extractor.diagnostics] == ["invalid-utf8"]
    assert "�" in units[0].source


def test_rust_attribute_scoping_matches_across_stacked_and_nested_items(tmp_path: Path) -> None:
    """Pins every attribute shape ``_preceding_attributes`` has to keep straight:
    stacked attributes, comments between them, and attributes on enclosing scopes."""
    units = _extract(
        tmp_path,
        "mixed.rs",
        """
        #[derive(Debug)]
        pub struct Widget;

        #[test]
        fn bare_test() { assert!(true); }

        #[cfg(test)]
        fn cfg_test_fn() { let x = 1; }

        #[cfg(all(test, feature = "x"))]
        fn cfg_all_test_fn() { let x = 1; }

        #[cfg(not(test))]
        pub fn not_test_fn() -> i32 { 7 }

        #[cfg(any(test, feature = "x"))]
        pub fn any_test_fn() -> i32 { 8 }

        // a comment above the attribute stack
        #[inline]
        // another comment below it
        pub fn commented_fn() -> i32 { 9 }

        #[cfg(test)]
        mod tests {
            #[test]
            fn nested_test() { assert!(true); }

            fn helper() -> i32 { 3 }
        }

        pub mod real {
            #[inline]
            pub fn inner(x: i32) -> i32 { x }

            #[cfg(test)]
            mod inner_tests {
                fn deep_helper() -> i32 { 1 }
            }
        }

        impl Widget {
            #[cfg(test)]
            fn test_only_method(&self) -> i32 { 1 }

            #[inline]
            pub fn real_method(&self) -> i32 { 2 }
        }

        fn outer_plain() -> i32 {
            #[cfg(test)]
            fn nested_in_fn() -> i32 { 4 }
            5
        }
        """,
    )

    assert {(unit.qualified_name, unit.is_public) for unit in units} == {
        ("mixed.any_test_fn", True),
        ("mixed.commented_fn", True),
        ("mixed.not_test_fn", True),
        ("mixed.outer_plain", False),
        ("mixed.real.inner", True),
        ("mixed.Widget.real_method", True),
    }


def test_rust_attribute_lookup_stays_linear_in_item_count(tmp_path: Path) -> None:
    """Locating each item among its siblings by scan made extraction quadratic; the
    bound is deliberately generous so a slow machine still passes."""
    items = []
    for index in range(3000):
        if index % 3 == 0:
            items.append(f"#[inline]\npub fn item_{index}(x: i32) -> i32 {{ x + {index} }}")
        elif index % 3 == 1:
            items.append(f"#[test]\nfn test_{index}() {{ assert_eq!(1, 1); }}")
        else:
            items.append(f"fn plain_{index}(x: i32) -> i32 {{\n    let y = x * {index};\n    y\n}}")
    path = tmp_path / "wide.rs"
    path.write_text("\n".join(items) + "\n", encoding="utf-8")

    extractor = CodeExtractor(tmp_path, include_private=True, languages=("rust",))
    started = time.perf_counter()
    units = list(extractor.extract_from_file(path))
    elapsed = time.perf_counter() - started

    assert len(units) == 2000
    assert elapsed < 5.0, f"3000-item Rust file took {elapsed:.1f}s"
