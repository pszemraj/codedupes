"""End-to-end extraction tests against the exact pinned grammar wheels."""

from __future__ import annotations

import codecs
import time
from pathlib import Path
from textwrap import dedent

import pytest

from codedupes.extractor import CodeExtractor
from codedupes.languages.base import BackendResult
from codedupes.languages.registry import get_grammar_statuses
from codedupes.languages.tree_sitter_backend import _PYTHON_BUILTINS
from codedupes.models import CodeUnit, CodeUnitType

pytestmark = pytest.mark.grammar


def test_every_pinned_grammar_probes_ready_on_this_interpreter() -> None:
    """The live probe must construct a real parser for all six dialects."""
    statuses = get_grammar_statuses()

    assert len(statuses) == 6
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
        ".py": "python",
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
    units = _extract(tmp_path, filename, source)
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


@pytest.mark.parametrize("suffix", ["js", "jsx", "ts", "tsx"])
@pytest.mark.parametrize(
    ("initializer", "expected_count"),
    [
        ("", 2),
        ("Worker.ready = true;", 2),
        ("if (enabled) { Worker.first = 1; Worker.second = 2; }", 4),
        ("function setup() { first(); second(); third(); }", 2),
    ],
)
def test_class_member_count_includes_static_initializer_bodies(
    tmp_path: Path, suffix: str, initializer: str, expected_count: int
) -> None:
    """Static blocks recurse through control flow but stop at nested definitions."""
    units = _extract(
        tmp_path,
        f"sample.{suffix}",
        f"""
        class Worker {{
            static {{ {initializer} }}
            run(value) {{ return value + 1; }}
        }}
        """,
    )
    worker = next(unit for unit in units if unit.unit_type == CodeUnitType.CLASS)

    assert worker.statement_count == expected_count


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
    """A filtered class takes its members with it in every backend, TypeScript included."""
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


# ---------------------------------------------------------------------------
# Python backend.
# ---------------------------------------------------------------------------


def _python_result(
    tmp_path: Path,
    source: str,
    *,
    filename: str = "sample.py",
    include_private: bool = True,
) -> BackendResult:
    """Extract one Python file through the extractor, keeping its diagnostics."""
    path = tmp_path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(source).strip() + "\n", encoding="utf-8")
    extractor = CodeExtractor(
        tmp_path,
        include_private=include_private,
        include_stubs=True,
        languages=("python",),
    )
    units = tuple(extractor.extract_from_file(path))
    return BackendResult(units, tuple(extractor.diagnostics))


def _python_units(
    tmp_path: Path,
    source: str,
    *,
    filename: str = "sample.py",
    include_private: bool = True,
) -> dict[str, CodeUnit]:
    result = _python_result(tmp_path, source, filename=filename, include_private=include_private)
    return {unit.qualified_name: unit for unit in result.units}


def _python_unit(tmp_path: Path, filename: str, source: str) -> CodeUnit:
    """Return the outermost unit of a single-definition source."""
    return min(
        _python_result(tmp_path, source, filename=filename).units, key=lambda u: u.start_byte
    )


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
    units = _python_units(tmp_path, source)
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
    units = _python_units(
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
    units = _python_units(
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
    units = _python_units(
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
    units = _python_units(
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
    result = _python_result(tmp_path, source)
    outer = min(result.units, key=lambda unit: unit.start_byte)

    assert outer.statement_count == expected_count


def test_python_identifiers_include_api_names_and_exclude_builtins(tmp_path: Path) -> None:
    units = _python_units(
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
    everything = _python_units(tmp_path, source)
    public_only = _python_units(tmp_path, source, include_private=False)

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
    units = _python_units(
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
    units = _python_units(
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
    result = _python_result(tmp_path, "def a():\n\ndef b():\n\nclass C:\n\nx = 1\n")

    assert result.units == ()
    assert result.diagnostics == ()


def test_python_builtins_exclude_the_site_injected_names(tmp_path: Path) -> None:
    """``exit``/``quit``/``help`` come from ``site``, not the language, so they stay identifiers."""
    units = _python_units(
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
    units = _python_units(
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
    units = _python_units(tmp_path, "def func():\n    return 1", filename=filename)

    assert list(units) == [expected_qualified_name]


def test_python_error_recovery_reports_partial_parse_and_skips_the_broken_unit(
    tmp_path: Path,
) -> None:
    result = _python_result(
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


def test_python_bom_keeps_on_disk_byte_offsets(tmp_path: Path) -> None:
    """The lexer skips the BOM, so the first unit starts at byte 3 and the byte
    range still slices the file as stored."""
    path = tmp_path / "bom_sample.py"
    body = 'def greet(name):\n    message = "héllo " + name\n    return message\n'
    path.write_bytes(codecs.BOM_UTF8 + body.encode("utf-8"))

    extractor = CodeExtractor(tmp_path, include_private=True, languages=("python",))
    units = list(extractor.extract_from_file(path))
    raw = path.read_bytes()

    assert extractor.diagnostics == []
    [unit] = units
    assert unit.start_byte == len(codecs.BOM_UTF8)
    assert raw[unit.start_byte : unit.end_byte].decode("utf-8") == unit.source
    assert not unit.source.startswith("﻿")


def test_python_crlf_source_stays_byte_exact(tmp_path: Path) -> None:
    path = tmp_path / "crlf_sample.py"
    path.write_bytes(
        b"# leading comment\r\ndef greet(name):\r\n"
        b'    message = "hi " + name\r\n'
        b"    return message\r\n"
    )

    units = list(
        CodeExtractor(tmp_path, include_private=True, languages=("python",)).extract_from_file(path)
    )
    raw = path.read_bytes()

    [unit] = units
    assert "\r\n" in unit.source
    assert raw[unit.start_byte : unit.end_byte].decode("utf-8") == unit.source
    assert (unit.lineno, unit.end_lineno) == (2, 4)
