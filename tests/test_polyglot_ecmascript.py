"""JavaScript, TypeScript, JSX, and TSX backend: unit extraction, scoping, exports, and privacy rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedupes.models import CodeUnitType
from tests.polyglot_helpers import extract

pytestmark = pytest.mark.grammar


def test_renamed_declarations_hash_structurally_equal(tmp_path: Path) -> None:
    """A declaration's own name normalizes like Python def/class names do."""
    units = extract(
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
    units = extract(
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
    units = extract(
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


def test_javascript_extracts_modern_stable_unit_forms(tmp_path: Path) -> None:
    units = extract(
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
    units = extract(
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
    units = extract(
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
    units = extract(
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
    units = extract(
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
    units = extract(
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
    units = extract(
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
    units = extract(
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
    units = extract(
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
    units = extract(
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
    units = extract(
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

    with_private = {unit.name for unit in extract(tmp_path, "sample.ts", source)}
    public_only = {
        unit.name for unit in extract(tmp_path, "sample.ts", source, include_private=False)
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
    units = extract(
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
    units = extract(
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
    [first] = extract(tmp_path, "first.tsx", "export const Card = () => <h1>Hello</h1>;\n")
    [second] = extract(
        tmp_path,
        "second.tsx",
        "export const Card = () => <h1>Goodbye, friend</h1>;\n",
    )
    [structural] = extract(
        tmp_path,
        "third.tsx",
        "export const Card = () => <h1>Hello<br /></h1>;\n",
    )

    assert first.structural_hash == second.structural_hash
    assert first.structural_hash != structural.structural_hash


def test_tsx_and_unicode_use_exact_byte_slices(tmp_path: Path) -> None:
    units = extract(
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


def test_javascript_suppression_directive_attachment(tmp_path: Path) -> None:
    """``codedupes: ignore`` attaches through JS's comment forms and export wrapping."""
    [leading_line] = extract(
        tmp_path,
        "leading_line.js",
        """
        // codedupes: ignore
        function foo() {
          return 1;
        }
        """,
    )
    assert leading_line.suppressions == {"unused", "duplicates"}

    [leading_block] = extract(
        tmp_path,
        "leading_block.js",
        """
        /* codedupes: ignore */
        function foo() {
          return 1;
        }
        """,
    )
    assert leading_block.suppressions == {"unused", "duplicates"}

    [trailing_brace] = extract(
        tmp_path,
        "trailing_brace.js",
        """
        function foo() {  // codedupes: ignore
          return 1;
        }
        """,
    )
    assert trailing_brace.suppressions == {"unused", "duplicates"}

    [body_statement_comment] = extract(
        tmp_path,
        "body_statement_comment.js",
        "function foo() { const value = 1; // codedupes: ignore[duplicates]\n return value; }",
    )
    assert body_statement_comment.suppressions == set()

    [export_arrow] = extract(
        tmp_path,
        "export_arrow.js",
        """
        // codedupes: ignore
        export const foo = () => {
          return 1;
        };
        """,
    )
    assert export_arrow.suppressions == {"unused", "duplicates"}
