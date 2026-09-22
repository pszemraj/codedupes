"""Rust backend: item extraction, statement counts, visibility, attribute scoping, and token hashing."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from codedupes.extractor import CodeExtractor
from codedupes.models import CodeUnitType
from tests.polyglot_helpers import extract

pytestmark = pytest.mark.grammar


def test_rust_extracts_free_impl_trait_and_nested_functions(tmp_path: Path) -> None:
    units = extract(
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
    units = extract(
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
    units = extract(
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
    units = extract(
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
    units = extract(
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
    units = extract(
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
    units = extract(
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
    units = extract(
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
    units = extract(
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
    units = extract(
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
    [plain] = extract(
        tmp_path,
        "plain.rs",
        """
        fn double_plus(value: i32) -> i32 {
            let doubled = value * 2;
            doubled + 1
        }
        """,
    )
    [commented] = extract(
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


def test_rust_attribute_scoping_matches_across_stacked_and_nested_items(tmp_path: Path) -> None:
    """Pins every attribute shape ``_preceding_attributes`` has to keep straight:
    stacked attributes, comments between them, and attributes on enclosing scopes."""
    units = extract(
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
