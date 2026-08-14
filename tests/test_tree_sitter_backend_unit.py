"""Parser-independent tests for Tree-sitter backend normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from codedupes.languages.tree_sitter_backend import (
    JavaScriptBackend,
    RustBackend,
    TypeScriptBackend,
    _same_node,
    _structural_hash,
    _token_hash,
)
from codedupes.models import CodeUnitType


@dataclass
class FakeNode:
    """Small Tree-sitter node double with byte-addressed children and fields."""

    type: str
    start_byte: int
    end_byte: int
    is_named: bool = True
    children: tuple[FakeNode, ...] = ()
    named_children: tuple[FakeNode, ...] = ()
    fields: dict[str, FakeNode] = field(default_factory=dict)
    parent: FakeNode | None = None
    has_error: bool = False
    is_error: bool = False
    is_missing: bool = False
    start_point: tuple[int, int] = (0, 0)
    end_point: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if not self.named_children:
            self.named_children = tuple(child for child in self.children if child.is_named)
        for child in self.children:
            child.parent = self

    def child_by_field_name(self, name: str) -> FakeNode | None:
        value = self.fields.get(name)
        if value is None:
            return None
        # Real bindings may hand back a fresh Python wrapper. Return a clone so
        # tests reject object-identity comparisons.
        return FakeNode(
            type=value.type,
            start_byte=value.start_byte,
            end_byte=value.end_byte,
            is_named=value.is_named,
            children=value.children,
            named_children=value.named_children,
            fields=value.fields,
            parent=value.parent,
        )


def _leaf(
    source: bytes,
    text: bytes,
    node_type: str,
    *,
    start: int = 0,
    named: bool = True,
) -> FakeNode:
    offset = source.index(text, start)
    return FakeNode(node_type, offset, offset + len(text), is_named=named)


def _flat_function(source: bytes, *, operator: bytes = b"+") -> FakeNode:
    tokens = [
        _leaf(source, b"fn", "fn", named=False),
        _leaf(source, source.split()[1], "identifier", start=2),
        _leaf(source, b"a" if b" a" in source else b"x", "identifier"),
        _leaf(source, b"b" if b" b" in source else b"y", "identifier"),
        _leaf(source, b"return", "return", named=False),
        _leaf(
            source,
            b"a" if b" a" in source else b"x",
            "identifier",
            start=source.index(b"return"),
        ),
        _leaf(source, operator, operator.decode(), named=False),
        _leaf(
            source,
            b"b" if b" b" in source else b"y",
            "identifier",
            start=source.index(operator),
        ),
    ]
    return FakeNode("function_item", 0, len(source), children=tuple(tokens))


def test_same_node_uses_source_identity_not_wrapper_identity() -> None:
    left = FakeNode("identifier", 10, 14)
    right = FakeNode("identifier", 10, 14)
    other = FakeNode("identifier", 11, 14)

    assert left is not right
    assert _same_node(left, right)
    assert not _same_node(left, other)


def test_structural_hash_normalizes_local_names_but_preserves_operators() -> None:
    first_source = b"fn add(a, b) { return a + b; }"
    second_source = b"fn total(x, y) { return x + y; }"
    changed_source = b"fn total(x, y) { return x - y; }"

    first = _structural_hash(
        _flat_function(first_source, operator=b"+"),
        first_source,
        "rust",
        CodeUnitType.FUNCTION,
    )
    renamed = _structural_hash(
        _flat_function(second_source, operator=b"+"),
        second_source,
        "rust",
        CodeUnitType.FUNCTION,
    )
    changed = _structural_hash(
        _flat_function(changed_source, operator=b"-"),
        changed_source,
        "rust",
        CodeUnitType.FUNCTION,
    )

    assert first == renamed
    assert renamed != changed


def test_token_hash_ignores_comments_but_retains_literal_text() -> None:
    source_a = b"name /* first */ 1"
    source_b = b"name /* second */ 1"
    source_c = b"name /* second */ 2"

    def tree(source: bytes) -> FakeNode:
        name = _leaf(source, b"name", "identifier")
        comment_start = source.index(b"/*")
        comment_end = source.index(b"*/") + 2
        comment = FakeNode("comment", comment_start, comment_end)
        number = _leaf(source, source[-1:], "number")
        return FakeNode("expression", 0, len(source), children=(name, comment, number))

    assert _token_hash(tree(source_a), source_a) == _token_hash(tree(source_b), source_b)
    assert _token_hash(tree(source_b), source_b) != _token_hash(tree(source_c), source_c)


def test_javascript_binding_accepts_fresh_field_wrappers(tmp_path: Path) -> None:
    source = b"const run = () => 1"
    name = _leaf(source, b"run", "identifier")
    arrow = FakeNode("arrow_function", source.index(b"()"), len(source))
    declaration = FakeNode(
        "variable_declarator",
        source.index(b"run"),
        len(source),
        children=(name, arrow),
        fields={"name": name, "value": arrow},
    )
    arrow.parent = declaration

    backend = JavaScriptBackend(tmp_path, "javascript", include_private=True)
    binding = backend._binding_for_value(arrow, source)

    assert binding is not None
    assert binding[0] == "run"
    assert binding[1] is declaration


def test_javascript_anonymous_default_export_gets_stable_name(tmp_path: Path) -> None:
    source = b"export default function () { return 7; }"
    function_start = source.index(b"function")
    body_start = source.index(b"{")
    body = FakeNode("statement_block", body_start, len(source))
    function = FakeNode(
        "function_declaration",
        function_start,
        len(source),
        children=(body,),
        fields={"body": body},
    )
    FakeNode("export_statement", 0, len(source), children=(function,))

    backend = JavaScriptBackend(tmp_path, "javascript", include_private=True)
    spec = backend._function_spec(function, source, "sample")

    assert spec is not None
    assert spec.name == "default"
    assert spec.qualified_name == "sample.default"
    assert spec.is_exported


def test_javascript_local_arrow_inside_method_remains_function(tmp_path: Path) -> None:
    source = b"class Worker { run() { const local = () => 1; } }"
    class_name = _leaf(source, b"Worker", "identifier")
    method_name = _leaf(source, b"run", "property_identifier")
    local_name = _leaf(source, b"local", "identifier")
    arrow_body = _leaf(source, b"1", "number")
    arrow = FakeNode(
        "arrow_function",
        source.index(b"()"),
        arrow_body.end_byte,
        children=(arrow_body,),
        fields={"body": arrow_body},
    )
    declaration = FakeNode(
        "variable_declarator",
        local_name.start_byte,
        arrow.end_byte,
        children=(local_name, arrow),
        fields={"name": local_name, "value": arrow},
    )
    method_body = FakeNode(
        "statement_block",
        source.index(b"{", source.index(b"run")),
        source.rindex(b"}"),
        children=(declaration,),
    )
    method = FakeNode(
        "method_definition",
        method_name.start_byte,
        method_body.end_byte,
        children=(method_name, method_body),
        fields={"name": method_name, "body": method_body},
    )
    class_body = FakeNode(
        "class_body",
        source.index(b"{"),
        len(source),
        children=(method,),
    )
    FakeNode(
        "class_declaration",
        0,
        len(source),
        children=(class_name, class_body),
        fields={"name": class_name, "body": class_body},
    )

    backend = JavaScriptBackend(tmp_path, "javascript", include_private=True)
    spec = backend._function_spec(arrow, source, "sample")

    assert spec is not None
    assert spec.qualified_name == "sample.Worker.run.local"
    assert spec.unit_type == CodeUnitType.FUNCTION


def test_javascript_class_field_arrow_is_method(tmp_path: Path) -> None:
    source = b"class Worker { handle = () => 1; }"
    class_name = _leaf(source, b"Worker", "identifier")
    field_name = _leaf(source, b"handle", "property_identifier")
    arrow_body = _leaf(source, b"1", "number")
    arrow = FakeNode(
        "arrow_function",
        source.index(b"()"),
        arrow_body.end_byte,
        children=(arrow_body,),
        fields={"body": arrow_body},
    )
    field_node = FakeNode(
        "field_definition",
        field_name.start_byte,
        arrow.end_byte,
        children=(field_name, arrow),
        fields={"name": field_name, "value": arrow},
    )
    class_body = FakeNode(
        "class_body",
        source.index(b"{"),
        len(source),
        children=(field_node,),
    )
    FakeNode(
        "class_declaration",
        0,
        len(source),
        children=(class_name, class_body),
        fields={"name": class_name, "body": class_body},
    )

    backend = JavaScriptBackend(tmp_path, "javascript", include_private=True)
    spec = backend._function_spec(arrow, source, "sample")

    assert spec is not None
    assert spec.qualified_name == "sample.Worker.handle"
    assert spec.unit_type == CodeUnitType.METHOD


def test_rust_nested_function_inside_impl_is_not_a_method(tmp_path: Path) -> None:
    source = b"impl Widget { fn outer() { fn inner() {} } }"
    target = _leaf(source, b"Widget", "type_identifier")
    outer_name = _leaf(source, b"outer", "identifier")
    inner_name = _leaf(source, b"inner", "identifier")
    inner_body = FakeNode(
        "block",
        source.index(b"{", source.index(b"inner")),
        source.index(b"}", source.index(b"inner")) + 1,
    )
    inner = FakeNode(
        "function_item",
        source.index(b"fn inner"),
        inner_body.end_byte,
        children=(inner_name, inner_body),
        fields={"name": inner_name, "body": inner_body},
    )
    outer_body = FakeNode(
        "block",
        source.index(b"{", source.index(b"outer")),
        source.rindex(b"}"),
        children=(inner,),
    )
    outer = FakeNode(
        "function_item",
        source.index(b"fn outer"),
        outer_body.end_byte,
        children=(outer_name, outer_body),
        fields={"name": outer_name, "body": outer_body},
    )
    impl_body = FakeNode(
        "declaration_list",
        source.index(b"{"),
        len(source),
        children=(outer,),
    )
    FakeNode(
        "impl_item",
        0,
        len(source),
        children=(target, impl_body),
        fields={"type": target, "body": impl_body},
    )

    contexts, inner_is_method = RustBackend._context(inner, source)
    outer_contexts, outer_is_method = RustBackend._context(outer, source)

    assert contexts == ["Widget", "outer"]
    assert not inner_is_method
    assert outer_contexts == ["Widget"]
    assert outer_is_method


def test_typescript_private_method_is_not_public(tmp_path: Path) -> None:
    source = b"class Worker { private run() {} }"
    class_name = _leaf(source, b"Worker", "type_identifier")
    access = _leaf(source, b"private", "accessibility_modifier")
    method_name = _leaf(source, b"run", "property_identifier")
    method_body = FakeNode(
        "statement_block",
        source.index(b"{", source.index(b"run")),
        source.index(b"}", source.index(b"run")) + 1,
    )
    method = FakeNode(
        "method_definition",
        access.start_byte,
        method_body.end_byte,
        children=(access, method_name, method_body),
        fields={"name": method_name, "body": method_body},
    )
    class_body = FakeNode(
        "class_body",
        source.index(b"{"),
        len(source),
        children=(method,),
    )
    FakeNode(
        "class_declaration",
        0,
        len(source),
        children=(class_name, class_body),
        fields={"name": class_name, "body": class_body},
    )

    backend = TypeScriptBackend(tmp_path, "typescript", include_private=True)
    spec = backend._method_spec(method, source, "sample")

    assert spec is not None
    assert spec.qualified_name == "sample.Worker.run"
    assert not spec.is_public
