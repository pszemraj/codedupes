"""Tree-sitter extraction backends for C, Rust, JavaScript, and TypeScript.

The parser packages are imported only when one of these languages is actually
encountered.  No grammar is downloaded or compiled at analysis time: the
project pins the official precompiled Python grammar wheels in ``pyproject``.
"""

from __future__ import annotations

import hashlib
import importlib
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from codedupes.languages.base import BackendResult
from codedupes.languages.registry import GRAMMAR_PACKAGES
from codedupes.models import CodeUnit, CodeUnitType, ExtractionDiagnostic

FINGERPRINT_SCHEMA_VERSION = 1
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_STABLE_PATH_RE = re.compile(
    r"^[A-Za-z_$#][A-Za-z0-9_$#]*(?:\.[A-Za-z_$#][A-Za-z0-9_$#]*)*$"
)

_COMMENT_TYPES = {
    "comment",
    "line_comment",
    "block_comment",
    "doc_comment",
    "inner_doc_comment_marker",
    "outer_doc_comment_marker",
}
_IDENTIFIER_TYPES = {
    "identifier",
    "field_identifier",
    "property_identifier",
    "private_property_identifier",
    "shorthand_property_identifier",
    "shorthand_property_identifier_pattern",
    "type_identifier",
    "scoped_identifier",
    "namespace_identifier",
}
_PRESERVED_IDENTIFIER_TYPES = {
    "field_identifier",
    "property_identifier",
    "private_property_identifier",
    "type_identifier",
    "namespace_identifier",
}
_STRING_MARKERS = ("string", "char_literal", "template_string", "raw_string")
_NUMBER_MARKERS = ("number", "integer", "float", "decimal", "hex", "octal", "binary")


class GrammarUnavailableError(RuntimeError):
    """Raised when a requested precompiled parser package cannot be loaded."""


class GrammarProvider:
    """Load and cache immutable ``tree_sitter.Language`` objects."""

    _languages: ClassVar[dict[str, Any]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def language(cls, dialect: str) -> Any:
        """Return a cached Tree-sitter language for one parser dialect."""
        grammar_key = "javascript" if dialect == "jsx" else dialect
        with cls._lock:
            if grammar_key in cls._languages:
                return cls._languages[grammar_key]

            try:
                tree_sitter = importlib.import_module("tree_sitter")
            except ModuleNotFoundError as exc:
                raise GrammarUnavailableError(
                    "Tree-sitter support is not installed. Reinstall codedupes so the pinned "
                    "tree-sitter and tree-sitter-* wheels are present."
                ) from exc

            module_name: str
            function_name: str
            if grammar_key == "c":
                module_name, function_name = "tree_sitter_c", "language"
            elif grammar_key == "rust":
                module_name, function_name = "tree_sitter_rust", "language"
            elif grammar_key == "javascript":
                module_name, function_name = "tree_sitter_javascript", "language"
            elif grammar_key == "typescript":
                module_name, function_name = "tree_sitter_typescript", "language_typescript"
            elif grammar_key == "tsx":
                module_name, function_name = "tree_sitter_typescript", "language_tsx"
            else:
                raise GrammarUnavailableError(f"No grammar provider is registered for {dialect!r}")

            try:
                grammar_module = importlib.import_module(module_name)
                capsule_factory = getattr(grammar_module, function_name)
                language = tree_sitter.Language(capsule_factory())
            except (ModuleNotFoundError, AttributeError, TypeError, ValueError) as exc:
                package, pinned = GRAMMAR_PACKAGES[grammar_key]
                raise GrammarUnavailableError(
                    f"Could not load the {grammar_key} grammar from {package}=={pinned}: {exc}"
                ) from exc

            cls._languages[grammar_key] = language
            return language

    @classmethod
    def parser(cls, dialect: str) -> Any:
        """Create a fresh parser for one parse operation."""
        try:
            tree_sitter = importlib.import_module("tree_sitter")
            return tree_sitter.Parser(cls.language(dialect))
        except GrammarUnavailableError:
            raise
        except (ModuleNotFoundError, TypeError, ValueError) as exc:
            raise GrammarUnavailableError(
                f"Could not create a parser for {dialect}: {exc}"
            ) from exc


@dataclass(frozen=True)
class UnitSpec:
    """Language-specific description of one executable code unit."""

    node: Any
    source_node: Any
    body: Any
    name: str
    qualified_name: str
    unit_type: CodeUnitType
    native_kind: str
    is_public: bool
    is_exported: bool


def _children(node: Any) -> tuple[Any, ...]:
    return tuple(getattr(node, "children", ()) or ())


def _named_children(node: Any) -> tuple[Any, ...]:
    return tuple(getattr(node, "named_children", ()) or ())


def _child_by_field(node: Any, field: str) -> Any | None:
    method = getattr(node, "child_by_field_name", None)
    if method is None:
        return None
    try:
        return method(field)
    except (TypeError, ValueError):
        return None


def _first_node(*nodes: Any | None) -> Any | None:
    """Return the first non-``None`` node without relying on extension-type truthiness."""
    return next((node for node in nodes if node is not None), None)


def _same_node(left: Any | None, right: Any | None) -> bool:
    """Compare Tree-sitter nodes by stable source identity, not Python wrapper identity."""
    if left is None or right is None:
        return False
    return (
        getattr(left, "type", None) == getattr(right, "type", None)
        and int(getattr(left, "start_byte", -1)) == int(getattr(right, "start_byte", -2))
        and int(getattr(left, "end_byte", -1)) == int(getattr(right, "end_byte", -2))
    )


def _walk(node: Any) -> Iterable[Any]:
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(_children(current)))


def _node_text(source: bytes, node: Any | None) -> str:
    if node is None:
        return ""
    start = max(0, int(getattr(node, "start_byte", 0)))
    end = max(start, int(getattr(node, "end_byte", start)))
    return source[start:end].decode("utf-8", errors="replace")


def _point_parts(point: Any) -> tuple[int, int]:
    if hasattr(point, "row") and hasattr(point, "column"):
        return int(point.row), int(point.column)
    if isinstance(point, tuple) and len(point) == 2:
        return int(point[0]), int(point[1])
    return 0, 0


def _first_descendant(node: Any | None, node_types: set[str]) -> Any | None:
    if node is None:
        return None
    for candidate in _walk(node):
        if getattr(candidate, "type", "") in node_types:
            return candidate
    return None


def _nearest_ancestor(node: Any, node_types: set[str]) -> Any | None:
    current = getattr(node, "parent", None)
    while current is not None:
        if getattr(current, "type", "") in node_types:
            return current
        current = getattr(current, "parent", None)
    return None


def _has_ancestor(node: Any, node_types: set[str]) -> bool:
    """Return whether ``node`` is nested beneath any syntax kind in ``node_types``."""
    return _nearest_ancestor(node, node_types) is not None


def _contains_error(node: Any) -> bool:
    if bool(getattr(node, "has_error", False)):
        return True
    if bool(getattr(node, "is_error", False)) or bool(getattr(node, "is_missing", False)):
        return True
    return getattr(node, "type", "") == "ERROR"


def _module_prefix(root: Path, file_path: Path, language: str) -> str:
    try:
        rel = file_path.relative_to(root)
    except ValueError:
        rel = Path(file_path.name)

    parts = list(rel.parts[:-1])
    stem = rel.name
    for suffix in (".d.ts", ".d.mts", ".d.cts", ".tsx", ".mts", ".cts", ".jsx", ".mjs", ".cjs"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    else:
        stem = Path(stem).stem

    conventional = {"index"}
    if language == "rust":
        conventional |= {"mod", "lib", "main"}
    if stem not in conventional or not parts:
        parts.append(stem)
    if not parts:
        parts.append(stem or file_path.stem)
    return ".".join(part for part in parts if part)


def _qualified(prefix: str, *parts: str) -> str:
    clean = [part for part in (prefix, *parts) if part]
    return ".".join(clean)


def _clean_name(text: str) -> str:
    return re.sub(r"\s+", "", text.strip())


def _stable_path(text: str) -> str | None:
    candidate = _clean_name(text)
    return candidate if _STABLE_PATH_RE.fullmatch(candidate) else None


def _leaf_nodes(node: Any) -> Iterable[Any]:
    for candidate in _walk(node):
        if not _children(candidate):
            yield candidate


def _structural_hash(node: Any, source: bytes, language: str, unit_type: CodeUnitType) -> str:
    normalized_names: dict[str, str] = {}
    pieces: list[str] = [
        f"schema={FINGERPRINT_SCHEMA_VERSION}",
        f"language={language}",
        f"unit={unit_type.name.lower()}",
    ]

    def visit(current: Any) -> None:
        node_type = str(getattr(current, "type", ""))
        if node_type in _COMMENT_TYPES or "comment" in node_type:
            return

        text = _node_text(source, current)
        lower_type = node_type.lower()
        children = _children(current)

        if any(marker in lower_type for marker in _STRING_MARKERS) and not (
            "template" in lower_type and children
        ):
            pieces.append(f"<{node_type}:STR>")
            return

        if not children:
            if node_type in _IDENTIFIER_TYPES:
                if node_type in _PRESERVED_IDENTIFIER_TYPES or (
                    text.startswith("__") and text.endswith("__")
                ):
                    value = text
                else:
                    value = normalized_names.setdefault(text, f"_v{len(normalized_names)}")
                pieces.append(f"<{node_type}:{value}>")
            elif any(marker in lower_type for marker in _NUMBER_MARKERS) or bool(getattr(current, "is_named", False)):
                pieces.append(f"<{node_type}:{text}>")
            else:
                pieces.append(text)
            return

        pieces.append(f"({node_type}")
        for child in children:
            visit(child)
        pieces.append(")")

    visit(node)
    return hashlib.sha256("\x1f".join(pieces).encode("utf-8")).hexdigest()[:16]


def _token_hash(node: Any, source: bytes) -> str:
    tokens: list[str] = []
    for leaf in _leaf_nodes(node):
        node_type = str(getattr(leaf, "type", ""))
        if node_type in _COMMENT_TYPES or "comment" in node_type:
            continue
        text = _node_text(source, leaf)
        if text.strip():
            tokens.append(f"{node_type}:{text}")
    return hashlib.sha256("\x1f".join(tokens).encode("utf-8")).hexdigest()[:16]


def _collect_identifiers(node: Any, source: bytes, builtins: frozenset[str]) -> frozenset[str]:
    identifiers: set[str] = set()
    for leaf in _leaf_nodes(node):
        if getattr(leaf, "type", "") not in _IDENTIFIER_TYPES:
            continue
        text = _node_text(source, leaf)
        if _IDENTIFIER_RE.fullmatch(text) and text not in builtins:
            identifiers.add(text)
    return frozenset(identifiers)


def _collect_calls(node: Any, source: bytes) -> set[str]:
    calls: set[str] = set()
    for candidate in _walk(node):
        node_type = getattr(candidate, "type", "")
        if node_type not in {
            "call_expression",
            "new_expression",
            "macro_invocation",
            "method_call_expression",
        }:
            continue
        callee = _first_node(
            _child_by_field(candidate, "function"),
            _child_by_field(candidate, "callee"),
            _child_by_field(candidate, "macro"),
        )
        if callee is None:
            named = _named_children(candidate)
            callee = named[0] if named else None
        text = _clean_name(_node_text(source, callee))
        if not text:
            continue
        calls.add(text)
        final = re.split(r"[.:]+", text)[-1]
        if final:
            calls.add(final)
    return calls


class TreeSitterBackend:
    """Shared parse, diagnostics, fingerprint, and unit-construction machinery."""

    language: str
    dialect: str
    statement_types: frozenset[str] = frozenset()
    nested_scope_types: frozenset[str] = frozenset()
    class_member_types: frozenset[str] = frozenset()
    builtins: frozenset[str] = frozenset()

    def __init__(self, root: Path, dialect: str, include_private: bool) -> None:
        self.root = root.resolve()
        self.dialect = dialect
        self.include_private = include_private

    def collect_specs(self, root_node: Any, source: bytes, file_path: Path) -> list[UnitSpec]:
        raise NotImplementedError

    def _include_spec(self, spec: UnitSpec) -> bool:
        if self.include_private:
            return True
        last = spec.name.rsplit(".", 1)[-1]
        return not (last.startswith("_") and not last.startswith("__")) and not last.startswith("#")

    def _statement_count(self, body: Any, unit_type: CodeUnitType) -> int:
        if body is None:
            return 0
        if unit_type == CodeUnitType.CLASS:
            return sum(
                1 for child in _named_children(body) if child.type in self.class_member_types
            )

        count = 0
        stack = list(reversed(_named_children(body)))
        while stack:
            current = stack.pop()
            node_type = getattr(current, "type", "")
            if node_type in self.nested_scope_types:
                count += 1
                continue
            if node_type in self.statement_types:
                count += 1
            stack.extend(reversed(_named_children(current)))
        if count == 0 and getattr(body, "type", "") not in {
            "compound_statement",
            "block",
            "statement_block",
            "class_body",
            "declaration_list",
        }:
            return 1
        return count

    def _diagnostic_for_node(
        self,
        file_path: Path,
        node: Any,
        message: str,
        *,
        code: str,
        severity: str = "warning",
    ) -> ExtractionDiagnostic:
        start_row, _ = _point_parts(getattr(node, "start_point", (0, 0)))
        end_row, _ = _point_parts(getattr(node, "end_point", (start_row, 0)))
        return ExtractionDiagnostic(
            file_path=file_path,
            language=self.language,
            message=message,
            severity=severity,  # type: ignore[arg-type]
            code=code,
            lineno=start_row + 1,
            end_lineno=end_row + 1,
        )

    def extract_file(self, file_path: Path) -> BackendResult:
        source = file_path.read_bytes()
        parser = GrammarProvider.parser(self.dialect)
        tree = parser.parse(source)
        root_node = tree.root_node

        diagnostics: list[ExtractionDiagnostic] = []
        if _contains_error(root_node):
            diagnostics.append(
                self._diagnostic_for_node(
                    file_path,
                    root_node,
                    "The syntax tree contains recovery nodes; unaffected units were "
                    "still analyzed.",
                    code="partial-parse",
                )
            )

        specs = self.collect_specs(root_node, source, file_path)
        deduped: dict[tuple[int, int, str, str], UnitSpec] = {}
        for spec in specs:
            key = (
                int(getattr(spec.source_node, "start_byte", 0)),
                int(getattr(spec.source_node, "end_byte", 0)),
                spec.native_kind,
                spec.qualified_name,
            )
            deduped[key] = spec

        units: list[CodeUnit] = []
        for spec in sorted(
            deduped.values(),
            key=lambda item: (
                int(getattr(item.source_node, "start_byte", 0)),
                int(getattr(item.source_node, "end_byte", 0)),
                item.qualified_name,
            ),
        ):
            if not self._include_spec(spec):
                continue
            if _contains_error(spec.node):
                diagnostics.append(
                    self._diagnostic_for_node(
                        file_path,
                        spec.node,
                        f"Skipped {spec.qualified_name} because its syntax subtree "
                        "contains an error.",
                        code="unit-parse-error",
                    )
                )
                continue

            start_byte = int(getattr(spec.source_node, "start_byte", 0))
            end_byte = int(getattr(spec.source_node, "end_byte", start_byte))
            start_row, start_column = _point_parts(getattr(spec.source_node, "start_point", (0, 0)))
            end_row, end_column = _point_parts(
                getattr(spec.source_node, "end_point", (start_row, start_column))
            )
            snippet = source[start_byte:end_byte].decode("utf-8", errors="replace")

            structural_hash = _structural_hash(
                spec.node,
                source,
                self.language,
                spec.unit_type,
            )
            units.append(
                CodeUnit(
                    name=spec.name,
                    qualified_name=spec.qualified_name,
                    unit_type=spec.unit_type,
                    file_path=file_path,
                    lineno=start_row + 1,
                    end_lineno=max(start_row + 1, end_row + 1),
                    source=snippet,
                    language=self.language,
                    dialect=self.dialect,
                    native_kind=spec.native_kind,
                    start_byte=start_byte,
                    end_byte=end_byte,
                    start_column=start_column,
                    end_column=end_column,
                    has_body=spec.body is not None,
                    statement_count=self._statement_count(spec.body, spec.unit_type),
                    _ast_hash=structural_hash,
                    _structural_hash=structural_hash,
                    _token_hash=_token_hash(spec.node, source),
                    identifiers=_collect_identifiers(spec.node, source, self.builtins),
                    calls=_collect_calls(spec.node, source),
                    is_public=spec.is_public,
                    is_dunder=spec.name.startswith("__") and spec.name.endswith("__"),
                    is_exported=spec.is_exported,
                )
            )

        return BackendResult(tuple(units), tuple(diagnostics))


class CBackend(TreeSitterBackend):
    """Extract C function definitions; prototypes and declarations are ignored."""

    language = "c"
    statement_types = frozenset(
        {
            "declaration",
            "expression_statement",
            "return_statement",
            "if_statement",
            "switch_statement",
            "case_statement",
            "for_statement",
            "while_statement",
            "do_statement",
            "break_statement",
            "continue_statement",
            "goto_statement",
            "labeled_statement",
            "seh_try_statement",
            "seh_leave_statement",
        }
    )
    nested_scope_types = frozenset({"function_definition"})
    builtins = frozenset(
        {
            "sizeof",
            "NULL",
            "true",
            "false",
            "stdin",
            "stdout",
            "stderr",
        }
    )

    def _include_spec(self, spec: UnitSpec) -> bool:
        return self.include_private or spec.is_public

    @staticmethod
    def _declarator_name(declarator: Any | None, source: bytes) -> str | None:
        if declarator is None:
            return None
        node_type = getattr(declarator, "type", "")
        if node_type in {"identifier", "field_identifier"}:
            return _node_text(source, declarator)

        nested = _child_by_field(declarator, "declarator")
        if nested is not None:
            found = CBackend._declarator_name(nested, source)
            if found:
                return found

        for child in _named_children(declarator):
            found = CBackend._declarator_name(child, source)
            if found:
                return found
        return None

    def collect_specs(self, root_node: Any, source: bytes, file_path: Path) -> list[UnitSpec]:
        prefix = _module_prefix(self.root, file_path, self.language)
        specs: list[UnitSpec] = []
        for node in _walk(root_node):
            if getattr(node, "type", "") != "function_definition":
                continue
            declarator = _child_by_field(node, "declarator")
            name = self._declarator_name(declarator, source)
            body = _first_node(
                _child_by_field(node, "body"),
                _first_descendant(node, {"compound_statement"}),
            )
            if not name or body is None:
                continue
            prefix_text = source[int(node.start_byte) : int(body.start_byte)].decode(
                "utf-8", errors="replace"
            )
            is_public = re.search(r"\bstatic\b", prefix_text) is None
            specs.append(
                UnitSpec(
                    node=node,
                    source_node=node,
                    body=body,
                    name=name,
                    qualified_name=_qualified(prefix, name),
                    unit_type=CodeUnitType.FUNCTION,
                    native_kind="function_definition",
                    is_public=is_public,
                    is_exported=is_public,
                )
            )
        return specs


class RustBackend(TreeSitterBackend):
    """Extract Rust free functions and body-bearing impl/trait methods."""

    language = "rust"
    statement_types = frozenset(
        {
            "let_declaration",
            "expression_statement",
            "return_expression",
            "if_expression",
            "match_expression",
            "loop_expression",
            "while_expression",
            "for_expression",
            "break_expression",
            "continue_expression",
        }
    )
    nested_scope_types = frozenset({"function_item", "closure_expression"})
    builtins = frozenset(
        {
            "self",
            "Self",
            "super",
            "crate",
            "Some",
            "None",
            "Ok",
            "Err",
            "true",
            "false",
        }
    )

    def _include_spec(self, spec: UnitSpec) -> bool:
        return self.include_private or spec.is_public

    @staticmethod
    def _visibility(node: Any, source: bytes) -> bool:
        for child in _named_children(node):
            if getattr(child, "type", "") == "visibility_modifier":
                return _node_text(source, child).strip().startswith("pub")
        return False

    @staticmethod
    def _context(node: Any, source: bytes) -> tuple[list[str], bool]:
        contexts: list[str] = []
        is_method = False
        inside_enclosing_function = False
        current = getattr(node, "parent", None)
        while current is not None:
            node_type = getattr(current, "type", "")
            if node_type == "impl_item":
                if not inside_enclosing_function:
                    is_method = True
                target = _child_by_field(current, "type")
                if target is None:
                    candidates = [
                        child
                        for child in _named_children(current)
                        if getattr(child, "type", "")
                        not in {"declaration_list", "type_parameters", "where_clause"}
                    ]
                    target = candidates[-1] if candidates else None
                target_text = _clean_name(_node_text(source, target))
                if target_text:
                    contexts.append(target_text)
            elif node_type in {"trait_item", "trait_declaration"}:
                if not inside_enclosing_function:
                    is_method = True
                name = _child_by_field(current, "name")
                name_text = _node_text(source, name).strip()
                if name_text:
                    contexts.append(name_text)
            elif node_type == "mod_item":
                name = _child_by_field(current, "name")
                name_text = _node_text(source, name).strip()
                if name_text:
                    contexts.append(name_text)
            elif node_type == "function_item":
                inside_enclosing_function = True
                name = _child_by_field(current, "name")
                name_text = _node_text(source, name).strip()
                if name_text:
                    contexts.append(name_text)
            current = getattr(current, "parent", None)
        contexts.reverse()
        return contexts, is_method

    def collect_specs(self, root_node: Any, source: bytes, file_path: Path) -> list[UnitSpec]:
        prefix = _module_prefix(self.root, file_path, self.language)
        specs: list[UnitSpec] = []
        for node in _walk(root_node):
            if getattr(node, "type", "") != "function_item":
                continue
            name_node = _child_by_field(node, "name")
            body = _first_node(
                _child_by_field(node, "body"),
                _first_descendant(node, {"block"}),
            )
            name = _node_text(source, name_node).strip()
            if not name or body is None:
                continue
            contexts, is_method = self._context(node, source)
            public = self._visibility(node, source)
            # Trait methods are part of the trait API even when they do not carry
            # an explicit `pub` modifier.
            if is_method and _nearest_ancestor(node, {"trait_item", "trait_declaration"}):
                public = True
            specs.append(
                UnitSpec(
                    node=node,
                    source_node=node,
                    body=body,
                    name=name,
                    qualified_name=_qualified(prefix, *contexts, name),
                    unit_type=CodeUnitType.METHOD if is_method else CodeUnitType.FUNCTION,
                    native_kind="function_item",
                    is_public=public,
                    is_exported=public,
                )
            )
        return specs


class ECMAScriptBackend(TreeSitterBackend):
    """Shared JavaScript/TypeScript extraction and stable lexical naming."""

    function_declarations = frozenset(
        {"function_declaration", "generator_function_declaration"}
    )
    function_expressions = frozenset(
        {"function_expression", "generator_function", "arrow_function"}
    )
    class_declarations = frozenset({"class_declaration"})
    class_expressions = frozenset({"class"})
    method_types = frozenset({"method_definition"})
    field_types = frozenset({"field_definition", "public_field_definition"})
    transparent_types = frozenset(
        {
            "parenthesized_expression",
            "as_expression",
            "satisfies_expression",
            "type_assertion",
            "non_null_expression",
            "instantiation_expression",
        }
    )
    statement_types = frozenset(
        {
            "expression_statement",
            "lexical_declaration",
            "variable_declaration",
            "return_statement",
            "throw_statement",
            "if_statement",
            "switch_statement",
            "for_statement",
            "for_in_statement",
            "while_statement",
            "do_statement",
            "try_statement",
            "break_statement",
            "continue_statement",
            "debugger_statement",
            "with_statement",
        }
    )
    nested_scope_types = frozenset(
        {
            "function_declaration",
            "generator_function_declaration",
            "function_expression",
            "generator_function",
            "arrow_function",
            "class_declaration",
            "class",
            "method_definition",
        }
    )
    class_member_types = frozenset(
        {"method_definition", "field_definition", "public_field_definition", "static_block"}
    )
    builtins = frozenset(
        {
            "undefined",
            "NaN",
            "Infinity",
            "console",
            "JSON",
            "Math",
            "Object",
            "Array",
            "String",
            "Number",
            "Boolean",
            "Promise",
            "Map",
            "Set",
            "Date",
            "RegExp",
            "Error",
            "require",
            "module",
            "exports",
        }
    )

    def _name_field(self, node: Any, source: bytes) -> str | None:
        name_node = _first_node(
            _child_by_field(node, "name"),
            _child_by_field(node, "property"),
            _child_by_field(node, "key"),
        )
        if name_node is None:
            return None
        text = _node_text(source, name_node).strip()
        if getattr(name_node, "type", "") in {"string", "string_fragment"}:
            text = text.strip("'\"")
        return _stable_path(text)

    def _unwrap_value(self, node: Any) -> Any:
        current = node
        while getattr(getattr(current, "parent", None), "type", "") in self.transparent_types:
            current = current.parent
        return current

    def _binding_for_value(self, node: Any, source: bytes) -> tuple[str, Any] | None:
        current = self._unwrap_value(node)
        parent = getattr(current, "parent", None)
        if parent is None:
            return None
        parent_type = getattr(parent, "type", "")

        if parent_type == "variable_declarator" and _same_node(
            _child_by_field(parent, "value"), current
        ):
            name = _child_by_field(parent, "name")
            stable = _stable_path(_node_text(source, name))
            return (stable, parent) if stable else None

        if parent_type in {"assignment_expression", "augmented_assignment_expression"}:
            right = _child_by_field(parent, "right")
            if _same_node(right, current):
                left = _child_by_field(parent, "left")
                stable = _stable_path(_node_text(source, left))
                return (stable, parent) if stable else None

        if parent_type == "pair" and _same_node(_child_by_field(parent, "value"), current):
            key = _child_by_field(parent, "key")
            key_text = _node_text(source, key).strip().strip("'\"")
            key_name = _stable_path(key_text)
            object_node = getattr(parent, "parent", None)
            object_binding = self._binding_for_value(object_node, source) if object_node else None
            if key_name and object_binding:
                return (f"{object_binding[0]}.{key_name}", parent)
            return None

        if parent_type in self.field_types and _same_node(
            _child_by_field(parent, "value"), current
        ):
            key_name = self._name_field(parent, source)
            class_name = self._class_context(parent, source)
            if key_name and class_name:
                return (f"{class_name}.{key_name}", parent)
            return None

        if parent_type == "export_statement":
            text = _node_text(source, parent)
            if re.match(r"\s*export\s+default\b", text):
                return ("default", parent)
        return None

    def _class_context(self, node: Any, source: bytes) -> str | None:
        class_node = _nearest_ancestor(node, set(self.class_declarations | self.class_expressions))
        if class_node is None:
            return None
        name = self._name_field(class_node, source)
        if name:
            return name
        binding = self._binding_for_value(class_node, source)
        return binding[0] if binding else None

    def _contextual_binding(self, node: Any, bound_name: str, source: bytes) -> str:
        """Qualify a stable binding with its enclosing lexical scopes."""
        if bound_name.startswith(("exports.", "module.exports")):
            return bound_name
        context = ".".join(self._lexical_context(node, source))
        if not context or bound_name == context or bound_name.startswith(f"{context}."):
            return bound_name
        return f"{context}.{bound_name}"

    def _object_context(self, node: Any, source: bytes) -> str | None:
        object_node = _nearest_ancestor(node, {"object"})
        if object_node is None:
            return None
        binding = self._binding_for_value(object_node, source)
        if binding is None:
            return None
        return self._contextual_binding(object_node, binding[0], source)

    def _lexical_context(self, node: Any, source: bytes) -> list[str]:
        contexts: list[str] = []
        current = getattr(node, "parent", None)
        while current is not None:
            node_type = getattr(current, "type", "")
            if node_type in self.class_declarations | self.class_expressions:
                name = self._name_field(current, source)
                if not name:
                    binding = self._binding_for_value(current, source)
                    name = binding[0].rsplit(".", 1)[-1] if binding else None
                if name:
                    contexts.append(name)
            elif node_type in self.function_declarations:
                name = self._name_field(current, source)
                if name:
                    contexts.append(name)
            elif node_type in self.function_expressions:
                binding = self._binding_for_value(current, source)
                if binding:
                    contexts.append(binding[0].rsplit(".", 1)[-1])
            elif node_type in self.method_types or node_type in {"internal_module", "module"}:
                name = self._name_field(current, source)
                if name:
                    contexts.append(name)
            current = getattr(current, "parent", None)
        contexts.reverse()
        return contexts

    @staticmethod
    def _is_public_member(node: Any, source: bytes, name: str) -> bool:
        """Apply naming and TypeScript accessibility rules to one member."""
        if name.startswith(("_", "#")):
            return False
        for child in _named_children(node):
            if getattr(child, "type", "") != "accessibility_modifier":
                continue
            if _node_text(source, child).strip() in {"private", "protected"}:
                return False
        return True

    @staticmethod
    def _is_exported(node: Any, source: bytes, name: str) -> bool:
        current = node
        while current is not None:
            if getattr(current, "type", "") == "export_statement":
                return True
            current = getattr(current, "parent", None)
        return name.startswith("exports.") or name.startswith("module.exports")

    def _function_spec(
        self,
        node: Any,
        source: bytes,
        prefix: str,
    ) -> UnitSpec | None:
        node_type = getattr(node, "type", "")
        if _has_ancestor(node, {"ambient_declaration"}):
            return None
        body = _child_by_field(node, "body")
        if body is None:
            return None

        source_node = node
        if node_type in self.function_declarations:
            name = self._name_field(node, source)
            if not name:
                export_parent = getattr(node, "parent", None)
                export_text = _node_text(source, export_parent)
                if getattr(export_parent, "type", "") != "export_statement" or not re.match(
                    r"\s*export\s+default\b", export_text
                ):
                    return None
                name = "default"
                source_node = export_parent
            context = self._lexical_context(node, source)
            qualified_name = _qualified(prefix, *context, name)
            unit_type = CodeUnitType.FUNCTION
        else:
            binding = self._binding_for_value(node, source)
            own_name = self._name_field(node, source)
            if binding:
                bound_name, source_node = binding
                contextual_name = self._contextual_binding(node, bound_name, source)
                name = bound_name.rsplit(".", 1)[-1]
                qualified_name = _qualified(prefix, contextual_name)
                source_kind = getattr(source_node, "type", "")
                unit_type = (
                    CodeUnitType.METHOD
                    if source_kind in self.field_types
                    else CodeUnitType.FUNCTION
                )
            elif own_name:
                name = own_name
                qualified_name = _qualified(prefix, *self._lexical_context(node, source), name)
                unit_type = CodeUnitType.FUNCTION
            else:
                return None

        exported = self._is_exported(source_node, source, qualified_name.removeprefix(prefix + "."))
        return UnitSpec(
            node=node,
            source_node=source_node,
            body=body,
            name=name,
            qualified_name=qualified_name,
            unit_type=unit_type,
            native_kind=node_type,
            is_public=self._is_public_member(source_node, source, name),
            is_exported=exported,
        )

    def _class_spec(self, node: Any, source: bytes, prefix: str) -> UnitSpec | None:
        node_type = getattr(node, "type", "")
        if _has_ancestor(node, {"ambient_declaration"}):
            return None
        body = _first_node(
            _child_by_field(node, "body"),
            _first_descendant(node, {"class_body"}),
        )
        if body is None:
            return None
        source_node = node
        name = self._name_field(node, source)
        if not name:
            binding = self._binding_for_value(node, source)
            if not binding:
                return None
            bound_name, source_node = binding
            contextual_name = self._contextual_binding(node, bound_name, source)
            name = bound_name.rsplit(".", 1)[-1]
            qualified_name = _qualified(prefix, contextual_name)
        else:
            qualified_name = _qualified(prefix, *self._lexical_context(node, source), name)
        exported = self._is_exported(source_node, source, qualified_name.removeprefix(prefix + "."))
        return UnitSpec(
            node=node,
            source_node=source_node,
            body=body,
            name=name,
            qualified_name=qualified_name,
            unit_type=CodeUnitType.CLASS,
            native_kind=node_type,
            is_public=self._is_public_member(source_node, source, name),
            is_exported=exported,
        )

    def _method_spec(self, node: Any, source: bytes, prefix: str) -> UnitSpec | None:
        if _has_ancestor(node, {"ambient_declaration"}):
            return None
        body = _child_by_field(node, "body")
        name = self._name_field(node, source)
        if body is None or not name:
            return None
        lexical_context = self._lexical_context(node, source)
        object_context = self._object_context(node, source)
        container = object_context or ".".join(lexical_context)
        if not container:
            return None
        qualified_name = _qualified(prefix, container, name)
        exported = self._is_exported(node, source, container)
        return UnitSpec(
            node=node,
            source_node=node,
            body=body,
            name=name,
            qualified_name=qualified_name,
            unit_type=CodeUnitType.METHOD,
            native_kind="method_definition",
            is_public=self._is_public_member(node, source, name),
            is_exported=exported,
        )

    def collect_specs(self, root_node: Any, source: bytes, file_path: Path) -> list[UnitSpec]:
        prefix = _module_prefix(self.root, file_path, self.language)
        specs: list[UnitSpec] = []
        for node in _walk(root_node):
            node_type = getattr(node, "type", "")
            spec: UnitSpec | None = None
            if node_type in self.function_declarations | self.function_expressions:
                spec = self._function_spec(node, source, prefix)
            elif node_type in self.class_declarations | self.class_expressions:
                spec = self._class_spec(node, source, prefix)
            elif node_type in self.method_types:
                spec = self._method_spec(node, source, prefix)
            if spec is not None:
                specs.append(spec)
        return specs


class JavaScriptBackend(ECMAScriptBackend):
    """JavaScript/JSX backend."""

    language = "javascript"


class TypeScriptBackend(ECMAScriptBackend):
    """TypeScript/TSX backend; bodyless and ambient declarations are excluded."""

    language = "typescript"
    class_declarations = ECMAScriptBackend.class_declarations | frozenset(
        {"abstract_class_declaration"}
    )


def create_backend(
    *,
    root: Path,
    language: str,
    dialect: str,
    include_private: bool,
) -> TreeSitterBackend:
    """Create the concrete backend for a canonical language."""
    backend_type: type[TreeSitterBackend]
    if language == "c":
        backend_type = CBackend
    elif language == "rust":
        backend_type = RustBackend
    elif language == "javascript":
        backend_type = JavaScriptBackend
    elif language == "typescript":
        backend_type = TypeScriptBackend
    else:
        raise ValueError(f"No Tree-sitter backend exists for language {language!r}")
    return backend_type(root=root, dialect=dialect, include_private=include_private)
