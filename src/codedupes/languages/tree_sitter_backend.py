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

# 1: initial canonical stream.
# 2: declaration names (methods, classes) normalize like local identifiers so
#    renamed copies stay in the deterministic tier; object-literal keys and
#    member-access names remain preserved as data/API shape.
FINGERPRINT_SCHEMA_VERSION = 2
# ECMAScript identifiers are Unicode from ES2015 on, and Rust accepts non-ASCII
# identifiers too, so an ASCII-only pattern would silently drop those units.
# ``[^\W\d]`` is Python's Unicode-aware "letter or underscore".
_IDENTIFIER_RE = re.compile(r"^(?:[^\W\d]|\$)(?:\w|\$)*$")
_STABLE_PATH_RE = re.compile(r"^(?:[^\W\d]|[$#])(?:\w|[$#])*(?:\.(?:[^\W\d]|[$#])(?:\w|[$#])*)*$")

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
# A declaration's own name is a definition name, not API shape, so it must
# normalize the way Python def/class names do even when its leaf type is
# otherwise preserved (TS class names are ``type_identifier``, method names
# are ``property_identifier``). Class names normalize anywhere; the member
# types normalize only inside a ``class_body`` or as the hashed unit root,
# so object-literal members keep key-shape sensitivity like Python dict keys.
_CLASS_DECLARATION_TYPES = {"class_declaration", "abstract_class_declaration", "class"}
_CLASS_MEMBER_CALLABLE_TYPES = {
    "method_definition",
    "method_signature",
    "abstract_method_signature",
}
# ``jsx_text`` is display copy, not structure: two otherwise identical React
# components must not fingerprint differently because their labels differ.
_STRING_MARKERS = ("string", "char_literal", "template_string", "raw_string", "jsx_text")
_NUMBER_MARKERS = ("number", "integer", "float", "decimal", "hex", "octal", "binary")


class GrammarUnavailableError(RuntimeError):
    """Raised when a requested precompiled parser package cannot be loaded."""


class GrammarProvider:
    """Load and cache immutable ``tree_sitter.Language`` objects."""

    _languages: ClassVar[dict[str, Any]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def language(cls, dialect: str) -> Any:
        """Return a cached Tree-sitter language for one parser dialect.

        :param dialect: Parser dialect name.
        :raises GrammarUnavailableError: If the pinned grammar package cannot be loaded.
        :return: Cached ``tree_sitter.Language`` object for the dialect.
        """
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
        """Create a fresh parser for one parse operation.

        :param dialect: Parser dialect name.
        :raises GrammarUnavailableError: If the parser cannot be constructed.
        :return: New ``tree_sitter.Parser`` bound to the dialect grammar.
        """
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


def _spec_span(spec: UnitSpec) -> tuple[int, int]:
    """Return the byte range covering both a spec's unit node and its source node.

    :param spec: Unit spec.
    :return: Start and end byte offsets of the whole spec.
    """
    starts = (
        int(getattr(spec.node, "start_byte", 0)),
        int(getattr(spec.source_node, "start_byte", 0)),
    )
    ends = (
        int(getattr(spec.node, "end_byte", 0)),
        int(getattr(spec.source_node, "end_byte", 0)),
    )
    return min(starts), max(ends)


def _children(node: Any) -> tuple[Any, ...]:
    """Return a node's children, tolerating parsers that expose none.

    :param node: Tree-sitter node.
    :return: Child nodes, empty when the node has no children.
    """
    return tuple(getattr(node, "children", ()) or ())


def _named_children(node: Any) -> tuple[Any, ...]:
    """Return a node's named children, tolerating parsers that expose none.

    :param node: Tree-sitter node.
    :return: Named child nodes, empty when the node has none.
    """
    return tuple(getattr(node, "named_children", ()) or ())


def _is_comment(node: Any) -> bool:
    """Report whether a node is a comment in any of the supported grammars.

    :param node: Tree-sitter node.
    :return: ``True`` when the node's syntax kind names a comment.
    """
    node_type = str(getattr(node, "type", ""))
    return node_type in _COMMENT_TYPES or "comment" in node_type


def _child_by_field(node: Any, field: str) -> Any | None:
    """Look up a child node by its grammar field name.

    :param node: Tree-sitter node.
    :param field: Grammar field name.
    :return: Matching child node, or ``None`` when the field is absent.
    """
    method = getattr(node, "child_by_field_name", None)
    if method is None:
        return None
    try:
        return method(field)
    except (TypeError, ValueError):
        return None


def _first_node(*nodes: Any | None) -> Any | None:
    """Return the first non-``None`` node without relying on extension-type truthiness.

    :param nodes: Candidate nodes in priority order.
    :return: First non-``None`` candidate, or ``None`` when all candidates are ``None``.
    """
    return next((node for node in nodes if node is not None), None)


def _same_node(left: Any | None, right: Any | None) -> bool:
    """Compare Tree-sitter nodes by stable source identity, not Python wrapper identity.

    :param left: First node.
    :param right: Second node.
    :return: ``True`` when both nodes share a syntax kind and byte span.
    """
    if left is None or right is None:
        return False
    return (
        getattr(left, "type", None) == getattr(right, "type", None)
        and int(getattr(left, "start_byte", -1)) == int(getattr(right, "start_byte", -2))
        and int(getattr(left, "end_byte", -1)) == int(getattr(right, "end_byte", -2))
    )


def _preceding_named_siblings(node: Any) -> Iterable[Any]:
    """Yield the named siblings that precede a node, nearest first.

    py-tree-sitter exposes ``prev_named_sibling``, which is O(1) per step; scanning
    the parent's named children instead is O(siblings) per lookup and turns
    attribute collection quadratic in the number of items in a file. Node doubles
    and parsers without that attribute fall back to the scan, which matches nodes
    by source identity because bindings may hand out fresh wrappers.

    :param node: Node whose preceding named siblings are walked.
    :return: Iterator over preceding named siblings, closest to ``node`` first.
    """
    if getattr(node, "is_named", True) and hasattr(node, "prev_named_sibling"):
        current = node.prev_named_sibling
        while current is not None:
            yield current
            current = getattr(current, "prev_named_sibling", None)
        return

    siblings = _named_children(getattr(node, "parent", None))
    index = next(
        (position for position, sibling in enumerate(siblings) if _same_node(sibling, node)),
        None,
    )
    if index is None:
        return
    yield from reversed(siblings[:index])


def _walk(node: Any) -> Iterable[Any]:
    """Walk a subtree in preorder without recursing.

    :param node: Root of the subtree.
    :return: Iterator over ``node`` followed by all of its descendants.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(_children(current)))


def _node_text(source: bytes, node: Any | None) -> str:
    """Decode the source text covered by one node.

    :param source: Full file source bytes.
    :param node: Node whose byte span is decoded, or ``None``.
    :return: Decoded text, empty when ``node`` is ``None``.
    """
    if node is None:
        return ""
    start = max(0, int(getattr(node, "start_byte", 0)))
    end = max(start, int(getattr(node, "end_byte", start)))
    return source[start:end].decode("utf-8", errors="replace")


def _point_parts(point: Any) -> tuple[int, int]:
    """Normalize a Tree-sitter point into a row/column pair.

    :param point: Point object or two-element tuple from the parser.
    :return: Zero-based row and column, ``(0, 0)`` for unrecognized shapes.
    """
    if hasattr(point, "row") and hasattr(point, "column"):
        return int(point.row), int(point.column)
    if isinstance(point, tuple) and len(point) == 2:
        return int(point[0]), int(point[1])
    return 0, 0


def _first_descendant(node: Any | None, node_types: set[str]) -> Any | None:
    """Find the first node in a subtree matching one of ``node_types``.

    :param node: Root of the search, or ``None``.
    :param node_types: Syntax kinds to match.
    :return: First matching node in preorder, or ``None``.
    """
    if node is None:
        return None
    for candidate in _walk(node):
        if getattr(candidate, "type", "") in node_types:
            return candidate
    return None


def _nearest_ancestor(node: Any, node_types: set[str]) -> Any | None:
    """Find the closest ancestor matching one of ``node_types``.

    :param node: Node whose ancestor chain is walked.
    :param node_types: Syntax kinds to match.
    :return: Nearest matching ancestor, or ``None``.
    """
    current = getattr(node, "parent", None)
    while current is not None:
        if getattr(current, "type", "") in node_types:
            return current
        current = getattr(current, "parent", None)
    return None


def _has_ancestor(node: Any, node_types: set[str]) -> bool:
    """Return whether ``node`` is nested beneath any syntax kind in ``node_types``.

    :param node: Node whose ancestor chain is walked.
    :param node_types: Syntax kinds to match.
    :return: ``True`` when a matching ancestor exists.
    """
    return _nearest_ancestor(node, node_types) is not None


def _contains_error(node: Any) -> bool:
    """Report whether a node is, or contains, a parse-error marker.

    :param node: Node to inspect.
    :return: ``True`` when the node is erroneous, missing, or covers recovery nodes.
    """
    if bool(getattr(node, "has_error", False)):
        return True
    if bool(getattr(node, "is_error", False)) or bool(getattr(node, "is_missing", False)):
        return True
    return getattr(node, "type", "") == "ERROR"


def _module_prefix(root: Path, file_path: Path, language: str) -> str:
    """Build the dotted module prefix that qualifies every unit in one file.

    :param root: Extraction root the file path is made relative to.
    :param file_path: File being extracted.
    :param language: Canonical language name.
    :return: Dotted prefix, with conventional entry-point stems collapsed away.
    """
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
    """Join a module prefix and name segments into one dotted name.

    :param prefix: Module prefix, possibly empty.
    :param parts: Name segments in outermost-first order.
    :return: Dotted qualified name with empty segments dropped.
    """
    clean = [part for part in (prefix, *parts) if part]
    return ".".join(clean)


def _push_context_segment(segments: list[str], segment: str) -> None:
    """Add one outer context segment unless the inner path already spells it out.

    :param segments: Innermost-first context segments collected so far, updated in place.
    :param segment: Candidate outer segment, itself possibly dotted.
    :return: ``None``.
    """
    if segments and (segments[-1] == segment or segments[-1].startswith(f"{segment}.")):
        return
    segments.append(segment)


def _clean_name(text: str) -> str:
    """Strip every whitespace run from a source fragment.

    :param text: Raw source text.
    :return: Text with all whitespace removed.
    """
    return re.sub(r"\s+", "", text.strip())


def _stable_path(text: str) -> str | None:
    """Accept a source fragment only when it forms a stable dotted identifier path.

    :param text: Raw source text.
    :return: Cleaned dotted path, or ``None`` when the text is not a stable name.
    """
    candidate = _clean_name(text)
    return candidate if _STABLE_PATH_RE.fullmatch(candidate) else None


def _leaf_nodes(node: Any) -> Iterable[Any]:
    """Yield every childless node in a subtree.

    :param node: Root of the subtree.
    :return: Iterator over the subtree's leaf nodes.
    """
    for candidate in _walk(node):
        if not _children(candidate):
            yield candidate


def _structural_hash(node: Any, source: bytes, language: str, unit_type: CodeUnitType) -> str:
    """Fingerprint a subtree with local identifiers, literals, and comments normalized.

    :param node: Unit node to fingerprint.
    :param source: Full file source bytes.
    :param language: Canonical language name, mixed into the fingerprint.
    :param unit_type: Unit kind, mixed into the fingerprint.
    :return: Truncated SHA-256 digest of the normalized structural token stream.
    """
    normalized_names: dict[str, str] = {}
    pieces: list[str] = [
        f"schema={FINGERPRINT_SCHEMA_VERSION}",
        f"language={language}",
        f"unit={unit_type.name.lower()}",
    ]

    # Byte spans of declaration-name leaves that normalize despite carrying a
    # preserved identifier type. Spans, not object identity: py-tree-sitter can
    # hand out distinct wrappers for the same underlying node.
    declaration_name_spans: set[tuple[int, int]] = set()

    def _mark_declaration_name(owner: Any) -> None:
        """Record the span of a declaration's leaf name so it normalizes.

        :param owner: Declaration node whose ``name`` field child is marked.
        """
        name_node = _child_by_field(owner, "name")
        if name_node is None or _children(name_node):
            return
        declaration_name_spans.add(
            (int(getattr(name_node, "start_byte", -1)), int(getattr(name_node, "end_byte", -1)))
        )

    # Iterative preorder walk with a close-paren sentinel: minified or
    # generated sources nest deeply enough to blow the Python recursion limit.
    close_marker = object()
    stack: list[Any] = [node]
    while stack:
        current = stack.pop()
        if current is close_marker:
            pieces.append(")")
            continue

        node_type = str(getattr(current, "type", ""))
        if node_type in _COMMENT_TYPES or "comment" in node_type:
            continue

        text = _node_text(source, current)
        lower_type = node_type.lower()
        children = _children(current)

        if any(marker in lower_type for marker in _STRING_MARKERS) and not (
            "template" in lower_type and children
        ):
            pieces.append(f"<{node_type}:STR>")
            continue

        if node_type in _CLASS_DECLARATION_TYPES or (
            node_type in _CLASS_MEMBER_CALLABLE_TYPES
            and (
                current is node
                or getattr(getattr(current, "parent", None), "type", "") == "class_body"
            )
        ):
            _mark_declaration_name(current)

        if not children:
            if node_type in _IDENTIFIER_TYPES:
                span = (
                    int(getattr(current, "start_byte", -1)),
                    int(getattr(current, "end_byte", -1)),
                )
                preserved = node_type in _PRESERVED_IDENTIFIER_TYPES or (
                    text.startswith("__") and text.endswith("__")
                )
                if preserved and span not in declaration_name_spans:
                    value = text
                else:
                    value = normalized_names.setdefault(text, f"_v{len(normalized_names)}")
                pieces.append(f"<{node_type}:{value}>")
            elif any(marker in lower_type for marker in _NUMBER_MARKERS) or bool(
                getattr(current, "is_named", False)
            ):
                pieces.append(f"<{node_type}:{text}>")
            else:
                pieces.append(text)
            continue

        pieces.append(f"({node_type}")
        stack.append(close_marker)
        stack.extend(reversed(children))

    return hashlib.sha256("\x1f".join(pieces).encode("utf-8")).hexdigest()[:16]


def _token_hash(node: Any, source: bytes) -> str:
    """Fingerprint a subtree's literal token stream, ignoring comments.

    Comment subtrees are pruned before flattening: some grammars (for example
    tree-sitter-rust) parse comments with delimiter children whose leaf types
    do not mention "comment", so a leaf-level filter alone would let ``//``
    and ``/* */`` markers leak into the token stream.

    :param node: Unit node to fingerprint.
    :param source: Full file source bytes.
    :return: Truncated SHA-256 digest of the typed token stream.
    """
    tokens: list[str] = []
    stack = [node]
    while stack:
        current = stack.pop()
        node_type = str(getattr(current, "type", ""))
        if node_type in _COMMENT_TYPES or "comment" in node_type:
            continue
        children = _children(current)
        if children:
            stack.extend(reversed(children))
            continue
        text = _node_text(source, current)
        if text.strip():
            tokens.append(f"{node_type}:{text}")
    return hashlib.sha256("\x1f".join(tokens).encode("utf-8")).hexdigest()[:16]


def _collect_identifiers(node: Any, source: bytes, builtins: frozenset[str]) -> frozenset[str]:
    """Collect identifier-like leaf names from a subtree, skipping language builtins.

    :param node: Unit node to scan.
    :param source: Full file source bytes.
    :param builtins: Names treated as builtins and excluded.
    :return: Identifier names found in the subtree.
    """
    identifiers: set[str] = set()
    for leaf in _leaf_nodes(node):
        if getattr(leaf, "type", "") not in _IDENTIFIER_TYPES:
            continue
        text = _node_text(source, leaf)
        if _IDENTIFIER_RE.fullmatch(text) and text not in builtins:
            identifiers.add(text)
    return frozenset(identifiers)


def _collect_calls(node: Any, source: bytes) -> set[str]:
    """Collect callee names for calls, constructions, and macro invocations.

    :param node: Unit node to scan.
    :param source: Full file source bytes.
    :return: Callee texts plus their trailing name segments.
    """
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
        """Store the extraction root, parser dialect, and visibility policy.

        :param root: Extraction root used for qualified naming.
        :param dialect: Parser dialect to parse files with.
        :param include_private: Whether non-public units are extracted.
        """
        self.root = root.resolve()
        self.dialect = dialect
        self.include_private = include_private

    def collect_specs(self, root_node: Any, source: bytes, file_path: Path) -> list[UnitSpec]:
        """Collect the language-specific unit specs for one parsed file.

        :param root_node: Root node of the parsed syntax tree.
        :param source: Full file source bytes.
        :param file_path: File being extracted.
        :raises NotImplementedError: Always; concrete backends override this.
        :return: Unit specs for every extractable unit in the file.
        """
        raise NotImplementedError

    def _include_spec(self, spec: UnitSpec) -> bool:
        """Decide whether one unit spec survives the visibility filter.

        :param spec: Candidate unit spec.
        :return: ``True`` when private units are included or the spec is public.
        """
        # ``is_public`` already encodes each language's visibility rules (C
        # ``static``, Rust ``pub``, naming conventions, and TypeScript
        # accessibility modifiers), so filtering must use it rather than
        # re-deriving a name-prefix subset of those rules.
        return self.include_private or spec.is_public

    def _statement_count(self, body: Any, unit_type: CodeUnitType) -> int:
        """Count the statements or class members inside one unit body.

        :param body: Body node of the unit, or ``None``.
        :param unit_type: Kind of unit the body belongs to.
        :return: Statement count, with nested scopes counted once each.
        """
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
        """Build an extraction diagnostic anchored to one node's line span.

        :param file_path: File the diagnostic refers to.
        :param node: Node whose span locates the diagnostic.
        :param message: Human-readable diagnostic text.
        :param code: Machine-readable diagnostic code.
        :param severity: Diagnostic severity, defaults to ``"warning"``.
        :return: Diagnostic describing the node.
        """
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
        """Parse one file and build its code units and parse diagnostics.

        :param file_path: File to extract.
        :return: Extracted units together with any diagnostics raised while parsing.
        """
        try:
            source = file_path.read_bytes()
        except OSError as exc:
            return BackendResult(
                (),
                (
                    ExtractionDiagnostic(
                        file_path=file_path,
                        language=self.language,
                        message=f"Could not read {file_path}: {exc}",
                        severity="warning",
                        code="read-error",
                    ),
                ),
            )

        diagnostics: list[ExtractionDiagnostic] = []
        # Recall-first: the file is still analyzed after lossy decoding, but the
        # replacement characters reach hashes, identifiers, and embeddings, so the
        # corruption is reported rather than left silent.
        try:
            source.decode("utf-8")
        except UnicodeDecodeError as exc:
            diagnostics.append(
                ExtractionDiagnostic(
                    file_path=file_path,
                    language=self.language,
                    message=(
                        f"{file_path} is not valid UTF-8 ({exc.reason} at byte {exc.start}); "
                        "it was decoded with replacement characters."
                    ),
                    severity="warning",
                    code="invalid-utf8",
                )
            )

        parser = GrammarProvider.parser(self.dialect)
        tree = parser.parse(source)
        root_node = tree.root_node

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

        # A filtered-out private class takes its members with it, matching the
        # Python extractor: emitting them would leak the container's internals
        # under a name whose owner was never reported.
        private_container_spans = [
            _spec_span(spec)
            for spec in deduped.values()
            if spec.unit_type == CodeUnitType.CLASS and not self._include_spec(spec)
        ]

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
            spec_start, spec_end = _spec_span(spec)
            if any(
                start <= spec_start and spec_end <= end for start, end in private_container_spans
            ):
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
                    statement_count=self._statement_count(spec.body, spec.unit_type),
                    structural_hash=structural_hash,
                    token_hash=_token_hash(spec.node, source),
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

    @staticmethod
    def _declarator_name(declarator: Any | None, source: bytes) -> str | None:
        """Find the identifier that a possibly nested C declarator declares.

        The search is an explicit preorder walk: generated sources parenthesize
        declarators deeply enough to blow the Python recursion limit. The
        ``declarator`` field is visited first so ``int (*fp)(int)`` reports the
        pointer name rather than a parameter name.

        :param declarator: Declarator node, or ``None``.
        :param source: Full file source bytes.
        :return: Declared name, or ``None`` when no identifier is reachable.
        """
        if declarator is None:
            return None
        stack = [declarator]
        while stack:
            current = stack.pop()
            if getattr(current, "type", "") in {"identifier", "field_identifier"}:
                return _node_text(source, current)
            nested = _child_by_field(current, "declarator")
            children = [
                child for child in _named_children(current) if not _same_node(child, nested)
            ]
            if nested is not None:
                children.insert(0, nested)
            stack.extend(reversed(children))
        return None

    @staticmethod
    def _has_static_storage_class(node: Any, source: bytes) -> bool:
        """Report whether a C definition declares internal linkage.

        Scanning the raw declaration text instead would misread C99
        ``int dst[static 4]`` parameters and interleaved comments.

        :param node: Function definition node to inspect.
        :param source: Full file source bytes.
        :return: ``True`` when the definition carries a ``static`` specifier.
        """
        return any(
            getattr(child, "type", "") == "storage_class_specifier"
            and _node_text(source, child).strip() == "static"
            for child in _children(node)
        )

    def collect_specs(self, root_node: Any, source: bytes, file_path: Path) -> list[UnitSpec]:
        """Collect C function definitions, skipping prototypes and declarations.

        :param root_node: Root node of the parsed syntax tree.
        :param source: Full file source bytes.
        :param file_path: File being extracted.
        :return: Unit specs for every named, body-bearing function definition.
        """
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
            is_public = not self._has_static_storage_class(node, source)
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
    # Rust's grammar wraps semicolon-terminated expressions and control-flow
    # statements in ``expression_statement``. Counting their inner expression
    # kinds as well would double-count the same statement.
    statement_types = frozenset({"let_declaration", "expression_statement"})
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

    def _statement_count(self, body: Any, unit_type: CodeUnitType) -> int:
        """Count Rust statements, including one semicolon-free tail expression.

        :param body: Rust function body node.
        :param unit_type: Kind of unit the body belongs to.
        :return: Recursive statement count with the tail expression counted once.
        """
        count = super()._statement_count(body, unit_type)
        if body is None or getattr(body, "type", "") != "block":
            return count

        # Comments are named children in tree-sitter-rust, so a trailing comment
        # would otherwise be mistaken for the block's tail expression.
        children = [child for child in _named_children(body) if not _is_comment(child)]
        if not children:
            return count

        tail = children[-1]
        tail_type = str(getattr(tail, "type", ""))
        is_block_item = tail_type.endswith(("_item", "_declaration", "_definition"))
        if (
            tail_type not in self.statement_types
            and tail_type not in self.nested_scope_types
            and not is_block_item
        ):
            count += 1
        return count

    @staticmethod
    def _visibility(node: Any, source: bytes) -> bool:
        """Report whether a Rust item carries a ``pub`` visibility modifier.

        :param node: Item node to inspect.
        :param source: Full file source bytes.
        :return: ``True`` when the item is declared ``pub``.
        """
        for child in _named_children(node):
            if getattr(child, "type", "") == "visibility_modifier":
                return _node_text(source, child).strip().startswith("pub")
        return False

    @staticmethod
    def _trait_impl_ancestor(node: Any) -> Any | None:
        """Find the ``impl Trait for Type`` block that directly contains one item.

        :param node: Item whose ancestor chain is walked.
        :return: Nearest trait-implementation ``impl_item``, or ``None``.
        """
        current = getattr(node, "parent", None)
        while current is not None:
            node_type = getattr(current, "type", "")
            if node_type == "function_item":
                # A helper nested inside a method is local scope, not trait API.
                return None
            if node_type == "impl_item":
                return current if _child_by_field(current, "trait") is not None else None
            current = getattr(current, "parent", None)
        return None

    def _local_trait_visibility(self, root_node: Any, source: bytes) -> dict[str, bool]:
        """Map trait names declared in this file to their ``pub`` visibility.

        Same-named traits in different modules of one file collapse
        recall-first: any public declaration marks the name public.

        :param root_node: Root node of the parsed syntax tree.
        :param source: Full file source bytes.
        :return: Trait name to visibility for every ``trait_item`` in the file.
        """
        visibility: dict[str, bool] = {}
        for candidate in _walk(root_node):
            if getattr(candidate, "type", "") != "trait_item":
                continue
            name_node = _child_by_field(candidate, "name")
            name = _node_text(source, name_node).strip() if name_node is not None else ""
            if not name:
                continue
            visibility[name] = visibility.get(name, False) or self._visibility(candidate, source)
        return visibility

    @staticmethod
    def _impl_trait_public(impl: Any, source: bytes, local_traits: dict[str, bool]) -> bool:
        """Decide the visibility of methods in one ``impl Trait for Type`` block.

        Impl methods cannot legally carry ``pub``: they are reachable through
        the trait, so they default to public. When the trait is a bare name
        declared in this same file, the trait's own visibility gates them
        instead. Path-qualified and unresolved traits stay public: cross-file
        resolution is out of scope, so unknown traits err recall-first.

        :param impl: Trait-implementation ``impl_item`` node.
        :param source: Full file source bytes.
        :param local_traits: Same-file trait visibility from :meth:`_local_trait_visibility`.
        :return: ``True`` when the impl's methods are treated as public.
        """
        base = _child_by_field(impl, "trait")
        if getattr(base, "type", "") == "generic_type":
            base = _first_node(_child_by_field(base, "type"), base)
        if getattr(base, "type", "") != "type_identifier":
            return True
        return local_traits.get(_node_text(source, base).strip(), True)

    @staticmethod
    def _preceding_attributes(node: Any) -> list[Any]:
        """Return the attribute_item siblings stacked directly above one item.

        tree-sitter-rust parses ``#[...]`` as a preceding named sibling of the
        item it annotates, not as a child of that item.

        :param node: Item whose stacked attributes are collected.
        :return: Attribute nodes directly above the item, nearest first.
        """
        if getattr(node, "parent", None) is None:
            return []
        attributes: list[Any] = []
        for sibling in _preceding_named_siblings(node):
            # Comments routinely sit between an attribute and the item it annotates.
            if _is_comment(sibling):
                continue
            if getattr(sibling, "type", "") != "attribute_item":
                break
            attributes.append(sibling)
        return attributes

    @staticmethod
    def _is_test_attribute(attribute: Any, source: bytes) -> bool:
        """Return whether an attribute restricts its item to test builds.

        :param attribute: Rust ``attribute_item`` node.
        :param source: Full file source bytes.
        :return: ``True`` for ``test``, ``cfg(test)``, or ``cfg(all(..., test, ...))``.
        """
        attribute_node = _first_descendant(attribute, {"attribute"})
        named = _named_children(attribute_node) if attribute_node is not None else ()
        if not named:
            return False

        name = _node_text(source, named[0]).strip()
        if name == "test":
            return True
        if name != "cfg":
            return False

        arguments = _child_by_field(attribute_node, "arguments")
        predicates = _named_children(arguments) if arguments is not None else ()
        if len(predicates) == 1:
            return _node_text(source, predicates[0]).strip() == "test"
        if not predicates or _node_text(source, predicates[0]).strip() != "all":
            return False

        all_arguments = next(
            (node for node in predicates[1:] if getattr(node, "type", "") == "token_tree"),
            None,
        )
        if all_arguments is None:
            return False
        return any(
            getattr(node, "type", "") == "identifier" and _node_text(source, node).strip() == "test"
            for node in _named_children(all_arguments)
        )

    @classmethod
    def _is_test_scoped(cls, node: Any, source: bytes) -> bool:
        """Return whether a function lives under ``#[test]`` or ``#[cfg(test)]``.

        Inline Rust test modules share source files with production code, so
        file-glob test exclusion cannot catch them. Matching is narrow: bare
        ``test``, ``cfg(test)``, and ``cfg(all(test, ...))``. ``cfg(not(test))``
        and ``cfg(any(test, ...))`` gate real production configurations and are
        not excluded.

        :param node: Function item to classify.
        :param source: Full file source bytes.
        :return: ``True`` when the item or an enclosing item is test-scoped.
        """
        current = node
        while current is not None:
            for attribute in cls._preceding_attributes(current):
                if cls._is_test_attribute(attribute, source):
                    return True
            current = getattr(current, "parent", None)
        return False

    @staticmethod
    def _context(node: Any, source: bytes) -> tuple[list[str], bool]:
        """Collect the impl, trait, module, and function names enclosing one item.

        :param node: Item whose enclosing scopes are walked.
        :param source: Full file source bytes.
        :return: Outermost-first context names and whether the item is a method.
        """
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
                # ``impl Display for W`` and ``impl Debug for W`` both define
                # ``fmt``, so the trait has to qualify the method or the two
                # units collide under one name. Inherent impls carry no trait
                # field and keep the plain ``W.method`` shape. Appending the
                # trait before the target puts it after the target once
                # ``contexts`` is reversed into outermost-first order.
                trait_node = _child_by_field(current, "trait")
                if trait_node is not None and not _same_node(trait_node, target):
                    trait_text = _clean_name(_node_text(source, trait_node))
                    if trait_text:
                        contexts.append(trait_text)
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
        """Collect Rust free functions and body-bearing impl/trait methods.

        :param root_node: Root node of the parsed syntax tree.
        :param source: Full file source bytes.
        :param file_path: File being extracted.
        :return: Unit specs for every non-test, body-bearing function item.
        """
        prefix = _module_prefix(self.root, file_path, self.language)
        specs: list[UnitSpec] = []
        local_trait_visibility = self._local_trait_visibility(root_node, source)
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
            if self._is_test_scoped(node, source):
                continue
            contexts, is_method = self._context(node, source)
            public = self._visibility(node, source)
            # Trait methods are part of the trait API even when they do not carry
            # an explicit `pub` modifier, but they cannot be more visible than
            # the trait that exposes them.
            trait = _nearest_ancestor(node, {"trait_item", "trait_declaration"})
            if trait is not None:
                public = self._visibility(trait, source)
            else:
                impl = self._trait_impl_ancestor(node)
                if impl is not None:
                    public = self._impl_trait_public(impl, source, local_trait_visibility)
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

    function_declarations = frozenset({"function_declaration", "generator_function_declaration"})
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
        {"method_definition", "field_definition", "public_field_definition", "class_static_block"}
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
        """Read a node's declared name from its name, property, or key field.

        :param node: Node whose declared name is read.
        :param source: Full file source bytes.
        :return: Stable dotted name, or ``None`` when missing or not stable.
        """
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
        """Climb past transparent wrapper expressions around a value.

        :param node: Value node to unwrap.
        :return: Outermost node that still describes the same value.
        """
        current = node
        while getattr(getattr(current, "parent", None), "type", "") in self.transparent_types:
            current = current.parent
        return current

    def _binding_for_value(self, node: Any, source: bytes) -> tuple[str, Any] | None:
        """Resolve the stable name that a value expression is bound to.

        The walk out through enclosing object literals is iterative: generated
        sources nest objects deeply enough to blow the Python recursion limit.

        :param node: Value node, typically a function or class expression.
        :param source: Full file source bytes.
        :return: Bound name with the node carrying the binding, or ``None``.
        """
        keys: list[str] = []
        binding_node: Any = None
        value = node
        while True:
            current = self._unwrap_value(value)
            parent = getattr(current, "parent", None)
            if parent is None:
                return None

            if getattr(parent, "type", "") == "pair" and _same_node(
                _child_by_field(parent, "value"), current
            ):
                key = _child_by_field(parent, "key")
                key_name = _stable_path(_node_text(source, key).strip().strip("'\""))
                object_node = getattr(parent, "parent", None)
                if not key_name or object_node is None:
                    return None
                keys.append(key_name)
                if binding_node is None:
                    binding_node = parent
                value = object_node
                continue

            base = self._direct_binding(current, parent, source)
            if base is None:
                return None
            if not keys:
                return base
            return (".".join([base[0], *reversed(keys)]), binding_node)

    def _direct_binding(self, current: Any, parent: Any, source: bytes) -> tuple[str, Any] | None:
        """Resolve the binding a value takes directly from its immediate parent.

        :param current: Value node, already unwrapped of transparent wrappers.
        :param parent: Immediate parent of ``current``.
        :param source: Full file source bytes.
        :return: Bound name with the node carrying the binding, or ``None``.
        """
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
        """Resolve the name of the class enclosing a node.

        :param node: Node nested inside a class body.
        :param source: Full file source bytes.
        :return: Class name, or ``None`` when there is no named enclosing class.
        """
        class_node = _nearest_ancestor(node, set(self.class_declarations | self.class_expressions))
        if class_node is None:
            return None
        identity = self._class_identity(class_node, source)
        return identity[0] if identity else None

    def _class_identity(self, node: Any, source: bytes) -> tuple[str, Any] | None:
        """Resolve a class's externally stable name and the node carrying it.

        A named class expression has two names: its internal lexical name and
        the binding through which surrounding code reaches it. The binding is
        the stable unit identity when one exists; the internal name remains the
        fallback for an unbound expression. Class declarations keep their
        declared name, except anonymous default exports, whose export binding is
        their only stable identity.

        :param node: Class declaration or expression node.
        :param source: Full file source bytes.
        :return: Stable name and binding node, or ``None`` when neither exists.
        """
        declared_name = self._name_field(node, source)
        if getattr(node, "type", "") in self.class_expressions or not declared_name:
            binding = self._binding_for_value(node, source)
            if binding is not None:
                return binding
        return (declared_name, node) if declared_name else None

    def _contextual_binding(self, node: Any, bound_name: str, source: bytes) -> str:
        """Qualify a stable binding with its enclosing lexical scopes.

        :param node: Node the binding was resolved from.
        :param bound_name: Stable binding name.
        :param source: Full file source bytes.
        :return: Binding qualified by lexical context when that adds information.
        """
        if bound_name.startswith(("exports.", "module.exports")):
            return bound_name
        context = ".".join(self._lexical_context(node, source))
        if not context or bound_name == context or bound_name.startswith(f"{context}."):
            return bound_name
        return f"{context}.{bound_name}"

    def _object_owner(self, node: Any, source: bytes) -> str | None:
        """Resolve the binding path of the object literal directly containing a member.

        Only a *direct* container counts. Reaching for the nearest enclosing
        object literal instead would short-circuit past intervening class and
        function scopes, dropping their segments and collapsing sibling classes
        of one registry literal onto a single qualified name.

        :param node: Member node whose immediate parent may be an object literal.
        :param source: Full file source bytes.
        :return: Dotted binding path of the object literal, or ``None``.
        """
        parent = getattr(node, "parent", None)
        if getattr(parent, "type", "") != "object":
            return None
        binding = self._binding_for_value(parent, source)
        return binding[0] if binding else None

    def _member_container(self, node: Any, source: bytes) -> str:
        """Resolve the dotted container path of one method definition.

        :param node: Method definition node.
        :param source: Full file source bytes.
        :return: Dotted container path, empty when no container is nameable.
        """
        owner = self._object_owner(node, source)
        if owner and owner.startswith(("exports.", "module.exports")):
            return owner
        segments: list[str] = [owner] if owner else []
        for name in reversed(self._lexical_context(node, source)):
            _push_context_segment(segments, name)
        segments.reverse()
        return ".".join(segments)

    def _lexical_context(self, node: Any, source: bytes) -> list[str]:
        """Collect the class, function, method, module, and object-literal names enclosing a node.

        Object literals take part in this one walk rather than competing with
        it: a shorthand method contributes its own name plus the binding path of
        the literal holding it, so intervening class and function scopes still
        add their segments in order. The walk is iterative because generated
        sources nest deeply enough to blow the Python recursion limit.

        :param node: Node whose lexical scope chain is walked.
        :param source: Full file source bytes.
        :return: Outermost-first scope names.
        """
        contexts: list[str] = []
        current = getattr(node, "parent", None)
        while current is not None:
            node_type = getattr(current, "type", "")
            binding_path: str | None = None
            if node_type in self.class_declarations | self.class_expressions:
                identity = self._class_identity(current, source)
                if identity is not None:
                    name, source_node = identity
                    if _same_node(source_node, current):
                        _push_context_segment(contexts, name)
                    else:
                        binding_path = name
            elif node_type in self.function_declarations:
                name = self._name_field(current, source)
                if name:
                    _push_context_segment(contexts, name)
            elif node_type in self.function_expressions:
                binding = self._binding_for_value(current, source)
                binding_path = binding[0] if binding else None
            elif node_type in self.method_types:
                name = self._name_field(current, source)
                if name:
                    _push_context_segment(contexts, name)
                    binding_path = self._object_owner(current, source)
            elif node_type in {"internal_module", "module"}:
                name = self._name_field(current, source)
                if name:
                    _push_context_segment(contexts, name)
            if binding_path:
                _push_context_segment(contexts, binding_path)
                if binding_path.startswith(("exports.", "module.exports")):
                    # A CommonJS export target is already rooted at the module.
                    break
            current = getattr(current, "parent", None)
        contexts.reverse()
        return contexts

    @staticmethod
    def _is_public_member(node: Any, source: bytes, name: str) -> bool:
        """Apply naming and TypeScript accessibility rules to one member.

        :param node: Member node to inspect.
        :param source: Full file source bytes.
        :param name: Member name.
        :return: ``True`` when the member counts as public.
        """
        if name.startswith(("_", "#")):
            return False
        for child in _named_children(node):
            if getattr(child, "type", "") != "accessibility_modifier":
                continue
            if _node_text(source, child).strip() in {"private", "protected"}:
                return False
        return True

    @staticmethod
    def _file_export_names(root_node: Any, source: bytes) -> frozenset[str]:
        """Collect the local names a file exports by reference rather than inline.

        ``export { alpha, beta as b }`` and ``export default alpha`` name units
        declared elsewhere in the file, so those units have no export ancestor of
        their own. Re-exports (``export { x } from "./other"``) name another
        module's units and are ignored.

        :param root_node: Root node of the parsed syntax tree.
        :param source: Full file source bytes.
        :return: Local names the file exports by reference.
        """
        names: set[str] = set()
        for node in _walk(root_node):
            if getattr(node, "type", "") != "export_statement":
                continue
            if _child_by_field(node, "source") is not None:
                continue
            for child in _children(node):
                if getattr(child, "type", "") != "export_clause":
                    continue
                for specifier in _named_children(child):
                    local = _stable_path(_node_text(source, _child_by_field(specifier, "name")))
                    if local:
                        names.add(local)
            value = _child_by_field(node, "value")
            if value is not None and getattr(value, "type", "") == "identifier":
                local = _stable_path(_node_text(source, value))
                if local:
                    names.add(local)
        return frozenset(names)

    @staticmethod
    def _is_exported(
        node: Any,
        source: bytes,
        name: str,
        exported_names: frozenset[str] = frozenset(),
    ) -> bool:
        """Report whether a unit is reachable as a module export.

        :param node: Unit node whose ancestors are walked.
        :param source: Full file source bytes.
        :param name: Unit name relative to the module prefix.
        :param exported_names: Top-level names the file exports by reference.
        :return: ``True`` when the unit is exported or assigned onto ``exports``.
        """
        current = node
        crossed_function_body = False
        while current is not None:
            node_type = getattr(current, "type", "")
            if node_type == "export_statement":
                return True
            if node_type == "statement_block":
                # A function body is an export boundary: units nested inside an
                # exported function are local scope, not module exports. A
                # class_body is deliberately not a boundary so members of an
                # exported class stay exported.
                crossed_function_body = True
                break
            current = getattr(current, "parent", None)
        if not crossed_function_body and name.split(".")[0] in exported_names:
            return True
        return name.startswith(("exports.", "module.exports"))

    def _function_spec(
        self,
        node: Any,
        source: bytes,
        prefix: str,
        exported_names: frozenset[str] = frozenset(),
    ) -> UnitSpec | None:
        """Build a unit spec for one function declaration or function expression.

        :param node: Function node.
        :param source: Full file source bytes.
        :param prefix: Module prefix for qualified names.
        :param exported_names: Top-level names the file exports by reference.
        :return: Unit spec, or ``None`` when the function has no body or stable name.
        """
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

        exported = self._is_exported(
            source_node,
            source,
            qualified_name.removeprefix(prefix + "."),
            exported_names,
        )
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

    def _class_spec(
        self,
        node: Any,
        source: bytes,
        prefix: str,
        exported_names: frozenset[str] = frozenset(),
    ) -> UnitSpec | None:
        """Build a unit spec for one class declaration or class expression.

        :param node: Class node.
        :param source: Full file source bytes.
        :param prefix: Module prefix for qualified names.
        :param exported_names: Top-level names the file exports by reference.
        :return: Unit spec, or ``None`` when the class has no body or stable name.
        """
        node_type = getattr(node, "type", "")
        if _has_ancestor(node, {"ambient_declaration"}):
            return None
        body = _first_node(
            _child_by_field(node, "body"),
            _first_descendant(node, {"class_body"}),
        )
        if body is None:
            return None
        identity = self._class_identity(node, source)
        if identity is None:
            return None
        name, source_node = identity
        if not _same_node(source_node, node):
            bound_name = name
            contextual_name = self._contextual_binding(node, bound_name, source)
            name = bound_name.rsplit(".", 1)[-1]
            qualified_name = _qualified(prefix, contextual_name)
        else:
            qualified_name = _qualified(prefix, *self._lexical_context(node, source), name)
        exported = self._is_exported(
            source_node,
            source,
            qualified_name.removeprefix(prefix + "."),
            exported_names,
        )
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

    def _method_spec(
        self,
        node: Any,
        source: bytes,
        prefix: str,
        exported_names: frozenset[str] = frozenset(),
    ) -> UnitSpec | None:
        """Build a unit spec for one class or object-literal method definition.

        :param node: Method definition node.
        :param source: Full file source bytes.
        :param prefix: Module prefix for qualified names.
        :param exported_names: Top-level names the file exports by reference.
        :return: Unit spec, or ``None`` when the method has no body, name, or container.
        """
        if _has_ancestor(node, {"ambient_declaration"}):
            return None
        body = _child_by_field(node, "body")
        name = self._name_field(node, source)
        if body is None or not name:
            return None
        container = self._member_container(node, source)
        if not container:
            return None
        qualified_name = _qualified(prefix, container, name)
        exported = self._is_exported(node, source, container, exported_names)
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
        """Collect JavaScript/TypeScript function, class, and method units.

        :param root_node: Root node of the parsed syntax tree.
        :param source: Full file source bytes.
        :param file_path: File being extracted.
        :return: Unit specs for every nameable, body-bearing unit in the file.
        """
        prefix = _module_prefix(self.root, file_path, self.language)
        exported_names = self._file_export_names(root_node, source)
        specs: list[UnitSpec] = []
        for node in _walk(root_node):
            node_type = getattr(node, "type", "")
            spec: UnitSpec | None = None
            if node_type in self.function_declarations | self.function_expressions:
                spec = self._function_spec(node, source, prefix, exported_names)
            elif node_type in self.class_declarations | self.class_expressions:
                spec = self._class_spec(node, source, prefix, exported_names)
            elif node_type in self.method_types:
                spec = self._method_spec(node, source, prefix, exported_names)
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
    nested_scope_types = ECMAScriptBackend.nested_scope_types | frozenset(
        {"abstract_class_declaration"}
    )


def create_backend(
    *,
    root: Path,
    language: str,
    dialect: str,
    include_private: bool,
) -> TreeSitterBackend:
    """Create the concrete backend for a canonical language.

    :param root: Extraction root used for qualified naming.
    :param language: Canonical language name.
    :param dialect: Parser dialect to parse files with.
    :param include_private: Whether non-public units are extracted.
    :raises ValueError: If no Tree-sitter backend exists for ``language``.
    :return: Backend instance for the language.
    """
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
