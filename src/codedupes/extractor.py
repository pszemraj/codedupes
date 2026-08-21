"""Language-aware extraction of functions, methods, and classes."""

from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import keyword
import logging
import os
from collections.abc import Iterator
from pathlib import Path

from codedupes.languages.registry import (
    get_backend,
    language_for_path,
    normalize_languages,
    repository_allows_c_headers,
)
from codedupes.models import CodeUnit, CodeUnitType, ExtractionDiagnostic

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    ".tox",
    ".nox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
    ".eggs",
    "build",
    "dist",
    "target",
    "node_modules",
    ".pnpm-store",
    ".yarn",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".gradle",
    ".idea",
    ".vscode",
    ".terraform",
    ".serverless",
    ".aws-sam",
    ".dart_tool",
}

DEFAULT_EXCLUDE_PATTERNS = [
    "**/test_*",
    "**/*_test.*",
    "**/*_tests.*",
    "**/*.test.*",
    "**/*.spec.*",
    "**/tests/**",
    "**/__tests__/**",
]


class NormalizedASTHasher(ast.NodeTransformer):
    """Transform AST into a normalized form for structural comparisons."""

    def __init__(self) -> None:
        """Initialize normalization state."""
        self._var_counter = 0
        self._name_map: dict[str, str] = {}

    def _get_normalized_name(self, name: str) -> str:
        """Return a stable synthetic name for identifier normalization.

        :param name: Original identifier.
        :return: Normalized synthetic name (or original for dunder names).
        """
        if name.startswith("__") and name.endswith("__"):
            return name  # Keep dunder names
        if name not in self._name_map:
            self._name_map[name] = f"_v{self._var_counter}"
            self._var_counter += 1
        return self._name_map[name]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Normalize identifier references in ``Name`` nodes.

        :param node: AST name node to normalize.
        :return: Updated node after generic visit.
        """
        node.id = self._get_normalized_name(node.id)
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        """Normalize function argument names in ``arg`` nodes.

        :param node: AST argument node to normalize.
        :return: Updated argument node after generic visit.
        """
        node.arg = self._get_normalized_name(node.arg)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Normalize function definition metadata and body for hash comparisons.

        :param node: FunctionDef node to normalize.
        :return: Updated function definition node after generic visit.
        """
        node.name = self._get_normalized_name(node.name)
        # Remove docstring
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        """Normalize async function definition metadata and body.

        :param node: AsyncFunctionDef node to normalize.
        :return: Updated function definition node after generic visit.
        """
        node.name = self._get_normalized_name(node.name)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        """Normalize class definition metadata and body.

        :param node: ClassDef node to normalize.
        :return: Updated class node after generic visit.
        """
        node.name = self._get_normalized_name(node.name)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        """Normalize string constants for structural comparison.

        :param node: Constant AST node.
        :return: Updated constant node with string values replaced by ``<STR>``.
        """
        # Normalize string constants (but not numeric)
        if isinstance(node.value, str):
            node.value = "<STR>"
        return node


class CallGraphVisitor(ast.NodeVisitor):
    """Extract function/method calls from an AST node."""

    def __init__(self) -> None:
        """Initialize a fresh call graph accumulator."""
        self.calls: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        """Collect direct and attribute call targets from ``Call`` nodes.

        :param node: AST call node.
        :return: ``None``.
        """
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # self.method() or obj.method()
            self.calls.add(node.func.attr)
            # Also track the full chain if it's a simple attribute access
            if isinstance(node.func.value, ast.Name):
                self.calls.add(f"{node.func.value.id}.{node.func.attr}")
        self.generic_visit(node)


class _CodeUnitCollector(ast.NodeVisitor):
    """Collect code units with deterministic scope tracking."""

    def __init__(
        self,
        extractor: CodeExtractor,
        file_path: Path,
        source_map: _PythonSourceMap,
        module_name: str,
        exported: set[str],
    ) -> None:
        """Create a collector bound to an extractor and source context.

        :param extractor: Owning code extractor.
        :param file_path: Source file path.
        :param source_map: Precomputed per-file line/byte tables.
        :param module_name: Deduced module name.
        :param exported: Export names from module-level ``__all__``.
        """
        self.extractor = extractor
        self.file_path = file_path
        self.source_map = source_map
        self.module_name = module_name
        self.exported = exported
        self.units: list[CodeUnit] = []

        # Scope stacks while walking AST:
        # - class_stack tracks nested class scope.
        # - function_stack tracks nested local function scope.
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Collect function code units and recurse into nested definitions."""
        is_method = bool(self.class_stack) and not self.function_stack
        scope_prefix = self.class_stack + self.function_stack

        if self.extractor._should_emit_function(node.name):
            self.units.extend(
                self.extractor._emit_function(
                    node,
                    self.file_path,
                    self.source_map,
                    self.module_name,
                    scope_prefix=scope_prefix,
                    class_member=is_method,
                    exported=self.exported,
                )
            )

        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Collect async functions using the same logic as normal functions."""
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Collect class units and descend into exported/visible class bodies."""
        scope_prefix = self.class_stack
        should_enter = self.extractor._should_emit_class(node.name)

        if should_enter:
            self.units.extend(
                self.extractor._emit_class(
                    node,
                    self.file_path,
                    self.source_map,
                    self.module_name,
                    scope_prefix=scope_prefix,
                    exported=self.exported,
                )
            )

            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()
        else:
            # If class is excluded, skip descendants to avoid leaking private internals.
            logger.debug(f"Skipping private class {node.name} in {self.file_path}")


def compute_ast_hash(node: ast.AST) -> str:
    """Compute a hash of the normalized AST structure.

    :param node: AST node to hash.
    :return: Stable short hash string.
    """
    hasher = NormalizedASTHasher()
    normalized = hasher.visit(copy.deepcopy(node))
    structure = ast.dump(normalized, annotate_fields=False)
    return hashlib.sha256(structure.encode()).hexdigest()[:16]


def compute_token_hash(source: str) -> str:
    """
    Compute hash based on tokenized source (ignoring whitespace/comments).
    Simpler than AST but catches reformatted duplicates.

    :param source: Source snippet.
    :return: Token-based short hash string.
    """
    import tokenize
    from io import StringIO

    tokens: list[tuple[int, str]] = []
    try:
        for tok in tokenize.generate_tokens(StringIO(source).readline):
            if tok.type not in (
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENCODING,
            ):
                tokens.append((tok.type, tok.string))
    except Exception:  # noqa: BLE001 - tokenize raises assorted parsing errors
        # Fall back to simple normalization
        tokens = [(0, w) for w in source.split()]

    return hashlib.sha256(str(tokens).encode()).hexdigest()[:16]


def extract_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str | None:
    """Extract docstring from a function or class node.

    :param node: AST node to inspect.
    :return: Leading docstring if present, otherwise ``None``.
    """
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        return node.body[0].value.value
    return None


def get_exported_names(tree: ast.Module) -> set[str]:
    """Extract names from ``__all__`` if present.

    :param tree: Parsed module AST.
    :return: Set of exported names.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "__all__"
                    and isinstance(node.value, (ast.List, ast.Tuple))
                ):
                    return {
                        elt.value
                        for elt in node.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    }
    return set()


class _PythonStatementCounter(ast.NodeVisitor):
    """Count executable Python statements without descending into nested scopes."""

    def __init__(self) -> None:
        """Start the visitor with an empty statement count."""
        self.count = 0

    def generic_visit(self, node: ast.AST) -> None:
        """Count ``node`` when it is a statement, then visit its children.

        :param node: AST node being visited.
        """
        if isinstance(node, ast.stmt):
            self.count += 1
        super().generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Count a nested function as one statement without entering its body.

        :param node: Nested function definition node.
        """
        self.count += 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Count a nested async function as one statement without entering its body.

        :param node: Nested async function definition node.
        """
        self.count += 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Count a nested class as one statement without entering its body.

        :param node: Nested class definition node.
        """
        self.count += 1


def _count_python_statements(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> int:
    """Count the executable statements in one definition, ignoring its docstring.

    :param node: Definition node whose body is counted.
    :return: Statement count, with nested scopes counted once each.
    """
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    counter = _PythonStatementCounter()
    for statement in body:
        counter.visit(statement)
    return counter.count


def _python_identifiers(node: ast.AST) -> frozenset[str]:
    """Collect identifier names bound or referenced under one AST subtree.

    :param node: AST subtree to scan.
    :return: Identifier names excluding Python keywords and builtins.
    """
    ignored = set(keyword.kwlist) | set(dir(builtins))
    identifiers: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            identifiers.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifiers.add(child.name)
        elif isinstance(child, ast.arg):
            identifiers.add(child.arg)
    return frozenset(
        identifier
        for identifier in identifiers
        if identifier and identifier not in ignored and not identifier.isdigit()
    )


class _PythonSourceMap:
    """Per-file line and byte-offset tables shared by every emitted unit.

    Python extraction returns complete source lines, including indentation and
    the final line ending; the byte range describes those exact bytes rather
    than the narrower AST column span. Splitting and encoding happen once per
    file so per-unit work stays O(unit size) instead of O(file size).
    """

    def __init__(self, source: str) -> None:
        """Build line tables for one file's source text.

        :param source: Entire file source text.
        """
        self.lines = source.splitlines(keepends=True)
        self._encoded_lines = source.encode("utf-8").splitlines(keepends=True)
        self._line_start_bytes = [0]
        for line in self._encoded_lines:
            self._line_start_bytes.append(self._line_start_bytes[-1] + len(line))

    def snippet(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
        """Return the complete-line source snippet for one definition node.

        :param node: Definition node with line span metadata.
        :return: Source lines covering the node, endings included.
        """
        return "".join(self.lines[node.lineno - 1 : node.end_lineno])

    def byte_range(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    ) -> tuple[int, int, int, int]:
        """Return ``(start_byte, end_byte, start_column, end_column)`` for a node.

        :param node: Definition node with line span metadata.
        :return: Byte range of the emitted snippet plus column bounds.
        """
        line_count = len(self._encoded_lines)
        start_line_index = max(0, node.lineno - 1)
        end_line_index = max(start_line_index, (node.end_lineno or node.lineno) - 1)
        start_byte = self._line_start_bytes[min(start_line_index, line_count)]
        end_byte = self._line_start_bytes[min(end_line_index + 1, line_count)]
        final_line = self._encoded_lines[end_line_index] if end_line_index < line_count else b""
        end_column = len(final_line.rstrip(b"\r\n"))
        return start_byte, end_byte, 0, end_column


class CodeExtractor:
    """Extract supported code units from a source tree or individual file."""

    def __init__(
        self,
        root: Path,
        exclude_patterns: list[str] | None = None,
        include_private: bool = True,
        include_stubs: bool = False,
        languages: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        """Construct an extractor for a project root.

        :param root: Root path to scan.
        :param exclude_patterns: Optional path glob patterns.
        :param include_private: Include private names when true.
        :param include_stubs: Include ``.pyi`` files.
        :param languages: Optional canonical/alias language filter. Auto-detects
            supported source files when omitted.
        """
        self.root = root.resolve()
        self.exclude_patterns = exclude_patterns or DEFAULT_EXCLUDE_PATTERNS.copy()
        self.include_private = include_private
        self.include_stubs = include_stubs
        self.languages = normalize_languages(languages)
        self.diagnostics: list[ExtractionDiagnostic] = []
        self._c_headers_allowed: bool | None = None

    @staticmethod
    def _is_excluded_dir_name(name: str) -> bool:
        """Return ``True`` when a directory name should be skipped by default.

        :param name: Directory name.
        :return: Whether the directory is excluded.
        """
        return name in DEFAULT_EXCLUDE_DIR_NAMES or name.endswith(".egg-info")

    def _should_exclude(self, path: Path) -> bool:
        """Check if path matches any exclude pattern.

        :param path: Candidate path.
        :return: ``True`` when extraction should skip this file.
        """
        from fnmatch import fnmatch

        rel = path.relative_to(self.root)
        if any(self._is_excluded_dir_name(part) for part in rel.parts[:-1]):
            return True

        rel_path = str(rel)
        for pattern in self.exclude_patterns:
            if fnmatch(rel_path, pattern):
                return True
            # ``fnmatch`` does not treat ``**/name.py`` as matching root-level ``name.py``.
            if pattern.startswith("**/") and fnmatch(rel_path, pattern[3:]):
                return True
        return False

    def _get_module_name(self, file_path: Path) -> str:
        """Convert file path to dotted module name.

        :param file_path: File path under the configured root.
        :return: Dotted module name.
        """
        rel = file_path.relative_to(self.root)
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = Path(parts[-1]).stem
        return ".".join(parts) if parts else ""

    def _allow_c_headers(self) -> bool:
        """Resolve the repository-level C-header ambiguity policy once.

        :return: ``True`` when ambiguous ``.h`` files may be parsed as C.
        """
        if self._c_headers_allowed is None:
            self._c_headers_allowed = repository_allows_c_headers(self.root, self.languages)
        return self._c_headers_allowed

    def extract_from_file(self, file_path: Path) -> Iterator[CodeUnit]:
        """Yield all supported code units from a single file.

        Python keeps the CPython AST backend. C, Rust, JavaScript, and TypeScript
        are routed to pinned Tree-sitter grammar packages. Missing grammars are a
        hard configuration error; codedupes never silently falls back to line
        chunking.

        :param file_path: File to extract code units from.
        :return: Iterator over the code units found in the file.
        """
        # Normalize relative caller paths, but keep the in-tree name for files
        # that are symlinks to targets outside the root: exclusion and module
        # naming are computed relative to the root, and the symlink is the
        # file's identity within the analyzed tree.
        resolved = file_path.resolve()
        if resolved.is_relative_to(self.root):
            file_path = resolved
        if self._should_exclude(file_path):
            logger.debug(f"Skipping excluded file {file_path}")
            return

        selection = language_for_path(
            file_path,
            include_stubs=self.include_stubs or file_path.suffix.lower() == ".pyi",
            selected_languages=self.languages,
            allow_c_header=self._allow_c_headers(),
        )
        if selection is None:
            return

        if selection.language != "python":
            backend = get_backend(
                root=self.root,
                selection=selection,
                include_private=self.include_private,
            )
            result = backend.extract_file(file_path)
            self.diagnostics.extend(result.diagnostics)
            yield from result.units
            return

        yield from self._extract_python_from_file(file_path)

    def _extract_python_from_file(self, file_path: Path) -> Iterator[CodeUnit]:
        """Yield Python units using the original CPython AST implementation.

        :param file_path: Python file to parse.
        :return: Iterator over the code units found in the file.
        """
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            message = f"Could not parse {file_path}: {exc}"
            logger.warning(message)
            self.diagnostics.append(
                ExtractionDiagnostic(
                    file_path=file_path,
                    language="python",
                    message=str(exc),
                    severity="warning",
                    code="parse-error",
                    lineno=getattr(exc, "lineno", None),
                    end_lineno=getattr(exc, "end_lineno", None),
                )
            )
            return

        module_name = self._get_module_name(file_path)
        exported = get_exported_names(tree)
        source_map = _PythonSourceMap(source)
        visitor = _CodeUnitCollector(self, file_path, source_map, module_name, exported)
        visitor.visit(tree)
        yield from visitor.units

    def _should_emit_function(self, name: str) -> bool:
        """Respect private-function filtering.

        :param name: Function name.
        :return: Whether to emit this function.
        """
        if self._is_private_name(name):
            return self.include_private
        return True

    @staticmethod
    def _is_private_name(name: str) -> bool:
        """Return whether a symbol name is private by convention.

        :param name: Name to classify.
        :return: ``True`` for names starting with single underscore.
        """
        return name.startswith("_") and not name.startswith("__")

    def _should_emit_class(self, name: str) -> bool:
        """Respect private-class filtering.

        :param name: Class name.
        :return: Whether to emit this class.
        """
        if self.include_private:
            return True
        return not self._is_private_name(name)

    def _qualified_name(
        self,
        module_name: str,
        scope_prefix: list[str],
        name: str,
    ) -> str:
        """Construct a dotted qualified symbol name.

        :param module_name: Module path prefix.
        :param scope_prefix: Nested class/function scope.
        :param name: Symbol name.
        :return: Qualified symbol name.
        """
        parts = [part for part in scope_prefix if part]
        if module_name:
            parts.insert(0, module_name)
        parts.append(name)
        return ".".join(parts)

    def _emit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: Path,
        source_map: _PythonSourceMap,
        module_name: str,
        scope_prefix: list[str],
        class_member: bool,
        exported: set[str],
    ) -> Iterator[CodeUnit]:
        """Emit one or more code units for a function node.

        :param node: Function or async function AST node.
        :param file_path: Source file path.
        :param source_map: Precomputed per-file line/byte tables.
        :param module_name: Module name.
        :param scope_prefix: Scope prefix stack.
        :param class_member: Whether node is a method.
        :param exported: Exported names from module __all__.
        :return: Iterator of constructed ``CodeUnit`` instances.
        """
        name = node.name
        qualified = self._qualified_name(module_name, scope_prefix, name)
        unit_type = CodeUnitType.METHOD if class_member else CodeUnitType.FUNCTION

        func_source = source_map.snippet(node)

        # Build call graph
        call_visitor = CallGraphVisitor()
        call_visitor.visit(node)

        ast_hash = compute_ast_hash(node)
        token_hash = compute_token_hash(func_source)
        start_byte, end_byte, start_column, end_column = source_map.byte_range(node)

        yield CodeUnit(
            name=name,
            qualified_name=qualified,
            unit_type=unit_type,
            file_path=file_path,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            source=func_source,
            docstring=extract_docstring(node),
            language="python",
            dialect="python",
            native_kind=type(node).__name__,
            start_byte=start_byte,
            end_byte=end_byte,
            start_column=start_column,
            end_column=end_column,
            statement_count=_count_python_statements(node),
            identifiers=_python_identifiers(node),
            calls=call_visitor.calls,
            is_public=not name.startswith("_"),
            is_dunder=name.startswith("__") and name.endswith("__"),
            is_exported=name in exported,
            structural_hash=ast_hash,
            token_hash=token_hash,
        )

    def _emit_class(
        self,
        node: ast.ClassDef,
        file_path: Path,
        source_map: _PythonSourceMap,
        module_name: str,
        scope_prefix: list[str],
        exported: set[str],
    ) -> Iterator[CodeUnit]:
        """Emit a class code unit with metadata and hashes.

        :param node: Class AST node.
        :param file_path: Source file path.
        :param source_map: Precomputed per-file line/byte tables.
        :param module_name: Module name.
        :param scope_prefix: Scope prefix stack.
        :param exported: Exported names from module __all__.
        :return: Iterator over emitted class ``CodeUnit`` values.
        """
        class_name = node.name
        qualified = self._qualified_name(module_name, scope_prefix, class_name)
        class_source = source_map.snippet(node)
        ast_hash = compute_ast_hash(node)
        token_hash = compute_token_hash(class_source)
        start_byte, end_byte, start_column, end_column = source_map.byte_range(node)

        yield CodeUnit(
            name=class_name,
            qualified_name=qualified,
            unit_type=CodeUnitType.CLASS,
            file_path=file_path,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            source=class_source,
            docstring=extract_docstring(node),
            language="python",
            dialect="python",
            native_kind=type(node).__name__,
            start_byte=start_byte,
            end_byte=end_byte,
            start_column=start_column,
            end_column=end_column,
            statement_count=_count_python_statements(node),
            identifiers=_python_identifiers(node),
            is_public=not class_name.startswith("_"),
            is_dunder=False,
            is_exported=class_name in exported,
            structural_hash=ast_hash,
            token_hash=token_hash,
        )

    def extract_all(self) -> list[CodeUnit]:
        """Extract all supported code units from the configured directory tree.

        :return: Every code unit extracted from the tree, in walk order.
        """
        units: list[CodeUnit] = []
        seen: set[Path] = set()
        allow_c_header = self._allow_c_headers()

        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            dirnames[:] = [name for name in dirnames if not self._is_excluded_dir_name(name)]
            current_dir = Path(dirpath)

            for filename in filenames:
                source_file = current_dir / filename
                selection = language_for_path(
                    source_file,
                    include_stubs=self.include_stubs,
                    selected_languages=self.languages,
                    allow_c_header=allow_c_header,
                )
                if selection is None:
                    continue

                try:
                    resolved = source_file.resolve()
                except OSError:
                    resolved = source_file
                if resolved in seen:
                    continue
                seen.add(resolved)

                if self._should_exclude(source_file):
                    continue

                units.extend(self.extract_from_file(source_file))

        return units
