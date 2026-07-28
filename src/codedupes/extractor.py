"""AST-based extraction of code units from Python files."""

from __future__ import annotations

import ast
import copy
import hashlib
import logging
import os
from collections.abc import Iterator
from pathlib import Path

from codedupes.models import CodeUnit, CodeUnitType

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
    "**/*_test.py",
    "**/tests/**",
]

DefinitionNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
AST_VISITOR_CLASS_NAMES = {"NodeVisitor", "NodeTransformer"}


def extract_docstring(node: DefinitionNode) -> str | None:
    """Extract an unmodified leading docstring from a definition.

    :param node: Function or class node to inspect.
    :return: Leading docstring if present, otherwise ``None``.
    """
    return ast.get_docstring(node, clean=False)


def _remove_leading_docstring(node: DefinitionNode) -> None:
    """Remove one leading docstring expression from a definition body.

    :param node: Function or class node to modify.
    :return: ``None``.
    """
    if extract_docstring(node) is not None:
        node.body = node.body[1:]


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
        return self._visit_definition(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        """Normalize async function definition metadata and body.

        :param node: AsyncFunctionDef node to normalize.
        :return: Updated function definition node after generic visit.
        """
        return self._visit_definition(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        """Normalize class definition metadata and body.

        :param node: ClassDef node to normalize.
        :return: Updated class node after generic visit.
        """
        return self._visit_definition(node)

    def _visit_definition(self, node: DefinitionNode) -> ast.AST:
        """Normalize one function or class definition.

        :param node: Definition node to normalize.
        :return: Updated definition after recursively visiting its body.
        """
        node.name = self._get_normalized_name(node.name)
        _remove_leading_docstring(node)
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


def _dotted_expression_name(node: ast.expr) -> str | None:
    """Return a dotted name for a simple class-base expression.

    :param node: Base-class expression.
    :return: Dotted name for ``Name``/``Attribute`` expressions, otherwise ``None``.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_expression_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _get_ast_visitor_base_names(tree: ast.Module) -> set[str]:
    """Return names bound to AST visitor classes by module-level imports.

    :param tree: Parsed module AST.
    :return: Base expressions proven to resolve to visitor classes from ``ast``.
    """
    base_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ast":
                    qualifier = alias.asname or alias.name
                    base_names.update(
                        f"{qualifier}.{class_name}" for class_name in AST_VISITOR_CLASS_NAMES
                    )
        elif isinstance(node, ast.ImportFrom) and node.module == "ast":
            for alias in node.names:
                if alias.name == "*":
                    base_names.update(AST_VISITOR_CLASS_NAMES)
                elif alias.name in AST_VISITOR_CLASS_NAMES:
                    base_names.add(alias.asname or alias.name)
    return base_names


class _CodeUnitCollector(ast.NodeVisitor):
    """Collect code units with deterministic scope tracking."""

    def __init__(
        self,
        extractor: CodeExtractor,
        file_path: Path,
        source: str,
        module_name: str,
        exported: set[str],
        ast_visitor_base_names: set[str],
    ) -> None:
        """Create a collector bound to an extractor and source context.

        :param extractor: Owning code extractor.
        :param file_path: Source file path.
        :param source: Full file source text.
        :param module_name: Deduced module name.
        :param exported: Export names from module-level ``__all__``.
        :param ast_visitor_base_names: Base names resolved from module-level ``ast`` imports.
        """
        self.extractor = extractor
        self.file_path = file_path
        self.source_lines = source.splitlines(keepends=True)
        self.module_name = module_name
        self.exported = exported
        self.ast_visitor_base_names = ast_visitor_base_names
        self.units: list[CodeUnit] = []

        # Scope stacks while walking AST:
        # - class_stack tracks nested class scope.
        # - function_stack tracks nested local function scope.
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []
        self.dynamic_dispatch_stack: list[bool] = []
        self.dynamic_dispatch_class_names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Collect function code units and recurse into nested definitions."""
        is_method = bool(self.class_stack) and not self.function_stack
        scope_prefix = self.class_stack + self.function_stack

        if self.extractor._should_emit_name(node.name):
            self.units.append(
                self.extractor._emit_function(
                    node,
                    self.file_path,
                    self.source_lines,
                    self.module_name,
                    scope_prefix=scope_prefix,
                    class_member=is_method,
                    is_dynamic_dispatch_class=(
                        self.dynamic_dispatch_stack[-1] if is_method else False
                    ),
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
        should_enter = self.extractor._should_emit_name(node.name)

        if should_enter:
            self.units.append(
                self.extractor._emit_class(
                    node,
                    self.file_path,
                    self.source_lines,
                    self.module_name,
                    scope_prefix=scope_prefix,
                    exported=self.exported,
                )
            )

            base_names = {
                base_name
                for base in node.bases
                if (base_name := _dotted_expression_name(base)) is not None
            }
            is_dynamic_dispatch_class = any(
                base_name in self.ast_visitor_base_names
                or base_name in self.dynamic_dispatch_class_names
                for base_name in base_names
            )
            if is_dynamic_dispatch_class:
                self.dynamic_dispatch_class_names.add(node.name)

            self.class_stack.append(node.name)
            self.dynamic_dispatch_stack.append(is_dynamic_dispatch_class)
            self.generic_visit(node)
            self.dynamic_dispatch_stack.pop()
            self.class_stack.pop()
        else:
            # If class is excluded, skip descendants to avoid leaking private internals.
            logger.debug("Skipping private class %s in %s", node.name, self.file_path)


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
    except Exception:  # noqa: BLE001 - arbitrary source can fail tokenize in many ways
        # Fall back to simple normalization
        tokens = [(0, w) for w in source.split()]

    return hashlib.sha256(str(tokens).encode()).hexdigest()[:16]


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


class CodeExtractor:
    """Extract all code units from a directory of Python files."""

    def __init__(
        self,
        root: Path,
        exclude_patterns: list[str] | None = None,
        include_private: bool = True,
        include_stubs: bool = False,
    ) -> None:
        """Construct an extractor for a project root.

        :param root: Root path to scan.
        :param exclude_patterns: Optional path glob patterns.
        :param include_private: Include private names when true.
        :param include_stubs: Include ``.pyi`` files.
        """
        self.root = root.resolve()
        self.exclude_patterns = exclude_patterns or DEFAULT_EXCLUDE_PATTERNS.copy()
        self.include_private = include_private
        self.include_stubs = include_stubs

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

    def extract_from_file(self, file_path: Path) -> Iterator[CodeUnit]:
        """Yield all code units from a single file.

        :param file_path: Source file to parse.
        :return: Iterator over discovered code units.
        """
        if self._should_exclude(file_path):
            logger.debug("Skipping excluded file %s", file_path)
            return

        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"Could not parse {file_path}: {e}")
            return

        module_name = self._get_module_name(file_path)
        exported = get_exported_names(tree)
        ast_visitor_base_names = _get_ast_visitor_base_names(tree)
        visitor = _CodeUnitCollector(
            self,
            file_path,
            source,
            module_name,
            exported,
            ast_visitor_base_names,
        )
        visitor.visit(tree)
        yield from visitor.units

    def _should_emit_name(self, name: str) -> bool:
        """Respect private symbol filtering.

        :param name: Function or class name.
        :return: Whether to emit this symbol.
        """
        return self.include_private or not self._is_private_name(name)

    @staticmethod
    def _is_private_name(name: str) -> bool:
        """Return whether a symbol name is private by convention.

        :param name: Name to classify.
        :return: ``True`` for names starting with single underscore.
        """
        return name.startswith("_") and not name.startswith("__")

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
        source_lines: list[str],
        module_name: str,
        scope_prefix: list[str],
        class_member: bool,
        is_dynamic_dispatch_class: bool,
        exported: set[str],
    ) -> CodeUnit:
        """Build one function or method code unit.

        :param node: Function or async function AST node.
        :param file_path: Source file path.
        :param source_lines: Entire file source split with line endings.
        :param module_name: Module name.
        :param scope_prefix: Scope prefix stack.
        :param class_member: Whether node is a method.
        :param is_dynamic_dispatch_class: Whether the containing class uses AST visitor dispatch.
        :param exported: Exported names from module __all__.
        :return: Constructed function or method unit.
        """
        unit_type = CodeUnitType.METHOD if class_member else CodeUnitType.FUNCTION

        call_visitor = CallGraphVisitor()
        call_visitor.visit(node)

        return self._build_code_unit(
            node,
            file_path,
            source_lines,
            module_name,
            scope_prefix,
            unit_type=unit_type,
            calls=call_visitor.calls,
            exported=exported,
            dynamic_dispatch_hook=(
                class_member and is_dynamic_dispatch_class and node.name.startswith("visit_")
            ),
        )

    def _emit_class(
        self,
        node: ast.ClassDef,
        file_path: Path,
        source_lines: list[str],
        module_name: str,
        scope_prefix: list[str],
        exported: set[str],
    ) -> CodeUnit:
        """Build one class code unit.

        :param node: Class AST node.
        :param file_path: Source file path.
        :param source_lines: Entire file source split with line endings.
        :param module_name: Module name.
        :param scope_prefix: Scope prefix stack.
        :param exported: Exported names from module __all__.
        :return: Constructed class unit.
        """
        return self._build_code_unit(
            node,
            file_path=file_path,
            source_lines=source_lines,
            module_name=module_name,
            scope_prefix=scope_prefix,
            unit_type=CodeUnitType.CLASS,
            calls=set(),
            exported=exported,
            dynamic_dispatch_hook=False,
        )

    def _build_code_unit(
        self,
        node: DefinitionNode,
        file_path: Path,
        source_lines: list[str],
        module_name: str,
        scope_prefix: list[str],
        unit_type: CodeUnitType,
        calls: set[str],
        exported: set[str],
        dynamic_dispatch_hook: bool,
    ) -> CodeUnit:
        """Build shared source and metadata fields for one code unit.

        :param node: Function or class definition node.
        :param file_path: Source file path.
        :param source_lines: Entire file source split with line endings.
        :param module_name: Module name.
        :param scope_prefix: Scope prefix stack.
        :param unit_type: Emitted unit type.
        :param calls: Direct call targets found in the definition.
        :param exported: Exported names from module ``__all__``.
        :param dynamic_dispatch_hook: Whether runtime visitor dispatch reaches this method.
        :return: Constructed code unit.
        """
        name = node.name
        unit_source = "".join(source_lines[node.lineno - 1 : node.end_lineno])
        return CodeUnit(
            name=name,
            qualified_name=self._qualified_name(module_name, scope_prefix, name),
            unit_type=unit_type,
            file_path=file_path,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            source=unit_source,
            docstring=extract_docstring(node),
            calls=calls,
            is_public=not name.startswith("_"),
            is_dunder=unit_type != CodeUnitType.CLASS
            and name.startswith("__")
            and name.endswith("__"),
            is_exported=name in exported,
            is_dynamic_dispatch_hook=dynamic_dispatch_hook,
            _ast_hash=compute_ast_hash(node),
            _token_hash=compute_token_hash(unit_source),
        )

    def extract_all(self) -> list[CodeUnit]:
        """Extract all code units from the configured directory tree.

        :return: List of extracted code units.
        """
        units: list[CodeUnit] = []
        valid_suffixes = {".py"}
        if self.include_stubs:
            valid_suffixes.add(".pyi")

        seen: set[Path] = set()
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            dirnames[:] = [name for name in dirnames if not self._is_excluded_dir_name(name)]
            current_dir = Path(dirpath)

            for filename in filenames:
                if Path(filename).suffix not in valid_suffixes:
                    continue

                py_file = current_dir / filename
                try:
                    resolved = py_file.resolve()
                except OSError:
                    resolved = py_file
                if resolved in seen:
                    continue
                seen.add(resolved)

                units.extend(self.extract_from_file(py_file))

        return units
