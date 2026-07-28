"""AST-based extraction of code units from Python files."""

from __future__ import annotations

import ast
import copy
import hashlib
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
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

# Sentinel identity seeded as "already proven" in the corpus-wide inheritance graph;
# any class with an edge to this node is a direct ast.NodeVisitor/NodeTransformer subclass.
_CROSS_FILE_AST_VISITOR_ROOT = "<ast-visitor-root>"


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


def _resolve_relative_module(package: str, level: int, module: str | None) -> str | None:
    """Resolve a relative import to an absolute dotted module path.

    :param package: Dotted package containing the importing module (its own name for
        ``__init__.py``, otherwise its parent package).
    :param level: Import level (``0`` for absolute imports, ``1`` for ``from . import``, etc.).
    :param module: Dotted module name following the leading dots, if any.
    :return: Absolute dotted module path, or ``None`` if it escapes the analyzed root.
    """
    if level <= 0:
        return module
    bits = package.rsplit(".", level - 1)
    if len(bits) < level:
        return None
    base = bits[0]
    if module:
        return f"{base}.{module}" if base else module
    return base or None


def _get_cross_file_import_map(
    tree: ast.Module, module_name: str, is_package_init: bool
) -> tuple[dict[str, str], set[str]]:
    """Resolve module-level imports to corpus-qualified identities for base-class lookup.

    Handles ``from pkg.mod import Base [as B]`` (including relative variants) and plain
    ``import pkg.mod`` (matched against ``pkg.mod.Base`` base expressions). Star imports,
    aliased ``import ... as`` module bindings, and imports guarded inside conditionals are
    left unresolved.

    :param tree: Parsed module AST.
    :param module_name: Dotted module name for this file, as returned by module-name resolution.
    :param is_package_init: Whether this file is a package ``__init__.py``.
    :return: Bare-name-to-identity map from ``from`` imports, and dotted module paths bound by
        plain ``import`` statements.
    """
    from_import_map: dict[str, str] = {}
    imported_module_names: set[str] = set()
    package = module_name if is_package_init else module_name.rpartition(".")[0]

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is None:
                    imported_module_names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            resolved_module = _resolve_relative_module(package, node.level, node.module)
            if resolved_module is None:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound_name = alias.asname or alias.name
                from_import_map[bound_name] = f"{resolved_module}.{alias.name}"

    return from_import_map, imported_module_names


def _resolve_base_identity(
    base_name: str, from_import_map: dict[str, str], imported_module_names: set[str]
) -> str | None:
    """Resolve a base-class expression to a corpus-qualified identity via this file's imports.

    :param base_name: Dotted or bare base-class expression from a class definition.
    :param from_import_map: Bare names bound to resolved identities via ``from`` imports.
    :param imported_module_names: Dotted module paths bound via plain ``import`` statements.
    :return: Resolved ``module.ClassName`` identity, or ``None`` if unresolved.
    """
    if base_name in from_import_map:
        return from_import_map[base_name]
    module_part, sep, class_part = base_name.rpartition(".")
    if sep and module_part in imported_module_names:
        return f"{module_part}.{class_part}"
    return None


@dataclass
class _ClassFact:
    """Cross-file inheritance evidence recorded for one emitted class."""

    qualified_name: str
    resolved_base_identities: set[str] = field(default_factory=set)


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
        from_import_map: dict[str, str],
        imported_module_names: set[str],
    ) -> None:
        """Create a collector bound to an extractor and source context.

        :param extractor: Owning code extractor.
        :param file_path: Source file path.
        :param source: Full file source text.
        :param module_name: Deduced module name.
        :param exported: Export names from module-level ``__all__``.
        :param ast_visitor_base_names: Base names resolved from module-level ``ast`` imports.
        :param from_import_map: Bare names bound to resolved identities via ``from`` imports.
        :param imported_module_names: Dotted module paths bound via plain ``import`` statements.
        """
        self.extractor = extractor
        self.file_path = file_path
        self.source_lines = source.splitlines(keepends=True)
        self.module_name = module_name
        self.exported = exported
        self.ast_visitor_base_names = ast_visitor_base_names
        self.from_import_map = from_import_map
        self.imported_module_names = imported_module_names
        self.units: list[CodeUnit] = []
        self.class_facts: list[_ClassFact] = []

        # Scope stacks while walking AST:
        # - class_stack tracks nested class scope.
        # - function_stack tracks nested local function scope.
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []
        self.dynamic_dispatch_stack: list[bool] = []
        self.dynamic_dispatch_class_names: set[str] = set()
        # Most recently defined class per bare name in this file, for same-file base lookups
        # feeding the corpus-wide cross-file inheritance graph (mirrors dynamic_dispatch_class_names'
        # document-order semantics but keeps qualified identities instead of a proven/not flag).
        self.local_class_qualified_by_name: dict[str, str] = {}

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

            qualified_name = self.extractor._qualified_name(
                self.module_name, scope_prefix, node.name
            )
            self.class_facts.append(
                _ClassFact(qualified_name, self._resolve_cross_file_base_identities(base_names))
            )
            self.local_class_qualified_by_name[node.name] = qualified_name

            self.class_stack.append(node.name)
            self.dynamic_dispatch_stack.append(is_dynamic_dispatch_class)
            self.generic_visit(node)
            self.dynamic_dispatch_stack.pop()
            self.class_stack.pop()
        else:
            # If class is excluded, skip descendants to avoid leaking private internals.
            logger.debug("Skipping private class %s in %s", node.name, self.file_path)

    def _resolve_cross_file_base_identities(self, base_names: set[str]) -> set[str]:
        """Resolve base-class expressions to identities usable by the corpus-wide graph.

        :param base_names: Dotted or bare base-class expressions for one class definition.
        :return: Resolved identities: the AST-visitor root sentinel, import-resolved
            ``module.ClassName`` identities, and/or same-file class qualified names.
        """
        resolved: set[str] = set()
        for base_name in base_names:
            if base_name in self.ast_visitor_base_names:
                resolved.add(_CROSS_FILE_AST_VISITOR_ROOT)
                continue
            imported_identity = _resolve_base_identity(
                base_name, self.from_import_map, self.imported_module_names
            )
            if imported_identity is not None:
                resolved.add(imported_identity)
                continue
            if "." not in base_name:
                local_identity = self.local_class_qualified_by_name.get(base_name)
                if local_identity is not None:
                    resolved.add(local_identity)
        return resolved


def _resolve_cross_file_dynamic_dispatch_hooks(
    units: list[CodeUnit], class_facts: list[_ClassFact]
) -> None:
    """Mark ``visit_*`` methods whose class is provably an AST visitor across files.

    Computes the transitive closure of "inherits from ast.NodeVisitor/NodeTransformer" over
    the corpus-wide class graph built from per-file inheritance evidence, then flags any
    not-yet-marked ``visit_*`` method belonging to a class proven only through that closure.
    Unresolvable bases (third-party imports, star imports, dynamic bases) never enter the
    graph, so they stay unproven, same as the existing same-file behavior.

    :param units: All code units collected across the corpus (mutated in place).
    :param class_facts: Per-class base-identity evidence gathered during extraction.
    :return: ``None``.
    """
    edges = {fact.qualified_name: fact.resolved_base_identities for fact in class_facts}
    proven: set[str] = {_CROSS_FILE_AST_VISITOR_ROOT}

    changed = True
    while changed:
        changed = False
        for qualified_name, base_identities in edges.items():
            if qualified_name not in proven and base_identities & proven:
                proven.add(qualified_name)
                changed = True

    if len(proven) <= 1:
        return

    for unit in units:
        if (
            unit.unit_type == CodeUnitType.METHOD
            and not unit.is_dynamic_dispatch_hook
            and unit.name.startswith("visit_")
            and unit.qualified_name.rsplit(".", 1)[0] in proven
        ):
            unit.is_dynamic_dispatch_hook = True


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

    def _collect_file(self, file_path: Path) -> _CodeUnitCollector | None:
        """Parse one file and run the code-unit collector over it.

        :param file_path: Source file to parse.
        :return: Populated collector, or ``None`` if the file was excluded or unparsable.
        """
        if self._should_exclude(file_path):
            logger.debug("Skipping excluded file %s", file_path)
            return None

        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"Could not parse {file_path}: {e}")
            return None

        module_name = self._get_module_name(file_path)
        exported = get_exported_names(tree)
        ast_visitor_base_names = _get_ast_visitor_base_names(tree)
        from_import_map, imported_module_names = _get_cross_file_import_map(
            tree, module_name, is_package_init=file_path.name == "__init__.py"
        )
        collector = _CodeUnitCollector(
            self,
            file_path,
            source,
            module_name,
            exported,
            ast_visitor_base_names,
            from_import_map,
            imported_module_names,
        )
        collector.visit(tree)
        return collector

    def extract_from_file(self, file_path: Path) -> Iterator[CodeUnit]:
        """Yield all code units from a single file.

        Base-class resolution is limited to evidence provable within this one file; it does
        not benefit from the corpus-wide cross-file inheritance pass that ``extract_all``
        performs, since no other files are available here.

        :param file_path: Source file to parse.
        :return: Iterator over discovered code units.
        """
        collector = self._collect_file(file_path)
        if collector is None:
            return
        yield from collector.units

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

        Runs a corpus-wide pass after per-file extraction to resolve base classes across
        files (imports, including relative imports), so ``visit_*`` methods on classes that
        inherit ``ast.NodeVisitor``/``NodeTransformer`` through another module are correctly
        exempted from potentially-unused reporting even when the proof spans multiple files.

        :return: List of extracted code units.
        """
        units: list[CodeUnit] = []
        class_facts: list[_ClassFact] = []
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

                collector = self._collect_file(py_file)
                if collector is None:
                    continue
                units.extend(collector.units)
                class_facts.extend(collector.class_facts)

        _resolve_cross_file_dynamic_dispatch_hooks(units, class_facts)
        return units
