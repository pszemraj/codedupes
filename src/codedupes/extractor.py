"""AST-based extraction of code units from Python files."""

from __future__ import annotations

import ast
import hashlib
import logging
import copy
from pathlib import Path
from typing import Iterator

from .models import CodeUnit, CodeUnitType

logger = logging.getLogger(__name__)


class NormalizedASTHasher(ast.NodeTransformer):
    """Transform AST into a normalized form for structural comparisons."""

    def __init__(self) -> None:
        self._var_counter = 0
        self._name_map: dict[str, str] = {}

    def _get_normalized_name(self, name: str) -> str:
        if name.startswith("__") and name.endswith("__"):
            return name  # Keep dunder names
        if name not in self._name_map:
            self._name_map[name] = f"_v{self._var_counter}"
            self._var_counter += 1
        return self._name_map[name]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._get_normalized_name(node.id)
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._get_normalized_name(node.arg)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
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
        # Normalize string constants (but not numeric)
        if isinstance(node.value, str):
            node.value = "<STR>"
        return node


class CallGraphVisitor(ast.NodeVisitor):
    """Extract function/method calls from an AST node."""

    def __init__(self) -> None:
        self.calls: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
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
        extractor: "CodeExtractor",
        file_path: Path,
        source: str,
        module_name: str,
        exported: set[str],
    ) -> None:
        self.extractor = extractor
        self.file_path = file_path
        self.source = source
        self.module_name = module_name
        self.exported = exported
        self.units: list[CodeUnit] = []

        # Scope stacks while walking AST:
        # - class_stack tracks nested class scope.
        # - function_stack tracks nested local function scope.
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        is_method = bool(self.class_stack) and not self.function_stack
        scope_prefix = self.class_stack + self.function_stack

        if self.extractor._should_emit_function(node.name, is_method=is_method):
            self.units.extend(
                self.extractor._emit_function(
                    node,
                    self.file_path,
                    self.source,
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
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        scope_prefix = self.class_stack
        should_enter = self.extractor._should_emit_class(node.name)

        if should_enter:
            self.units.extend(
                self.extractor._emit_class(
                    node,
                    self.file_path,
                    self.source,
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
            logger.debug("Skipping private class %s in %s", node.name, self.file_path)


def compute_ast_hash(node: ast.AST) -> str:
    """Compute a hash of the normalized AST structure."""
    hasher = NormalizedASTHasher()
    normalized = hasher.visit(copy.deepcopy(node))
    structure = ast.dump(normalized, annotate_fields=False)
    return hashlib.sha256(structure.encode()).hexdigest()[:16]


def compute_token_hash(source: str) -> str:
    """
    Compute hash based on tokenized source (ignoring whitespace/comments).
    Simpler than AST but catches reformatted duplicates.
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
    except Exception:  # tokenize.TokenizeError and other parsing errors
        # Fall back to simple normalization
        tokens = [(0, w) for w in source.split()]

    return hashlib.sha256(str(tokens).encode()).hexdigest()[:16]


def extract_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str | None:
    """Extract docstring from a function or class node."""
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        return node.body[0].value.value
    return None


def get_exported_names(tree: ast.Module) -> set[str]:
    """Extract names from __all__ if present."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
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
    ) -> None:
        self.root = root.resolve()
        self.exclude_patterns = exclude_patterns or ["**/test_*", "**/*_test.py", "**/tests/**"]
        self.include_private = include_private

    def _should_exclude(self, path: Path) -> bool:
        """Check if path matches any exclude pattern."""
        from fnmatch import fnmatch

        rel_path = str(path.relative_to(self.root))
        return any(fnmatch(rel_path, pat) for pat in self.exclude_patterns)

    def _get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        rel = file_path.relative_to(self.root)
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1].removesuffix(".py")
        return ".".join(parts) if parts else ""

    def extract_from_file(self, file_path: Path) -> Iterator[CodeUnit]:
        """Extract all code units from a single file."""
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"Could not parse {file_path}: {e}")
            return

        module_name = self._get_module_name(file_path)
        exported = get_exported_names(tree)
        visitor = _CodeUnitCollector(self, file_path, source, module_name, exported)
        visitor.visit(tree)
        for unit in visitor.units:
            yield unit

    def _should_emit_function(self, name: str, is_method: bool) -> bool:
        """Respect private-function filtering."""
        _ = is_method
        if self._is_private_name(name):
            return self.include_private
        return True

    @staticmethod
    def _is_private_name(name: str) -> bool:
        return name.startswith("_") and not name.startswith("__")

    def _should_emit_class(self, name: str) -> bool:
        """Respect private-class filtering."""
        if self.include_private:
            return True
        return not self._is_private_name(name)

    def _qualified_name(
        self,
        module_name: str,
        scope_prefix: list[str],
        name: str,
    ) -> str:
        parts = [part for part in scope_prefix if part]
        if module_name:
            parts.insert(0, module_name)
        parts.append(name)
        return ".".join(parts)

    def _emit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: Path,
        source: str,
        module_name: str,
        scope_prefix: list[str],
        class_member: bool,
        exported: set[str],
    ) -> Iterator[CodeUnit]:
        """Extract a function or method."""
        name = node.name
        qualified = self._qualified_name(module_name, scope_prefix, name)
        unit_type = CodeUnitType.METHOD if class_member else CodeUnitType.FUNCTION

        # Extract source for this node
        lines = source.splitlines(keepends=True)
        func_source = "".join(lines[node.lineno - 1 : node.end_lineno])

        # Build call graph
        call_visitor = CallGraphVisitor()
        call_visitor.visit(node)

        ast_hash = compute_ast_hash(node)
        token_hash = compute_token_hash(func_source)

        yield CodeUnit(
            name=name,
            qualified_name=qualified,
            unit_type=unit_type,
            file_path=file_path,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            source=func_source,
            docstring=extract_docstring(node),
            calls=call_visitor.calls,
            is_public=not name.startswith("_"),
            is_dunder=name.startswith("__") and name.endswith("__"),
            is_exported=name in exported,
            _ast_hash=ast_hash,
            _token_hash=token_hash,
        )

    def _emit_class(
        self,
        node: ast.ClassDef,
        file_path: Path,
        source: str,
        module_name: str,
        scope_prefix: list[str],
        exported: set[str],
    ) -> Iterator[CodeUnit]:
        """Extract a class and its methods."""
        class_name = node.name
        qualified = self._qualified_name(module_name, scope_prefix, class_name)
        lines = source.splitlines(keepends=True)
        class_source = "".join(lines[node.lineno - 1 : node.end_lineno])
        ast_hash = compute_ast_hash(node)
        token_hash = compute_token_hash(class_source)

        yield CodeUnit(
            name=class_name,
            qualified_name=qualified,
            unit_type=CodeUnitType.CLASS,
            file_path=file_path,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            source=class_source,
            docstring=extract_docstring(node),
            is_public=not class_name.startswith("_"),
            is_dunder=False,
            is_exported=class_name in exported,
            _ast_hash=ast_hash,
            _token_hash=token_hash,
        )

    def extract_all(self) -> list[CodeUnit]:
        """Extract all code units from the directory."""
        units: list[CodeUnit] = []
        for py_file in self.root.rglob("*.py"):
            if self._should_exclude(py_file):
                logger.debug(f"Excluding {py_file}")
                continue
            units.extend(self.extract_from_file(py_file))
        return units
