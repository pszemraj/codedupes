"""AST-based extraction of code units from Python files."""

from __future__ import annotations

import ast
import hashlib
import logging
from pathlib import Path
from typing import Iterator

from .models import CodeUnit, CodeUnitType

logger = logging.getLogger(__name__)


class NormalizedASTHasher(ast.NodeTransformer):
    """
    Transforms AST to normalize variable names, remove docstrings/comments.
    This allows structural comparison independent of naming.
    """

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
        # Remove docstring
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


def compute_ast_hash(node: ast.AST) -> str:
    """Compute a hash of the normalized AST structure."""
    hasher = NormalizedASTHasher()
    normalized = hasher.visit(ast.parse(ast.unparse(node)))
    # Use ast.dump for structural representation
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

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip methods (handled via class)
                if not any(
                    isinstance(parent, ast.ClassDef)
                    for parent in ast.walk(tree)
                    if node in ast.walk(parent) and parent is not node
                ):
                    yield from self._extract_function(
                        node, file_path, source, module_name, None, exported
                    )
            elif isinstance(node, ast.ClassDef):
                yield from self._extract_class(node, file_path, source, module_name, exported)

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: Path,
        source: str,
        module_name: str,
        class_name: str | None,
        exported: set[str],
    ) -> Iterator[CodeUnit]:
        """Extract a function or method."""
        name = node.name
        is_private = name.startswith("_") and not name.startswith("__")

        if not self.include_private and is_private:
            return

        if class_name:
            qualified = (
                f"{module_name}.{class_name}.{name}" if module_name else f"{class_name}.{name}"
            )
            unit_type = CodeUnitType.METHOD
        else:
            qualified = f"{module_name}.{name}" if module_name else name
            unit_type = CodeUnitType.FUNCTION

        # Extract source for this node
        lines = source.splitlines(keepends=True)
        func_source = "".join(lines[node.lineno - 1 : node.end_lineno])

        # Build call graph
        call_visitor = CallGraphVisitor()
        call_visitor.visit(node)

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
        )

    def _extract_class(
        self,
        node: ast.ClassDef,
        file_path: Path,
        source: str,
        module_name: str,
        exported: set[str],
    ) -> Iterator[CodeUnit]:
        """Extract a class and its methods."""
        class_name = node.name
        is_private = class_name.startswith("_") and not class_name.startswith("__")

        if not self.include_private and is_private:
            return

        qualified = f"{module_name}.{class_name}" if module_name else class_name
        lines = source.splitlines(keepends=True)
        class_source = "".join(lines[node.lineno - 1 : node.end_lineno])

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
        )

        # Extract methods
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield from self._extract_function(
                    item, file_path, source, module_name, class_name, exported
                )

    def extract_all(self) -> list[CodeUnit]:
        """Extract all code units from the directory."""
        units: list[CodeUnit] = []
        for py_file in self.root.rglob("*.py"):
            if self._should_exclude(py_file):
                logger.debug(f"Excluding {py_file}")
                continue
            units.extend(self.extract_from_file(py_file))

        # Compute hashes
        for unit in units:
            try:
                parsed = ast.parse(unit.source)
                if parsed.body:
                    unit._ast_hash = compute_ast_hash(parsed.body[0])
            except SyntaxError:
                pass
            unit._token_hash = compute_token_hash(unit.source)

        return units
