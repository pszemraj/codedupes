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


def _remove_leading_docstring(node: DefinitionNode) -> None:
    """Remove one leading docstring expression from a definition body.

    :param node: Function or class node to modify.
    :return: ``None``.
    """
    if ast.get_docstring(node) is not None:
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


class ReferenceVisitor(ast.NodeVisitor):
    """Collect names a definition references via calls, loads, or attribute access.

    Load-context ``Name`` and ``Attribute`` nodes subsume call targets (a call's
    ``func`` is itself a loaded name or attribute) while also covering non-call
    references the reference graph must see: callback-style arguments
    (``callback=validate``), property access (``self.cached_value``), decorators,
    and type annotations.
    """

    def __init__(self) -> None:
        """Initialize a fresh reference accumulator."""
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        """Collect loaded bare-name references, including non-call uses.

        :param node: AST name node.
        :return: ``None``.
        """
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Collect loaded attribute references such as method and property access.

        :param node: AST attribute node.
        :return: ``None``.
        """
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.attr)
            # Also track the full chain for simple obj.attr access so alias
            # resolution can map module-qualified references.
            if isinstance(node.value, ast.Name):
                self.names.add(f"{node.value.id}.{node.attr}")
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


# Deliberately NOT importlib.util.resolve_name: this must never raise, must return
# None for beyond-top-level imports, and must best-effort resolve relative imports
# from root-level files (package == ""), where the stdlib raises ImportError.
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


@dataclass(frozen=True)
class _NameBinding:
    """Identity currently bound to one name in a lexical scope."""

    identity: str | None = None
    module: str | None = None
    is_ast_visitor: bool = False


@dataclass
class _ClassFact:
    """Cross-file inheritance evidence recorded for one emitted class."""

    qualified_name: str
    inheritance_identity: str
    source_path: Path
    resolved_base_identities: set[str] = field(default_factory=set)
    # True when the definition executes conditionally (branch, loop, or
    # handler), so the import identity may not be this class at runtime.
    conditionally_defined: bool = False


class _CodeUnitCollector(ast.NodeVisitor):
    """Collect code units with deterministic scope tracking."""

    def __init__(
        self,
        extractor: CodeExtractor,
        file_path: Path,
        source: str,
        module_name: str,
        inheritance_module_name: str,
        exported: set[str],
    ) -> None:
        """Create a collector bound to an extractor and source context.

        :param extractor: Owning code extractor.
        :param file_path: Source file path.
        :param source: Full file source text.
        :param module_name: Deduced module name.
        :param inheritance_module_name: Importable module identity used by the
            corpus-wide inheritance graph.
        :param exported: Export names from module-level ``__all__``.
        """
        self.extractor = extractor
        self.file_path = file_path
        self.source_lines = source.splitlines(keepends=True)
        self.module_name = module_name
        self.inheritance_module_name = inheritance_module_name
        self.import_package = (
            inheritance_module_name
            if file_path.name in {"__init__.py", "__init__.pyi"}
            else inheritance_module_name.rpartition(".")[0]
        )
        self.exported = exported
        self.units: list[CodeUnit] = []
        self.class_facts: list[_ClassFact] = []

        # Scope stacks while walking AST:
        # - class_stack tracks nested class scope.
        # - function_stack tracks nested local function scope.
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []
        self.dynamic_dispatch_stack: list[bool] = []
        self.binding_scopes: list[dict[str, _NameBinding]] = [{}]
        self.conditional_depth = 0

    def _bind_name(self, name: str, binding: _NameBinding) -> None:
        """Update one name in the current lexical scope.

        :param name: Bound bare name.
        :param binding: Identity now visible through ``name``.
        :return: ``None``.
        """
        self.binding_scopes[-1][name] = binding

    def _lookup_name(self, name: str) -> _NameBinding | None:
        """Resolve the nearest currently visible binding for a bare name.

        :param name: Bare name to look up.
        :return: Current binding, or ``None`` when no visited scope binds it.
        """
        for scope in reversed(self.binding_scopes):
            if name in scope:
                return scope[name]
        return None

    def _resolve_base_binding(self, base_name: str) -> _NameBinding | None:
        """Resolve a dotted class-base expression through current lexical bindings.

        :param base_name: Dotted or bare base-class expression.
        :return: Resolved binding, or ``None`` when the expression is unbound.
        """
        bound_name, _, suffix = base_name.partition(".")
        binding = self._lookup_name(bound_name)
        if binding is None or not suffix:
            return binding
        if binding.module is not None:
            identity = f"{binding.module}.{suffix}"
            if binding.module == "ast" and suffix in AST_VISITOR_CLASS_NAMES:
                return _NameBinding(
                    identity=_CROSS_FILE_AST_VISITOR_ROOT,
                    is_ast_visitor=True,
                )
            return _NameBinding(identity=identity)
        if binding.identity is not None:
            return _NameBinding(identity=f"{binding.identity}.{suffix}")
        return binding

    def visit_Import(self, node: ast.Import) -> None:
        """Record module bindings at their document-order import position.

        :param node: Import statement.
        :return: ``None``.
        """
        for alias in node.names:
            bound_name = alias.asname or alias.name.partition(".")[0]
            module = alias.name if alias.asname else bound_name
            self._bind_name(bound_name, _NameBinding(module=module))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Record from-import bindings at their document-order position.

        :param node: From-import statement.
        :return: ``None``.
        """
        resolved_module = _resolve_relative_module(
            self.import_package,
            node.level,
            node.module,
        )
        for alias in node.names:
            if alias.name == "*":
                if resolved_module == "ast":
                    for class_name in AST_VISITOR_CLASS_NAMES:
                        self._bind_name(
                            class_name,
                            _NameBinding(
                                identity=_CROSS_FILE_AST_VISITOR_ROOT,
                                is_ast_visitor=True,
                            ),
                        )
                continue
            bound_name = alias.asname or alias.name
            if resolved_module == "ast" and alias.name in AST_VISITOR_CLASS_NAMES:
                binding = _NameBinding(
                    identity=_CROSS_FILE_AST_VISITOR_ROOT,
                    is_ast_visitor=True,
                )
            elif resolved_module is None:
                binding = _NameBinding()
            else:
                binding = _NameBinding(identity=f"{resolved_module}.{alias.name}")
            self._bind_name(bound_name, binding)

    def visit_Name(self, node: ast.Name) -> None:
        """Track assignment and deletion of names in the current lexical scope.

        :param node: Name expression.
        :return: ``None``.
        """
        if isinstance(node.ctx, ast.Store):
            self._bind_name(node.id, _NameBinding())
        elif isinstance(node.ctx, ast.Del):
            self.binding_scopes[-1].pop(node.id, None)

    @staticmethod
    def _joined_bindings(
        outcomes: list[dict[str, _NameBinding]],
    ) -> dict[str, _NameBinding]:
        """Join alternative branch outcomes into one conservative binding state.

        A name keeps its binding only when every reachable path agrees; any
        disagreement, including being unbound on some path, degrades it to an
        anonymous binding so neither visitor proof nor an import identity
        survives control-flow uncertainty. The anonymous binding also shadows
        outer scopes, matching how a conditional local assignment poisons the
        whole enclosing scope at runtime.

        :param outcomes: Innermost-scope binding states, one per reachable path.
        :return: Joined binding state.
        """
        joined: dict[str, _NameBinding] = {}
        for name in {name for outcome in outcomes for name in outcome}:
            bindings = [outcome.get(name) for outcome in outcomes]
            first = bindings[0]
            if first is not None and all(binding == first for binding in bindings[1:]):
                joined[name] = first
            else:
                joined[name] = _NameBinding()
        return joined

    def _visit_statement_suite(self, suite: list[ast.stmt]) -> None:
        """Visit the statements of one suite in document order.

        :param suite: Statement suite.
        :return: ``None``.
        """
        for statement in suite:
            self.visit(statement)

    def _visit_conditional_suite(self, suite: list[ast.stmt]) -> None:
        """Visit a suite that may be skipped at runtime and join both outcomes.

        :param suite: Possibly skipped statement suite.
        :return: ``None``.
        """
        if not suite:
            return
        pre_state = dict(self.binding_scopes[-1])
        self.conditional_depth += 1
        self._visit_statement_suite(suite)
        self.conditional_depth -= 1
        self.binding_scopes[-1] = self._joined_bindings([pre_state, self.binding_scopes[-1]])

    def visit_If(self, node: ast.If) -> None:
        """Join name bindings across both paths of a conditional.

        :param node: If statement.
        :return: ``None``.
        """
        self.visit(node.test)
        pre_state = dict(self.binding_scopes[-1])
        outcomes: list[dict[str, _NameBinding]] = []
        self.conditional_depth += 1
        for suite in (node.body, node.orelse):
            self.binding_scopes[-1] = dict(pre_state)
            self._visit_statement_suite(suite)
            outcomes.append(self.binding_scopes[-1])
        self.conditional_depth -= 1
        self.binding_scopes[-1] = self._joined_bindings(outcomes)

    def visit_While(self, node: ast.While) -> None:
        """Join bindings across the untaken, looped, and else paths of a while.

        :param node: While statement.
        :return: ``None``.
        """
        self.visit(node.test)
        self._visit_conditional_suite(node.body)
        self._visit_conditional_suite(node.orelse)

    def visit_For(self, node: ast.For) -> None:
        """Join bindings across the zero-iteration and looped paths."""
        self._visit_loop(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Join async-for bindings using the same logic as ``for``."""
        self._visit_loop(node)

    def _visit_loop(self, node: ast.For | ast.AsyncFor) -> None:
        """Join bindings across a loop's skipped, iterated, and else paths.

        The iterable expression always evaluates; the target assignment and
        body run zero or more times; ``break`` skips the ``else`` suite.

        :param node: For or async-for statement.
        :return: ``None``.
        """
        self.visit(node.iter)
        pre_state = dict(self.binding_scopes[-1])
        self.conditional_depth += 1
        self.visit(node.target)
        self._visit_statement_suite(node.body)
        self.conditional_depth -= 1
        self.binding_scopes[-1] = self._joined_bindings([pre_state, self.binding_scopes[-1]])
        self._visit_conditional_suite(node.orelse)

    def visit_Try(self, node: ast.Try) -> None:
        """Join bindings across the body, handler, else, and finally paths.

        A handler can begin after any prefix of the body, so its entry state
        joins every statement boundary the body can reach. The ``finally``
        suite runs unconditionally after the join.

        :param node: Try statement (``try*`` shares the binding structure).
        :return: ``None``.
        """
        prefix_states = [dict(self.binding_scopes[-1])]
        for statement in node.body:
            self.visit(statement)
            prefix_states.append(dict(self.binding_scopes[-1]))
        self.conditional_depth += 1
        self._visit_statement_suite(node.orelse)
        outcomes = [self.binding_scopes[-1]]
        handler_entry = self._joined_bindings(prefix_states)
        for handler in node.handlers:
            self.binding_scopes[-1] = dict(handler_entry)
            if handler.type is not None:
                self.visit(handler.type)
            if handler.name is not None:
                self._bind_name(handler.name, _NameBinding())
            self._visit_statement_suite(handler.body)
            outcome = self.binding_scopes[-1]
            if handler.name is not None:
                # Python unbinds the exception alias when the handler exits.
                outcome.pop(handler.name, None)
            outcomes.append(outcome)
        self.conditional_depth -= 1
        self.binding_scopes[-1] = self._joined_bindings(outcomes)
        self._visit_statement_suite(node.finalbody)

    visit_TryStar = visit_Try

    def visit_Match(self, node: ast.Match) -> None:
        """Join bindings across the match cases and the no-match fall-through.

        Case patterns bind their capture names even when a later guard rejects
        the match, so every capture degrades to an anonymous binding here.

        :param node: Match statement.
        :return: ``None``.
        """
        self.visit(node.subject)
        pre_state = dict(self.binding_scopes[-1])
        outcomes = [dict(pre_state)]
        self.conditional_depth += 1
        for case in node.cases:
            self.binding_scopes[-1] = dict(pre_state)
            self._bind_match_captures(case.pattern)
            if case.guard is not None:
                self.visit(case.guard)
            self._visit_statement_suite(case.body)
            outcomes.append(self.binding_scopes[-1])
        self.conditional_depth -= 1
        self.binding_scopes[-1] = self._joined_bindings(outcomes)

    def _bind_match_captures(self, pattern: ast.pattern) -> None:
        """Bind every capture name one match pattern can introduce.

        :param pattern: Match pattern node.
        :return: ``None``.
        """
        if isinstance(pattern, ast.MatchAs):
            if pattern.pattern is not None:
                self._bind_match_captures(pattern.pattern)
            if pattern.name is not None:
                self._bind_name(pattern.name, _NameBinding())
        elif isinstance(pattern, ast.MatchStar):
            if pattern.name is not None:
                self._bind_name(pattern.name, _NameBinding())
        elif isinstance(pattern, ast.MatchMapping):
            for sub_pattern in pattern.patterns:
                self._bind_match_captures(sub_pattern)
            if pattern.rest is not None:
                self._bind_name(pattern.rest, _NameBinding())
        elif isinstance(pattern, (ast.MatchSequence, ast.MatchOr)):
            for sub_pattern in pattern.patterns:
                self._bind_match_captures(sub_pattern)
        elif isinstance(pattern, ast.MatchClass):
            for sub_pattern in (*pattern.patterns, *pattern.kwd_patterns):
                self._bind_match_captures(sub_pattern)

    def _bind_function_arguments(self, arguments: ast.arguments) -> None:
        """Seed a function scope with its parameter bindings.

        :param arguments: Parsed function arguments.
        :return: ``None``.
        """
        positional = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        for argument in positional:
            self._bind_name(argument.arg, _NameBinding())
        if arguments.vararg is not None:
            self._bind_name(arguments.vararg.arg, _NameBinding())
        if arguments.kwarg is not None:
            self._bind_name(arguments.kwarg.arg, _NameBinding())

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Collect function code units and recurse into nested definitions."""
        self._visit_function_definition(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Collect async functions using the same logic as normal functions."""
        self._visit_function_definition(node)

    def _visit_function_definition(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        """Collect one function and visit its body in a fresh binding scope.

        :param node: Function or async-function definition.
        :return: ``None``.
        """
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

        self._bind_name(node.name, _NameBinding())
        self.function_stack.append(node.name)
        self.binding_scopes.append({})
        self._bind_function_arguments(node.args)
        for statement in node.body:
            self.visit(statement)
        self.binding_scopes.pop()
        self.function_stack.pop()

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

        base_bindings = [
            binding
            for base in node.bases
            if (base_name := _dotted_expression_name(base)) is not None
            and (binding := self._resolve_base_binding(base_name)) is not None
        ]
        is_dynamic_dispatch_class = any(binding.is_ast_visitor for binding in base_bindings)
        qualified_name = self.extractor._qualified_name(self.module_name, scope_prefix, node.name)
        inheritance_identity = self.extractor._qualified_name(
            self.inheritance_module_name, scope_prefix, node.name
        )
        resolved_base_identities = {
            binding.identity for binding in base_bindings if binding.identity is not None
        }
        # Function-local classes are invisible to importers, and their method
        # units carry the function scope in their qualified prefix, so they can
        # neither confer nor receive cross-file proof. Recording them would only
        # pollute the module-level identity they appear to share.
        if not self.function_stack:
            self.class_facts.append(
                _ClassFact(
                    qualified_name,
                    inheritance_identity,
                    self.file_path,
                    resolved_base_identities,
                    conditionally_defined=self.conditional_depth > 0,
                )
            )

        if should_enter:
            self.class_stack.append(node.name)
            self.dynamic_dispatch_stack.append(is_dynamic_dispatch_class)
            self.binding_scopes.append({})
            for statement in node.body:
                self.visit(statement)
            self.binding_scopes.pop()
            self.dynamic_dispatch_stack.pop()
            self.class_stack.pop()
        else:
            # If class is excluded, skip descendants to avoid leaking private internals.
            logger.debug("Skipping private class %s in %s", node.name, self.file_path)
        self._bind_name(
            node.name,
            _NameBinding(
                identity=inheritance_identity,
                is_ast_visitor=is_dynamic_dispatch_class,
            ),
        )


def _resolve_cross_file_dynamic_dispatch_hooks(
    units: list[CodeUnit], class_facts: list[_ClassFact]
) -> None:
    """Mark ``visit_*`` methods whose class is provably an AST visitor across files.

    Computes the transitive closure of "inherits from ast.NodeVisitor/NodeTransformer" over
    the corpus-wide class graph built from per-file inheritance evidence, then flags any
    not-yet-marked ``visit_*`` method belonging to a class proven only through that closure.
    Unresolvable bases (third-party imports, star imports, dynamic bases) never enter the
    graph, so they stay unproven, same as the existing same-file behavior. Identities whose
    definition is control-flow dependent are likewise barred from conferring proof, since
    the runtime attribute may be a different object on the executed path.

    :param units: All code units collected across the corpus (mutated in place).
    :param class_facts: Per-class base-identity evidence gathered during extraction.
    :return: ``None``.
    """
    identity_paths: dict[str, set[Path]] = {}
    for fact in class_facts:
        identity_paths.setdefault(fact.inheritance_identity, set()).add(fact.source_path)
    collision_identities = {
        identity for identity, source_paths in identity_paths.items() if len(source_paths) > 1
    }
    for identity in sorted(collision_identities):
        source_paths = ", ".join(str(path) for path in sorted(identity_paths[identity]))
        logger.warning(
            "Disabling cross-file inheritance proof for ambiguous import identity %s: %s",
            identity,
            source_paths,
        )

    # A physical-file collision or a control-flow-dependent definition (branch,
    # loop, or handler) makes both the class identity and any import of that
    # identity ambiguous: the runtime attribute may not be the recorded class.
    # Excluding both sides keeps this optimization proof-based: ambiguity may
    # create an unused-code false positive, but never suppresses a genuinely
    # unused method as though inheritance were proven.
    ambiguous_identities = collision_identities | {
        fact.inheritance_identity for fact in class_facts if fact.conditionally_defined
    }
    edges = {
        fact.inheritance_identity: fact.resolved_base_identities - ambiguous_identities
        for fact in class_facts
        if fact.inheritance_identity not in ambiguous_identities
    }
    proven: set[str] = {_CROSS_FILE_AST_VISITOR_ROOT}

    changed = True
    while changed:
        changed = False
        for qualified_name, base_identities in edges.items():
            if qualified_name not in proven and base_identities & proven:
                proven.add(qualified_name)
                changed = True

    # Marking is per definition, not per identity: a redefined class name is
    # proven only when every physical definition's own base evidence is proven,
    # because emitted method units share the qualified-name prefix across
    # redefinitions. Ambiguous identities can still receive proof through their
    # unambiguous bases; they only stop conferring proof to importers.
    facts_by_qualified_name: dict[str, list[_ClassFact]] = {}
    for fact in class_facts:
        facts_by_qualified_name.setdefault(fact.qualified_name, []).append(fact)
    proven_qualified_names = {
        qualified_name
        for qualified_name, facts in facts_by_qualified_name.items()
        if all((fact.resolved_base_identities - ambiguous_identities) & proven for fact in facts)
    }
    if not proven_qualified_names:
        return
    for unit in units:
        if (
            unit.unit_type == CodeUnitType.METHOD
            and not unit.is_dynamic_dispatch_hook
            and unit.name.startswith("visit_")
            and unit.qualified_name.rsplit(".", 1)[0] in proven_qualified_names
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


def _iter_module_level_statements(tree: ast.Module) -> Iterator[ast.stmt]:
    """Yield statements executing in module scope, including inside control flow.

    Function and class bodies are excluded: a name assigned there is a local or
    a class attribute, never a module-level binding.

    :param tree: Parsed module AST.
    :return: Iterator over module-scope statements in breadth-first order.
    """
    pending: list[ast.stmt] = list(tree.body)
    while pending:
        statement = pending.pop(0)
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield statement
        pending.extend(
            child for child in ast.iter_child_nodes(statement) if isinstance(child, ast.stmt)
        )


def get_exported_names(tree: ast.Module) -> set[str]:
    """Extract names from a module-level ``__all__`` if present.

    An ``__all__`` local to a function or class body is not the module export
    list and never exempts names from unused analysis.

    :param tree: Parsed module AST.
    :return: Set of exported names.
    """
    for node in _iter_module_level_statements(tree):
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

    def _get_inheritance_module_name(self, file_path: Path) -> str:
        """Resolve the importable module identity used for inheritance matching.

        Directory names above the nearest regular Python package are source roots,
        not import path components. Keeping them out of the graph identity lets a
        file discovered as ``src.pkg.base`` match ``from pkg.base import Base``
        without changing the user-facing qualified names emitted for that file.

        :param file_path: File path under the configured root.
        :return: Importable dotted module name for inheritance resolution.
        """
        rel = file_path.relative_to(self.root)
        if len(rel.parts) > 1 and rel.parts[0] == "src":
            source_root = self.root / "src"
            if not any(
                (source_root / init_name).is_file() for init_name in ("__init__.py", "__init__.pyi")
            ):
                source_parts = list(rel.parts[1:])
                if source_parts[-1] == "__init__.py":
                    source_parts = source_parts[:-1]
                else:
                    source_parts[-1] = Path(source_parts[-1]).stem
                return ".".join(source_parts)

        module_parts = [] if file_path.name == "__init__.py" else [file_path.stem]
        package_dir = file_path.parent
        while package_dir == self.root or self.root in package_dir.parents:
            if not any(
                (package_dir / init_name).is_file() for init_name in ("__init__.py", "__init__.pyi")
            ):
                break
            module_parts.insert(0, package_dir.name)
            if package_dir == self.root:
                break
            package_dir = package_dir.parent
        return ".".join(module_parts)

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
        inheritance_module_name = self._get_inheritance_module_name(file_path)
        exported = get_exported_names(tree)
        collector = _CodeUnitCollector(
            self,
            file_path,
            source,
            module_name,
            inheritance_module_name,
            exported,
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

        reference_visitor = ReferenceVisitor()
        reference_visitor.visit(node)

        return self._build_code_unit(
            node,
            file_path,
            source_lines,
            module_name,
            scope_prefix,
            unit_type=unit_type,
            referenced_names=reference_visitor.names,
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
        reference_visitor = ReferenceVisitor()
        reference_visitor.visit(node)

        return self._build_code_unit(
            node,
            file_path=file_path,
            source_lines=source_lines,
            module_name=module_name,
            scope_prefix=scope_prefix,
            unit_type=CodeUnitType.CLASS,
            referenced_names=reference_visitor.names,
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
        referenced_names: set[str],
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
        :param referenced_names: Names the definition references.
        :param exported: Exported names from module ``__all__``.
        :param dynamic_dispatch_hook: Whether runtime visitor dispatch reaches this method.
        :return: Constructed code unit.
        """
        name = node.name
        source_start_lineno = min(
            (decorator.lineno for decorator in node.decorator_list),
            default=node.lineno,
        )
        unit_source = "".join(source_lines[source_start_lineno - 1 : node.end_lineno])
        return CodeUnit(
            name=name,
            qualified_name=self._qualified_name(module_name, scope_prefix, name),
            unit_type=unit_type,
            file_path=file_path,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            source=unit_source,
            referenced_names=referenced_names,
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
