"""AST-based extraction of code units from Python files."""

from __future__ import annotations

import ast
import copy
import hashlib
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

from codedupes._reference_flow import (
    BranchingVisitor,
    comprehension_definitely_runs,
    comprehension_exact_result_count,
    iterable_definitely_empty,
    iterable_definitely_nonempty,
)
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
_OPAQUE_REFERENCE_TARGET = "\0opaque"
_PRESERVE_REFERENCE_TARGET = "\0preserve"
_FREE_REFERENCE_PREFIX = "\0free:"
_PARAMETER_REFERENCE_PREFIX = "\0param:"
_CALL_RESULT_PREFIX = "\0call-result:"
_GENERATOR_RESUME_PREFIX = "\0generator-resume:"
_COROUTINE_INSTANCE_PREFIX = "\0coroutine-instance:"
_FALLBACK_RESULT_PREFIX = "\0fallback-result:"
_DEFERRED_ACTIVATOR_PREFIX = "\0deferred-activator:"
_NEXT_ACTIVATOR_TARGET = "\0next-activator"

_ReferenceConditions = frozenset[tuple[str, bool]]
_ReferenceFlowPoint = tuple[int, _ReferenceConditions]

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


@dataclass(frozen=True)
class _ReferenceScope:
    """Compile-time lexical slots and resolvable code targets for one function."""

    local_names: frozenset[str]
    global_names: frozenset[str]
    nonlocal_names: frozenset[str]
    targets: dict[str, frozenset[str]]
    qualified_name: str = ""
    invocation_targets: dict[str, dict[str, frozenset[str]]] = field(default_factory=dict)
    escape_targets: dict[str, dict[str, frozenset[str]]] = field(default_factory=dict)
    nonlocal_effects: dict[str, frozenset[str]] = field(default_factory=dict)
    free_calls: tuple[str, ...] = ()
    free_call_events: dict[int, _ReferenceCallEvent] = field(default_factory=dict)
    terminal_free_calls: frozenset[_ReferenceFlowPoint] = frozenset()
    called_parameters: frozenset[str] = frozenset()
    activated_parameters: frozenset[str] = frozenset()
    exhausted_parameters: frozenset[str] = frozenset()
    delegated_parameters: frozenset[str] = frozenset()
    escaped_parameters: frozenset[str] = frozenset()
    positional_parameters: tuple[str, ...] = ()
    keyword_parameters: frozenset[str] = frozenset()
    variadic_positional_parameter: str | None = None
    variadic_keyword_parameter: str | None = None
    parameter_defaults: dict[str, frozenset[str]] = field(default_factory=dict)
    return_targets: frozenset[str] = frozenset()
    call_results: dict[str, _ReferenceCall] = field(default_factory=dict)
    deferred_instance_parameters: dict[str, dict[str, frozenset[str]]] = field(default_factory=dict)
    fallback_results: dict[str, tuple[str, frozenset[str]]] = field(default_factory=dict)
    yield_effects: tuple[dict[str, frozenset[str]], ...] = ()
    yield_free_calls: tuple[frozenset[_ReferenceFlowPoint], ...] = ()
    deferred_execution: bool = False
    generator_execution: bool = False
    may_raise: bool = True
    deferred_scopes: dict[str, _ReferenceScope] = field(default_factory=dict)


@dataclass(frozen=True)
class _ReferenceCall:
    """Symbolic call operands retained for interprocedural resolution."""

    callable_targets: frozenset[str]
    positional_targets: tuple[frozenset[str], ...] = ()
    positional_unpacking: tuple[bool, ...] = ()
    keyword_targets: tuple[tuple[str | None, frozenset[str]], ...] = ()


@dataclass(frozen=True)
class _ReferenceCallEvent:
    """One branch-aware free call in a nested-function summary."""

    name: str
    predecessors: frozenset[_ReferenceFlowPoint]
    direct_effects: dict[str, frozenset[str]]
    call: _ReferenceCall
    activate: bool = False
    exhaust: bool = False
    conditions: _ReferenceConditions = frozenset()


def _function_header_expressions(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[ast.AST]:
    """Yield expressions evaluated outside a function body's lexical scope.

    :param node: Function definition whose header is being inspected.
    :return: Decorators, defaults, annotations, and type parameters.
    """
    yield from node.decorator_list
    yield from node.args.defaults
    yield from (default for default in node.args.kw_defaults if default is not None)
    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    if node.args.vararg is not None:
        arguments.append(node.args.vararg)
    if node.args.kwarg is not None:
        arguments.append(node.args.kwarg)
    yield from (argument.annotation for argument in arguments if argument.annotation is not None)
    if node.returns is not None:
        yield node.returns
    yield from getattr(node, "type_params", ())


def _type_parameter_names(node: DefinitionNode) -> frozenset[str]:
    """Return PEP 695 type-parameter names introduced by a definition.

    :param node: Function or class definition.
    :return: Names local to the definition's annotation scope.
    """
    return frozenset(
        parameter.name
        for parameter in getattr(node, "type_params", ())
        if isinstance(parameter.name, str)
    )


def _function_is_generator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether one function body can suspend at a yield expression.

    :param node: Function definition to inspect.
    :return: Whether the function is a synchronous or asynchronous generator.
    """

    class YieldFinder(ast.NodeVisitor):
        """Find yields in one function body without entering nested scopes."""

        found = False

        def visit_Yield(self, child: ast.Yield) -> None:
            """Mark a yield expression in the current function."""
            self.found = True

        visit_YieldFrom = visit_Yield

        def visit_FunctionDef(self, child: ast.FunctionDef) -> None:
            """Skip nested function bodies."""

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, child: ast.ClassDef) -> None:
            """Skip nested class bodies."""

        def visit_Lambda(self, child: ast.Lambda) -> None:
            """Skip nested lambda bodies."""

    finder = YieldFinder()
    for statement in node.body:
        finder.visit(statement)
    return finder.found


def _function_defers_execution(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a call creates a coroutine or generator without running the body.

    :param node: Function definition to inspect.
    :return: Whether body execution is deferred beyond the call expression.
    """
    return isinstance(node, ast.AsyncFunctionDef) or _function_is_generator(node)


def _loop_body_can_terminate_early(statements: list[ast.stmt]) -> bool:
    """Return whether control can leave a loop before iterable exhaustion.

    :param statements: Current loop body.
    :return: Whether a reachable break, return, or raise can leave the loop.
    """

    class BreakFinder(ast.NodeVisitor):
        """Find breaks without entering nested loops or deferred scopes."""

        found = False

        def visit_Break(self, node: ast.Break) -> None:
            """Record a break targeting the current loop."""
            self.found = True

        visit_Return = visit_Break
        visit_Raise = visit_Break

        def visit_If(self, node: ast.If) -> None:
            """Skip statically unreachable conditional branches."""
            if isinstance(node.test, ast.Constant):
                branch = node.body if bool(node.test.value) else node.orelse
                for statement in branch:
                    self.visit(statement)
                return
            self.generic_visit(node)

        def visit_For(self, node: ast.For) -> None:
            """Skip a nested loop with its own break target."""

        visit_AsyncFor = visit_For
        visit_While = visit_For

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            """Skip deferred nested function bodies."""

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            """Skip nested class bodies."""

    finder = BreakFinder()
    for statement in statements:
        finder.visit(statement)
    return finder.found


def _function_body_may_raise(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a function contains an operation that may raise.

    :param node: Function definition to inspect.
    :return: Whether its body is not provably exception-free.
    """

    class RaisingOperationFinder(ast.NodeVisitor):
        """Find potentially raising operations without entering nested scopes."""

        found = False

        def visit(self, child: ast.AST) -> None:
            """Stop descending once a potentially raising operation is found."""
            if not self.found:
                super().visit(child)

        def generic_visit(self, child: ast.AST) -> None:
            """Mark runtime operations whose evaluation can raise."""
            if isinstance(
                child,
                (
                    ast.Assert,
                    ast.Attribute,
                    ast.Await,
                    ast.BinOp,
                    ast.Call,
                    ast.Compare,
                    ast.Raise,
                    ast.Subscript,
                    ast.UnaryOp,
                    ast.YieldFrom,
                ),
            ):
                self.found = True
                return
            super().generic_visit(child)

        def visit_FunctionDef(self, child: ast.FunctionDef) -> None:
            """Skip nested deferred function bodies."""

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef

        def visit_ClassDef(self, child: ast.ClassDef) -> None:
            """Treat executed class construction as potentially raising."""
            self.found = True

    finder = RaisingOperationFinder()
    for statement in node.body:
        finder.visit(statement)
    return finder.found


def _function_outer_header_expressions(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[ast.AST]:
    """Yield function header expressions outside any type-parameter scope.

    :param node: Function definition to inspect.
    :return: Decorators and default-value expressions.
    """
    yield from node.decorator_list
    yield from node.args.defaults
    yield from (default for default in node.args.kw_defaults if default is not None)


def _function_annotation_expressions(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[ast.AST]:
    """Yield type parameters and annotations evaluated in their annotation scope.

    :param node: Function definition to inspect.
    :return: Type parameters and annotation expressions.
    """
    yield from getattr(node, "type_params", ())
    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    if node.args.vararg is not None:
        arguments.append(node.args.vararg)
    if node.args.kwarg is not None:
        arguments.append(node.args.kwarg)
    yield from (argument.annotation for argument in arguments if argument.annotation is not None)
    if node.returns is not None:
        yield node.returns


def _class_header_expressions(node: ast.ClassDef) -> Iterator[ast.AST]:
    """Yield expressions evaluated before a class namespace exists.

    :param node: Class definition whose header is being inspected.
    :return: Decorators, bases, keyword values, and type parameters.
    """
    yield from node.decorator_list
    yield from node.bases
    yield from (keyword.value for keyword in node.keywords)
    yield from getattr(node, "type_params", ())


def _bound_target_names(target: ast.AST) -> set[str]:
    """Collect bare names bound by one assignment-style target.

    :param target: Assignment, loop, with, or comprehension target.
    :return: Bare bound names.
    """
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {name for element in target.elts for name in _bound_target_names(element)}
    if isinstance(target, ast.Starred):
        return _bound_target_names(target.value)
    return set()


def _match_capture_names(pattern: ast.pattern) -> set[str]:
    """Collect names captured by one structural-match pattern.

    :param pattern: Match pattern to inspect.
    :return: Bare capture names.
    """
    if isinstance(pattern, ast.MatchAs):
        names = _match_capture_names(pattern.pattern) if pattern.pattern is not None else set()
        if pattern.name is not None:
            names.add(pattern.name)
        return names
    if isinstance(pattern, ast.MatchStar):
        return {pattern.name} if pattern.name is not None else set()
    if isinstance(pattern, ast.MatchMapping):
        names = {name for child in pattern.patterns for name in _match_capture_names(child)}
        if pattern.rest is not None:
            names.add(pattern.rest)
        return names
    if isinstance(pattern, (ast.MatchSequence, ast.MatchOr)):
        return {name for child in pattern.patterns for name in _match_capture_names(child)}
    if isinstance(pattern, ast.MatchClass):
        return {
            name
            for child in (*pattern.patterns, *pattern.kwd_patterns)
            for name in _match_capture_names(child)
        }
    return set()


class _FunctionLexicalSlotCollector(ast.NodeVisitor):
    """Collect function symbol-table slots from syntax, regardless of reachability."""

    def __init__(self, arguments: ast.arguments) -> None:
        """Initialize lexical slots with the function parameters.

        :param arguments: Function or lambda parameters.
        """
        self.local_names: set[str] = set()
        self.global_names: set[str] = set()
        self.nonlocal_names: set[str] = set()
        positional = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        self.local_names.update(argument.arg for argument in positional)
        if arguments.vararg is not None:
            self.local_names.add(arguments.vararg.arg)
        if arguments.kwarg is not None:
            self.local_names.add(arguments.kwarg.arg)

    def visit_Name(self, node: ast.Name) -> None:
        """Record every syntactic store or delete as a function-local slot."""
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.local_names.add(node.id)

    def visit_Global(self, node: ast.Global) -> None:
        """Record module-scope declarations from the entire function block."""
        self.global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """Record enclosing-scope declarations from the entire function block."""
        self.nonlocal_names.update(node.names)

    def visit_Import(self, node: ast.Import) -> None:
        """Record names introduced by imports."""
        for alias in node.names:
            self.local_names.add(alias.asname or alias.name.partition(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Record names introduced by explicit from-imports."""
        self.local_names.update(
            alias.asname or alias.name for alias in node.names if alias.name != "*"
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Bind a nested function without entering its separate lexical body."""
        self.local_names.add(node.name)
        for expression in _function_header_expressions(node):
            self.visit(expression)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Bind a nested class and inspect only its outer-evaluated header."""
        self.local_names.add(node.name)
        for expression in _class_header_expressions(node):
            self.visit(expression)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Inspect lambda defaults while leaving its body in the lambda scope."""
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Record an exception alias and inspect the complete handler suite."""
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            self.local_names.add(node.name)
        for statement in node.body:
            self.visit(statement)

    def visit_match_case(self, node: ast.match_case) -> None:
        """Record pattern captures and inspect the guard and complete case suite."""
        self.local_names.update(_match_capture_names(node.pattern))
        if node.guard is not None:
            self.visit(node.guard)
        for statement in node.body:
            self.visit(statement)

    def _visit_comprehension(
        self,
        generators: list[ast.comprehension],
        values: tuple[ast.AST, ...],
    ) -> None:
        """Collect enclosing walrus slots without leaking implicit-scope targets.

        :param generators: Comprehension generator clauses.
        :param values: Result expressions evaluated in the implicit scope.
        :return: ``None``.
        """
        for generator in generators:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        for value in values:
            self.visit(value)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Collect enclosing slots from a list comprehension."""
        self._visit_comprehension(node.generators, (node.elt,))

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Collect enclosing slots from a dictionary comprehension."""
        self._visit_comprehension(node.generators, (node.key, node.value))


_FunctionScopeState = tuple[
    dict[str, set[str]],
    frozenset[_ReferenceFlowPoint],
    frozenset[int],
]


class _FunctionScopeCollector(BranchingVisitor[_FunctionScopeState]):
    """Collect compile-time function slots without descending into nested bodies."""

    def __init__(
        self,
        qualified_name: str,
        import_package: str,
        arguments: ast.arguments,
        *,
        deferred_execution: bool = False,
        generator_execution: bool = False,
        visible_deferred_scopes: dict[str, _ReferenceScope] | None = None,
    ) -> None:
        """Initialize a function-scope prepass.

        :param qualified_name: Full name used to qualify nested definitions.
        :param import_package: Package used to resolve relative imports.
        :param arguments: Function or lambda parameters.
        :param deferred_execution: Whether calling creates a deferred coroutine/generator.
        :param generator_execution: Whether each activation can suspend at a yield.
        :param visible_deferred_scopes: Already defined sibling callables visible by name.
        """
        super().__init__()
        self.qualified_name = qualified_name
        self.import_package = import_package
        self.local_names: set[str] = set()
        self.global_names: set[str] = set()
        self.nonlocal_names: set[str] = set()
        self.targets: dict[str, set[str]] = {}
        self.invocation_states: dict[str, list[dict[str, set[str]]]] = {}
        self.escape_states: dict[str, list[dict[str, set[str]]]] = {}
        self.deferred_scopes: dict[str, _ReferenceScope] = {}
        self.visible_deferred_scopes = dict(visible_deferred_scopes or {})
        self.free_calls: list[str] = []
        self.called_parameters: set[str] = set()
        self.activated_parameters: set[str] = set()
        self.exhausted_parameters: set[str] = set()
        self.delegated_parameters: set[str] = set()
        self.return_targets: set[str] = set()
        self.call_results: dict[str, _ReferenceCall] = {}
        self.deferred_instance_parameters: dict[str, dict[str, set[str]]] = {}
        self.fallback_results: dict[str, tuple[str, frozenset[str]]] = {}
        self.yield_effects: dict[int, list[dict[str, set[str]]]] = {}
        self.yield_free_calls: dict[int, set[_ReferenceFlowPoint]] = {}
        self._yield_phases: set[int] = {0}
        self._free_call_frontier: set[_ReferenceFlowPoint] = {(0, frozenset())}
        self._free_call_ids: dict[tuple[int, str, _ReferenceConditions], int] = {}
        self._free_call_predecessors: dict[int, set[_ReferenceFlowPoint]] = {}
        self._free_call_effect_states: dict[int, list[dict[str, set[str]]]] = {}
        self._free_call_descriptors: dict[int, tuple[str, _ReferenceCall]] = {}
        self._free_call_conditions: dict[int, _ReferenceConditions] = {}
        self._condition_dependencies: dict[str, frozenset[str]] = {}
        self._activation_call_nodes: set[int] = set()
        self._exhaustion_call_nodes: set[int] = set()
        self._activated_free_call_ids: set[int] = set()
        self._exhausted_free_call_ids: set[int] = set()
        self.deferred_execution = deferred_execution
        self.generator_execution = generator_execution
        self._suppress_escape = False
        positional = [*arguments.posonlyargs, *arguments.args]
        keyword = [*arguments.args, *arguments.kwonlyargs]
        self.positional_parameters = tuple(argument.arg for argument in positional)
        self.keyword_parameters = {argument.arg for argument in keyword}
        self.variadic_positional_parameter = (
            arguments.vararg.arg if arguments.vararg is not None else None
        )
        self.variadic_keyword_parameter = (
            arguments.kwarg.arg if arguments.kwarg is not None else None
        )
        self.parameter_names = {
            *self.positional_parameters,
            *self.keyword_parameters,
        }
        if arguments.vararg is not None:
            self.parameter_names.add(arguments.vararg.arg)
        if arguments.kwarg is not None:
            self.parameter_names.add(arguments.kwarg.arg)
        self.local_names.update(self.parameter_names)
        if arguments.vararg is not None:
            self.local_names.add(arguments.vararg.arg)
        if arguments.kwarg is not None:
            self.local_names.add(arguments.kwarg.arg)

    def _bind(self, name: str, target: str | None = None) -> None:
        """Record one compile-time local slot and optional code identity.

        :param name: Local binding name.
        :param target: Optional qualified code identity.
        :return: ``None``.
        """
        self.local_names.add(name)
        self.targets[name] = {target} if target is not None else set()
        self._invalidate_conditions({name})

    def _bind_targets(self, name: str, targets: set[str]) -> None:
        """Record one local slot with a set of possible code identities.

        :param name: Local binding name.
        :param targets: Possible qualified code identities.
        :return: ``None``.
        """
        self.local_names.add(name)
        self.targets[name] = set(targets)
        self._invalidate_conditions({name})

    def _invalidate_conditions(self, names: set[str]) -> None:
        """Drop predicate facts whose loaded names were rebound.

        :param names: Rebound lexical names.
        :return: ``None``.
        """
        if not names:
            return
        self._free_call_frontier = {
            (
                event_id,
                frozenset(
                    (key, value)
                    for key, value in conditions
                    if self._condition_dependencies.get(key, frozenset()).isdisjoint(names)
                ),
            )
            for event_id, conditions in self._free_call_frontier
        }

    def _expression_targets(self, node: ast.AST) -> set[str]:
        """Resolve simple callable aliases through current local target bindings.

        :param node: Expression whose possible identities are needed.
        :return: Possible qualified identities, excluding opaque values.
        """
        if isinstance(node, ast.Name):
            if node.id in {"next", "anext"} and node.id not in self.targets:
                return {_NEXT_ACTIVATOR_TARGET}
            if node.id in self.targets:
                return set(self.targets[node.id]) - {
                    _OPAQUE_REFERENCE_TARGET,
                    _PRESERVE_REFERENCE_TARGET,
                }
            prefix = (
                _PARAMETER_REFERENCE_PREFIX
                if node.id in self.parameter_names
                else _FREE_REFERENCE_PREFIX
            )
            return {f"{prefix}{node.id}"}
        if isinstance(node, ast.Attribute):
            receiver_targets = self._expression_targets(node.value)
            if node.attr in {"send", "__anext__", "__next__"}:
                deferred_targets = {
                    target
                    for target in receiver_targets
                    if (scope := self._lookup_deferred_scope(target)) is not None
                    and scope.deferred_execution
                }
                if deferred_targets:
                    return {f"{_DEFERRED_ACTIVATOR_PREFIX}{target}" for target in deferred_targets}
            return {f"{target}.{node.attr}" for target in receiver_targets}
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.parameter_names
        ):
            try:
                projection = ast.literal_eval(node.slice)
            except (ValueError, TypeError):
                return {f"{_PARAMETER_REFERENCE_PREFIX}{node.value.id}"}
            if not isinstance(projection, (str, int)):
                return {f"{_PARAMETER_REFERENCE_PREFIX}{node.value.id}"}
            return {f"{_PARAMETER_REFERENCE_PREFIX}{node.value.id}[{projection!r}]"}
        if isinstance(node, ast.Subscript):
            return self._expression_targets(node.value)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return {
                target
                for element in node.elts
                for target in self._expression_targets(element)
            }
        if isinstance(node, ast.IfExp):
            return self._expression_targets(node.body) | self._expression_targets(node.orelse)
        if isinstance(node, ast.BoolOp):
            return {
                target
                for value in node.values
                for target in self._expression_targets(value)
            }
        if isinstance(node, ast.NamedExpr):
            return self._expression_targets(node.value)
        if isinstance(node, ast.Starred):
            return self._expression_targets(node.value)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"iter", "next"} and node.args:
                projected = self._expression_targets(node.args[0])
                if projected and all(
                    target.startswith(_PARAMETER_REFERENCE_PREFIX) for target in projected
                ):
                    return projected
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in self.parameter_names
                and node.func.attr in {"get", "pop"}
                and node.args
            ):
                if not (
                    isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, (str, int))
                ):
                    primary = f"{_PARAMETER_REFERENCE_PREFIX}{node.func.value.id}"
                else:
                    primary = (
                        f"{_PARAMETER_REFERENCE_PREFIX}{node.func.value.id}"
                        f"[{node.args[0].value!r}]"
                    )
                if len(node.args) < 2:
                    return {primary}
                token = (
                    f"{_FALLBACK_RESULT_PREFIX}{self.qualified_name}:"
                    f"{getattr(node, 'lineno', 0)}:{getattr(node, 'col_offset', 0)}"
                )
                self.fallback_results[token] = (
                    primary,
                    frozenset(self._expression_targets(node.args[1])),
                )
                return {token}
            targets: set[str] = set()
            callable_targets = self._expression_targets(node.func)
            descriptor = _ReferenceCall(
                callable_targets=frozenset(callable_targets),
                positional_targets=tuple(
                    frozenset(self._expression_targets(argument)) for argument in node.args
                ),
                positional_unpacking=tuple(
                    isinstance(argument, ast.Starred) for argument in node.args
                ),
                keyword_targets=tuple(
                    (keyword.arg, frozenset(self._expression_targets(keyword.value)))
                    for keyword in node.keywords
                ),
            )
            unresolved = False
            for callable_target in callable_targets:
                deferred = self._lookup_deferred_scope(callable_target)
                if deferred is None:
                    unresolved = True
                    continue
                parameter_targets = self._bind_reference_call(deferred, descriptor)
                if deferred.deferred_execution:
                    instance = (
                        f"{self.qualified_name}:{getattr(node, 'lineno', 0)}:"
                        f"{getattr(node, 'col_offset', 0)}"
                    )
                    self.deferred_instance_parameters[instance] = {
                        name: set(targets) for name, targets in parameter_targets.items()
                    }
                    if deferred.generator_execution:
                        targets.add(f"{_GENERATOR_RESUME_PREFIX}0\0{instance}\0{callable_target}")
                    else:
                        targets.add(f"{_COROUTINE_INSTANCE_PREFIX}{instance}\0{callable_target}")
                else:
                    targets.update(
                        self._resolve_summary_targets(
                            set(deferred.return_targets),
                            parameter_targets=parameter_targets,
                            instance_context=(
                                f"{self.qualified_name}:{getattr(node, 'lineno', 0)}:"
                                f"{getattr(node, 'col_offset', 0)}:{callable_target}"
                            ),
                        )
                    )
            if unresolved:
                token = (
                    f"{_CALL_RESULT_PREFIX}{self.qualified_name}:"
                    f"{getattr(node, 'lineno', 0)}:{getattr(node, 'col_offset', 0)}"
                )
                self.call_results[token] = descriptor
                targets.add(token)
            return targets
        return set()

    def _bind_reference_call(
        self,
        scope: _ReferenceScope,
        call: _ReferenceCall,
        *,
        state: dict[str, set[str]] | None = None,
        parameter_targets: dict[str, set[str]] | None = None,
    ) -> dict[str, set[str]]:
        """Bind symbolic call operands to a known function signature.

        :param scope: Callee summary whose signature receives the arguments.
        :param call: Symbolic call operands evaluated by the caller.
        :param state: Caller bindings used to resolve free targets.
        :param parameter_targets: Caller parameter bindings used for substitution.
        :return: Callee parameter targets keyed by formal name.
        """
        bound: dict[str, set[str]] = {}
        extra_positional: set[str] = set()
        extra_positional_values: list[set[str]] = []
        variadic_position = 0
        for index, targets in enumerate(call.positional_targets):
            unpacking = (
                index < len(call.positional_unpacking)
                and call.positional_unpacking[index]
            )
            resolved = self._resolve_summary_targets(
                set(targets),
                state=state,
                parameter_targets=parameter_targets,
            )
            if unpacking:
                if scope.variadic_positional_parameter is not None:
                    extra_positional.update(resolved)
                    bound.setdefault(scope.variadic_positional_parameter, set()).update(
                        resolved
                    )
                    for target in targets:
                        if not target.startswith(_PARAMETER_REFERENCE_PREFIX):
                            continue
                        source = target.removeprefix(_PARAMETER_REFERENCE_PREFIX)
                        for name, values in (parameter_targets or {}).items():
                            if name.startswith(f"{source}["):
                                suffix = name.removeprefix(source)
                                bound[
                                    f"{scope.variadic_positional_parameter}{suffix}"
                                ] = set(values)
                else:
                    for parameter in scope.positional_parameters[index:]:
                        bound.setdefault(parameter, set()).update(resolved)
                continue
            if index < len(scope.positional_parameters):
                bound.setdefault(scope.positional_parameters[index], set()).update(resolved)
            else:
                extra_positional.update(resolved)
                extra_positional_values.append(set(resolved))
                if scope.variadic_positional_parameter is not None:
                    bound[f"{scope.variadic_positional_parameter}[{variadic_position!r}]"] = set(
                        resolved
                    )
                    variadic_position += 1
        if scope.variadic_positional_parameter is not None and extra_positional:
            bound[scope.variadic_positional_parameter] = extra_positional
            for index, resolved in enumerate(reversed(extra_positional_values), start=1):
                bound[f"{scope.variadic_positional_parameter}[{-index}]"] = set(resolved)

        extra_keywords: set[str] = set()
        for name, targets in call.keyword_targets:
            resolved = self._resolve_summary_targets(
                set(targets),
                state=state,
                parameter_targets=parameter_targets,
            )
            if name is not None and name in scope.keyword_parameters:
                bound.setdefault(name, set()).update(resolved)
            else:
                extra_keywords.update(resolved)
                if scope.variadic_keyword_parameter is not None and name is not None:
                    bound[f"{scope.variadic_keyword_parameter}[{name!r}]"] = set(resolved)
                elif scope.variadic_keyword_parameter is not None:
                    for target in targets:
                        if not target.startswith(_PARAMETER_REFERENCE_PREFIX):
                            continue
                        source = target.removeprefix(_PARAMETER_REFERENCE_PREFIX)
                        for parameter, values in (parameter_targets or {}).items():
                            if parameter.startswith(f"{source}["):
                                suffix = parameter.removeprefix(source)
                                bound[
                                    f"{scope.variadic_keyword_parameter}{suffix}"
                                ] = set(values)
        if scope.variadic_keyword_parameter is not None and extra_keywords:
            bound[scope.variadic_keyword_parameter] = extra_keywords
        for name, targets in scope.parameter_defaults.items():
            if name not in bound:
                bound[name] = self._resolve_summary_targets(
                    set(targets),
                    state=state,
                    parameter_targets=parameter_targets,
                )
        return bound

    def _resolve_summary_targets(
        self,
        targets: set[str],
        *,
        state: dict[str, set[str]] | None = None,
        parameter_targets: dict[str, set[str]] | None = None,
        active_results: frozenset[str] = frozenset(),
        instance_context: str | None = None,
    ) -> set[str]:
        """Resolve symbolic free/parameter callable identities in one summary.

        :param targets: Concrete and symbolic summary targets.
        :param state: Enclosing bindings used for free-name resolution.
        :param parameter_targets: Call arguments keyed by parameter name.
        :param active_results: Call-result markers already expanded on this path.
        :param instance_context: Caller identity used to clone returned deferred instances.
        :return: Concrete callable identities visible to the caller.
        """
        state = self.targets if state is None else state
        parameter_targets = {} if parameter_targets is None else parameter_targets
        resolved: set[str] = set()
        for target in targets:
            if instance_context is not None and target.startswith(
                (_GENERATOR_RESUME_PREFIX, _COROUTINE_INSTANCE_PREFIX)
            ):
                if target.startswith(_GENERATOR_RESUME_PREFIX):
                    function_target, phase, source_instance = self._generator_target(target)
                else:
                    source_instance, function_target = target.removeprefix(
                        _COROUTINE_INSTANCE_PREFIX
                    ).split("\0", 1)
                    phase = 0
                cloned_instance = f"{instance_context}:{source_instance or function_target}"
                source_parameters = self._lookup_instance_parameters(
                    source_instance or function_target
                )
                self.deferred_instance_parameters[cloned_instance] = {
                    name: self._resolve_summary_targets(
                        set(values),
                        state=state,
                        parameter_targets=parameter_targets,
                        active_results=active_results,
                    )
                    for name, values in source_parameters.items()
                }
                if target.startswith(_GENERATOR_RESUME_PREFIX):
                    resolved.add(
                        f"{_GENERATOR_RESUME_PREFIX}{phase}\0{cloned_instance}\0"
                        f"{function_target}"
                    )
                else:
                    resolved.add(
                        f"{_COROUTINE_INSTANCE_PREFIX}{cloned_instance}\0"
                        f"{function_target}"
                    )
            elif target.startswith(_FALLBACK_RESULT_PREFIX):
                fallback = self._lookup_fallback_result(target)
                if fallback is None:
                    continue
                primary, fallback_targets = fallback
                selected = self._resolve_summary_targets(
                    {primary},
                    state=state,
                    parameter_targets=parameter_targets,
                    active_results=active_results,
                    instance_context=instance_context,
                )
                if selected:
                    resolved.update(selected)
                else:
                    resolved.update(
                        self._resolve_summary_targets(
                            set(fallback_targets),
                            state=state,
                            parameter_targets=parameter_targets,
                            active_results=active_results,
                            instance_context=instance_context,
                        )
                    )
            elif target.startswith(_CALL_RESULT_PREFIX):
                if target in active_results:
                    continue
                call = self._lookup_call_result(target)
                if call is None:
                    continue
                nested_active_results = active_results | {target}
                callable_targets = self._resolve_summary_targets(
                    set(call.callable_targets),
                    state=state,
                    parameter_targets=parameter_targets,
                    active_results=nested_active_results,
                    instance_context=instance_context,
                )
                for callable_target in callable_targets:
                    deferred = self._lookup_deferred_scope(callable_target)
                    if deferred is None:
                        continue
                    bound = self._bind_reference_call(
                        deferred,
                        call,
                        state=state,
                        parameter_targets=parameter_targets,
                    )
                    if deferred.deferred_execution:
                        instance = target.removeprefix(_CALL_RESULT_PREFIX)
                        self.deferred_instance_parameters[instance] = {
                            name: set(targets) for name, targets in bound.items()
                        }
                        if deferred.generator_execution:
                            resolved.add(
                                f"{_GENERATOR_RESUME_PREFIX}0\0{instance}\0"
                                f"{callable_target}"
                            )
                        else:
                            resolved.add(
                                f"{_COROUTINE_INSTANCE_PREFIX}{instance}\0"
                                f"{callable_target}"
                            )
                        continue
                    resolved.update(
                        self._resolve_summary_targets(
                            set(deferred.return_targets),
                            state=state,
                            parameter_targets=bound,
                            active_results=nested_active_results,
                            instance_context=instance_context,
                        )
                    )
            elif target.startswith(_FREE_REFERENCE_PREFIX):
                resolved.update(
                    state.get(target.removeprefix(_FREE_REFERENCE_PREFIX), set())
                    - {_OPAQUE_REFERENCE_TARGET, _PRESERVE_REFERENCE_TARGET}
                )
            elif target.startswith(_PARAMETER_REFERENCE_PREFIX):
                parameter = target.removeprefix(_PARAMETER_REFERENCE_PREFIX)
                if parameter in parameter_targets:
                    resolved.update(parameter_targets[parameter])
                else:
                    resolved.add(target)
            elif target not in {_OPAQUE_REFERENCE_TARGET, _PRESERVE_REFERENCE_TARGET}:
                resolved.add(target)
        return resolved

    def _lookup_fallback_result(
        self,
        target: str,
    ) -> tuple[str, frozenset[str]] | None:
        """Find one symbolic mapping fallback in nested summaries.

        :param target: Fallback-result marker to resolve.
        :return: Primary projection and fallback targets, if known.
        """
        if target in self.fallback_results:
            return self.fallback_results[target]
        pending = list(self.deferred_scopes.values())
        while pending:
            scope = pending.pop()
            if target in scope.fallback_results:
                return scope.fallback_results[target]
            pending.extend(scope.deferred_scopes.values())
        return None

    def _lookup_call_result(self, target: str) -> _ReferenceCall | None:
        """Find one symbolic call-result descriptor in nested summaries.

        :param target: Call-result marker to resolve.
        :return: Matching symbolic call, if known.
        """
        if target in self.call_results:
            return self.call_results[target]
        pending = list(self.deferred_scopes.values())
        while pending:
            scope = pending.pop()
            if target in scope.call_results:
                return scope.call_results[target]
            pending.extend(scope.deferred_scopes.values())
        return None

    def _lookup_deferred_scope(self, target: str) -> _ReferenceScope | None:
        """Find one nested callable summary at any descendant depth.

        :param target: Qualified nested callable identity.
        :return: Matching callable scope, if known.
        """
        if target.startswith(_GENERATOR_RESUME_PREFIX):
            target, _, _ = self._generator_target(target)
        elif target.startswith(_COROUTINE_INSTANCE_PREFIX):
            _, target = target.removeprefix(_COROUTINE_INSTANCE_PREFIX).split("\0", 1)
        pending = list(self.deferred_scopes.values())
        if target in self.deferred_scopes:
            return self.deferred_scopes[target]
        if target in self.visible_deferred_scopes:
            return self.visible_deferred_scopes[target]
        pending.extend(self.visible_deferred_scopes.values())
        while pending:
            scope = pending.pop()
            if target in scope.deferred_scopes:
                return scope.deferred_scopes[target]
            pending.extend(scope.deferred_scopes.values())
        return None

    def _lookup_instance_parameters(self, instance: str) -> dict[str, set[str]]:
        """Find arguments captured by one deferred callable instance.

        :param instance: Stable deferred-instance identity.
        :return: Concrete parameter targets, or an empty mapping.
        """
        if instance in self.deferred_instance_parameters:
            return {
                name: set(targets)
                for name, targets in self.deferred_instance_parameters[instance].items()
            }
        pending = list(self.deferred_scopes.values())
        while pending:
            scope = pending.pop()
            if instance in scope.deferred_instance_parameters:
                return {
                    name: set(targets)
                    for name, targets in scope.deferred_instance_parameters[instance].items()
                }
            pending.extend(scope.deferred_scopes.values())
        return {}

    @staticmethod
    def _generator_target(target: str) -> tuple[str, int, str | None]:
        """Split a generator identity into its function and activation phase.

        :param target: Function identity or generator-resume marker.
        :return: Function identity, zero-based activation phase, and instance identity.
        """
        if not target.startswith(_GENERATOR_RESUME_PREFIX):
            return target, 0, None
        phase, instance, function_target = target.removeprefix(_GENERATOR_RESUME_PREFIX).split(
            "\0", 2
        )
        return function_target, int(phase), instance

    def _record_escape(self, targets: set[str]) -> None:
        """Record bindings visible when local callables leave direct-call tracking.

        :param targets: Callable identities that may escape.
        :return: ``None``.
        """
        if not targets:
            return
        state = self._snapshot_targets()
        for target in targets:
            self.escape_states.setdefault(target, []).append(state)

    def _record_free_call_events(
        self,
        node: ast.Call,
        names: list[str],
        call: _ReferenceCall,
    ) -> None:
        """Append alternative symbolic calls to the branch-aware summary graph.

        :param node: Call expression supplying stable source identity.
        :param names: Possible free callable names at this program point.
        :param call: Symbolic argument targets for later parameter binding.
        :return: ``None``.
        """
        if not names:
            return
        next_frontier: set[_ReferenceFlowPoint] = set()
        direct_effects = {
            name: set(self.targets[name]) for name in self.nonlocal_names if name in self.targets
        }
        for conditions in {conditions for _, conditions in self._free_call_frontier}:
            predecessors = {point for point in self._free_call_frontier if point[1] == conditions}
            for name in names:
                key = (id(node), name, conditions)
                event_id = self._free_call_ids.setdefault(key, len(self._free_call_ids) + 1)
                self._free_call_predecessors.setdefault(event_id, set()).update(predecessors)
                self._free_call_effect_states.setdefault(event_id, []).append(direct_effects)
                self._free_call_descriptors[event_id] = (name, call)
                self._free_call_conditions[event_id] = conditions
                if id(node) in self._activation_call_nodes:
                    self._activated_free_call_ids.add(event_id)
                if id(node) in self._exhaustion_call_nodes:
                    self._exhausted_free_call_ids.add(event_id)
                next_frontier.add((event_id, conditions))
        self._free_call_frontier = next_frontier

    def _record_invocation(
        self,
        target: str,
        state: dict[str, set[str]],
        active: set[str] | None = None,
        parameter_targets: dict[str, set[str]] | None = None,
    ) -> dict[str, set[str]]:
        """Record a nested call and propagate its free calls and nonlocal effects.

        :param target: Qualified nested callable identity.
        :param state: Enclosing bindings at invocation.
        :param active: Callable identities already expanded on this path.
        :param parameter_targets: Concrete arguments keyed by callee parameter.
        :return: Bindings after applying the callable's summarized effects.
        """
        instance: str | None = None
        if target.startswith(_COROUTINE_INSTANCE_PREFIX):
            instance, function_target = target.removeprefix(_COROUTINE_INSTANCE_PREFIX).split(
                "\0", 1
            )
            generator_phase = 0
        else:
            function_target, generator_phase, instance = self._generator_target(target)
        self.invocation_states.setdefault(function_target, []).append(state)
        deferred = self._lookup_deferred_scope(function_target)
        if deferred is None:
            return state
        if parameter_targets is None and instance is not None:
            parameter_targets = self._lookup_instance_parameters(instance)
        parameter_targets = {} if parameter_targets is None else parameter_targets
        active = set() if active is None else set(active)
        if function_target in active:
            return state
        active.add(function_target)
        working_state = {name: set(targets) for name, targets in state.items()}
        effects = deferred.nonlocal_effects
        terminal_free_calls = deferred.terminal_free_calls
        if deferred.generator_execution and generator_phase < len(deferred.yield_effects):
            effects = deferred.yield_effects[generator_phase]
            terminal_free_calls = deferred.yield_free_calls[generator_phase]
        if deferred.free_call_events:
            working_state = self._replay_free_call_events(
                deferred,
                working_state,
                active,
                parameter_targets,
                terminal_free_calls,
            )
        else:
            for free_name in deferred.free_calls:
                for free_target in working_state.get(free_name, set()) - {
                    _OPAQUE_REFERENCE_TARGET,
                    _PRESERVE_REFERENCE_TARGET,
                }:
                    nested = self._lookup_deferred_scope(free_target)
                    if nested is not None and not nested.deferred_execution:
                        working_state = self._record_invocation(
                            free_target,
                            working_state,
                            active,
                        )
        working_state = self._apply_summary_effects(
            effects,
            working_state,
            parameter_targets,
        )
        for parameter in deferred.called_parameters:
            for argument_target in parameter_targets.get(parameter, set()):
                nested = self._lookup_deferred_scope(argument_target)
                if nested is not None and not nested.deferred_execution:
                    working_state = self._record_invocation(
                        argument_target,
                        working_state,
                        active,
                    )
        for parameter in deferred.activated_parameters:
            next_arguments: set[str] = set()
            for argument_target in parameter_targets.get(parameter, set()):
                function_target, phase, argument_instance = self._generator_target(
                    argument_target
                )
                nested = self._lookup_deferred_scope(function_target)
                if nested is not None and nested.deferred_execution:
                    invocation_target = argument_target
                    exhausts_delegate = (
                        parameter in deferred.delegated_parameters
                        and deferred.generator_execution
                        and generator_phase >= len(deferred.yield_effects)
                        and nested.generator_execution
                    )
                    if exhausts_delegate:
                        invocation_target = (
                            f"{_GENERATOR_RESUME_PREFIX}{len(nested.yield_effects)}\0"
                            f"{argument_instance or parameter}\0{function_target}"
                        )
                    working_state = self._record_invocation(
                        invocation_target,
                        working_state,
                        active,
                    )
                    if nested.generator_execution and not exhausts_delegate:
                        next_arguments.add(
                            f"{_GENERATOR_RESUME_PREFIX}{phase + 1}\0"
                            f"{argument_instance or parameter}\0{function_target}"
                        )
                    else:
                        next_arguments.add(invocation_target)
                else:
                    next_arguments.add(argument_target)
            if next_arguments:
                parameter_targets[parameter] = next_arguments
        for parameter in deferred.exhausted_parameters:
            for argument_target in parameter_targets.get(parameter, set()):
                function_target, _, instance = self._generator_target(argument_target)
                nested = self._lookup_deferred_scope(function_target)
                if nested is not None and nested.generator_execution:
                    exhausted_target = (
                        f"{_GENERATOR_RESUME_PREFIX}{len(nested.yield_effects)}\0"
                        f"{instance or parameter}\0{function_target}"
                    )
                    working_state = self._record_invocation(
                        exhausted_target,
                        working_state,
                        active,
                    )
        if instance is not None and parameter_targets:
            self.deferred_instance_parameters[instance] = {
                name: set(targets) for name, targets in parameter_targets.items()
            }
        changed_names = {
            name
            for name in working_state.keys() | state.keys()
            if working_state.get(name, set()) != state.get(name, set())
        }
        self._invalidate_conditions(changed_names)
        self.targets.update({name: set(targets) for name, targets in working_state.items()})
        return working_state

    def _apply_summary_effects(
        self,
        effects: dict[str, frozenset[str]],
        state: dict[str, set[str]],
        parameter_targets: dict[str, set[str]],
    ) -> dict[str, set[str]]:
        """Apply direct nonlocal bindings from one summary program point.

        :param effects: Symbolic nonlocal values at the program point.
        :param state: Incoming enclosing bindings.
        :param parameter_targets: Concrete callee arguments.
        :return: Updated independent binding state.
        """
        working_state = {name: set(targets) for name, targets in state.items()}
        for name, targets in effects.items():
            effect_targets = set(targets)
            preserves_incoming = _PRESERVE_REFERENCE_TARGET in effect_targets
            effect_targets.discard(_PRESERVE_REFERENCE_TARGET)
            effect_targets = self._resolve_summary_targets(
                effect_targets,
                state=working_state,
                parameter_targets=parameter_targets,
            )
            if preserves_incoming:
                effect_targets.update(working_state.get(name, set()))
            working_state[name] = effect_targets
        return working_state

    def _replay_free_call_events(
        self,
        scope: _ReferenceScope,
        state: dict[str, set[str]],
        active: set[str],
        parameter_targets: dict[str, set[str]],
        terminal_free_calls: frozenset[_ReferenceFlowPoint],
    ) -> dict[str, set[str]]:
        """Replay branch-aware free calls in their reachable partial order.

        :param scope: Callee summary containing the event graph.
        :param state: Enclosing bindings at function entry.
        :param active: Callable identities already expanded on this path.
        :param parameter_targets: Concrete arguments keyed by callee parameter.
        :param terminal_free_calls: Event frontiers reached in this execution segment.
        :return: Bindings after all reachable terminal call events.
        """
        outputs: dict[_ReferenceFlowPoint, dict[str, set[str]]] = {}
        for _ in range(max(1, len(scope.free_call_events) * 2)):
            changed = False
            for event_id, event in scope.free_call_events.items():
                output_point = (event_id, event.conditions)
                if event.predecessors:
                    predecessor_states = [
                        state if predecessor[0] == 0 else outputs[predecessor]
                        for predecessor in event.predecessors
                        if predecessor[0] == 0 or predecessor in outputs
                    ]
                    if not predecessor_states:
                        continue
                    event_state = self._merge_target_states(predecessor_states)
                else:
                    event_state = {name: set(targets) for name, targets in state.items()}
                event_state = self._apply_summary_effects(
                    event.direct_effects,
                    event_state,
                    parameter_targets,
                )
                call_targets = event_state.get(event.name, set()) - {
                    _OPAQUE_REFERENCE_TARGET,
                    _PRESERVE_REFERENCE_TARGET,
                }
                result_state = event_state
                for call_target in call_targets:
                    nested = self._lookup_deferred_scope(call_target)
                    if nested is None or (nested.deferred_execution and not event.activate):
                        continue
                    invocation_target = call_target
                    if event.exhaust and nested.generator_execution:
                        function_target, _, instance = self._generator_target(call_target)
                        invocation_target = (
                            f"{_GENERATOR_RESUME_PREFIX}{len(nested.yield_effects)}\0"
                            f"{instance or event_id}\0{function_target}"
                        )
                    bound = self._bind_reference_call(
                        nested,
                        event.call,
                        state=event_state,
                        parameter_targets=parameter_targets,
                    )
                    result_state = self._record_invocation(
                        invocation_target,
                        result_state,
                        active,
                        parameter_targets=bound,
                    )
                previous = outputs.get(output_point)
                combined = (
                    result_state
                    if previous is None
                    else self._merge_target_states([previous, result_state])
                )
                if previous != combined:
                    outputs[output_point] = combined
                    changed = True
            if not changed:
                break
        terminal_states = [
            state if point[0] == 0 else outputs[point]
            for point in terminal_free_calls
            if point[0] == 0 or point in outputs
        ]
        return self._merge_target_states(terminal_states) if terminal_states else state

    def _snapshot_targets(self) -> dict[str, set[str]]:
        """Copy possible code identities without summary-control metadata."""
        return {name: set(targets) for name, targets in self.targets.items()}

    def _snapshot_bindings(self) -> _FunctionScopeState:
        """Copy code identities and the current free-call frontier.

        :return: Independent flow-state copy.
        """
        return (
            self._snapshot_targets(),
            frozenset(self._free_call_frontier),
            frozenset(self._yield_phases),
        )

    def _restore_bindings(self, state: _FunctionScopeState) -> None:
        """Restore possible code identities and the free-call frontier."""
        targets, frontier, yield_phases = state
        self.targets = {name: set(values) for name, values in targets.items()}
        self._free_call_frontier = set(frontier)
        self._yield_phases = set(yield_phases)

    def _merge_target_states(
        self,
        states: list[dict[str, set[str]]],
    ) -> dict[str, set[str]]:
        """Join possible code identities from alternative control-flow paths."""
        return {
            name: {
                *(target for state in states for target in state.get(name, set())),
                *(
                    {_OPAQUE_REFERENCE_TARGET}
                    if any(name in state and not state[name] for state in states)
                    else set()
                ),
                *(
                    {_PRESERVE_REFERENCE_TARGET}
                    if name in self.nonlocal_names and any(name not in state for state in states)
                    else set()
                ),
            }
            for name in {name for state in states for name in state}
        }

    def _merge_bindings(self, states: list[_FunctionScopeState]) -> _FunctionScopeState:
        """Join bindings and branch-aware free-call frontiers."""
        return (
            self._merge_target_states([state[0] for state in states]),
            frozenset(event for state in states for event in state[1]),
            frozenset(phase for state in states for phase in state[2]),
        )

    def _stable_condition(self, node: ast.expr) -> tuple[str, bool] | None:
        """Return a reusable identity for one side-effect-free branch predicate.

        :param node: Conditional expression to identify.
        :return: Structural predicate identity and its non-negated polarity.
        """
        polarity = True
        while isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            polarity = not polarity
            node = node.operand
        if not isinstance(node, (ast.Name, ast.Attribute, ast.Subscript, ast.Compare)):
            return None
        if any(
            isinstance(child, (ast.Call, ast.Await, ast.NamedExpr, ast.Yield, ast.YieldFrom))
            for child in ast.walk(node)
        ):
            return None
        key = ast.dump(node, include_attributes=False)
        self._condition_dependencies[key] = frozenset(
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        )
        return key, polarity

    def _restrict_condition(self, key: str, value: bool) -> bool:
        """Restrict every current path to one stable predicate outcome.

        :param key: Structural predicate identity.
        :param value: Required truth value.
        :return: Whether at least one path remains reachable.
        """
        restricted: set[_ReferenceFlowPoint] = set()
        for event_id, conditions in self._free_call_frontier:
            known = dict(conditions)
            if key in known and known[key] != value:
                continue
            known[key] = value
            restricted.add((event_id, frozenset(known.items())))
        self._free_call_frontier = restricted
        return bool(restricted)

    def visit_If(self, node: ast.If) -> None:
        """Join branches while retaining reusable side-effect-free predicates."""
        condition = self._stable_condition(node.test)
        if condition is None or isinstance(node.test, ast.Constant):
            super().visit_If(node)
            return
        self.visit(node.test)
        base = self._snapshot_bindings()
        states: list[_FunctionScopeState] = []
        key, polarity = condition
        for truth, statements in ((True, node.body), (False, node.orelse)):
            self._restore_bindings(base)
            if not self._restrict_condition(key, truth == polarity):
                continue
            self._visit_suite(statements)
            states.append(self._snapshot_bindings())
        self._restore_bindings(self._merge_bindings(states or [base]))

    def _statement_cannot_raise(self, statement: ast.stmt) -> bool:
        """Recognize calls to visible helpers proven exception-free."""
        if super()._statement_cannot_raise(statement):
            return True
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
        ):
            return False
        parent = self.qualified_name.rpartition(".")[0]
        target = f"{parent}.{statement.value.func.id}" if parent else statement.value.func.id
        scope = self._lookup_deferred_scope(target)
        return scope is not None and not scope.may_raise

    def _exception_occurs_after_effects(self, statement: ast.stmt) -> bool:
        """Recognize generator exhaustion after a summarized resume segment."""
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "next"
            and len(statement.value.args) == 1
        ):
            return False
        for target in self._expression_targets(statement.value.args[0]):
            function_target, phase, _ = self._generator_target(target)
            scope = self._lookup_deferred_scope(function_target)
            if (
                scope is not None
                and scope.generator_execution
                and phase > len(scope.yield_effects)
            ):
                return True
        return False

    def visit_Name(self, node: ast.Name) -> None:
        """Record assignment and deletion targets as local slots."""
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._bind(node.id)
        elif isinstance(node.ctx, ast.Load) and not self._suppress_escape:
            self._record_escape(self._expression_targets(node))

    def visit_Assign(self, node: ast.Assign) -> None:
        """Propagate simple local callable aliases without treating them as escapes."""
        targets = self._expression_targets(node.value)
        simple_targets = all(isinstance(target, ast.Name) for target in node.targets)
        previous_suppression = self._suppress_escape
        self._suppress_escape = previous_suppression or (bool(targets) and simple_targets)
        self.visit(node.value)
        self._suppress_escape = previous_suppression
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._bind_targets(target.id, targets)
            else:
                self.visit(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Propagate a simple annotated callable alias."""
        self.visit(node.annotation)
        targets = self._expression_targets(node.value) if node.value is not None else set()
        if node.value is not None:
            previous_suppression = self._suppress_escape
            self._suppress_escape = previous_suppression or (
                bool(targets) and isinstance(node.target, ast.Name)
            )
            self.visit(node.value)
            self._suppress_escape = previous_suppression
        if isinstance(node.target, ast.Name):
            self._bind_targets(node.target.id, targets)
        else:
            self.visit(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """Propagate a simple assignment-expression callable alias."""
        targets = self._expression_targets(node.value)
        previous_suppression = self._suppress_escape
        self._suppress_escape = previous_suppression or (
            bool(targets) and isinstance(node.target, ast.Name)
        )
        self.visit(node.value)
        self._suppress_escape = previous_suppression
        if isinstance(node.target, ast.Name):
            self._bind_targets(node.target.id, targets)
        else:
            self.visit(node.target)

    def visit_Call(self, node: ast.Call) -> None:
        """Record enclosing bindings visible to a directly invoked local callable."""
        callable_targets = self._expression_targets(node.func)
        call = _ReferenceCall(
            callable_targets=frozenset(callable_targets),
            positional_targets=tuple(
                frozenset(self._expression_targets(argument)) for argument in node.args
            ),
            positional_unpacking=tuple(
                isinstance(argument, ast.Starred) for argument in node.args
            ),
            keyword_targets=tuple(
                (keyword.arg, frozenset(self._expression_targets(keyword.value)))
                for keyword in node.keywords
            ),
        )
        symbolic_free_calls = [
            target.removeprefix(_FREE_REFERENCE_PREFIX)
            for target in callable_targets
            if target.startswith(_FREE_REFERENCE_PREFIX)
        ]
        symbolic_parameter_calls = {
            target.removeprefix(_PARAMETER_REFERENCE_PREFIX)
            for target in callable_targets
            if target.startswith(_PARAMETER_REFERENCE_PREFIX)
        }
        bound_activated_targets = {
            target.removeprefix(_DEFERRED_ACTIVATOR_PREFIX)
            for target in callable_targets
            if target.startswith(_DEFERRED_ACTIVATOR_PREFIX)
        }
        calls_next_alias = _NEXT_ACTIVATOR_TARGET in callable_targets
        concrete_targets = {
            target
            for target in callable_targets
            if not target.startswith(
                (
                    _FREE_REFERENCE_PREFIX,
                    _PARAMETER_REFERENCE_PREFIX,
                    _DEFERRED_ACTIVATOR_PREFIX,
                )
            )
            and target != _NEXT_ACTIVATOR_TARGET
        }
        self._record_free_call_events(node, symbolic_free_calls, call)
        self.free_calls.extend(symbolic_free_calls)
        self.called_parameters.update(symbolic_parameter_calls)
        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "send",
            "__anext__",
            "__next__",
        }:
            self.activated_parameters.update(
                target.removeprefix(_PARAMETER_REFERENCE_PREFIX)
                for target in self._expression_targets(node.func.value)
                if target.startswith(_PARAMETER_REFERENCE_PREFIX)
            )
        deferred_targets = {
            target
            for target in concrete_targets
            if (scope := self._lookup_deferred_scope(target)) is not None
            and scope.deferred_execution
        }
        active_targets = concrete_targets - deferred_targets
        activated_argument_targets: set[str] = set(bound_activated_targets)
        activated_binding_name: str | None = None
        if (
            isinstance(node.func, ast.Name)
            and (node.func.id in {"next", "anext"} or calls_next_alias)
            and node.args
        ):
            argument_targets = self._expression_targets(node.args[0])
            self.activated_parameters.update(
                target.removeprefix(_PARAMETER_REFERENCE_PREFIX)
                for target in argument_targets
                if target.startswith(_PARAMETER_REFERENCE_PREFIX)
            )
            activated_argument_targets.update(
                {
                    target
                    for target in argument_targets
                    if (scope := self._lookup_deferred_scope(target)) is not None
                    and scope.deferred_execution
                }
            )
            if isinstance(node.args[0], ast.Name):
                activated_binding_name = node.args[0].id
        elif isinstance(node.func, ast.Attribute) and node.func.attr in {
            "send",
            "__anext__",
            "__next__",
        }:
            activated_argument_targets = {
                target
                for target in self._expression_targets(node.func.value)
                if (scope := self._lookup_deferred_scope(target)) is not None
                and scope.deferred_execution
            }
            if isinstance(node.func.value, ast.Name):
                activated_binding_name = node.func.value.id
        handled_argument_indexes: set[int] = set()
        handled_keyword_indexes: set[int] = set()
        invoked_argument_indexes: set[int] = set()
        escaped_argument_targets: set[str] = set()
        invoked_argument_targets: set[str] = set()
        invocation_parameters: dict[str, dict[str, set[str]]] = {}
        for target in active_targets:
            deferred = self._lookup_deferred_scope(target)
            if deferred is None:
                continue
            bound = self._bind_reference_call(deferred, call)
            invocation_parameters[target] = bound
            handled_argument_indexes.update(
                range(min(len(node.args), len(deferred.positional_parameters)))
            )
            if deferred.variadic_positional_parameter is not None:
                handled_argument_indexes.update(range(len(node.args)))
            handled_keyword_indexes.update(
                index
                for index, keyword in enumerate(node.keywords)
                if (
                    keyword.arg in deferred.keyword_parameters
                    or deferred.variadic_keyword_parameter is not None
                )
            )
            invoked_argument_indexes.update(
                index
                for index, parameter in enumerate(deferred.positional_parameters)
                if parameter in deferred.called_parameters
                or parameter in deferred.activated_parameters
            )
            invoked_argument_targets.update(
                argument_target
                for parameter in deferred.called_parameters | deferred.activated_parameters
                for argument_target in bound.get(parameter, set())
            )
            escaped_argument_targets.update(
                argument_target
                for parameter in deferred.escaped_parameters
                for argument_target in bound.get(parameter, set())
            )
        previous_suppression = self._suppress_escape
        self._suppress_escape = True
        self.visit(node.func)
        self._suppress_escape = previous_suppression
        for index, argument in enumerate(node.args):
            if index in handled_argument_indexes:
                argument_targets = self._expression_targets(argument)
                if index in invoked_argument_indexes:
                    invoked_argument_targets.update(argument_targets)
                previous_suppression = self._suppress_escape
                self._suppress_escape = True
                self.visit(argument)
                self._suppress_escape = previous_suppression
                continue
            self.visit(argument)
        for index, keyword in enumerate(node.keywords):
            if index in handled_keyword_indexes:
                previous_suppression = self._suppress_escape
                self._suppress_escape = True
                self.visit(keyword.value)
                self._suppress_escape = previous_suppression
            else:
                self.visit(keyword.value)
        if escaped_argument_targets:
            self._record_escape(escaped_argument_targets)
        if active_targets or activated_argument_targets or invoked_argument_targets:
            state = self._snapshot_targets()
            for target in active_targets | activated_argument_targets | invoked_argument_targets:
                state = self._record_invocation(
                    target,
                    state,
                    parameter_targets=invocation_parameters.get(target),
                )
            if activated_argument_targets:
                continuation_targets: dict[str, str] = {}
                for target in activated_argument_targets:
                    function_target, phase, instance = self._generator_target(target)
                    deferred = self._lookup_deferred_scope(function_target)
                    if deferred is not None and deferred.generator_execution:
                        continuation_targets[target] = (
                            f"{_GENERATOR_RESUME_PREFIX}{phase + 1}\0"
                            f"{instance or activated_binding_name or function_target}\0"
                            f"{function_target}"
                        )
                    else:
                        continuation_targets[target] = target
                for name, targets in self.targets.items():
                    if targets & continuation_targets.keys():
                        self.targets[name] = {
                            continuation_targets.get(target, target) for target in targets
                        }

    def visit_Await(self, node: ast.Await) -> None:
        """Activate a statically known coroutine at the await program point."""
        if isinstance(node.value, ast.Call):
            self._activation_call_nodes.add(id(node.value))
        self.visit(node.value)
        state = self._snapshot_targets()
        targets = self._expression_targets(node.value)
        invocation_parameters: dict[str, dict[str, set[str]]] = {}
        if isinstance(node.value, ast.Call):
            call = _ReferenceCall(
                callable_targets=frozenset(self._expression_targets(node.value.func)),
                positional_targets=tuple(
                    frozenset(self._expression_targets(argument)) for argument in node.value.args
                ),
                positional_unpacking=tuple(
                    isinstance(argument, ast.Starred) for argument in node.value.args
                ),
                keyword_targets=tuple(
                    (keyword.arg, frozenset(self._expression_targets(keyword.value)))
                    for keyword in node.value.keywords
                ),
            )
            for callable_target in call.callable_targets:
                deferred = self._lookup_deferred_scope(callable_target)
                if deferred is not None:
                    invocation_parameters[callable_target] = self._bind_reference_call(
                        deferred,
                        call,
                    )
        self.activated_parameters.update(
            target.removeprefix(_PARAMETER_REFERENCE_PREFIX)
            for target in targets
            if target.startswith(_PARAMETER_REFERENCE_PREFIX)
        )
        for target in targets:
            deferred = self._lookup_deferred_scope(target)
            if deferred is not None and deferred.deferred_execution:
                self._record_invocation(
                    target,
                    state,
                    parameter_targets=invocation_parameters.get(target),
                )

    def visit_Yield(self, node: ast.Yield) -> None:
        """Capture direct nonlocal effects visible at one suspension boundary."""
        if node.value is not None:
            self.visit(node.value)
        effects = {
            name: set(self.targets[name]) for name in self.nonlocal_names if name in self.targets
        }
        for phase in self._yield_phases:
            self.yield_effects.setdefault(phase, []).append(effects)
            self.yield_free_calls.setdefault(phase, set()).update(self._free_call_frontier)
        self._yield_phases = {phase + 1 for phase in self._yield_phases}

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        """Activate a delegated generator before capturing suspension state."""
        if isinstance(node.value, ast.Call):
            self._activation_call_nodes.add(id(node.value))
        self.visit(node.value)
        delegated = {
            target.removeprefix(_PARAMETER_REFERENCE_PREFIX)
            for target in self._expression_targets(node.value)
            if target.startswith(_PARAMETER_REFERENCE_PREFIX)
        }
        self.activated_parameters.update(delegated)
        self.delegated_parameters.update(delegated)
        effects = {
            name: set(self.targets[name]) for name in self.nonlocal_names if name in self.targets
        }
        for phase in self._yield_phases:
            self.yield_effects.setdefault(phase, []).append(effects)
            self.yield_free_calls.setdefault(phase, set()).update(self._free_call_frontier)
        self._yield_phases = {phase + 1 for phase in self._yield_phases}

    def visit_Return(self, node: ast.Return) -> None:
        """Record callable return identities before capturing the transfer state."""
        targets = self._expression_targets(node.value) if node.value is not None else set()
        self.return_targets.update(targets)
        self._record_escape(
            {
                target
                for target in targets
                if not target.startswith((_FREE_REFERENCE_PREFIX, _PARAMETER_REFERENCE_PREFIX))
            }
        )
        if node.value is not None:
            previous_suppression = self._suppress_escape
            self._suppress_escape = previous_suppression or bool(targets)
            self.visit(node.value)
            self._suppress_escape = previous_suppression
        self._control_transfers.append(("return", self._snapshot_bindings()))

    def visit_Global(self, node: ast.Global) -> None:
        """Record names explicitly resolved in module scope."""
        self.global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """Record names explicitly resolved in an enclosing function scope."""
        self.nonlocal_names.update(node.names)

    def visit_Import(self, node: ast.Import) -> None:
        """Record function-local module import identities."""
        for alias in node.names:
            name = alias.asname or alias.name.partition(".")[0]
            target = alias.name if alias.asname else name
            self._bind(name, target)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Record function-local from-import identities."""
        module = _resolve_relative_module(self.import_package, node.level, node.module)
        for alias in node.names:
            if alias.name == "*":
                continue
            name = alias.asname or alias.name
            target = f"{module}.{alias.name}" if module else alias.name
            self._bind(name, target)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Bind a nested function and inspect only its enclosing-scope header."""
        for expression in node.decorator_list:
            self.visit(expression)
        positional = [*node.args.posonlyargs, *node.args.args]
        default_targets: dict[str, frozenset[str]] = {}
        default_parameters = positional[-len(node.args.defaults) :] if node.args.defaults else []
        for argument, expression in zip(
            default_parameters,
            node.args.defaults,
            strict=True,
        ):
            default_targets[argument.arg] = frozenset(self._expression_targets(expression))
            previous_suppression = self._suppress_escape
            self._suppress_escape = True
            self.visit(expression)
            self._suppress_escape = previous_suppression
        for argument, expression in zip(
            node.args.kwonlyargs,
            node.args.kw_defaults,
            strict=True,
        ):
            if expression is None:
                continue
            default_targets[argument.arg] = frozenset(self._expression_targets(expression))
            previous_suppression = self._suppress_escape
            self._suppress_escape = True
            self.visit(expression)
            self._suppress_escape = previous_suppression
        for expression in _function_annotation_expressions(node):
            self.visit(expression)
        target = f"{self.qualified_name}.{node.name}"
        self.deferred_scopes[target] = replace(
            _collect_function_reference_scope(
                node,
                target,
                self.import_package,
                visible_deferred_scopes={
                    **self.visible_deferred_scopes,
                    **self.deferred_scopes,
                },
            ),
            parameter_defaults=default_targets,
        )
        self._bind(node.name, target)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Execute a nested class body and apply its nonlocal side effects."""
        for expression in _class_header_expressions(node):
            self.visit(expression)
        class_collector = _FunctionScopeCollector(
            f"{self.qualified_name}.{node.name}",
            self.import_package,
            ast.arguments(
                posonlyargs=[],
                args=[],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
        )
        class_collector._visit_suite(node.body)
        class_scope = class_collector.build()
        for name, targets in class_scope.nonlocal_effects.items():
            self._bind_targets(name, set(targets))
        self._bind(node.name, f"{self.qualified_name}.{node.name}")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Inspect lambda defaults but keep its body in the lambda's own scope."""
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)

    def _visit_comprehension(
        self,
        generators: list[ast.comprehension],
        values: tuple[ast.AST, ...],
        *,
        eager: bool,
    ) -> None:
        """Inspect comprehension expressions without leaking generator targets.

        :param generators: Comprehension generator clauses.
        :param values: Result expressions evaluated in the implicit scope.
        :param eager: Whether result evaluation begins during construction.
        :return: ``None``.
        """
        if not generators:
            return
        self.visit(generators[0].iter)
        if iterable_definitely_empty(generators[0].iter):
            return
        base = self._snapshot_bindings()
        guaranteed_state = base
        guaranteed_reach = iterable_definitely_nonempty(generators[0].iter)
        stopped = False
        conditions_are_conditional = False
        for condition in generators[0].ifs:
            condition_base = self._snapshot_bindings()
            self.visit(condition)
            if conditions_are_conditional:
                self._restore_bindings(
                    self._merge_bindings([condition_base, self._snapshot_bindings()])
                )
            if isinstance(condition, ast.Constant) and not bool(condition.value):
                stopped = True
                break
            if not (isinstance(condition, ast.Constant) and bool(condition.value)):
                conditions_are_conditional = True
        if guaranteed_reach:
            guaranteed_state = self._snapshot_bindings()
        if generators[0].ifs:
            guaranteed_reach = False
        for generator in generators[1:] if not stopped else ():
            self.visit(generator.iter)
            if guaranteed_reach:
                guaranteed_state = self._snapshot_bindings()
            if iterable_definitely_empty(generator.iter):
                stopped = True
                break
            conditions_are_conditional = False
            for condition in generator.ifs:
                condition_base = self._snapshot_bindings()
                self.visit(condition)
                if conditions_are_conditional:
                    self._restore_bindings(
                        self._merge_bindings([condition_base, self._snapshot_bindings()])
                    )
                if isinstance(condition, ast.Constant) and not bool(condition.value):
                    stopped = True
                    break
                if not (isinstance(condition, ast.Constant) and bool(condition.value)):
                    conditions_are_conditional = True
            if stopped:
                break
            if guaranteed_reach:
                guaranteed_state = self._snapshot_bindings()
            if generator.ifs or not iterable_definitely_nonempty(generator.iter):
                guaranteed_reach = False
        exact_count = None if stopped else comprehension_exact_result_count(generators)
        repetitions = exact_count if exact_count is not None else int(not stopped)
        for _ in range(repetitions):
            for value in values:
                self.visit(value)
        if not eager:
            self._restore_bindings(base)
        elif stopped or not comprehension_definitely_runs(generators):
            self._restore_bindings(
                self._merge_bindings([guaranteed_state, self._snapshot_bindings()])
            )

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Inspect list-comprehension expressions for outer walrus bindings."""
        self._visit_comprehension(node.generators, (node.elt,), eager=True)

    visit_SetComp = visit_ListComp

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """Join deferred generator-expression assignment targets with the base state."""
        self._visit_comprehension(node.generators, (node.elt,), eager=False)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Inspect dictionary-comprehension expressions for outer bindings."""
        self._visit_comprehension(node.generators, (node.key, node.value), eager=True)

    def _enter_exception_handler(self, handler: ast.ExceptHandler) -> None:
        """Bind a function-local exception alias on handler entry."""
        if handler.name is not None:
            self._bind(handler.name)

    def _exit_exception_handler(self, handler: ast.ExceptHandler) -> None:
        """Leave a function-local exception alias unbound after cleanup."""
        if handler.name is not None:
            self._bind(handler.name)

    def _visit_loop_target(self, target: ast.expr) -> None:
        """Apply a loop target to the function's compile-time slots."""
        self.visit(target)

    def _visit_generator_loop(self, node: ast.For | ast.AsyncFor) -> None:
        """Activate or exhaust a statically known generator iterable.

        :param node: Synchronous or asynchronous loop.
        :return: ``None``.
        """
        exhausts = not _loop_body_can_terminate_early(node.body)
        if isinstance(node.iter, ast.Call):
            self._activation_call_nodes.add(id(node.iter))
            if exhausts:
                self._exhaustion_call_nodes.add(id(node.iter))
        if isinstance(node, ast.AsyncFor):
            super().visit_AsyncFor(node)
        else:
            super().visit_For(node)
        state = self._snapshot_targets()
        targets = self._expression_targets(node.iter)
        parameter_names = {
            target.removeprefix(_PARAMETER_REFERENCE_PREFIX)
            for target in targets
            if target.startswith(_PARAMETER_REFERENCE_PREFIX)
        }
        if exhausts:
            self.exhausted_parameters.update(parameter_names)
        else:
            self.activated_parameters.update(parameter_names)
        for target in targets:
            function_target, _, instance = self._generator_target(target)
            deferred = self._lookup_deferred_scope(function_target)
            if deferred is not None and deferred.deferred_execution:
                invocation_target = target
                if exhausts and deferred.generator_execution:
                    invocation_target = (
                        f"{_GENERATOR_RESUME_PREFIX}{len(deferred.yield_effects)}\0"
                        f"{instance or id(node)}\0{function_target}"
                    )
                state = self._record_invocation(invocation_target, state)

    def visit_For(self, node: ast.For) -> None:
        """Analyze a synchronous loop and its generator activation semantics."""
        self._visit_generator_loop(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Analyze an asynchronous loop and its generator activation semantics."""
        self._visit_generator_loop(node)

    def _visit_match_pattern(self, pattern: ast.pattern) -> None:
        """Bind every capture introduced by a match pattern."""
        for name in _match_capture_names(pattern):
            self._bind(name)

    def build(self) -> _ReferenceScope:
        """Freeze the collected scope after applying declarations.

        :return: Immutable reference-scope description.
        """
        local_names = self.local_names - self.global_names - self.nonlocal_names
        return _ReferenceScope(
            local_names=frozenset(local_names),
            global_names=frozenset(self.global_names),
            nonlocal_names=frozenset(self.nonlocal_names),
            targets={
                name: frozenset(targets)
                for name, targets in self.targets.items()
                if name in local_names
            },
            qualified_name=self.qualified_name,
            invocation_targets={
                target: {
                    name: frozenset(targets)
                    for name, targets in self._merge_target_states(states).items()
                    if name in local_names
                }
                for target, states in self.invocation_states.items()
            },
            escape_targets={
                target: {
                    name: frozenset(targets)
                    for name, targets in self._merge_target_states(states).items()
                    if name in local_names
                }
                for target, states in self.escape_states.items()
            },
            nonlocal_effects={
                name: frozenset(self.targets[name])
                for name in self.nonlocal_names
                if name in self.targets
            },
            free_calls=tuple(self.free_calls),
            free_call_events={
                event_id: _ReferenceCallEvent(
                    name=self._free_call_descriptors[event_id][0],
                    predecessors=frozenset(self._free_call_predecessors[event_id]),
                    direct_effects={
                        name: frozenset(targets)
                        for name, targets in self._merge_target_states(
                            self._free_call_effect_states[event_id]
                        ).items()
                        if name in self.nonlocal_names
                    },
                    call=self._free_call_descriptors[event_id][1],
                    activate=event_id in self._activated_free_call_ids,
                    exhaust=event_id in self._exhausted_free_call_ids,
                    conditions=self._free_call_conditions[event_id],
                )
                for event_id in self._free_call_descriptors
            },
            terminal_free_calls=frozenset(
                {
                    *self._free_call_frontier,
                    *(event for _, state in self._control_transfers for event in state[1]),
                }
            ),
            called_parameters=frozenset(self.called_parameters),
            activated_parameters=frozenset(self.activated_parameters),
            exhausted_parameters=frozenset(self.exhausted_parameters),
            delegated_parameters=frozenset(self.delegated_parameters),
            escaped_parameters=frozenset(
                target.removeprefix(_PARAMETER_REFERENCE_PREFIX)
                for target in self.escape_states
                if target.startswith(_PARAMETER_REFERENCE_PREFIX)
            ),
            positional_parameters=self.positional_parameters,
            keyword_parameters=frozenset(self.keyword_parameters),
            variadic_positional_parameter=self.variadic_positional_parameter,
            variadic_keyword_parameter=self.variadic_keyword_parameter,
            return_targets=frozenset(self.return_targets),
            call_results=dict(self.call_results),
            deferred_instance_parameters={
                instance: {name: frozenset(targets) for name, targets in parameters.items()}
                for instance, parameters in self.deferred_instance_parameters.items()
            },
            fallback_results=dict(self.fallback_results),
            yield_effects=tuple(
                {
                    name: frozenset(targets)
                    for name, targets in self._merge_target_states(
                        self.yield_effects.get(phase, [{}])
                    ).items()
                }
                for phase in range(max(self.yield_effects, default=-1) + 1)
            ),
            yield_free_calls=tuple(
                frozenset(self.yield_free_calls.get(phase, {(0, frozenset())}))
                for phase in range(max(self.yield_effects, default=-1) + 1)
            ),
            deferred_execution=self.deferred_execution,
            generator_execution=self.generator_execution,
            deferred_scopes=dict(self.deferred_scopes),
        )


def _collect_function_reference_scope(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    qualified_name: str,
    import_package: str,
    *,
    visible_deferred_scopes: dict[str, _ReferenceScope] | None = None,
) -> _ReferenceScope:
    """Build lexical reference metadata for one emitted function.

    :param node: Function definition to inspect.
    :param qualified_name: Full emitted name.
    :param import_package: Package used for relative imports.
    :param visible_deferred_scopes: Already defined sibling callables visible by name.
    :return: Function lexical scope.
    """
    slot_collector = _FunctionLexicalSlotCollector(node.args)
    slot_collector.local_names.update(_type_parameter_names(node))
    for statement in node.body:
        slot_collector.visit(statement)

    collector = _FunctionScopeCollector(
        qualified_name,
        import_package,
        node.args,
        deferred_execution=_function_defers_execution(node),
        generator_execution=_function_is_generator(node),
        visible_deferred_scopes=visible_deferred_scopes,
    )
    collector.local_names.update(slot_collector.local_names)
    collector.global_names.update(slot_collector.global_names)
    collector.nonlocal_names.update(slot_collector.nonlocal_names)
    collector._visit_suite(node.body)
    return replace(
        collector.build(),
        may_raise=_function_body_may_raise(node),
    )


_ReferenceBindingState = tuple[
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, set[str]],
]


class ReferenceVisitor(BranchingVisitor[_ReferenceBindingState]):
    """Resolve references through Python lexical scopes without flattening locals."""

    def __init__(
        self,
        *,
        qualified_name: str = "",
        import_package: str = "",
        scope: _ReferenceScope | None = None,
        enclosing_scopes: tuple[_ReferenceScope, ...] = (),
        walrus_owner: ReferenceVisitor | None = None,
        module_bindings: dict[str, set[str]] | None = None,
    ) -> None:
        """Initialize a scope-aware reference accumulator.

        :param qualified_name: Qualified identity of the unit being inspected.
        :param import_package: Package used to resolve relative imports.
        :param scope: Function lexical scope, or ``None`` for module/class execution.
        :param enclosing_scopes: Lexically enclosing function scopes, outermost first.
        :param walrus_owner: Containing scope updated by comprehension assignment expressions.
        :param module_bindings: Import identities visible when immediate code executes.
        """
        super().__init__()
        self.qualified_name = qualified_name
        self.import_package = import_package
        self.scope = scope
        self.enclosing_scopes = enclosing_scopes
        self.walrus_owner = walrus_owner
        self.module_bindings = {
            name: set(targets) for name, targets in (module_bindings or {}).items()
        }
        self.names: set[str] = set()
        self.resolved_names: set[str] = set()
        self.attributes: set[str] = set()
        self.module_attributes: set[str] = set()
        self._current_bindings: dict[str, set[str]] = {}
        self._global_bindings: dict[str, set[str]] = {}
        self._nonlocal_bindings: dict[str, set[str]] = {}
        self._in_header = False
        self._type_parameter_bindings: set[str] = set()

    def collect_definition(self, node: DefinitionNode, *, include_header: bool = True) -> None:
        """Collect one definition while separating its header and body scopes.

        :param node: Root emitted definition.
        :param include_header: Whether this unit owns header evaluation; enclosing
            function/class units own headers for nested definitions.
        :return: ``None``.
        """
        type_parameter_names = _type_parameter_names(node)
        if isinstance(node, ast.ClassDef):
            if include_header:
                self._in_header = True
                for decorator in node.decorator_list:
                    self.visit(decorator)
                self._in_header = False
            if type_parameter_names:
                type_scope = _ReferenceScope(
                    local_names=type_parameter_names,
                    global_names=frozenset(),
                    nonlocal_names=frozenset(),
                    targets={},
                )
                self.enclosing_scopes = (*self.enclosing_scopes, type_scope)
                for name in type_parameter_names:
                    self._current_bindings[name] = set()
            if include_header:
                for expression in (
                    *getattr(node, "type_params", ()),
                    *node.bases,
                    *(keyword.value for keyword in node.keywords),
                ):
                    self.visit(expression)
        elif include_header:
            self._in_header = True
            for expression in _function_outer_header_expressions(node):
                self.visit(expression)
            previous_type_parameters = set(self._type_parameter_bindings)
            self._type_parameter_bindings.update(type_parameter_names)
            for expression in _function_annotation_expressions(node):
                self.visit(expression)
            self._type_parameter_bindings = previous_type_parameters
            self._in_header = False
        self._visit_suite(node.body)

    def _resolve_enclosing_name(self, name: str) -> set[str]:
        """Resolve a free/nonlocal slot through enclosing functions or the module.

        :param name: Loaded bare name.
        :return: Possible qualified targets or the unresolved spelling.
        """
        leaf_descendant = self.qualified_name
        descendant = leaf_descendant
        for enclosing in reversed(self.enclosing_scopes):
            if name in enclosing.global_names:
                break
            if name in enclosing.local_names:
                invocation = enclosing.invocation_targets.get(leaf_descendant)
                escape = enclosing.escape_targets.get(leaf_descendant)
                if invocation is None and escape is None:
                    invocation = enclosing.invocation_targets.get(descendant)
                    escape = enclosing.escape_targets.get(descendant)
                targets = (
                    set(invocation.get(name, frozenset()))
                    if invocation is not None
                    else set(enclosing.targets.get(name, frozenset()))
                )
                if escape is not None:
                    targets.update(escape.get(name, frozenset()))
                    targets.update(enclosing.targets.get(name, frozenset()))
                return targets
            descendant = enclosing.qualified_name
        if (self._in_header or self.scope is None) and name in self.module_bindings:
            return set(self.module_bindings[name])
        return {name}

    def _enclosing_name_is_resolved(self, name: str) -> bool:
        """Return whether an enclosing lexical slot supplies a code identity.

        :param name: Loaded bare name.
        :return: Whether an enclosing scope proves a target identity.
        """
        leaf_descendant = self.qualified_name
        descendant = leaf_descendant
        for enclosing in reversed(self.enclosing_scopes):
            if name in enclosing.global_names:
                return False
            if name in enclosing.local_names:
                invocation = enclosing.invocation_targets.get(leaf_descendant)
                escape = enclosing.escape_targets.get(leaf_descendant)
                if invocation is None and escape is None:
                    invocation = enclosing.invocation_targets.get(descendant)
                    escape = enclosing.escape_targets.get(descendant)
                targets = (
                    set(invocation.get(name, frozenset()))
                    if invocation is not None
                    else set(enclosing.targets.get(name, frozenset()))
                )
                if escape is not None:
                    targets.update(escape.get(name, frozenset()))
                    targets.update(enclosing.targets.get(name, frozenset()))
                visible_targets = targets - {_OPAQUE_REFERENCE_TARGET}
                return (
                    bool(visible_targets)
                    and name not in visible_targets
                    and _OPAQUE_REFERENCE_TARGET not in targets
                )
            descendant = enclosing.qualified_name
        if (self._in_header or self.scope is None) and name in self.module_bindings:
            targets = self.module_bindings[name]
            return bool(targets) and name not in targets
        return False

    def _resolve_name(self, name: str) -> set[str]:
        """Resolve one loaded name to code identities visible at this location.

        :param name: Loaded bare name.
        :return: Possible qualified targets or the unresolved spelling.
        """
        if name in self._type_parameter_bindings:
            return set()
        if self._in_header:
            return self._resolve_enclosing_name(name)
        if self.scope is not None:
            if name in self.scope.global_names:
                return set(self._global_bindings.get(name, {name}))
            if name in self.scope.nonlocal_names and name in self._nonlocal_bindings:
                return set(self._nonlocal_bindings[name])
            if name in self.scope.local_names:
                return set(self._current_bindings.get(name, set()))
            return self._resolve_enclosing_name(name)
        if name in self._current_bindings:
            return set(self._current_bindings[name])
        return self._resolve_enclosing_name(name)

    def _name_is_resolved(self, name: str) -> bool:
        """Return whether a loaded name came from a proven lexical binding.

        :param name: Loaded bare name.
        :return: Whether current lexical state proves a target identity.
        """
        if name in self._type_parameter_bindings:
            return False
        if self._in_header:
            return self._enclosing_name_is_resolved(name)
        if self.scope is not None:
            if name in self.scope.global_names:
                targets = self._global_bindings.get(name, set())
                return bool(targets) and name not in targets
            if name in self.scope.nonlocal_names and name in self._nonlocal_bindings:
                targets = self._nonlocal_bindings[name]
                visible_targets = targets - {_OPAQUE_REFERENCE_TARGET}
                return (
                    bool(visible_targets)
                    and name not in visible_targets
                    and _OPAQUE_REFERENCE_TARGET not in targets
                )
            if name in self.scope.local_names:
                targets = self._current_bindings.get(name, set())
                visible_targets = targets - {_OPAQUE_REFERENCE_TARGET}
                return (
                    bool(visible_targets)
                    and name not in visible_targets
                    and _OPAQUE_REFERENCE_TARGET not in targets
                )
            return self._enclosing_name_is_resolved(name)
        if name in self._current_bindings:
            targets = self._current_bindings[name]
            return bool(targets) and name not in targets
        return self._enclosing_name_is_resolved(name)

    def _record_name_load(self, name: str) -> None:
        """Record one loaded name with its lexical-resolution provenance."""
        targets = self._resolve_name(name)
        visible_targets = targets - {_OPAQUE_REFERENCE_TARGET}
        self.names.update(visible_targets)
        # A mixed branch can retain the raw name for an opaque runtime value
        # alongside exact import targets. Preserve provenance per target: the
        # exact identities bypass later module-alias rewriting while the raw
        # spelling remains unresolved.
        self.resolved_names.update(visible_targets - {name})

    def _expression_is_resolved(self, node: ast.AST) -> bool:
        """Return whether a dotted receiver has a proven lexical identity.

        :param node: Receiver expression.
        :return: Whether its root name has a proven target.
        """
        if isinstance(node, ast.Name):
            return self._name_is_resolved(node.id)
        if isinstance(node, ast.Attribute):
            return self._expression_is_resolved(node.value)
        return False

    def _name_uses_module_aliases(self, name: str) -> bool:
        """Return whether a free name should use the file's final alias map.

        :param name: Root name of an unresolved dotted expression.
        :return: Whether module-level alias resolution applies.
        """
        if name in self._type_parameter_bindings:
            return False
        if self.scope is not None:
            if name in self.scope.global_names:
                return True
            if name in self.scope.local_names or name in self.scope.nonlocal_names:
                return False
            for enclosing in reversed(self.enclosing_scopes):
                if name in enclosing.global_names:
                    return True
                if name in enclosing.local_names:
                    return False
            return True
        return name not in self._current_bindings

    def _bind_name(self, name: str, targets: set[str] | None = None) -> None:
        """Update a name's current runtime value category.

        :param name: Name being rebound.
        :param targets: Possible qualified code identities.
        :return: ``None``.
        """
        targets = set() if targets is None else set(targets)
        if self.scope is not None and name in self.scope.global_names:
            self._global_bindings[name] = targets
        elif self.scope is not None and name in self.scope.nonlocal_names:
            self._nonlocal_bindings[name] = targets
        else:
            self._current_bindings[name] = targets

    def _bind_target(self, target: ast.AST, targets: set[str] | None = None) -> None:
        """Bind every bare name in an assignment-style target.

        :param target: Assignment-style target expression.
        :param targets: Possible qualified code identities.
        :return: ``None``.
        """
        for name in _bound_target_names(target):
            self._bind_name(name, targets)

    def _visit_target_lookups(self, target: ast.AST) -> None:
        """Collect receiver/index loads required to evaluate a store or delete target."""
        if isinstance(target, ast.Attribute):
            self.visit(target.value)
        elif isinstance(target, ast.Subscript):
            self.visit(target.value)
            self.visit(target.slice)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._visit_target_lookups(element)
        elif isinstance(target, ast.Starred):
            self._visit_target_lookups(target.value)

    def _snapshot_bindings(
        self,
    ) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
        """Copy the current local, global, and nonlocal target bindings.

        :return: Independent snapshots of the three binding maps.
        """
        return tuple(
            {name: set(targets) for name, targets in bindings.items()}
            for bindings in (
                self._current_bindings,
                self._global_bindings,
                self._nonlocal_bindings,
            )
        )

    def _restore_bindings(
        self,
        state: tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]],
    ) -> None:
        """Restore a previously captured binding state."""
        current, global_bindings, nonlocal_bindings = state
        self._current_bindings = {name: set(targets) for name, targets in current.items()}
        self._global_bindings = {name: set(targets) for name, targets in global_bindings.items()}
        self._nonlocal_bindings = {
            name: set(targets) for name, targets in nonlocal_bindings.items()
        }

    def _merge_bindings(
        self,
        states: list[tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]],
    ) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
        """Join possible control-flow states without discarding any code target.

        :param states: Alternative local, global, and nonlocal binding states.
        :return: The conservative union of every alternative state.
        """

        def merge_map(index: int) -> dict[str, set[str]]:
            """Merge one binding-map position from every state.

            :param index: Tuple position of the binding map to merge.
            :return: The merged binding map.
            """
            maps = [state[index] for state in states]
            names = {name for mapping in maps for name in mapping}
            merged: dict[str, set[str]] = {}
            for name in names:
                targets = {target for mapping in maps for target in mapping.get(name, set())}
                if any(name in mapping and not mapping[name] for mapping in maps):
                    # Preserve the possibility of an arbitrary runtime value;
                    # the sentinel keeps attribute dispatch conservative without
                    # allowing a local slot to match a module-level definition.
                    targets.add(name if index == 1 else _OPAQUE_REFERENCE_TARGET)
                if any(name not in mapping for mapping in maps):
                    if index == 1:
                        targets.add(name)
                    elif index == 2 or self.scope is None:
                        targets.update(self._resolve_enclosing_name(name))
                merged[name] = targets
            return merged

        return merge_map(0), merge_map(1), merge_map(2)

    def visit_Name(self, node: ast.Name) -> None:
        """Collect only targets reached by a loaded name's lexical binding.

        :param node: AST name node.
        :return: ``None``.
        """
        if isinstance(node.ctx, ast.Load):
            self._record_name_load(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self._bind_name(node.id)

    def _expression_targets(self, node: ast.AST) -> set[str]:
        """Resolve a simple dotted expression without recording its loads twice.

        :param node: Expression whose possible qualified targets are needed.
        :return: Possible qualified targets for the expression.
        """
        if isinstance(node, ast.Name):
            return self._resolve_name(node.id) - {_OPAQUE_REFERENCE_TARGET}
        if isinstance(node, ast.Attribute):
            return {f"{target}.{node.attr}" for target in self._expression_targets(node.value)}
        return set()

    def _resolved_expression_targets(self, node: ast.AST) -> set[str]:
        """Return only proven identities within a dotted expression.

        :param node: Expression whose exact targets are needed.
        :return: Proven qualified targets for the expression.
        """
        if isinstance(node, ast.Name):
            return self._resolve_name(node.id) - {node.id, _OPAQUE_REFERENCE_TARGET}
        if isinstance(node, ast.Attribute):
            return {
                f"{target}.{node.attr}" for target in self._resolved_expression_targets(node.value)
            }
        return set()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Keep resolved receiver identities separate from unknown attribute tails.

        :param node: AST attribute node.
        :return: ``None``.
        """
        if isinstance(node.ctx, ast.Load):
            receiver_is_resolved = self._expression_is_resolved(node.value)
            if not receiver_is_resolved:
                dotted_name = _dotted_expression_name(node)
                root_name = dotted_name.partition(".")[0] if dotted_name else ""
                if dotted_name and self._name_uses_module_aliases(root_name):
                    self.module_attributes.add(dotted_name)
                else:
                    # ``self``/arbitrary objects can dispatch through subclasses
                    # or dynamic attributes, so their tails stay broad.
                    self.attributes.add(node.attr)
            receiver_targets = self._expression_targets(node.value)
            attribute_targets = {f"{target}.{node.attr}" for target in receiver_targets}
            self.names.update(attribute_targets)
            self.resolved_names.update(
                f"{target}.{node.attr}" for target in self._resolved_expression_targets(node.value)
            )
        self.visit(node.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Evaluate assignment values before replacing target bindings."""
        self.visit(node.value)
        for target in node.targets:
            self._visit_target_lookups(target)
            self._bind_target(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Collect annotation/value references before binding the target."""
        self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)
        self._visit_target_lookups(node.target)
        self._bind_target(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """Evaluate a walrus value before updating its target."""
        self.visit(node.value)
        self._visit_target_lookups(node.target)
        self._bind_target(node.target)
        if self.walrus_owner is not None:
            self.walrus_owner._bind_target(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Treat an augmented-assignment target as both loaded and rebound."""
        if isinstance(node.target, ast.Name):
            self._record_name_load(node.target.id)
        elif isinstance(node.target, ast.Attribute):
            loaded = ast.Attribute(value=node.target.value, attr=node.target.attr, ctx=ast.Load())
            self.visit(loaded)
        else:
            self.visit(node.target)
        self.visit(node.value)
        self._bind_target(node.target)

    def visit_Delete(self, node: ast.Delete) -> None:
        """Remove class/module bindings or leave function slots unbound."""
        for target in node.targets:
            self._visit_target_lookups(target)
            for name in _bound_target_names(target):
                if self.scope is None:
                    self._current_bindings.pop(name, None)
                else:
                    self._bind_name(name)

    def visit_Import(self, node: ast.Import) -> None:
        """Bind imports to qualified module identities."""
        for alias in node.names:
            name = alias.asname or alias.name.partition(".")[0]
            target = alias.name if alias.asname else name
            self._bind_name(name, {target})

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Bind from-imports to qualified object identities."""
        module = _resolve_relative_module(self.import_package, node.level, node.module)
        for alias in node.names:
            if alias.name == "*":
                continue
            name = alias.asname or alias.name
            target = f"{module}.{alias.name}" if module else alias.name
            self._bind_name(name, {target})

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Inspect a nested function and aggregate its deferred body references."""
        for expression in _function_outer_header_expressions(node):
            self.visit(expression)
        previous_type_parameters = set(self._type_parameter_bindings)
        self._type_parameter_bindings.update(_type_parameter_names(node))
        for expression in _function_annotation_expressions(node):
            self.visit(expression)
        self._type_parameter_bindings = previous_type_parameters
        target = f"{self.qualified_name}.{node.name}"
        enclosing = self.enclosing_scopes
        if self.scope is not None:
            enclosing = (*enclosing, self.scope)
        child = ReferenceVisitor(
            qualified_name=target,
            import_package=self.import_package,
            scope=_collect_function_reference_scope(
                node,
                target,
                self.import_package,
            ),
            enclosing_scopes=enclosing,
            module_bindings=self.module_bindings,
        )
        child.collect_definition(node, include_header=False)
        self.names.update(child.names)
        self.resolved_names.update(child.resolved_names)
        self.attributes.update(child.attributes)
        self.module_attributes.update(child.module_attributes)
        self._bind_name(node.name, {target})

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Inspect a nested class and aggregate its executed body references."""
        for decorator in node.decorator_list:
            self.visit(decorator)
        previous_type_parameters = set(self._type_parameter_bindings)
        self._type_parameter_bindings.update(_type_parameter_names(node))
        for expression in (
            *getattr(node, "type_params", ()),
            *node.bases,
            *(keyword.value for keyword in node.keywords),
        ):
            self.visit(expression)
        self._type_parameter_bindings = previous_type_parameters
        target = f"{self.qualified_name}.{node.name}"
        child = ReferenceVisitor(
            qualified_name=target,
            import_package=self.import_package,
            enclosing_scopes=self.enclosing_scopes
            if self.scope is None
            else (*self.enclosing_scopes, self.scope),
            module_bindings=self.module_bindings,
        )
        child.collect_definition(node, include_header=False)
        self.names.update(child.names)
        self.resolved_names.update(child.resolved_names)
        self.attributes.update(child.attributes)
        self.module_attributes.update(child.module_attributes)
        self._bind_name(node.name, {target})

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Evaluate defaults outside, then collect the deferred lambda body."""
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        collector = _FunctionScopeCollector(
            self.qualified_name,
            self.import_package,
            node.args,
        )
        collector.visit(node.body)
        enclosing = self.enclosing_scopes
        if self.scope is not None:
            enclosing = (*enclosing, self.scope)
        child = ReferenceVisitor(
            qualified_name=self.qualified_name,
            import_package=self.import_package,
            scope=collector.build(),
            enclosing_scopes=enclosing,
            module_bindings=self.module_bindings,
        )
        child.visit(node.body)
        self.names.update(child.names)
        self.resolved_names.update(child.resolved_names)
        self.attributes.update(child.attributes)
        self.module_attributes.update(child.module_attributes)

    def _visit_comprehension(
        self,
        generators: list[ast.comprehension],
        values: tuple[ast.AST, ...],
        *,
        eager: bool,
    ) -> None:
        """Collect a comprehension with its first iterable in the outer scope.

        :param generators: Comprehension generators in evaluation order.
        :param values: Result expressions evaluated in the comprehension scope.
        :param eager: Whether the comprehension executes eagerly.
        :return: ``None``.
        """
        if not generators:
            return
        self.visit(generators[0].iter)
        if iterable_definitely_empty(generators[0].iter):
            return
        owner = self.walrus_owner or self
        owner_base = owner._snapshot_bindings()
        guaranteed_state = owner_base
        guaranteed_reach = iterable_definitely_nonempty(generators[0].iter)
        local_names = {
            name for generator in generators for name in _bound_target_names(generator.target)
        }
        scope = _ReferenceScope(
            local_names=frozenset(local_names),
            global_names=frozenset(),
            nonlocal_names=frozenset(),
            targets={},
        )
        enclosing = self.enclosing_scopes
        if self.scope is not None:
            enclosing = (*enclosing, self.scope)
        child = ReferenceVisitor(
            qualified_name=self.qualified_name,
            import_package=self.import_package,
            scope=scope,
            enclosing_scopes=enclosing,
            walrus_owner=owner,
            module_bindings=self.module_bindings,
        )
        child._bind_target(generators[0].target)
        stopped = False
        conditions_are_conditional = False
        for condition in generators[0].ifs:
            condition_base = owner._snapshot_bindings()
            child.visit(condition)
            if conditions_are_conditional:
                owner._restore_bindings(
                    owner._merge_bindings([condition_base, owner._snapshot_bindings()])
                )
            if isinstance(condition, ast.Constant) and not bool(condition.value):
                stopped = True
                break
            if not (isinstance(condition, ast.Constant) and bool(condition.value)):
                conditions_are_conditional = True
        if guaranteed_reach:
            guaranteed_state = owner._snapshot_bindings()
        if generators[0].ifs:
            guaranteed_reach = False
        for generator in generators[1:] if not stopped else ():
            child.visit(generator.iter)
            if guaranteed_reach:
                guaranteed_state = owner._snapshot_bindings()
            if iterable_definitely_empty(generator.iter):
                stopped = True
                break
            child._bind_target(generator.target)
            conditions_are_conditional = False
            for condition in generator.ifs:
                condition_base = owner._snapshot_bindings()
                child.visit(condition)
                if conditions_are_conditional:
                    owner._restore_bindings(
                        owner._merge_bindings([condition_base, owner._snapshot_bindings()])
                    )
                if isinstance(condition, ast.Constant) and not bool(condition.value):
                    stopped = True
                    break
                if not (isinstance(condition, ast.Constant) and bool(condition.value)):
                    conditions_are_conditional = True
            if stopped:
                break
            if guaranteed_reach:
                guaranteed_state = owner._snapshot_bindings()
            if generator.ifs or not iterable_definitely_nonempty(generator.iter):
                guaranteed_reach = False
        exact_count = None if stopped else comprehension_exact_result_count(generators)
        repetitions = exact_count if exact_count is not None else int(not stopped)
        for _ in range(repetitions):
            for value in values:
                child.visit(value)
        self.names.update(child.names)
        self.resolved_names.update(child.resolved_names)
        self.attributes.update(child.attributes)
        self.module_attributes.update(child.module_attributes)
        if not eager:
            owner._restore_bindings(owner_base)
        elif stopped or not comprehension_definitely_runs(generators):
            owner._restore_bindings(
                owner._merge_bindings([guaranteed_state, owner._snapshot_bindings()])
            )

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Collect a list comprehension in its implicit function scope."""
        self._visit_comprehension(node.generators, (node.elt,), eager=True)

    visit_SetComp = visit_ListComp

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """Collect a deferred generator expression without forcing its walrus effects."""
        self._visit_comprehension(node.generators, (node.elt,), eager=False)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Collect a dictionary comprehension in its implicit function scope."""
        self._visit_comprehension(node.generators, (node.key, node.value), eager=True)

    def visit_With(self, node: ast.With) -> None:
        """Evaluate context managers before binding ``as`` targets."""
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._visit_target_lookups(item.optional_vars)
                self._bind_target(item.optional_vars)
        self._visit_suite(node.body)

    visit_AsyncWith = visit_With

    def _visit_loop_target(self, target: ast.expr) -> None:
        """Collect loop-target receiver loads and bind its bare names."""
        self._visit_target_lookups(target)
        self._bind_target(target)

    def _enter_exception_handler(self, handler: ast.ExceptHandler) -> None:
        """Bind an exception alias on handler entry."""
        if handler.name is not None:
            self._bind_name(handler.name)

    def _exit_exception_handler(self, handler: ast.ExceptHandler) -> None:
        """Clear an exception alias after its handler."""
        if handler.name is not None:
            self._bind_name(handler.name)

    def _visit_match_pattern(self, pattern: ast.pattern) -> None:
        """Collect pattern expression loads and bind every capture."""
        self.visit(pattern)
        for name in _match_capture_names(pattern):
            self._bind_name(name)


def _dotted_expression_name(node: ast.expr) -> str | None:
    """Return a dotted name for a simple class-base expression.

    :param node: Base-class expression.
    :return: Dotted name for ``Name``/``Attribute`` expressions, otherwise ``None``.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_expression_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
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
    reference_targets: frozenset[str] = frozenset()
    may_be_opaque: bool = False


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

        # Ordered lexical frames preserve interleaved function/class nesting.
        self.scope_stack: list[tuple[str, str]] = []
        self.dynamic_dispatch_stack: list[bool] = []
        self.binding_scopes: list[dict[str, _NameBinding]] = [{}]
        self.reference_function_scopes: list[_ReferenceScope] = []
        self.conditional_depth = 0

    def _bind_name(self, name: str, binding: _NameBinding) -> None:
        """Update one name in the current lexical scope.

        :param name: Bound bare name.
        :param binding: Identity now visible through ``name``.
        :return: ``None``.
        """
        self.binding_scopes[-1][name] = binding

    @staticmethod
    def _reference_targets(name: str, binding: _NameBinding) -> set[str]:
        """Convert a collector binding into static reference targets.

        :param name: Bare name holding the binding.
        :param binding: Current collector binding.
        :return: Known identities plus a raw fallback for opaque values.
        """
        targets = set(binding.reference_targets)
        if binding.identity is not None:
            targets.add(binding.identity)
        if binding.module is not None:
            targets.add(binding.module)
        if binding.may_be_opaque:
            targets.add(name)
        return targets

    def _module_reference_bindings(self) -> dict[str, set[str]]:
        """Return import identities visible at the current module position.

        :return: Program-point module bindings for immediate definition code.
        """
        return {
            name: self._reference_targets(name, binding)
            for name, binding in self.binding_scopes[0].items()
        }

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
                binding = _NameBinding(may_be_opaque=True)
            else:
                binding = _NameBinding(identity=f"{resolved_module}.{alias.name}")
            self._bind_name(bound_name, binding)

    def visit_Name(self, node: ast.Name) -> None:
        """Track assignment and deletion of names in the current lexical scope.

        :param node: Name expression.
        :return: ``None``.
        """
        if isinstance(node.ctx, ast.Store):
            self._bind_name(node.id, _NameBinding(may_be_opaque=True))
        elif isinstance(node.ctx, ast.Del):
            self.binding_scopes[-1].pop(node.id, None)

    def _reference_expression_binding(self, node: ast.AST) -> _NameBinding:
        """Resolve a simple assignment value for program-point references.

        :param node: Assignment value expression.
        :return: Known reference targets and opacity for the resulting value.
        """
        if isinstance(node, ast.Name):
            binding = self._lookup_name(node.id)
            return binding if binding is not None else _NameBinding(may_be_opaque=True)
        if isinstance(node, ast.Attribute):
            parent = self._reference_expression_binding(node.value)
            targets = {
                f"{target}.{node.attr}" for target in self._reference_targets("", parent) if target
            }
            return _NameBinding(
                reference_targets=frozenset(targets),
                may_be_opaque=parent.may_be_opaque,
            )
        return _NameBinding(may_be_opaque=True)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Propagate simple assignment aliases in document order.

        :param node: Assignment statement.
        :return: ``None``.
        """
        self.visit(node.value)
        binding = self._reference_expression_binding(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._bind_name(target.id, binding)
            else:
                self.visit(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Propagate a simple annotated assignment alias.

        :param node: Annotated assignment statement.
        :return: ``None``.
        """
        self.visit(node.annotation)
        if node.value is None:
            return
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._bind_name(node.target.id, self._reference_expression_binding(node.value))
        else:
            self.visit(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """Propagate an assignment-expression alias after its value.

        :param node: Assignment expression.
        :return: ``None``.
        """
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._bind_name(node.target.id, self._reference_expression_binding(node.value))

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Treat an augmented-assignment result as opaque.

        :param node: Augmented assignment statement.
        :return: ``None``.
        """
        self.visit(node.target)
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._bind_name(node.target.id, _NameBinding(may_be_opaque=True))

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
                concrete = [binding for binding in bindings if binding is not None]
                targets = {
                    target
                    for binding in concrete
                    for target in _CodeUnitCollector._reference_targets(name, binding)
                    if target != name
                }
                joined[name] = _NameBinding(
                    reference_targets=frozenset(targets),
                    may_be_opaque=any(binding.may_be_opaque for binding in concrete),
                )
        return joined

    def _visit_statement_suite(self, suite: list[ast.stmt]) -> None:
        """Visit the statements of one suite in document order.

        :param suite: Statement suite.
        :return: ``None``.
        """
        for statement in suite:
            self.visit(statement)
            if not BranchingVisitor._statement_falls_through(statement):
                break

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
                self._bind_name(handler.name, _NameBinding(may_be_opaque=True))
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
                self._bind_name(pattern.name, _NameBinding(may_be_opaque=True))
        elif isinstance(pattern, ast.MatchStar):
            if pattern.name is not None:
                self._bind_name(pattern.name, _NameBinding(may_be_opaque=True))
        elif isinstance(pattern, ast.MatchMapping):
            for sub_pattern in pattern.patterns:
                self._bind_match_captures(sub_pattern)
            if pattern.rest is not None:
                self._bind_name(pattern.rest, _NameBinding(may_be_opaque=True))
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
            self._bind_name(argument.arg, _NameBinding(may_be_opaque=True))
        if arguments.vararg is not None:
            self._bind_name(arguments.vararg.arg, _NameBinding(may_be_opaque=True))
        if arguments.kwarg is not None:
            self._bind_name(arguments.kwarg.arg, _NameBinding(may_be_opaque=True))

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
        is_method = bool(self.scope_stack) and self.scope_stack[-1][0] == "class"
        scope_prefix = [name for _, name in self.scope_stack]
        qualified_name = self.extractor._qualified_name(
            self.module_name,
            scope_prefix,
            node.name,
        )
        reference_scope = _collect_function_reference_scope(
            node,
            qualified_name,
            self.import_package,
        )
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
                    import_package=self.import_package,
                    import_module_name=self.inheritance_module_name,
                    reference_scope=reference_scope,
                    enclosing_reference_scopes=tuple(self.reference_function_scopes),
                    module_bindings=self._module_reference_bindings(),
                )
            )

        for expression in (
            *_function_outer_header_expressions(node),
            *_function_annotation_expressions(node),
        ):
            self.visit(expression)

        import_identity = self.extractor._qualified_name(
            self.inheritance_module_name,
            scope_prefix,
            node.name,
        )
        self._bind_name(
            node.name,
            _NameBinding(reference_targets=frozenset({import_identity})),
        )
        self.scope_stack.append(("function", node.name))
        self.binding_scopes.append({})
        self.reference_function_scopes.append(reference_scope)
        self._bind_function_arguments(node.args)
        for statement in node.body:
            self.visit(statement)
        self.reference_function_scopes.pop()
        self.binding_scopes.pop()
        self.scope_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Collect class units and descend into exported/visible class bodies."""
        scope_prefix = [name for _, name in self.scope_stack]
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
                    import_package=self.import_package,
                    import_module_name=self.inheritance_module_name,
                    enclosing_reference_scopes=tuple(self.reference_function_scopes),
                    module_bindings=self._module_reference_bindings(),
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
        for expression in (
            *node.decorator_list,
            *getattr(node, "type_params", ()),
            *node.bases,
            *(keyword.value for keyword in node.keywords),
        ):
            self.visit(expression)
        # Function-local classes are invisible to importers, and their method
        # units carry the function scope in their qualified prefix, so they can
        # neither confer nor receive cross-file proof. Recording them would only
        # pollute the module-level identity they appear to share.
        if not any(kind == "function" for kind, _ in self.scope_stack):
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
            self.scope_stack.append(("class", node.name))
            self.dynamic_dispatch_stack.append(is_dynamic_dispatch_class)
            self.binding_scopes.append({})
            type_parameter_names = _type_parameter_names(node)
            if type_parameter_names:
                self.reference_function_scopes.append(
                    _ReferenceScope(
                        local_names=type_parameter_names,
                        global_names=frozenset(),
                        nonlocal_names=frozenset(),
                        targets={},
                    )
                )
            for statement in node.body:
                self.visit(statement)
            if type_parameter_names:
                self.reference_function_scopes.pop()
            self.binding_scopes.pop()
            self.dynamic_dispatch_stack.pop()
            self.scope_stack.pop()
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
        import_package: str,
        import_module_name: str,
        reference_scope: _ReferenceScope,
        enclosing_reference_scopes: tuple[_ReferenceScope, ...],
        module_bindings: dict[str, set[str]],
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
        :param import_package: Package used to resolve relative imports.
        :param import_module_name: Importable identity of the containing module.
        :param reference_scope: Function compile-time lexical scope.
        :param enclosing_reference_scopes: Lexically enclosing function scopes.
        :param module_bindings: Import identities visible at the definition statement.
        :return: Constructed function or method unit.
        """
        unit_type = CodeUnitType.METHOD if class_member else CodeUnitType.FUNCTION
        qualified_name = self._qualified_name(module_name, scope_prefix, node.name)

        reference_visitor = ReferenceVisitor(
            qualified_name=qualified_name,
            import_package=import_package,
            scope=reference_scope,
            enclosing_scopes=enclosing_reference_scopes,
            module_bindings=module_bindings,
        )
        reference_visitor.collect_definition(node, include_header=not scope_prefix)

        return self._build_code_unit(
            node,
            file_path,
            source_lines,
            module_name,
            scope_prefix,
            unit_type=unit_type,
            referenced_names=reference_visitor.names,
            resolved_referenced_names=reference_visitor.resolved_names,
            referenced_attributes=reference_visitor.attributes,
            module_attribute_references=reference_visitor.module_attributes,
            exported=exported,
            dynamic_dispatch_hook=(
                class_member and is_dynamic_dispatch_class and node.name.startswith("visit_")
            ),
            import_module_name=import_module_name,
        )

    def _emit_class(
        self,
        node: ast.ClassDef,
        file_path: Path,
        source_lines: list[str],
        module_name: str,
        scope_prefix: list[str],
        exported: set[str],
        import_package: str,
        import_module_name: str,
        enclosing_reference_scopes: tuple[_ReferenceScope, ...],
        module_bindings: dict[str, set[str]],
    ) -> CodeUnit:
        """Build one class code unit.

        :param node: Class AST node.
        :param file_path: Source file path.
        :param source_lines: Entire file source split with line endings.
        :param module_name: Module name.
        :param scope_prefix: Scope prefix stack.
        :param exported: Exported names from module __all__.
        :param import_package: Package used to resolve relative imports.
        :param import_module_name: Importable identity of the containing module.
        :param enclosing_reference_scopes: Lexically enclosing function scopes.
        :param module_bindings: Import identities visible at the class statement.
        :return: Constructed class unit.
        """
        qualified_name = self._qualified_name(module_name, scope_prefix, node.name)
        reference_visitor = ReferenceVisitor(
            qualified_name=qualified_name,
            import_package=import_package,
            enclosing_scopes=enclosing_reference_scopes,
            module_bindings=module_bindings,
        )
        reference_visitor.collect_definition(node, include_header=not scope_prefix)

        return self._build_code_unit(
            node,
            file_path=file_path,
            source_lines=source_lines,
            module_name=module_name,
            scope_prefix=scope_prefix,
            unit_type=CodeUnitType.CLASS,
            referenced_names=reference_visitor.names,
            resolved_referenced_names=reference_visitor.resolved_names,
            referenced_attributes=reference_visitor.attributes,
            module_attribute_references=reference_visitor.module_attributes,
            exported=exported,
            dynamic_dispatch_hook=False,
            import_module_name=import_module_name,
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
        resolved_referenced_names: set[str],
        referenced_attributes: set[str],
        module_attribute_references: set[str],
        exported: set[str],
        dynamic_dispatch_hook: bool,
        import_module_name: str,
    ) -> CodeUnit:
        """Build shared source and metadata fields for one code unit.

        :param node: Function or class definition node.
        :param file_path: Source file path.
        :param source_lines: Entire file source split with line endings.
        :param module_name: Module name.
        :param scope_prefix: Scope prefix stack.
        :param unit_type: Emitted unit type.
        :param referenced_names: Names the definition references.
        :param resolved_referenced_names: References with proven lexical identities.
        :param referenced_attributes: Attribute names whose receiver type is unresolved.
        :param module_attribute_references: Dotted paths rooted in unresolved module globals.
        :param exported: Exported names from module ``__all__``.
        :param dynamic_dispatch_hook: Whether runtime visitor dispatch reaches this method.
        :param import_module_name: Importable identity of the containing module.
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
            module_name=module_name,
            import_module_name=import_module_name,
            referenced_names=referenced_names,
            resolved_referenced_names=resolved_referenced_names,
            referenced_attributes=referenced_attributes,
            module_attribute_references=module_attribute_references,
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
