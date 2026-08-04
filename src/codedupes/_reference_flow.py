"""Shared control-flow traversal for static reference binding analysis."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

BindingState = TypeVar("BindingState")


def comprehension_definitely_runs(generators: list[ast.comprehension]) -> bool:
    """Return whether literal iterables guarantee at least one result evaluation.

    Starred sequence elements and dictionary unpacking do not prove that an
    iterable is nonempty; an ordinary literal element does.

    :param generators: Comprehension generator clauses.
    :return: Whether each unfiltered iterable is provably nonempty.
    """
    if not generators or any(generator.ifs for generator in generators):
        return False

    return all(iterable_definitely_nonempty(generator.iter) for generator in generators)


def iterable_definitely_nonempty(expression: ast.expr) -> bool:
    """Return whether one literal contains an unconditional element.

    :param expression: Candidate iterable expression.
    :return: Whether the iterable is statically nonempty.
    """
    if isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
        return any(not isinstance(element, ast.Starred) for element in expression.elts)
    if isinstance(expression, ast.Dict):
        return any(key is not None for key in expression.keys)
    return False


def iterable_definitely_empty(expression: ast.expr) -> bool:
    """Return whether one literal is statically empty.

    :param expression: Candidate iterable expression.
    :return: Whether the iterable contains no possible elements.
    """
    if isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
        return not expression.elts
    if isinstance(expression, ast.Dict):
        return not expression.keys
    return False


def comprehension_exact_result_count(generators: list[ast.comprehension]) -> int | None:
    """Return an exact result count for one unfiltered plain-literal generator.

    :param generators: Comprehension generator clauses.
    :return: Exact count, or ``None`` when execution is not statically simple.
    """
    if len(generators) != 1 or generators[0].ifs:
        return None
    expression = generators[0].iter
    if isinstance(expression, (ast.List, ast.Tuple, ast.Set)) and not any(
        isinstance(element, ast.Starred) for element in expression.elts
    ):
        return len(expression.elts)
    if isinstance(expression, ast.Dict) and all(key is not None for key in expression.keys):
        return len(expression.keys)
    return None


class BranchingVisitor(ast.NodeVisitor, ABC, Generic[BindingState]):
    """Traverse Python control flow while joining implementation-defined bindings."""

    _MAX_LOOP_FIXPOINT_PASSES = 64

    def __init__(self) -> None:
        """Initialize loop-control state used by every binding implementation."""
        self._control_transfers: list[tuple[str, BindingState]] = []

    @abstractmethod
    def _snapshot_bindings(self) -> BindingState:
        """Return an independent copy of the current binding state."""

    @abstractmethod
    def _restore_bindings(self, state: BindingState) -> None:
        """Replace current bindings with a previously captured state."""

    @abstractmethod
    def _merge_bindings(self, states: list[BindingState]) -> BindingState:
        """Join bindings from alternative reachable states."""

    @abstractmethod
    def _visit_loop_target(self, target: ast.expr) -> None:
        """Apply one loop-target binding in the implementation's scope model."""

    @abstractmethod
    def _visit_match_pattern(self, pattern: ast.pattern) -> None:
        """Collect pattern loads and apply its capture bindings."""

    def _enter_exception_handler(self, handler: ast.ExceptHandler) -> None:
        """Apply an exception handler's alias binding.

        :param handler: Handler being entered.
        :return: ``None``.
        """

    def _exit_exception_handler(self, handler: ast.ExceptHandler) -> None:
        """Apply Python's exception-alias cleanup.

        :param handler: Handler being exited.
        :return: ``None``.
        """

    def _visit_statement_alternatives(
        self,
        alternatives: list[list[ast.stmt]],
        *,
        include_base: bool,
    ) -> None:
        """Visit mutually exclusive suites and join their possible bindings.

        :param alternatives: Statement suites reached on exclusive paths.
        :param include_base: Whether control may skip every suite.
        :return: ``None``.
        """
        base = self._snapshot_bindings()
        states = [base] if include_base else []
        for statements in alternatives:
            self._restore_bindings(base)
            self._visit_suite(statements)
            states.append(self._snapshot_bindings())
        self._restore_bindings(self._merge_bindings(states))

    @classmethod
    def _suite_falls_through(cls, statements: list[ast.stmt]) -> bool:
        """Return whether a statement suite has a structural fall-through path.

        :param statements: Statement suite to inspect.
        :return: Whether execution can reach the following statement.
        """
        return all(cls._statement_falls_through(statement) for statement in statements)

    @classmethod
    def _suite_has_explicit_raise(cls, statements: list[ast.stmt]) -> bool:
        """Return whether a reachable statement has an explicit raise path.

        :param statements: Statement suite to inspect.
        :return: Whether a context manager could suppress an explicit exception.
        """
        for statement in statements:
            if cls._statement_has_explicit_raise(statement):
                return True
            if not cls._statement_falls_through(statement):
                break
        return False

    @classmethod
    def _statement_has_explicit_raise(cls, statement: ast.stmt) -> bool:
        """Return whether one statement contains a reachable explicit raise.

        :param statement: Statement to inspect.
        :return: Whether an explicit raise path exists.
        """
        if isinstance(statement, ast.Raise):
            return True
        if isinstance(statement, ast.If):
            if isinstance(statement.test, ast.Constant):
                branch = statement.body if bool(statement.test.value) else statement.orelse
                return cls._suite_has_explicit_raise(branch)
            return cls._suite_has_explicit_raise(statement.body) or cls._suite_has_explicit_raise(
                statement.orelse
            )
        if isinstance(statement, (ast.Try, ast.TryStar)):
            return any(
                cls._suite_has_explicit_raise(suite)
                for suite in (
                    statement.body,
                    statement.orelse,
                    statement.finalbody,
                    *(handler.body for handler in statement.handlers),
                )
            )
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            return cls._suite_has_explicit_raise(statement.body)
        return False

    @staticmethod
    def _statement_cannot_raise(statement: ast.stmt) -> bool:
        """Return whether one simple statement is provably exception-free.

        :param statement: Candidate statement.
        :return: Whether evaluating it cannot transfer to an exception handler.
        """
        if isinstance(statement, (ast.Pass, ast.Global, ast.Nonlocal)):
            return True
        if isinstance(statement, ast.Assign):
            return isinstance(statement.value, ast.Constant) and all(
                isinstance(target, ast.Name) for target in statement.targets
            )
        return False

    def _exception_occurs_after_effects(self, statement: ast.stmt) -> bool:
        """Return whether a known exception can occur only after statement effects.

        :param statement: Candidate statement.
        :return: Whether handler entry should exclude the pre-statement state.
        """
        return False

    @classmethod
    def _statement_falls_through(cls, statement: ast.stmt) -> bool:
        """Return whether one statement has a structural fall-through path.

        :param statement: Statement to inspect.
        :return: Whether execution can continue after the statement.
        """
        if isinstance(statement, (ast.Break, ast.Continue, ast.Raise, ast.Return)):
            return False
        if isinstance(statement, ast.If):
            if isinstance(statement.test, ast.Constant):
                branch = statement.body if bool(statement.test.value) else statement.orelse
                return cls._suite_falls_through(branch)
            return cls._suite_falls_through(statement.body) or (
                not statement.orelse or cls._suite_falls_through(statement.orelse)
            )
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            if cls._suite_falls_through(statement.body):
                return True
            # ``__exit__``/``__aexit__`` can suppress an exception, but cannot
            # suppress return, break, or continue.
            return cls._suite_has_explicit_raise(statement.body)
        if isinstance(statement, (ast.Try, ast.TryStar)):
            if statement.finalbody and not cls._suite_falls_through(statement.finalbody):
                return False
            normal_path = cls._suite_falls_through(statement.body) and cls._suite_falls_through(
                statement.orelse
            )
            return normal_path or any(
                cls._suite_falls_through(handler.body) for handler in statement.handlers
            )
        return True

    def _visit_suite(self, statements: list[ast.stmt]) -> None:
        """Visit reachable statement prefixes and stop after an unconditional transfer.

        :param statements: Statement suite to visit.
        :return: ``None``.
        """
        for statement in statements:
            self.visit(statement)
            if not self._statement_falls_through(statement):
                break

    def visit_If(self, node: ast.If) -> None:
        """Join bindings from mutually exclusive conditional branches."""
        self.visit(node.test)
        if isinstance(node.test, ast.Constant):
            branch = node.body if bool(node.test.value) else node.orelse
            self._visit_suite(branch)
            return
        self._visit_statement_alternatives(
            [node.body, node.orelse],
            include_base=not node.orelse,
        )

    def visit_IfExp(self, node: ast.IfExp) -> None:
        """Join bindings produced by either conditional-expression arm."""
        self.visit(node.test)
        base = self._snapshot_bindings()
        states = []
        for expression in (node.body, node.orelse):
            self._restore_bindings(base)
            self.visit(expression)
            states.append(self._snapshot_bindings())
        self._restore_bindings(self._merge_bindings(states))

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        """Join each short-circuited operand with its evaluated outcome."""
        if not node.values:
            return
        self.visit(node.values[0])
        for value in node.values[1:]:
            short_circuit_state = self._snapshot_bindings()
            self.visit(value)
            self._restore_bindings(
                self._merge_bindings([short_circuit_state, self._snapshot_bindings()])
            )

    def visit_Assert(self, node: ast.Assert) -> None:
        """Collect a conditional assertion message without leaking its bindings."""
        self.visit(node.test)
        continuing_state = self._snapshot_bindings()
        if node.msg is not None:
            self.visit(node.msg)
        self._restore_bindings(continuing_state)

    def visit_Compare(self, node: ast.Compare) -> None:
        """Join bindings from short-circuited chained-comparison operands."""
        self.visit(node.left)
        if not node.comparators:
            return
        self.visit(node.comparators[0])
        for comparator in node.comparators[1:]:
            short_circuit_state = self._snapshot_bindings()
            self.visit(comparator)
            self._restore_bindings(
                self._merge_bindings([short_circuit_state, self._snapshot_bindings()])
            )

    def _visit_loop(self, node: ast.For | ast.AsyncFor) -> None:
        """Evaluate a loop iterable and iterate bindings to a conservative fixed point."""
        self.visit(node.iter)
        base = self._snapshot_bindings()
        if iterable_definitely_empty(node.iter):
            self._visit_suite(node.orelse)
            return
        iteration_entry = base
        normal_states: list[BindingState] = []
        break_states: list[BindingState] = []
        continue_states: list[BindingState] = []
        for _ in range(self._MAX_LOOP_FIXPOINT_PASSES):
            self._restore_bindings(iteration_entry)
            self._visit_loop_target(node.target)
            transfer_mark = len(self._control_transfers)
            self._visit_suite(node.body)
            iteration_transfers = self._control_transfers[transfer_mark:]
            del self._control_transfers[transfer_mark:]
            iteration_breaks = [state for kind, state in iteration_transfers if kind == "break"]
            iteration_continues = [
                state for kind, state in iteration_transfers if kind == "continue"
            ]
            self._control_transfers.extend(
                transfer
                for transfer in iteration_transfers
                if transfer[0] not in {"break", "continue"}
            )
            break_states.extend(iteration_breaks)
            continue_states.extend(iteration_continues)
            iteration_normal: list[BindingState] = []
            if self._suite_falls_through(node.body):
                iteration_normal.append(self._snapshot_bindings())
                normal_states.extend(iteration_normal)
            next_entry = self._merge_bindings(
                [base, iteration_entry, *iteration_normal, *iteration_continues]
            )
            if next_entry == iteration_entry:
                break
            iteration_entry = next_entry
        normal_exit_states = [*normal_states, *continue_states]
        if not iterable_definitely_nonempty(node.iter):
            normal_exit_states.insert(0, base)
        after_states = list(break_states)
        if normal_exit_states:
            self._restore_bindings(self._merge_bindings(normal_exit_states))
            self._visit_suite(node.orelse)
            if self._suite_falls_through(node.orelse):
                after_states.append(self._snapshot_bindings())
        self._restore_bindings(self._merge_bindings(after_states or [base]))

    def visit_For(self, node: ast.For) -> None:
        """Join bindings from zero or more for-loop iterations."""
        self._visit_loop(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Join bindings from zero or more async-for iterations."""
        self._visit_loop(node)

    def visit_While(self, node: ast.While) -> None:
        """Join zero or more while iterations, including abrupt-exit states."""
        base = self._snapshot_bindings()
        iteration_entry = base
        normal_test_states: list[BindingState] = []
        break_states: list[BindingState] = []
        continue_states: list[BindingState] = []
        test_is_true = isinstance(node.test, ast.Constant) and bool(node.test.value)
        test_is_false = isinstance(node.test, ast.Constant) and not bool(node.test.value)
        for _ in range(self._MAX_LOOP_FIXPOINT_PASSES):
            self._restore_bindings(iteration_entry)
            self.visit(node.test)
            test_state = self._snapshot_bindings()
            if not test_is_true:
                normal_test_states.append(test_state)
            if test_is_false:
                break
            transfer_mark = len(self._control_transfers)
            self._visit_suite(node.body)
            iteration_transfers = self._control_transfers[transfer_mark:]
            del self._control_transfers[transfer_mark:]
            iteration_breaks = [state for kind, state in iteration_transfers if kind == "break"]
            iteration_continues = [
                state for kind, state in iteration_transfers if kind == "continue"
            ]
            self._control_transfers.extend(
                transfer
                for transfer in iteration_transfers
                if transfer[0] not in {"break", "continue"}
            )
            break_states.extend(iteration_breaks)
            continue_states.extend(iteration_continues)
            iteration_normal: list[BindingState] = []
            if self._suite_falls_through(node.body):
                iteration_normal.append(self._snapshot_bindings())
            next_entry = self._merge_bindings(
                [base, iteration_entry, *iteration_normal, *iteration_continues]
            )
            if next_entry == iteration_entry:
                break
            iteration_entry = next_entry
        after_states = list(break_states)
        if normal_test_states:
            self._restore_bindings(self._merge_bindings(normal_test_states))
            self._visit_suite(node.orelse)
            if self._suite_falls_through(node.orelse):
                after_states.append(self._snapshot_bindings())
        self._restore_bindings(self._merge_bindings(after_states or [base]))

    def visit_Break(self, node: ast.Break) -> None:
        """Capture bindings that can leave the innermost analyzed loop.

        :param node: Break statement.
        :return: ``None``.
        """
        self._control_transfers.append(("break", self._snapshot_bindings()))

    def visit_Continue(self, node: ast.Continue) -> None:
        """Capture bindings that can begin another loop iteration.

        :param node: Continue statement.
        :return: ``None``.
        """
        self._control_transfers.append(("continue", self._snapshot_bindings()))

    def visit_Return(self, node: ast.Return) -> None:
        """Collect a return value and capture its binding state.

        :param node: Return statement.
        :return: ``None``.
        """
        if node.value is not None:
            self.visit(node.value)
        self._control_transfers.append(("return", self._snapshot_bindings()))

    def visit_Raise(self, node: ast.Raise) -> None:
        """Collect raised expressions and capture their binding state.

        :param node: Raise statement.
        :return: ``None``.
        """
        if node.exc is not None:
            self.visit(node.exc)
        if node.cause is not None:
            self.visit(node.cause)
        self._control_transfers.append(("raise", self._snapshot_bindings()))

    def visit_Try(self, node: ast.Try) -> None:
        """Join successful, handled-exception, and finally binding paths."""
        initial_state = self._snapshot_bindings()
        transfer_mark = len(self._control_transfers)
        prefix_states = [self._snapshot_bindings()]
        exceptional_states: list[BindingState] = []
        for statement in node.body:
            before_statement = self._snapshot_bindings()
            self.visit(statement)
            after_statement = self._snapshot_bindings()
            prefix_states.append(after_statement)
            if not self._statement_cannot_raise(statement):
                if self._exception_occurs_after_effects(statement):
                    exceptional_states.append(after_statement)
                else:
                    exceptional_states.extend((before_statement, after_statement))
            if not self._statement_falls_through(statement):
                break
        else_prefixes = [self._snapshot_bindings()]
        for statement in node.orelse:
            before_statement = self._snapshot_bindings()
            self.visit(statement)
            after_statement = self._snapshot_bindings()
            else_prefixes.append(after_statement)
            if not self._statement_cannot_raise(statement):
                if self._exception_occurs_after_effects(statement):
                    exceptional_states.append(after_statement)
                else:
                    exceptional_states.extend((before_statement, after_statement))
            if not self._statement_falls_through(statement):
                break
        states = []
        if self._suite_falls_through(node.body) and self._suite_falls_through(node.orelse):
            states.append(self._snapshot_bindings())
        handler_base = self._merge_bindings(exceptional_states or [initial_state])
        for handler in node.handlers:
            if not exceptional_states:
                break
            self._restore_bindings(handler_base)
            if handler.type is not None:
                self.visit(handler.type)
            self._enter_exception_handler(handler)
            handler_prefixes = [self._snapshot_bindings()]
            for statement in handler.body:
                self.visit(statement)
                handler_prefixes.append(self._snapshot_bindings())
                if not self._statement_falls_through(statement):
                    break
            exceptional_states.extend(handler_prefixes)
            self._exit_exception_handler(handler)
            if self._suite_falls_through(handler.body):
                states.append(self._snapshot_bindings())
        continuing_state = self._merge_bindings(states or [initial_state])
        pending_transfers = self._control_transfers[transfer_mark:]
        del self._control_transfers[transfer_mark:]
        if not node.finalbody:
            self._control_transfers.extend(pending_transfers)
            self._restore_bindings(continuing_state)
            return

        after_finally: list[BindingState] = []

        def run_finally_path(kind: str | None, state: BindingState) -> None:
            """Run the finally suite for one normal or abrupt incoming path.

            :param kind: Incoming transfer kind, or ``None`` for normal flow.
            :param state: Binding state entering the finally suite.
            :return: ``None``.
            """
            self._restore_bindings(state)
            self._visit_suite(node.finalbody)
            if not self._suite_falls_through(node.finalbody):
                return
            final_state = self._snapshot_bindings()
            if kind is None:
                after_finally.append(final_state)
            else:
                self._control_transfers.append((kind, final_state))

        if states:
            run_finally_path(None, continuing_state)
        for kind, state in pending_transfers:
            run_finally_path(kind, state)
        if exceptional_states:
            run_finally_path("raise", self._merge_bindings(exceptional_states))
        self._restore_bindings(self._merge_bindings(after_finally or [continuing_state]))

    visit_TryStar = visit_Try

    def visit_Match(self, node: ast.Match) -> None:
        """Join bindings from every match case and the no-match path."""
        self.visit(node.subject)
        base = self._snapshot_bindings()
        states = [base]
        for case in node.cases:
            self._restore_bindings(base)
            self._visit_match_pattern(case.pattern)
            if case.guard is not None:
                self.visit(case.guard)
            self._visit_suite(case.body)
            states.append(self._snapshot_bindings())
        self._restore_bindings(self._merge_bindings(states))
