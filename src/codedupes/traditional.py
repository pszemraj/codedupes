"""Traditional (non-ML) duplicate detection methods."""

from __future__ import annotations

import ast
import builtins
import logging
import tomllib
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from codedupes._reference_flow import (
    BranchingVisitor,
    comprehension_definitely_runs,
    comprehension_exact_result_count,
    iterable_definitely_empty,
    iterable_definitely_nonempty,
)
from codedupes.constants import DEFAULT_TRADITIONAL_THRESHOLD
from codedupes.models import CodeUnit, CodeUnitType, DuplicatePair
from codedupes.pairs import ordered_pair_key

logger = logging.getLogger(__name__)

_OPAQUE_ALIAS = "\0opaque"


def _find_exact_duplicates(
    units: list[CodeUnit], hash_attr: str, method: str
) -> list[DuplicatePair]:
    """Find duplicate pairs by grouping units by a stored hash attribute.

    :param units: Candidate units to compare.
    :param hash_attr: Unit attribute name containing a hash.
    :param method: Duplicate classification label.
    :return: Exact duplicate pairs for the selected hash field.
    """
    by_hash: dict[str, list[CodeUnit]] = defaultdict(list)

    for unit in units:
        value = getattr(unit, hash_attr, None)
        if value:
            by_hash[value].append(unit)

    duplicates = []
    for group in by_hash.values():
        if len(group) <= 1:
            continue
        for a, b in combinations(group, 2):
            duplicates.append(DuplicatePair(unit_a=a, unit_b=b, similarity=1.0, method=method))

    return duplicates


def find_exact_pair_keys(units: list[CodeUnit]) -> set[tuple[str, str]]:
    """Return ordered uid pair keys for every exact-duplicate pair.

    Uses the same predicate as :func:`run_traditional_analysis` exact detection:
    two units are exact duplicates when they share ``_ast_hash`` or ``_token_hash``.

    :param units: Candidate units to compare.
    :return: Ordered uid pair keys covering all exact-duplicate pairs.
    """
    pairs = _find_exact_duplicates(units, "_ast_hash", "ast_hash") + _find_exact_duplicates(
        units, "_token_hash", "token_hash"
    )
    return {ordered_pair_key(pair.unit_a, pair.unit_b) for pair in pairs}


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets.

    :param set_a: First identifier set.
    :param set_b: Second identifier set.
    :return: Intersection-over-union score.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def extract_identifiers(source: str) -> set[str]:
    """Extract all identifiers from source code.

    :param source: Source text.
    :return: Identifier names found in the AST.
    """
    identifiers = set()
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                identifiers.add(node.name)
            elif isinstance(node, ast.arg):
                identifiers.add(node.arg)
    except SyntaxError:
        pass
    return _normalize_identifiers(identifiers)


def _normalize_identifiers(identifiers: set[str]) -> set[str]:
    """Normalize identifier sets for stable near-duplicate matching.

    :param identifiers: Raw identifier names.
    :return: Normalized filtered identifiers.
    """
    ignored = set(dir(builtins))
    return identifiers - ignored


def find_near_duplicates_jaccard(
    units: list[CodeUnit],
    threshold: float = DEFAULT_TRADITIONAL_THRESHOLD,
) -> list[DuplicatePair]:
    """Find near-duplicates via Jaccard similarity on identifiers.

    :param units: Candidate units.
    :param threshold: Jaccard cutoff.
    :return: Near-duplicate pairs above threshold.
    """
    identifier_sets = {unit.uid: extract_identifiers(unit.source) for unit in units}

    duplicates = []
    for i, a in enumerate(units):
        for b in units[i + 1 :]:
            # Skip if same file and overlapping lines (parent/child)
            if a.file_path == b.file_path and not (
                a.end_lineno < b.lineno or b.end_lineno < a.lineno
            ):
                continue

            set_a = identifier_sets[a.uid]
            set_b = identifier_sets[b.uid]
            if not set_a or not set_b:
                continue

            size_ratio = min(len(set_a), len(set_b)) / max(len(set_a), len(set_b), 1)
            if size_ratio < threshold / 2:
                continue

            sim = jaccard_similarity(set_a, set_b)
            if sim >= threshold:
                duplicates.append(
                    DuplicatePair(unit_a=a, unit_b=b, similarity=sim, method="jaccard")
                )

    return duplicates


def _dedupe_duplicate_pairs(duplicates: list[DuplicatePair]) -> list[DuplicatePair]:
    """Deduplicate unordered duplicate pairs.

    :param duplicates: Duplicate candidates.
    :return: Unique duplicate pairs.
    """
    seen: set[tuple[str, str]] = set()
    deduped: list[DuplicatePair] = []
    for dup in duplicates:
        key = ordered_pair_key(dup.unit_a, dup.unit_b)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dup)
    return deduped


def _resolve_reference_targets(reference: str, aliases: dict[str, set[str]]) -> set[str]:
    """Resolve a reference through every possible module alias.

    :param reference: Raw referenced-name string.
    :param aliases: Alias map from local symbols to full targets.
    :return: Candidate reference target names.
    """
    if reference in aliases:
        return {target for target in aliases[reference] if not target.startswith(_OPAQUE_ALIAS)}
    if "." in reference:
        head, _, tail = reference.partition(".")
        if head in aliases:
            return {
                f"{target}.{tail}"
                for target in aliases[head]
                if not target.startswith(_OPAQUE_ALIAS)
            }
    return {reference}


def _reference_candidates(
    target: str,
    module_name: str,
    import_module_name: str,
    by_qualified_name: dict[str, list[CodeUnit]],
    by_import_name: dict[str, list[CodeUnit]],
) -> list[CodeUnit]:
    """Resolve one direct target, preferring the caller's own module namespace.

    :param target: Alias-resolved bare or dotted target.
    :param module_name: Root-relative module containing the caller.
    :param import_module_name: Importable module containing the caller.
    :param by_qualified_name: Exact emitted qualified-name index.
    :param by_import_name: Exact importable qualified-name index.
    :return: Best available candidate units.
    """
    exact = {unit.uid: unit for unit in by_qualified_name.get(target, [])}
    exact.update({unit.uid: unit for unit in by_import_name.get(target, [])})
    if exact:
        return list(exact.values())
    for containing_module in (import_module_name, module_name):
        if not containing_module or target.startswith(f"{containing_module}."):
            continue
        scoped_target = f"{containing_module}.{target}"
        scoped = {unit.uid: unit for unit in by_qualified_name.get(scoped_target, [])}
        scoped.update({unit.uid: unit for unit in by_import_name.get(scoped_target, [])})
        if scoped:
            return list(scoped.values())
    return []


def _extract_main_block_references(
    file_path: Path,
    module_name: str,
) -> tuple[set[str], set[str], set[str]]:
    """Extract names referenced from an if-``__main__`` block.

    :param file_path: Path to inspect.
    :param module_name: Root-relative identity of the containing module.
    :return: Unresolved names, resolved names, and unresolved attribute names.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set(), set(), set()

    from codedupes.extractor import ReferenceVisitor

    import_package = (
        module_name
        if file_path.name in {"__init__.py", "__init__.pyi"}
        else module_name.rpartition(".")[0]
    )
    visitor = ReferenceVisitor(
        qualified_name=module_name,
        import_package=import_package,
    )

    for node in tree.body:
        if not isinstance(node, ast.If):
            continue

        is_main = False
        test = node.test
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
            left = test.left
            comparators = test.comparators
            if len(comparators) == 1:
                right = comparators[0]
                if (
                    isinstance(left, ast.Name)
                    and left.id == "__name__"
                    and isinstance(right, ast.Constant)
                    and right.value == "__main__"
                ) or (
                    isinstance(left, ast.Constant)
                    and left.value == "__main__"
                    and isinstance(right, ast.Name)
                    and right.id == "__name__"
                ):
                    is_main = True

        if not is_main:
            continue

        for stmt in node.body:
            visitor.visit(stmt)

    return visitor.names, visitor.resolved_names, visitor.attributes


def _extract_pyproject_entry_points(project_root: Path) -> set[str]:
    """Collect callable targets from ``[project.scripts]`` and ``[project.gui-scripts]``.

    :param project_root: Project root path.
    :return: Entry point callable names.
    """
    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.is_file():
        return set()

    try:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeError):
        return set()

    project_cfg = data.get("project", {})
    if not isinstance(project_cfg, dict):
        return set()

    targets: set[str] = set()
    for section in ("scripts", "gui-scripts"):
        script_entries = project_cfg.get(section, {})
        if not isinstance(script_entries, dict):
            continue
        for value in script_entries.values():
            if not isinstance(value, str):
                continue
            module, separator, callable_path = value.partition(":")
            module = module.strip()
            callable_path = callable_path.strip()
            if separator and module and callable_path:
                targets.add(f"{module}.{callable_path}")

    return targets


def build_reference_graph(units: list[CodeUnit], project_root: Path | None = None) -> None:
    """Populate references from collected name references, entrypoints, and ``__main__`` blocks.

    :param units: Collected code units.
    :param project_root: Optional root for entry point resolution.
    :return: ``None``.
    """
    by_qualified_name: dict[str, list[CodeUnit]] = defaultdict(list)
    by_import_name: dict[str, list[CodeUnit]] = defaultdict(list)
    by_attribute: dict[str, list[CodeUnit]] = defaultdict(list)
    for unit in units:
        unit.references.clear()
        by_qualified_name[unit.qualified_name].append(unit)
        import_name = unit.qualified_name
        if unit.import_module_name:
            suffix = unit.qualified_name
            if unit.module_name:
                suffix = suffix.removeprefix(f"{unit.module_name}.")
            import_name = f"{unit.import_module_name}.{suffix}"
        by_import_name[import_name].append(unit)
        if unit.unit_type in {CodeUnitType.METHOD, CodeUnitType.CLASS}:
            by_attribute[unit.name].append(unit)

    alias_map_by_file: dict[Path, dict[str, set[str]]] = {}
    for unit in units:
        if unit.file_path not in alias_map_by_file:
            alias_map_by_file[unit.file_path] = _extract_aliases(
                unit.file_path,
                unit.import_module_name or unit.module_name,
            )

    # Populate references from each unit's collected name references.
    for unit in units:
        file_aliases = alias_map_by_file.get(unit.file_path, {})
        for reference in unit.referenced_names - unit.resolved_referenced_names:
            for target in _resolve_reference_targets(reference, file_aliases):
                for candidate in _reference_candidates(
                    target,
                    unit.module_name,
                    unit.import_module_name,
                    by_qualified_name,
                    by_import_name,
                ):
                    if candidate.uid != unit.uid:
                        candidate.references.add(unit.uid)
        for target in unit.resolved_referenced_names:
            for candidate in _reference_candidates(
                target,
                unit.module_name,
                unit.import_module_name,
                by_qualified_name,
                by_import_name,
            ):
                if candidate.uid != unit.uid:
                    candidate.references.add(unit.uid)
        for reference in unit.module_attribute_references:
            head = reference.partition(".")[0]
            targets = file_aliases.get(head)
            if targets is not None and (
                not targets or any(target.startswith(_OPAQUE_ALIAS) for target in targets)
            ):
                attribute = reference.rpartition(".")[2]
                for candidate in by_attribute.get(attribute, []):
                    if candidate.uid != unit.uid:
                        candidate.references.add(unit.uid)
        for attribute in unit.referenced_attributes:
            for candidate in by_attribute.get(attribute, []):
                if candidate.uid != unit.uid:
                    candidate.references.add(unit.uid)
    # Seed references from __main__ blocks.
    for file_path, file_aliases in alias_map_by_file.items():
        caller_uid = f"__main__::{file_path}"
        file_unit = next((unit for unit in units if unit.file_path == file_path), None)
        module_name = file_unit.module_name if file_unit is not None else ""
        import_module_name = file_unit.import_module_name if file_unit is not None else module_name
        main_references, resolved_main_references, main_attributes = _extract_main_block_references(
            file_path,
            import_module_name,
        )
        for reference in main_references - resolved_main_references:
            for target in _resolve_reference_targets(reference, file_aliases):
                for candidate in _reference_candidates(
                    target,
                    module_name,
                    import_module_name,
                    by_qualified_name,
                    by_import_name,
                ):
                    candidate.references.add(caller_uid)
        for target in resolved_main_references:
            for candidate in _reference_candidates(
                target,
                module_name,
                import_module_name,
                by_qualified_name,
                by_import_name,
            ):
                candidate.references.add(caller_uid)
        for attribute in main_attributes:
            for candidate in by_attribute.get(attribute, []):
                candidate.references.add(caller_uid)
    # Seed references from project entry points.
    if project_root is not None:
        root = project_root if project_root.is_dir() else project_root.parent
        for target in _extract_pyproject_entry_points(root):
            candidates = {unit.uid: unit for unit in by_qualified_name.get(target, [])}
            candidates.update({unit.uid: unit for unit in by_import_name.get(target, [])})
            for candidate in candidates.values():
                candidate.references.add("project.entrypoint")


class _ModuleAliasVisitor(BranchingVisitor[dict[str, set[str]]]):
    """Resolve possible module bindings without entering function or class bodies."""

    def __init__(self, module_name: str, import_package: str) -> None:
        """Initialize a module alias collector.

        :param module_name: Importable identity of the current module.
        :param import_package: Package used to resolve relative imports.
        """
        super().__init__()
        self.module_name = module_name
        self.import_package = import_package
        self.aliases: dict[str, set[str]] = {}

    def _snapshot_bindings(self) -> dict[str, set[str]]:
        """Copy possible bindings at the current program point.

        :return: An independent alias-binding snapshot.
        """
        return {name: set(targets) for name, targets in self.aliases.items()}

    def _restore_bindings(self, state: dict[str, set[str]]) -> None:
        """Restore possible bindings at one program point."""
        self.aliases = {name: set(targets) for name, targets in state.items()}

    @staticmethod
    def _merge_bindings(states: list[dict[str, set[str]]]) -> dict[str, set[str]]:
        """Join possible bindings from alternative control-flow paths.

        :param states: Alternative alias-binding states.
        :return: The conservative union of all aliases.
        """
        return {
            name: {
                *(target for state in states for target in state.get(name, set())),
                *(
                    {_OPAQUE_ALIAS}
                    if any(name in state and not state[name] for state in states)
                    else set()
                ),
            }
            for name in {name for state in states for name in state}
        }

    @staticmethod
    def _target_names(target: ast.AST) -> set[str]:
        """Collect bare names rebound by an assignment-style target.

        :param target: Assignment-style target expression.
        :return: Bare names bound by the target.
        """
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            return {
                name
                for element in target.elts
                for name in _ModuleAliasVisitor._target_names(element)
            }
        if isinstance(target, ast.Starred):
            return _ModuleAliasVisitor._target_names(target.value)
        return set()

    def _expression_targets(self, node: ast.AST) -> set[str]:
        """Resolve simple name and attribute expressions through known aliases.

        :param node: Expression whose possible qualified targets are needed.
        :return: Possible qualified targets for the expression.
        """
        if isinstance(node, ast.Name):
            return set(self.aliases[node.id]) if node.id in self.aliases else {node.id}
        if isinstance(node, ast.Attribute):
            return {f"{target}.{node.attr}" for target in self._expression_targets(node.value)}
        return set()

    def visit_Import(self, node: ast.Import) -> None:
        """Collect module import bindings."""
        for alias in node.names:
            name = alias.asname or alias.name.partition(".")[0]
            target = alias.name if alias.asname else name
            self.aliases[name] = {target}

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Collect absolute identities for from-import bindings."""
        from codedupes.extractor import _resolve_relative_module

        module = _resolve_relative_module(self.import_package, node.level, node.module)
        for alias in node.names:
            if alias.name == "*":
                continue
            name = alias.asname or alias.name
            target = f"{module}.{alias.name}" if module else alias.name
            self.aliases[name] = {target}

    def visit_Assign(self, node: ast.Assign) -> None:
        """Collect simple alias assignments and inspect their value expressions."""
        targets = self._expression_targets(node.value)
        self.visit(node.value)
        for target in node.targets:
            for name in self._target_names(target):
                self.aliases[name] = set(targets)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Collect a simple annotated alias assignment."""
        if node.value is not None:
            targets = self._expression_targets(node.value)
            self.visit(node.value)
            for name in self._target_names(node.target):
                self.aliases[name] = set(targets)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """Apply a module assignment-expression binding after its value."""
        targets = self._expression_targets(node.value)
        self.visit(node.value)
        for name in self._target_names(node.target):
            self.aliases[name] = set(targets)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Treat augmented assignment results as opaque bindings."""
        self.visit(node.target)
        self.visit(node.value)
        for name in self._target_names(node.target):
            self.aliases[name] = set()

    def visit_Delete(self, node: ast.Delete) -> None:
        """Record deleted bare names as unbound rather than stale aliases."""
        for target in node.targets:
            for name in self._target_names(target):
                self.aliases[name] = set()

    def visit_With(self, node: ast.With) -> None:
        """Apply ``with ... as`` rebinding before visiting the suite."""
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                for name in self._target_names(item.optional_vars):
                    self.aliases[name] = set()
        self._visit_suite(node.body)

    visit_AsyncWith = visit_With

    def _visit_comprehension(
        self,
        generators: list[ast.comprehension],
        values: tuple[ast.AST, ...],
        *,
        eager: bool,
    ) -> None:
        """Collect module walrus bindings without leaking generator targets.

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
        """Collect an eager list comprehension's module walrus bindings."""
        self._visit_comprehension(node.generators, (node.elt,), eager=True)

    visit_SetComp = visit_ListComp

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """Join deferred generator-expression bindings with the module base state."""
        self._visit_comprehension(node.generators, (node.elt,), eager=False)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Collect an eager dictionary comprehension's module walrus bindings."""
        self._visit_comprehension(node.generators, (node.key, node.value), eager=True)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Bind a module function after evaluating its header."""
        for expression in (
            *node.decorator_list,
            *node.args.defaults,
            *(default for default in node.args.kw_defaults if default is not None),
        ):
            self.visit(expression)
        target = f"{self.module_name}.{node.name}" if self.module_name else node.name
        self.aliases[node.name] = {target}

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Bind a module class after evaluating its header."""
        for expression in (
            *node.decorator_list,
            *node.bases,
            *(keyword.value for keyword in node.keywords),
        ):
            self.visit(expression)
        target = f"{self.module_name}.{node.name}" if self.module_name else node.name
        self.aliases[node.name] = {target}

    def _visit_loop_target(self, target: ast.expr) -> None:
        """Clear aliases rebound by a loop target."""
        for name in self._target_names(target):
            self.aliases[name] = set()

    def _enter_exception_handler(self, handler: ast.ExceptHandler) -> None:
        """Clear an imported alias rebound by an exception handler."""
        if handler.name is not None:
            self.aliases[handler.name] = set()

    def _exit_exception_handler(self, handler: ast.ExceptHandler) -> None:
        """Keep an exception alias unbound after handler cleanup."""
        if handler.name is not None:
            self.aliases[handler.name] = set()

    def _visit_match_pattern(self, pattern: ast.pattern) -> None:
        """Clear aliases rebound by match captures."""
        for part in ast.walk(pattern):
            if isinstance(part, (ast.MatchAs, ast.MatchStar)) and part.name is not None:
                self.aliases[part.name] = set()
            elif isinstance(part, ast.MatchMapping) and part.rest is not None:
                self.aliases[part.rest] = set()


def _extract_aliases(file_path: Path, module_name: str) -> dict[str, set[str]]:
    """Extract a conservative alias map from module-level imports and assignments.

    :param file_path: Python source path.
    :param module_name: Root-relative identity of the containing module.
    :return: Possible alias targets for name resolution.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return {}

    import_package = (
        module_name
        if file_path.name in {"__init__.py", "__init__.pyi"}
        else module_name.rpartition(".")[0]
    )
    visitor = _ModuleAliasVisitor(module_name, import_package)
    for node in tree.body:
        visitor.visit(node)
    return dict(visitor.aliases)


def find_potentially_unused(units: list[CodeUnit], strict_unused: bool = False) -> list[CodeUnit]:
    """Find code units that are never referenced and are not likely API.

    :param units: Candidate code units.
    :param strict_unused: Whether to include likely public functions in results.
    :return: Units with no references and not classified as API.
    """
    unused = []
    for unit in units:
        if not strict_unused and unit.unit_type == CodeUnitType.FUNCTION and unit.is_public:
            continue

        if unit.references:
            continue

        if unit.is_dynamic_dispatch_hook:
            continue

        source = unit.source.lower()
        if "noqa: codedupes" in source or "codedupes: ignore" in source:
            continue

        if unit.is_likely_api:
            continue
        if unit.name.startswith("get_") or unit.name.startswith("set_"):
            continue
        if "@abstractmethod" in unit.source or "@abc.abstractmethod" in unit.source:
            continue
        # Deliberately broader than analyzer._is_test_function_unit (any unit type,
        # plus file-name matching): unused reporting should stay quiet for anything
        # test-shaped, while duplicate suppression must stay narrow.
        if unit.name.startswith("test_") or "_test" in unit.file_path.name:
            continue

        unused.append(unit)

    return unused


def run_traditional_analysis(
    units: list[CodeUnit],
    jaccard_threshold: float = DEFAULT_TRADITIONAL_THRESHOLD,
) -> tuple[list[DuplicatePair], list[DuplicatePair]]:
    """Run traditional exact and near-duplicate detection.

    :param units: Candidate code units.
    :param jaccard_threshold: Similarity threshold for near-duplicate detection.
    :return: Exact and near-duplicate lists.
    """
    logger.info(f"Running traditional analysis on {len(units)} code units")

    ast_dupes = _find_exact_duplicates(units, "_ast_hash", "ast_hash")
    token_dupes = _find_exact_duplicates(units, "_token_hash", "token_hash")
    exact = _dedupe_duplicate_pairs(ast_dupes + token_dupes)
    logger.info(f"Found {len(exact)} exact duplicates")

    near = find_near_duplicates_jaccard(units, threshold=jaccard_threshold)
    exact_pairs = {ordered_pair_key(d.unit_a, d.unit_b) for d in exact}
    near = [d for d in near if ordered_pair_key(d.unit_a, d.unit_b) not in exact_pairs]
    logger.info(f"Found {len(near)} near duplicates (Jaccard)")

    return exact, near
