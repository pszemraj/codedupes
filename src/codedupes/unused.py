"""Potentially-unused code detection (Python-only reference graph)."""

from __future__ import annotations

import ast
import codecs
import logging
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from codedupes.models import CodeUnit, CodeUnitType

logger = logging.getLogger(__name__)


@dataclass
class DefinitionReferences:
    """Names one ``def``/``class`` statement references, keyed to the unit it maps to.

    ``references`` holds the names loaded directly in this definition's body;
    ``nested`` holds the names loaded inside definitions nested in it, keyed by
    the nested definition, so that a nested definition's reference to itself
    can be excluded when the enclosing unit is credited with it.
    """

    name: str
    # (first decorator line, def line) or (def line,): the tree-sitter backend
    # starts a decorated unit at its decorator; the def line is kept so a unit
    # built any other way still resolves.
    linenos: tuple[int, ...]
    references: set[str] = field(default_factory=set)
    nested: dict[int, tuple[DefinitionReferences, set[str]]] = field(default_factory=dict)


@dataclass
class ClassInfo:
    """A class definition with its base expressions and public methods."""

    definition: DefinitionReferences
    bases: tuple[str, ...]
    public_methods: list[DefinitionReferences] = field(default_factory=list)


@dataclass
class ModuleReferences:
    """Everything one module parse contributes to the reference graph."""

    aliases: dict[str, str] = field(default_factory=dict)
    module_references: set[str] = field(default_factory=set)
    definitions: list[DefinitionReferences] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)


def _definition_linenos(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> tuple[int, ...]:
    """Return the line numbers a unit for this definition may start on.

    :param node: Definition node.
    :return: ``(first decorator line, def line)`` or ``(def line,)``.
    """
    if node.decorator_list:
        return (node.decorator_list[0].lineno, node.lineno)
    return (node.lineno,)


def _dotted_name(node: ast.expr) -> str | None:
    """Render a ``Name``/``Attribute`` chain as dotted text.

    :param node: Expression node.
    :return: Dotted name, or ``None`` when the chain contains anything else.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _dotted_name(node.value)
        return None if head is None else f"{head}.{node.attr}"
    return None


def _base_text(node: ast.expr) -> str:
    """Render a class base expression as the name it resolves through.

    :param node: Base expression.
    :return: Dotted name of the base, or ``"<unknown>"`` for anything else.
    """
    if isinstance(node, ast.Subscript):
        return _base_text(node.value)
    if isinstance(node, ast.Call):
        return _base_text(node.func)
    return _dotted_name(node) or "<unknown>"


class _ReferenceCollector(ast.NodeVisitor):
    """Attribute every loaded name in a module to the definitions that contain it.

    A name is recorded on every scope on the stack, so a class sees what its
    methods use and an outer function sees what its nested functions use.
    Names outside any definition belong to the module itself.
    """

    def __init__(self) -> None:
        """Start with an empty module scope."""
        self.module_references: set[str] = set()
        self.definitions: list[DefinitionReferences] = []
        self.classes: list[ClassInfo] = []
        self._scopes: list[DefinitionReferences] = []
        # Parallel to _scopes: the ClassInfo when that scope is a class body.
        self._class_scopes: list[ClassInfo | None] = []
        self._annotation_depth = 0

    def _record(self, name: str) -> None:
        """Attribute one referenced name to the enclosing scopes or the module.

        The innermost definition owns the reference; every enclosing definition
        records it as nested, tagged with its origin.

        :param name: Referenced name or dotted attribute path.
        :return: ``None``.
        """
        if not self._scopes:
            self.module_references.add(name)
            return
        origin = self._scopes[-1]
        origin.references.add(name)
        for scope in self._scopes[:-1]:
            scope.nested.setdefault(id(origin), (origin, set()))[1].add(name)

    def _visit_annotation(self, node: ast.expr) -> None:
        """Visit an annotation, unquoting string forward references on the way.

        :param node: Annotation expression.
        :return: ``None``.
        """
        self._annotation_depth += 1
        try:
            self.visit(node)
        finally:
            self._annotation_depth -= 1

    def visit_Name(self, node: ast.Name) -> None:
        """Record loaded names; stores and deletes are not uses."""
        if isinstance(node.ctx, ast.Load):
            self._record(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Record attribute access in any context (a property setter is a use too)."""
        self._record(node.attr)
        if isinstance(node.value, ast.Name):
            self._record(f"{node.value.id}.{node.attr}")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        """Parse quoted forward references inside annotations."""
        if self._annotation_depth == 0 or not isinstance(node.value, str):
            return
        try:
            tree = ast.parse(node.value, mode="eval")
        except (SyntaxError, ValueError):
            return
        self.visit(tree)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Visit a subscript, skipping the string values of ``Literal[...]``."""
        self.visit(node.value)
        if (_dotted_name(node.value) or "").rsplit(".", 1)[-1] != "Literal":
            self.visit(node.slice)

    def visit_Import(self, node: ast.Import) -> None:
        """Count an import as a reference to what it imports."""
        for alias in node.names:
            self._record(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Count a from-import as a reference to each imported name."""
        for alias in node.names:
            self._record(alias.name)

    def visit_arg(self, node: ast.arg) -> None:
        """Visit a parameter annotation."""
        if node.annotation is not None:
            self._visit_annotation(node.annotation)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit an annotated assignment with the annotation unquoted."""
        self._visit_annotation(node.annotation)
        self.visit(node.target)
        if node.value is not None:
            self.visit(node.value)

    def _enter(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> DefinitionReferences:
        """Register a definition and visit its scope-external parts.

        Decorators, defaults, annotations, bases, and type parameters are
        evaluated in the enclosing namespace, so they are attributed there.

        :param node: Definition node.
        :return: The registered definition.
        """
        definition = DefinitionReferences(name=node.name, linenos=_definition_linenos(node))
        self.definitions.append(definition)
        for decorator in node.decorator_list:
            self.visit(decorator)
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                self.visit(base)
            for keyword in node.keywords:
                self.visit(keyword)
        else:
            self.visit(node.args)
            if node.returns is not None:
                self._visit_annotation(node.returns)
        for type_param in getattr(node, "type_params", ()):
            self.visit(type_param)
        return definition

    def _visit_body(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        definition: DefinitionReferences,
        class_info: ClassInfo | None,
    ) -> None:
        """Visit a definition body inside its own scope.

        :param node: Definition node.
        :param definition: Scope to attribute body references to.
        :param class_info: Class record when the scope is a class body.
        :return: ``None``.
        """
        self._scopes.append(definition)
        self._class_scopes.append(class_info)
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self._scopes.pop()
            self._class_scopes.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Register a function or method and collect its references.

        :param node: Function definition node.
        :return: ``None``.
        """
        definition = self._enter(node)
        enclosing_class = self._class_scopes[-1] if self._class_scopes else None
        if enclosing_class is not None and not node.name.startswith("_"):
            enclosing_class.public_methods.append(definition)
        self._visit_body(node, definition, None)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Register a function or method and collect its references."""
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Register an async function exactly like a plain one."""
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Register a class, its bases, and collect its body references."""
        definition = self._enter(node)
        class_info = ClassInfo(
            definition=definition,
            bases=tuple(_base_text(base) for base in node.bases),
        )
        self.classes.append(class_info)
        self._visit_body(node, definition, class_info)


def _parse_module(file_path: Path) -> ast.Module | None:
    """Parse one Python file the way the extractor reads it, warning when ``ast`` cannot.

    The extractor skips a BOM and decodes invalid UTF-8 lossily, so the same
    bytes are parsed here; a file ``ast`` still rejects (a syntax error the
    grammar recovered from, syntax newer than the interpreter) contributes no
    references, which the warning makes visible.

    :param file_path: Python source path.
    :return: Parsed module, or ``None`` when the file cannot be read or parsed.
    """
    try:
        raw = file_path.read_bytes()
    except OSError as error:
        logger.warning(f"Unused analysis skipped {file_path}: {error}")
        return None
    source = raw.removeprefix(codecs.BOM_UTF8).decode("utf-8", errors="replace")
    try:
        # ValueError covers Python 3.11's embedded-NUL report.
        return ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as error:
        logger.warning(
            f"Unused analysis collected no references from {file_path}: "
            f"{type(error).__name__}: {error}"
        )
        return None


def _extract_aliases(tree: ast.Module) -> dict[str, str]:
    """Extract a conservative alias map from module-level imports and assignments.

    :param tree: Parsed module.
    :return: Alias map for name resolution.
    """
    aliases: dict[str, str] = {}

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                asname = alias.asname or name.rsplit(".", 1)[-1]
                aliases[asname] = name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            for alias in node.names:
                imported = alias.name
                asname = alias.asname or imported
                aliases[asname] = f"{base}.{imported}" if base else imported
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target = node.targets[0].id
            value = node.value
            if isinstance(value, ast.Name):
                aliases[target] = value.id
            elif isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                aliases[target] = f"{value.value.id}.{value.attr}"
    return aliases


def collect_module_references(file_path: Path) -> ModuleReferences:
    """Parse one module once and collect aliases plus per-scope references.

    :param file_path: Python source path.
    :return: Module references; empty when the file cannot be parsed.
    """
    tree = _parse_module(file_path)
    if tree is None:
        return ModuleReferences()
    collector = _ReferenceCollector()
    try:
        collector.visit(tree)
    except RecursionError:
        # ast.NodeVisitor recurses per node; a generated elif or operator
        # chain a few hundred deep overflows it while tree-sitter copes.
        logger.warning(
            f"Unused analysis collected no references from {file_path}: "
            "expression nesting exceeds the interpreter recursion limit"
        )
        return ModuleReferences()
    return ModuleReferences(
        aliases=_extract_aliases(tree),
        module_references=collector.module_references,
        definitions=collector.definitions,
        classes=collector.classes,
    )


def _resolve_reference_targets(name: str, aliases: dict[str, str]) -> set[str]:
    """Expand a referenced name through the module's import and assignment aliases.

    :param name: Referenced name or dotted attribute path.
    :param aliases: Alias map from local symbols to full targets.
    :return: Candidate target names.
    """
    candidates = {name}
    if name in aliases:
        candidates.add(aliases[name])
    if "." in name:
        head, _, tail = name.partition(".")
        if head in aliases:
            candidates.add(f"{aliases[head]}.{tail}")
    return candidates


def _extract_pyproject_entry_points(project_root: Path) -> set[str]:
    """Collect callable targets from ``[project.scripts]``, ``gui-scripts``, and ``entry-points``.

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

    tables = [project_cfg.get("scripts", {}), project_cfg.get("gui-scripts", {})]
    groups = project_cfg.get("entry-points", {})
    if isinstance(groups, dict):
        tables.extend(groups.values())

    targets: set[str] = set()
    for table in tables:
        if not isinstance(table, dict):
            continue
        for value in table.values():
            if not isinstance(value, str):
                continue
            target = value.split(":", 1)[-1]
            if "." in target:
                target = target.rsplit(".", 1)[-1]
            target = target.strip()
            if target:
                targets.add(target)

    return targets


def _framework_derived_classes(
    classes: list[tuple[Path, ClassInfo, dict[str, str]]],
) -> list[tuple[Path, ClassInfo, str]]:
    """Find classes with a base that does not resolve to a project class.

    Each base is expanded through its module's import and assignment aliases,
    then resolved by name: a last dotted segment is a project class when some
    collected class carries that name, so an external base whose name collides
    with a project class resolves as project. ``object`` never counts.
    Derivation is transitive, so subclasses of a derived class are derived too.

    :param classes: Every collected class with its file and module alias map.
    :return: Each derived class with the base that made it derived.
    """
    project_names = {class_info.definition.name for _path, class_info, _aliases in classes}
    derived_names: set[str] = set()
    derived_base: dict[int, str] = {}
    changed = True
    while changed:
        changed = False
        for _path, class_info, aliases in classes:
            if id(class_info) in derived_base:
                continue
            for base in class_info.bases:
                tails = {
                    target.rsplit(".", 1)[-1]
                    for target in _resolve_reference_targets(base, aliases)
                }
                external = "object" not in tails and not (tails & project_names)
                if tails & derived_names or external:
                    derived_base[id(class_info)] = base
                    derived_names.add(class_info.definition.name)
                    changed = True
                    break
    return [
        (path, class_info, derived_base[id(class_info)])
        for path, class_info, _aliases in classes
        if id(class_info) in derived_base
    ]


def build_reference_graph(
    units: list[CodeUnit],
    project_root: Path | None = None,
    source_files: list[Path] | None = None,
) -> None:
    """Populate ``unit.references`` from every name each Python module loads.

    Matching is by name: a reference to ``helper`` marks every unit named
    ``helper`` (or whose qualified name ends in the referenced dotted path),
    except the referring unit itself.

    :param units: Collected code units; non-Python units are ignored.
    :param project_root: Optional root for pyproject entry-point resolution.
    :param source_files: Every Python file the extractor visited; files without
        units (re-export modules, scripts) still contribute references.
    :return: ``None``.
    """
    units = [unit for unit in units if unit.language == "python"]
    if not units:
        return

    by_name: dict[str, list[CodeUnit]] = defaultdict(list)
    by_location: dict[tuple[Path, int, str], list[CodeUnit]] = defaultdict(list)
    for unit in units:
        by_name[unit.name].append(unit)
        parts = unit.qualified_name.split(".")
        for i in range(len(parts)):
            by_name[".".join(parts[i:])].append(unit)
        by_location[(unit.file_path, unit.lineno, unit.name)].append(unit)

    def mark(
        referrer_uid: str,
        names: set[str],
        aliases: dict[str, str],
        excluded: frozenset[str] = frozenset(),
    ) -> None:
        """Add one referrer to every unit a set of names resolves to.

        :param referrer_uid: Unit uid or synthetic scope id doing the referencing.
        :param names: Referenced names.
        :param aliases: Alias map of the referring module.
        :param excluded: Unit uids a match must not credit (the referrer and the
            definition whose own body produced the names).
        :return: ``None``.
        """
        excluded = excluded | {referrer_uid}
        for name in names:
            for target in _resolve_reference_targets(name, aliases):
                for candidate in by_name.get(target, []):
                    if candidate.uid not in excluded:
                        candidate.references.add(referrer_uid)

    def units_for(file_path: Path, definition: DefinitionReferences) -> list[CodeUnit]:
        """Find the units extracted for one definition.

        :param file_path: Module the definition lives in.
        :param definition: Collected definition.
        :return: Matching units (none when the extractor filtered the symbol).
        """
        return [
            unit
            for lineno in definition.linenos
            for unit in by_location.get((file_path, lineno, definition.name), [])
        ]

    module_paths = {unit.file_path for unit in units} | set(source_files or ())
    modules = {
        file_path: collect_module_references(file_path) for file_path in sorted(module_paths)
    }
    for file_path, module in modules.items():
        mark(f"__module__::{file_path}", module.module_references, module.aliases)
        for definition in module.definitions:
            for unit in units_for(file_path, definition):
                mark(unit.uid, definition.references, module.aliases)
                # A nested definition's reference to itself must not surface
                # as the enclosing unit referencing it.
                for origin, names in definition.nested.values():
                    own = frozenset(item.uid for item in units_for(file_path, origin))
                    mark(unit.uid, names, module.aliases, own)

    # Public methods of classes deriving from outside the project are reached
    # by the framework's dispatch (NodeVisitor.visit_*, logging.Filter.filter),
    # which no in-project name can show.
    all_classes = [
        (path, cls, module.aliases) for path, module in modules.items() for cls in module.classes
    ]
    for file_path, class_info, base in _framework_derived_classes(all_classes):
        for method in class_info.public_methods:
            for unit in units_for(file_path, method):
                unit.references.add(f"framework::{base}")

    # Seed references from project entry points.
    if project_root is not None:
        root = project_root if project_root.is_dir() else project_root.parent
        for target in _extract_pyproject_entry_points(root):
            for candidate in by_name.get(target, []):
                candidate.references.add("project.entrypoint")


def _is_public_surface(unit: CodeUnit) -> bool:
    """Return whether default mode treats the unit as public API that callers outside the tree may use.

    A public name reachable only through a private module, class, or function
    (``pkg._impl.helper``, ``_Outer.Inner.method``, ``_factory.inner``) is not
    surface, so every segment of the qualified name must be public.

    :param unit: Candidate unit.
    :return: ``True`` for public functions and public methods of public classes.
    """
    if unit.unit_type not in (CodeUnitType.FUNCTION, CodeUnitType.METHOD):
        return False
    return not any(part.startswith("_") for part in unit.qualified_name.split("."))


def _decorators(unit: CodeUnit) -> str:
    """Return the decorator lines that precede a unit's own ``def``/``class`` line.

    :param unit: Unit whose source starts at its first decorator when decorated.
    :return: The decorator text, empty when the unit is not decorated.
    """
    lines: list[str] = []
    for line in unit.source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("def ", "async def ", "class ")):
            break
        lines.append(stripped)
    return "\n".join(lines)


def find_potentially_unused(units: list[CodeUnit], strict_unused: bool = False) -> list[CodeUnit]:
    """Find code units that are never referenced and are not likely API.

    :param units: Candidate code units.
    :param strict_unused: Whether to report public functions and public methods of public classes too.
    :return: Units with no references and not classified as API.
    """
    unused = []
    for unit in units:
        if unit.language != "python":
            continue
        if not strict_unused and _is_public_surface(unit):
            continue

        if unit.references:
            continue

        source = unit.source.lower()
        if "noqa: codedupes" in source or "codedupes: ignore" in source:
            continue

        if unit.is_likely_api:
            continue
        if unit.name == "__init__":
            continue
        if unit.name.startswith("get_") or unit.name.startswith("set_"):
            continue
        decorators = _decorators(unit)
        if "@abstractmethod" in decorators or "@abc.abstractmethod" in decorators:
            continue
        if unit.name.startswith("test_") or "_test" in unit.file_path.name:
            continue

        unused.append(unit)

    return unused


def run_unused_analysis(
    units: list[CodeUnit],
    *,
    project_root: Path | None,
    strict_unused: bool,
    source_files: list[Path] | None = None,
) -> list[CodeUnit]:
    """Build the reference graph and report the units it leaves unreferenced.

    :param units: Collected code units; non-Python units are ignored.
    :param project_root: Project root for pyproject entry-point resolution, or ``None``.
    :param source_files: Every Python file the extractor visited, units or not.
    :param strict_unused: Whether to report public functions and public methods of public classes too.
    :return: Potentially unused Python units.
    """
    build_reference_graph(units, project_root=project_root, source_files=source_files)
    unused = find_potentially_unused(units, strict_unused=strict_unused)
    logger.info(f"Found {len(unused)} potentially unused code units")
    return unused
