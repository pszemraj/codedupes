"""Potentially-unused code detection (Python-only reference graph)."""

from __future__ import annotations

import ast
import codecs
import logging
import re
import sys
import tomllib
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from codedupes.extractor import git_work_tree
from codedupes.models import CodeUnit, CodeUnitType, ExtractionDiagnostic

logger = logging.getLogger(__name__)

# ast.NodeVisitor recurses per syntax node; the interpreter's default limit
# overflows on a generated elif or operator chain a few hundred deep, well
# inside what tree-sitter extracts without complaint. This is raised for the
# reference walk only, never lowered, and restored afterward.
_VISIT_RECURSION_LIMIT = 15_000


@dataclass
class DefinitionReferences:
    """A ``def``/``class`` statement and its lexical reference scope."""

    name: str
    # (first decorator line, def line) or (def line,): the tree-sitter backend
    # starts a decorated unit at its decorator; the def line is kept so a unit
    # built any other way still resolves.
    linenos: tuple[int, ...]
    is_class: bool = False
    decorators: tuple[str, ...] = ()
    parent: DefinitionReferences | None = field(default=None, repr=False, compare=False)
    children_by_name: dict[str, list[DefinitionReferences]] = field(
        default_factory=dict, repr=False, compare=False
    )
    global_names: set[str] = field(default_factory=set, repr=False, compare=False)
    nonlocal_names: set[str] = field(default_factory=set, repr=False, compare=False)
    uses: list[ReferenceUse] = field(default_factory=list, repr=False, compare=False)


@dataclass
class ReferenceUse:
    """One loaded name, the definition it was loaded in, and whether it was a bare name."""

    name: str
    origin: DefinitionReferences = field(repr=False)
    bare: bool


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
    diagnostic: ExtractionDiagnostic | None = None


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

    def _record(self, name: str, *, bare: bool = False) -> None:
        """Attribute one referenced name to the enclosing scopes or the module.

        Every enclosing definition records the use with its original scope.

        :param name: Referenced name or dotted attribute path.
        :param bare: Whether this was a bare ``Name`` load.
        :return: ``None``.
        """
        if not self._scopes:
            self.module_references.add(name)
            return
        use = ReferenceUse(name, self._scopes[-1], bare)
        for scope in self._scopes:
            scope.uses.append(use)

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
            self._record(node.id, bare=True)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Record attribute access in any context (a property setter is a use too)."""
        self._record(node.attr)
        if isinstance(node.value, ast.Name):
            self._record(f"{node.value.id}.{node.attr}")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        """Record names that bypass enclosing definition scopes."""
        if self._scopes:
            self._scopes[-1].global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """Record names whose bindings belong to an enclosing function scope."""
        if self._scopes:
            self._scopes[-1].nonlocal_names.update(node.names)

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
        parent = self._scopes[-1] if self._scopes else None
        definition = DefinitionReferences(
            name=node.name,
            linenos=_definition_linenos(node),
            is_class=isinstance(node, ast.ClassDef),
            decorators=tuple(_base_text(decorator) for decorator in node.decorator_list),
            parent=parent,
        )
        if parent is not None:
            parent.children_by_name.setdefault(node.name, []).append(definition)
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


@contextmanager
def _recursion_limit(minimum: int) -> Iterator[None]:
    """Temporarily raise the interpreter recursion limit, never lower it.

    :param minimum: Recursion limit floor for the wrapped block.
    :return: Context manager restoring the prior recursion limit on exit.
    """
    current = sys.getrecursionlimit()
    if minimum > current:
        sys.setrecursionlimit(minimum)
    try:
        yield
    finally:
        sys.setrecursionlimit(current)


def _diagnostic(
    file_path: Path, code: str, message: str, lineno: int | None = None
) -> ExtractionDiagnostic:
    """Build a Python-language diagnostic for one unused-analysis failure.

    :param file_path: File the diagnostic refers to.
    :param code: Machine-readable diagnostic code.
    :param message: Human-readable diagnostic text.
    :param lineno: Optional 1-based line number the diagnostic anchors to.
    :return: The constructed diagnostic.
    """
    return ExtractionDiagnostic(
        file_path=file_path, language="python", message=message, code=code, lineno=lineno
    )


def _parse_module(file_path: Path) -> ast.Module | ExtractionDiagnostic:
    """Parse one Python file the way the extractor reads it.

    The extractor skips a BOM and decodes invalid UTF-8 lossily, so the same
    bytes are parsed here. A file ``ast`` cannot parse (a syntax error the
    grammar recovered from, syntax newer than the interpreter, a parser stack
    overflow on deeply nested expressions that tree-sitter copes with) yields
    a diagnostic instead of a module.

    :param file_path: Python source path.
    :return: Parsed module, or a diagnostic describing why parsing failed.
    """
    try:
        raw = file_path.read_bytes()
    except OSError as error:
        return _diagnostic(file_path, "unused-read-error", f"Could not read {file_path}: {error}")
    source = raw.removeprefix(codecs.BOM_UTF8).decode("utf-8", errors="replace")
    try:
        # ValueError covers Python 3.11's embedded-NUL report.
        return ast.parse(source)
    except (SyntaxError, ValueError) as error:
        message = f"{type(error).__name__}: {getattr(error, 'msg', str(error))}"
        return _diagnostic(
            file_path, "unused-parse-error", message, lineno=getattr(error, "lineno", None)
        )
    except (RecursionError, MemoryError) as error:
        # CPython's C parser overflows its own stack on pathological nesting
        # (~5,950 chained ``elif``s) and raises MemoryError, not RecursionError.
        return _diagnostic(file_path, "unused-recursion-limit", f"{type(error).__name__}: {error}")


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
    :return: Module references; carries a diagnostic when the file could not
        be parsed or its syntax tree could not be walked.
    """
    parsed = _parse_module(file_path)
    if isinstance(parsed, ExtractionDiagnostic):
        logger.warning(
            f"Unused analysis collected no references from {file_path}: {parsed.message}"
        )
        return ModuleReferences(diagnostic=parsed)
    tree = parsed
    collector = _ReferenceCollector()
    try:
        with _recursion_limit(_VISIT_RECURSION_LIMIT):
            collector.visit(tree)
    except RecursionError as error:
        # ast.NodeVisitor recurses per node; a chain deeper than even the
        # raised limit overflows it while tree-sitter copes.
        diagnostic = _diagnostic(
            file_path,
            "unused-recursion-limit",
            f"expression nesting exceeds the interpreter recursion limit ({error})",
        )
        logger.warning(
            f"Unused analysis collected no references from {file_path}: {diagnostic.message}"
        )
        return ModuleReferences(diagnostic=diagnostic)
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


def find_pyproject(target: Path) -> Path | None:
    """Find the nearest ``pyproject.toml`` at or above a scan target.

    The walk stops after checking the git work-tree root (if the target is
    inside one), so a ``pyproject.toml`` that happens to sit further up an
    unrelated ancestor directory is not mistaken for the project's own.

    :param target: Scan root or single-file target.
    :return: The nearest ``pyproject.toml``, or ``None`` when none exists at
        or above the target within that boundary.
    """
    directory = (target if target.is_dir() else target.parent).resolve()
    boundary = git_work_tree(directory)
    current = directory
    while True:
        candidate = current / "pyproject.toml"
        if candidate.is_file():
            return candidate
        if boundary is not None and current == boundary:
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _entry_point_module_files(project_dir: Path, module_parts: tuple[str, ...]) -> set[Path]:
    """Resolve an entry-point module to its file under the standard project layouts.

    The src layout (``src/pkg/cli.py``) and the flat layout (``pkg/cli.py``)
    beside ``pyproject.toml`` are checked on disk, whether or not the scan
    covers them.

    :param project_dir: Directory holding ``pyproject.toml``.
    :param module_parts: Dotted module path split into segments.
    :return: Resolved ``<root>/<module>.py`` or ``<root>/<module>/__init__.py``
        files that exist; empty when the module lives under another source root.
    """
    files: set[Path] = set()
    for root in (project_dir / "src", project_dir):
        base = root.joinpath(*module_parts)
        for candidate in (base.parent / f"{base.name}.py", base / "__init__.py"):
            if candidate.is_file():
                files.add(candidate.resolve())
    return files


def _entry_point_targets(pyproject: Path) -> set[tuple[str, str]]:
    """Collect ``(module, object)`` targets from a ``pyproject.toml``.

    Reads ``[project.scripts]``, ``[project.gui-scripts]``, and every
    ``[project.entry-points.*]`` group. A target with no object (a bare
    module, or one missing the ``:`` separator entirely) is skipped rather
    than falling back to crediting anything sharing its last dotted segment.

    :param pyproject: Path to the ``pyproject.toml`` file.
    :return: Set of ``(dotted module path, dotted object path)`` pairs.
    """
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeError):
        return set()

    project_cfg = data.get("project", {})
    if not isinstance(project_cfg, dict):
        return set()

    tables = [project_cfg.get("scripts", {}), project_cfg.get("gui-scripts", {})]
    groups = project_cfg.get("entry-points", {})
    if isinstance(groups, dict):
        tables.extend(groups.values())

    targets: set[tuple[str, str]] = set()
    for table in tables:
        if not isinstance(table, dict):
            continue
        for value in table.values():
            if not isinstance(value, str):
                continue
            # Strip a trailing extras marker (`pkg.mod:obj [extra1,extra2]`).
            value = value.split("[", 1)[0].strip()
            module, sep, obj = value.partition(":")
            module = module.strip()
            obj = obj.strip()
            if sep and module and obj:
                targets.add((module, obj))

    return targets


# Standard-library decorators that wrap or mark a definition without registering
# it anywhere; every other decorator may be a registration.
_WRAPPER_DECORATORS = frozenset(
    {
        "abc.abstractmethod",
        "classmethod",
        "contextlib.asynccontextmanager",
        "contextlib.contextmanager",
        "dataclasses.dataclass",
        "enum.unique",
        "functools.cache",
        "functools.cached_property",
        "functools.lru_cache",
        "functools.singledispatch",
        "functools.singledispatchmethod",
        "functools.total_ordering",
        "functools.wraps",
        "property",
        "staticmethod",
        "typing.final",
        "typing.no_type_check",
        "typing.overload",
        "typing.override",
        "typing.runtime_checkable",
        "typing_extensions.final",
        "typing_extensions.overload",
        "typing_extensions.override",
        "typing_extensions.runtime_checkable",
    }
)
_PROPERTY_ACCESSORS = (".setter", ".getter", ".deleter")


def _is_wrapper_decorator(decorator: str, aliases: dict[str, str]) -> bool:
    """Return whether a decorator is a standard-library wrapper rather than a possible registration.

    :param decorator: Dotted decorator target, call arguments stripped.
    :param aliases: Module alias map used to expand imported names.
    :return: ``True`` for a property accessor or a name resolving to ``_WRAPPER_DECORATORS``.
    """
    if decorator.endswith(_PROPERTY_ACCESSORS):
        return True
    return bool(_resolve_reference_targets(decorator, aliases) & _WRAPPER_DECORATORS)


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


def _lexical_path(definition: DefinitionReferences) -> str:
    """Return a definition's dotted path from its module's top level.

    :param definition: Collected definition.
    :return: Names from the outermost enclosing definition inward (``App.run``).
    """
    names: list[str] = []
    scope: DefinitionReferences | None = definition
    while scope is not None:
        names.append(scope.name)
        scope = scope.parent
    return ".".join(reversed(names))


def build_reference_graph(
    units: list[CodeUnit],
    project_root: Path | None = None,
    source_files: list[Path] | None = None,
) -> list[ExtractionDiagnostic]:
    """Populate ``unit.references`` from every name each Python module loads.

    Matching is by name: a reference to ``helper`` marks every unit named
    ``helper`` (or whose qualified name ends in the referenced dotted path),
    except the referring unit itself.

    :param units: Collected code units; non-Python units are ignored.
    :param project_root: Optional scan root or single-file target; entry
        points are read from the nearest ``pyproject.toml`` at or above it,
        bounded by the git work tree (see :func:`find_pyproject`).
    :param source_files: Every Python file the extractor visited; files without
        units (re-export modules, scripts) still contribute references.
    :return: One diagnostic per file the reference walk could not process.
    """
    units = [unit for unit in units if unit.language == "python"]
    if not units:
        return []

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

    def local_definitions(
        use: ReferenceUse,
        top_level_by_name: dict[str, list[DefinitionReferences]],
        redirected: set[str],
    ) -> list[DefinitionReferences] | None:
        """Resolve a bare name to the module's own definitions that can bind it.

        A class-body load sees the class's own definitions before the
        enclosing scopes; statement order is ignored, so both are candidates.
        Enclosing function scopes are searched innermost first, then the
        module's top level. Class bodies above the load are skipped: their
        names are not visible to the functions nested in them.

        :param use: Bare name load and its originating definition.
        :param top_level_by_name: Module-level definitions indexed by name.
        :param redirected: Names a ``def``/``class`` rebinds under ``global``
            or ``nonlocal``, whose binding scope is not its lexical parent.
        :return: Candidate definitions, or ``None`` when name-based matching
            applies (it only over-credits).
        """
        if use.name in redirected:
            return None
        scope: DefinitionReferences | None = use.origin
        class_bound: list[DefinitionReferences] = []
        if scope.is_class and use.name not in scope.global_names:
            class_bound = scope.children_by_name.get(use.name, [])
            scope = scope.parent
        while scope is not None and use.name not in scope.global_names:
            if not scope.is_class and use.name in scope.children_by_name:
                return class_bound + scope.children_by_name[use.name]
            scope = scope.parent
        module_bound = top_level_by_name.get(use.name)
        # Without a module-level definition, an import or builtin may bind
        # the name after a class's own definitions.
        return None if module_bound is None else class_bound + module_bound

    module_paths = {unit.file_path for unit in units} | set(source_files or ())
    modules = {
        file_path: collect_module_references(file_path) for file_path in sorted(module_paths)
    }
    for file_path, module in modules.items():
        mark(f"__module__::{file_path}", module.module_references, module.aliases)
        top_level_by_name: dict[str, list[DefinitionReferences]] = defaultdict(list)
        for definition in module.definitions:
            if definition.parent is None:
                top_level_by_name[definition.name].append(definition)
        redirected = {
            name
            for definition in module.definitions
            for name in definition.global_names | definition.nonlocal_names
            if name in definition.children_by_name
        }
        extracted_by_definition = {
            id(definition): units_for(file_path, definition) for definition in module.definitions
        }
        for definition in module.definitions:
            # A definition the extractor dropped (a filtered-out private
            # symbol, or one nested in a private container) still has a
            # body that references other units; credit it from a synthetic
            # id so those references aren't lost.
            extracted = extracted_by_definition[id(definition)]
            referrers = [unit.uid for unit in extracted] or [
                f"__definition__::{file_path}::{definition.name}::{definition.linenos[-1]}"
            ]
            for use in definition.uses:
                # A nested definition's reference to itself must not surface
                # as the enclosing unit referencing it.
                excluded = frozenset(unit.uid for unit in extracted_by_definition[id(use.origin)])
                bound = local_definitions(use, top_level_by_name, redirected) if use.bare else None
                for referrer_uid in referrers:
                    if bound is None:
                        mark(referrer_uid, {use.name}, module.aliases, excluded)
                        continue
                    for local in bound:
                        for target in extracted_by_definition[id(local)]:
                            if target.uid not in excluded and target.uid != referrer_uid:
                                target.references.add(referrer_uid)

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

    # Any other decorator receives the function object and may register it
    # (@app.route, @receiver(...), @cli.command()), which no in-project name shows.
    for file_path, module in modules.items():
        for definition in module.definitions:
            registration = next(
                (
                    decorator
                    for decorator in definition.decorators
                    if not _is_wrapper_decorator(decorator, module.aliases)
                ),
                None,
            )
            if registration is not None:
                for unit in units_for(file_path, definition):
                    unit.references.add(f"decorator::{registration}")

    # Seed references from the entry points of the nearest project above the
    # scan target, bounded by the git work tree.
    if project_root is not None:
        pyproject_path = find_pyproject(project_root)
        if pyproject_path is not None:
            for module, obj in _entry_point_targets(pyproject_path):
                module_parts = tuple(module.split("."))
                home_files = _entry_point_module_files(pyproject_path.parent, module_parts)
                for file_path, module_references in modules.items():
                    path_parts = (
                        file_path.parent.parts
                        if file_path.name == "__init__.py"
                        else (*file_path.parent.parts, file_path.stem)
                    )
                    if path_parts[-len(module_parts) :] != module_parts:
                        continue
                    # A module found in the src or flat layout is that one file,
                    # so a same-shaped copy elsewhere (examples/pkg/cli.py) gets
                    # no credit; under any other source root the module path
                    # suffix is all there is to go on.
                    if home_files and file_path.resolve() not in home_files:
                        continue
                    # ``module:object`` names one definition exactly: a nested
                    # ``factory._main`` or ``Outer.App.run`` is not ``_main`` or
                    # ``App.run``.
                    for definition in module_references.definitions:
                        if _lexical_path(definition) == obj:
                            for unit in units_for(file_path, definition):
                                unit.references.add("project.entrypoint")

    return [module.diagnostic for module in modules.values() if module.diagnostic is not None]


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


_ABSTRACT_DECORATOR_RE = re.compile(r"^@\s*(?:abc\.)?abstractmethod\b")


def _is_abstract(unit: CodeUnit) -> bool:
    """Return whether one of the unit's own decorator lines is ``@abstractmethod``.

    :param unit: Candidate unit.
    :return: ``True`` when a decorator line matches ``@abstractmethod`` or ``@abc.abstractmethod`` exactly.
    """
    return any(_ABSTRACT_DECORATOR_RE.match(line) for line in _decorators(unit).splitlines())


def _is_test_file(path: Path) -> bool:
    """Return whether a path is a pytest ``conftest.py`` or matches the default test-file shapes.

    :param path: File path to inspect.
    :return: ``True`` for ``conftest.py``, a ``test_*`` prefix, or a ``_test``/``_tests`` stem suffix.
    """
    stem = path.stem
    return (
        path.name == "conftest.py" or stem.startswith("test_") or stem.endswith(("_test", "_tests"))
    )


def _is_unused_candidate(unit: CodeUnit, strict_unused: bool) -> bool:
    """Apply every unused heuristic except the ``codedupes: ignore`` directive.

    :param unit: Candidate code unit.
    :param strict_unused: Whether to report public functions and public methods of public classes too.
    :return: ``True`` when the unit would be reported absent a suppression directive.
    """
    if unit.language != "python":
        return False
    if not strict_unused and _is_public_surface(unit):
        return False
    if unit.references:
        return False
    if unit.is_likely_api:
        return False
    if unit.name.startswith("get_") or unit.name.startswith("set_"):
        return False
    if _is_abstract(unit):
        return False
    return not (unit.name.startswith(("test_", "pytest_")) or _is_test_file(unit.file_path))


def find_potentially_unused(units: list[CodeUnit], strict_unused: bool = False) -> list[CodeUnit]:
    """Find code units that are never referenced, not likely API, and not suppressed.

    :param units: Candidate code units.
    :param strict_unused: Whether to report public functions and public methods of public classes too.
    :return: Candidates carrying no ``unused`` suppression directive.
    """
    return [
        unit
        for unit in units
        if _is_unused_candidate(unit, strict_unused) and "unused" not in unit.suppressions
    ]


@dataclass
class UnusedReport:
    """Result of one unused-code analysis pass."""

    unused: list[CodeUnit]
    suppressed: int = 0
    diagnostics: list[ExtractionDiagnostic] = field(default_factory=list)


def run_unused_analysis(
    units: list[CodeUnit],
    *,
    project_root: Path | None,
    strict_unused: bool,
    source_files: list[Path] | None = None,
) -> UnusedReport:
    """Build the reference graph and report the units it leaves unreferenced.

    :param units: Collected code units; non-Python units are ignored.
    :param project_root: Project root for pyproject entry-point resolution, or ``None``.
    :param source_files: Every Python file the extractor visited, units or not.
    :param strict_unused: Whether to report public functions and public methods of public classes too.
    :return: Potentially unused units, the count suppressed by directive, and per-file diagnostics.
    """
    diagnostics = build_reference_graph(units, project_root=project_root, source_files=source_files)
    unused = find_potentially_unused(units, strict_unused=strict_unused)
    suppressed = sum(
        1
        for unit in units
        if _is_unused_candidate(unit, strict_unused) and "unused" in unit.suppressions
    )
    logger.info(f"Found {len(unused)} potentially unused code units")
    return UnusedReport(unused=unused, suppressed=suppressed, diagnostics=diagnostics)
