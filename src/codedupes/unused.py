"""Potentially-unused code detection (Python-only reference graph)."""

from __future__ import annotations

import ast
import logging
import tomllib
from collections import defaultdict
from pathlib import Path

from codedupes.models import CodeUnit, CodeUnitType

logger = logging.getLogger(__name__)


def _resolve_call_targets(call: str, aliases: dict[str, str]) -> set[str]:
    """Resolve direct and alias-mapped call targets.

    :param call: Raw call expression string.
    :param aliases: Alias map from local symbols to full targets.
    :return: Candidate call target names.
    """
    candidates = {call}
    if call in aliases:
        candidates.add(aliases[call])
    if "." in call:
        head, _, tail = call.partition(".")
        if head in aliases:
            candidates.add(f"{aliases[head]}.{tail}")
    return candidates


def _extract_main_block_calls(file_path: Path) -> set[str]:
    """Extract function names called from an if-``__main__`` block.

    :param file_path: Path to inspect.
    :return: Function names called from the module entry block.
    """
    try:
        # utf-8-sig matches the BOM-tolerant extractor read: a file that
        # extraction accepts must not silently lose its __main__ references.
        # ValueError covers CPython 3.11's embedded-NUL report.
        source = file_path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return set()

    from codedupes.extractor import CallGraphVisitor

    calls: set[str] = set()
    visitor = CallGraphVisitor()

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

    calls.update(visitor.calls)
    return calls


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
            target = value.split(":", 1)[-1]
            if "." in target:
                target = target.rsplit(".", 1)[-1]
            target = target.strip()
            if target:
                targets.add(target)

    return targets


def build_reference_graph(units: list[CodeUnit], project_root: Path | None = None) -> None:
    """Populate references from direct calls, entrypoints, and ``__main__`` blocks.

    :param units: Collected code units.
    :param project_root: Optional root for entry point resolution.
    :return: ``None``.
    """
    units = [unit for unit in units if unit.language == "python"]
    if not units:
        return

    by_name: dict[str, list[CodeUnit]] = defaultdict(list)
    for unit in units:
        by_name[unit.name].append(unit)
        parts = unit.qualified_name.split(".")
        for i in range(len(parts)):
            by_name[".".join(parts[i:])].append(unit)

    alias_map_by_file: dict[Path, dict[str, str]] = {}
    for unit in units:
        if unit.file_path not in alias_map_by_file:
            alias_map_by_file[unit.file_path] = _extract_aliases(unit.file_path)

    # Populate references from call graph.
    for unit in units:
        file_aliases = alias_map_by_file.get(unit.file_path, {})
        for call in unit.calls:
            for target in _resolve_call_targets(call, file_aliases):
                for candidate in by_name.get(target, []):
                    if candidate.uid != unit.uid:
                        candidate.references.add(unit.uid)

    # Seed references from __main__ blocks.
    main_block_calls_by_file: dict[Path, set[str]] = {}
    for file_path in alias_map_by_file:
        main_block_calls_by_file[file_path] = _extract_main_block_calls(file_path)

    for unit in units:
        caller_uid = f"__main__::{unit.file_path}"
        for call in main_block_calls_by_file.get(unit.file_path, set()):
            for target in _resolve_call_targets(call, alias_map_by_file.get(unit.file_path, {})):
                for candidate in by_name.get(target, []):
                    candidate.references.add(caller_uid)

    # Seed references from project entry points.
    if project_root is not None:
        root = project_root if project_root.is_dir() else project_root.parent
        for target in _extract_pyproject_entry_points(root):
            for candidate in by_name.get(target, []):
                candidate.references.add("project.entrypoint")


def _extract_aliases(file_path: Path) -> dict[str, str]:
    """Extract a conservative alias map from module-level imports and assignments.

    :param file_path: Python source path.
    :return: Alias map for name resolution.
    """
    try:
        # utf-8-sig matches the BOM-tolerant extractor read: a file that
        # extraction accepts must not silently lose its alias map.
        # ValueError covers CPython 3.11's embedded-NUL report.
        source = file_path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return {}

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


def find_potentially_unused(units: list[CodeUnit], strict_unused: bool = False) -> list[CodeUnit]:
    """Find code units that are never referenced and are not likely API.

    :param units: Candidate code units.
    :param strict_unused: Whether to include likely public functions in results.
    :return: Units with no references and not classified as API.
    """
    unused = []
    for unit in units:
        if unit.language != "python":
            continue
        if not strict_unused and unit.unit_type == CodeUnitType.FUNCTION and unit.is_public:
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
        if "@abstractmethod" in unit.source or "@abc.abstractmethod" in unit.source:
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
) -> list[CodeUnit]:
    """Build the reference graph and report the units it leaves unreferenced.

    :param units: Collected code units; non-Python units are ignored.
    :param project_root: Project root for pyproject entry-point resolution, or ``None``.
    :param strict_unused: Whether to keep public functions in the results.
    :return: Potentially unused Python units.
    """
    build_reference_graph(units, project_root=project_root)
    unused = find_potentially_unused(units, strict_unused=strict_unused)
    logger.info(f"Found {len(unused)} potentially unused code units")
    return unused
