"""Traditional (non-ML) duplicate detection methods."""

from __future__ import annotations

import ast
import keyword
import logging
import tomllib
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from codedupes.models import CodeUnit, CodeUnitType, DuplicatePair

logger = logging.getLogger(__name__)


def find_exact_ast_duplicates(units: list[CodeUnit]) -> list[DuplicatePair]:
    """Find exact structural duplicates via normalized AST hash."""
    return _find_exact_duplicates(units, "_ast_hash", "ast_hash")


def find_exact_token_duplicates(units: list[CodeUnit]) -> list[DuplicatePair]:
    """Find duplicates via token hash.

    Catches reformatted code with identical token sequence.
    """
    return _find_exact_duplicates(units, "_token_hash", "token_hash")


def _find_exact_duplicates(
    units: list[CodeUnit], hash_attr: str, method: str
) -> list[DuplicatePair]:
    """Find duplicate pairs by grouping units by a stored hash attribute."""
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


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def extract_identifiers(source: str) -> set[str]:
    """Extract all identifiers from source code."""
    identifiers = set()
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.FunctionDef):
                identifiers.add(node.name)
            elif isinstance(node, ast.ClassDef):
                identifiers.add(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                identifiers.add(node.name)
            elif isinstance(node, ast.arg):
                identifiers.add(node.arg)
    except SyntaxError:
        pass
    return _normalize_identifiers(identifiers)


def _normalize_identifiers(identifiers: set[str]) -> set[str]:
    """Normalize identifier sets for stable near-duplicate matching."""
    ignored = set(keyword.kwlist) | set(dir(__builtins__))
    normalized = set()
    for ident in identifiers:
        if not ident:
            continue
        if ident in ignored:
            continue
        if ident.isdigit():
            continue
        normalized.add(ident)
    return normalized


def find_near_duplicates_jaccard(
    units: list[CodeUnit],
    threshold: float = 0.8,
) -> list[DuplicatePair]:
    """Find near-duplicates via Jaccard similarity on identifiers."""
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
    """Deduplicate unordered duplicate pairs."""
    seen = set()
    deduped: list[DuplicatePair] = []
    for dup in duplicates:
        key = tuple(sorted((dup.unit_a.uid, dup.unit_b.uid)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dup)
    return deduped


def _resolve_call_targets(call: str, aliases: dict[str, str]) -> set[str]:
    """Resolve direct and alias-mapped call targets."""
    candidates = {call}
    if call in aliases:
        candidates.add(aliases[call])
    if "." in call:
        head, _, tail = call.partition(".")
        if head in aliases:
            candidates.add(f"{aliases[head]}.{tail}")
    return candidates


def _extract_main_block_calls(file_path: Path) -> set[str]:
    """Extract function names called from an if-`__main__` block."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError):
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
                ):
                    is_main = True
                elif (
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
    """Collect callable targets from `[project.scripts]` and `[project.gui-scripts]`."""
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
    """Populate references from direct calls, entrypoints, and `__main__` blocks."""
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
    for unit in units:
        caller_uid = f"__main__::{unit.file_path}"
        for call in _extract_main_block_calls(unit.file_path):
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
    """Extract a conservative alias map from module-level imports and assignments."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError):
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
        elif isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
                value = node.value
                if isinstance(value, ast.Name):
                    aliases[target] = value.id
                elif isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                    aliases[target] = f"{value.value.id}.{value.attr}"
    return aliases


def find_potentially_unused(units: list[CodeUnit], strict_unused: bool = False) -> list[CodeUnit]:
    """Find code units that are never referenced and are not likely API."""
    unused = []
    for unit in units:
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


def run_traditional_analysis(
    units: list[CodeUnit],
    jaccard_threshold: float = 0.85,
    compute_unused: bool = True,
    strict_unused: bool = False,
    project_root: Path | None = None,
) -> tuple[list[DuplicatePair], list[DuplicatePair], list[CodeUnit]]:
    """Run all traditional duplicate detection methods."""
    logger.info(f"Running traditional analysis on {len(units)} code units")

    if compute_unused:
        build_reference_graph(units, project_root=project_root)

    ast_dupes = find_exact_ast_duplicates(units)
    token_dupes = find_exact_token_duplicates(units)
    exact = _dedupe_duplicate_pairs(ast_dupes + token_dupes)
    logger.info(f"Found {len(exact)} exact duplicates")

    near = find_near_duplicates_jaccard(units, threshold=jaccard_threshold)
    exact_pairs = {(d.unit_a.uid, d.unit_b.uid) for d in exact}
    exact_pairs |= {(d.unit_b.uid, d.unit_a.uid) for d in exact}
    near = [d for d in near if (d.unit_a.uid, d.unit_b.uid) not in exact_pairs]
    logger.info(f"Found {len(near)} near duplicates (Jaccard)")

    unused = find_potentially_unused(units, strict_unused=strict_unused) if compute_unused else []
    logger.info(f"Found {len(unused)} potentially unused code units")

    return exact, _dedupe_duplicate_pairs(near), unused
