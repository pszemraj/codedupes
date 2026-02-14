"""Traditional (non-ML) duplicate detection methods."""

from __future__ import annotations

import logging
import ast
import keyword
from pathlib import Path
from collections import defaultdict
from itertools import combinations

from .models import CodeUnit, DuplicatePair

logger = logging.getLogger(__name__)


def find_exact_ast_duplicates(units: list[CodeUnit]) -> list[DuplicatePair]:
    """
    Find exact structural duplicates via normalized AST hash.
    These are functionally identical regardless of variable naming.
    """
    return _find_exact_duplicates(units, "_ast_hash", "ast_hash")


def find_exact_token_duplicates(units: list[CodeUnit]) -> list[DuplicatePair]:
    """
    Find duplicates via token hash.
    Catches reformatted code that has identical token sequence.
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
    import ast

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
    """
    Find near-duplicates via Jaccard similarity on identifiers.
    Faster than embeddings but less semantic.
    """
    # Pre-compute identifier sets
    identifier_sets = {unit.uid: extract_identifiers(unit.source) for unit in units}

    duplicates = []
    for i, a in enumerate(units):
        for b in units[i + 1 :]:
            # Skip if same file and overlapping lines (parent/child)
            if a.file_path == b.file_path:
                if not (a.end_lineno < b.lineno or b.end_lineno < a.lineno):
                    continue

            set_a = identifier_sets[a.uid]
            set_b = identifier_sets[b.uid]

            # Skip tiny/noisy comparisons quickly
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


def build_reference_graph(units: list[CodeUnit]) -> None:
    """
    Populate the `references` field on each unit based on call graph analysis.
    Modifies units in place.
    """
    # Build lookup by name
    by_name: dict[str, list[CodeUnit]] = defaultdict(list)
    for unit in units:
        by_name[unit.name].append(unit)
        # Also index by qualified name parts
        parts = unit.qualified_name.split(".")
        for i in range(len(parts)):
            by_name[".".join(parts[i:])].append(unit)

    alias_map_by_file: dict[Path, dict[str, str]] = {}
    for unit in units:
        if unit.file_path in alias_map_by_file:
            continue
        alias_map_by_file[unit.file_path] = _extract_aliases(unit.file_path)

    # Populate references
    for unit in units:
        file_aliases = alias_map_by_file.get(unit.file_path, {})
        for call in unit.calls:
            candidates = {call}
            if "." in call:
                head, _, tail = call.partition(".")
                if head in file_aliases:
                    candidates.add(f"{file_aliases[head]}.{tail}")
            elif call in file_aliases:
                candidates.add(file_aliases[call])

            for candidate in candidates:
                for target in by_name.get(candidate, []):
                    if target.uid != unit.uid:  # Don't self-reference
                        target.references.add(unit.uid)


def _extract_aliases(file_path: Path) -> dict[str, str]:
    """Extract a conservative alias map from module-level import/assignment statements."""
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
                if base:
                    aliases[asname] = f"{base}.{imported}"
                else:
                    aliases[asname] = imported
        elif isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
                value = node.value
                if isinstance(value, ast.Name):
                    aliases[target] = value.id
                elif isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                    aliases[target] = f"{value.value.id}.{value.attr}"
    return aliases


def find_potentially_unused(units: list[CodeUnit]) -> list[CodeUnit]:
    """
    Find code units that are never referenced and are not likely API.

    Be conservative: only flag things that are clearly internal.
    """
    unused = []
    for unit in units:
        # Skip if it has references
        if unit.references:
            continue

        # Skip if it's likely part of public API
        if unit.is_likely_api:
            continue

        # Skip __init__ methods (always "called" implicitly)
        if unit.name == "__init__":
            continue

        # Skip if it's a property getter/setter (often not "called" explicitly)
        # This is a heuristic - could be improved with decorator analysis
        if unit.name.startswith("get_") or unit.name.startswith("set_"):
            continue

        # Skip abstract methods (meant to be overridden)
        if "@abstractmethod" in unit.source or "@abc.abstractmethod" in unit.source:
            continue

        # Skip if it's a test
        if unit.name.startswith("test_") or "_test" in unit.file_path.name:
            continue

        unused.append(unit)

    return unused


def run_traditional_analysis(
    units: list[CodeUnit],
    jaccard_threshold: float = 0.85,
    compute_unused: bool = True,
) -> tuple[list[DuplicatePair], list[DuplicatePair], list[CodeUnit]]:
    """
    Run all traditional duplicate detection methods.

    Returns:
        (exact_duplicates, near_duplicates, potentially_unused)
        potentially_unused is empty when compute_unused=False
    """
    logger.info(f"Running traditional analysis on {len(units)} code units")

    # Build reference graph first if we need unused references
    if compute_unused:
        build_reference_graph(units)

    # Find exact duplicates
    ast_dupes = find_exact_ast_duplicates(units)
    token_dupes = find_exact_token_duplicates(units)
    exact = _dedupe_duplicate_pairs(ast_dupes + token_dupes)
    logger.info(f"Found {len(exact)} exact duplicates")

    # Find near duplicates
    near = find_near_duplicates_jaccard(units, threshold=jaccard_threshold)
    # Filter out any that were already found as exact
    exact_pairs = {(d.unit_a.uid, d.unit_b.uid) for d in exact}
    exact_pairs |= {(d.unit_b.uid, d.unit_a.uid) for d in exact}
    near = [d for d in near if (d.unit_a.uid, d.unit_b.uid) not in exact_pairs]
    logger.info(f"Found {len(near)} near duplicates (Jaccard)")

    # Find unused
    unused = find_potentially_unused(units) if compute_unused else []
    logger.info(f"Found {len(unused)} potentially unused code units")

    return exact, _dedupe_duplicate_pairs(near), unused
