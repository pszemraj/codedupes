"""Traditional (non-ML) duplicate detection methods."""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations

from .models import CodeUnit, DuplicatePair

logger = logging.getLogger(__name__)


def find_exact_ast_duplicates(units: list[CodeUnit]) -> list[DuplicatePair]:
    """
    Find exact structural duplicates via normalized AST hash.
    These are functionally identical regardless of variable naming.
    """
    by_hash: dict[str, list[CodeUnit]] = defaultdict(list)

    for unit in units:
        if unit._ast_hash:
            by_hash[unit._ast_hash].append(unit)

    duplicates = []
    for hash_val, group in by_hash.items():
        if len(group) > 1:
            for a, b in combinations(group, 2):
                duplicates.append(
                    DuplicatePair(unit_a=a, unit_b=b, similarity=1.0, method="ast_hash")
                )

    return duplicates


def find_exact_token_duplicates(units: list[CodeUnit]) -> list[DuplicatePair]:
    """
    Find duplicates via token hash.
    Catches reformatted code that has identical token sequence.
    """
    by_hash: dict[str, list[CodeUnit]] = defaultdict(list)

    for unit in units:
        if unit._token_hash:
            by_hash[unit._token_hash].append(unit)

    duplicates = []
    for hash_val, group in by_hash.items():
        if len(group) > 1:
            for a, b in combinations(group, 2):
                # Don't double-count if already found by AST
                if a._ast_hash and a._ast_hash == b._ast_hash:
                    continue
                duplicates.append(
                    DuplicatePair(unit_a=a, unit_b=b, similarity=1.0, method="token_hash")
                )

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
    return identifiers


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

            sim = jaccard_similarity(identifier_sets[a.uid], identifier_sets[b.uid])
            if sim >= threshold:
                duplicates.append(
                    DuplicatePair(unit_a=a, unit_b=b, similarity=sim, method="jaccard")
                )

    return duplicates


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

    # Populate references
    for unit in units:
        for call in unit.calls:
            for target in by_name.get(call, []):
                if target.uid != unit.uid:  # Don't self-reference
                    target.references.add(unit.uid)


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
) -> tuple[list[DuplicatePair], list[DuplicatePair], list[CodeUnit]]:
    """
    Run all traditional duplicate detection methods.

    Returns:
        (exact_duplicates, near_duplicates, potentially_unused)
    """
    logger.info(f"Running traditional analysis on {len(units)} code units")

    # Build reference graph first
    build_reference_graph(units)

    # Find exact duplicates
    ast_dupes = find_exact_ast_duplicates(units)
    token_dupes = find_exact_token_duplicates(units)
    exact = ast_dupes + token_dupes
    logger.info(f"Found {len(exact)} exact duplicates")

    # Find near duplicates
    near = find_near_duplicates_jaccard(units, threshold=jaccard_threshold)
    # Filter out any that were already found as exact
    exact_pairs = {(d.unit_a.uid, d.unit_b.uid) for d in exact}
    exact_pairs |= {(d.unit_b.uid, d.unit_a.uid) for d in exact}
    near = [d for d in near if (d.unit_a.uid, d.unit_b.uid) not in exact_pairs]
    logger.info(f"Found {len(near)} near duplicates (Jaccard)")

    # Find unused
    unused = find_potentially_unused(units)
    logger.info(f"Found {len(unused)} potentially unused code units")

    return exact, near, unused
