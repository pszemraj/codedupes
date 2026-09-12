"""Traditional (non-ML) duplicate detection methods."""

from __future__ import annotations

import ast
import builtins
import keyword
import logging
import math
from collections import Counter, defaultdict
from itertools import combinations

from codedupes.models import CodeUnit, CodeUnitType, DuplicatePair
from codedupes.pairs import ordered_pair_key

logger = logging.getLogger(__name__)

# dir(builtins) rather than dir(__builtins__): the latter is a plain dict
# inside imported modules, so it would filter dict methods instead of
# builtin names.
_IGNORED_IDENTIFIERS = frozenset(keyword.kwlist) | frozenset(dir(builtins))


def _block_kind(unit_type: CodeUnitType) -> str:
    """Map a unit type to its comparison-blocking kind.

    Functions and methods share a kind so that code moved between module level
    and a class body stays comparable, matching semantic pairing.

    :param unit_type: Unit type to classify.
    :return: Blocking-kind label.
    """
    if unit_type in (CodeUnitType.FUNCTION, CodeUnitType.METHOD):
        return "callable"
    return unit_type.name.lower()


def _find_exact_duplicates(
    units: list[CodeUnit], hash_attr: str, method: str
) -> list[DuplicatePair]:
    """Find duplicate pairs by grouping units by a stored hash attribute.

    :param units: Candidate units to compare.
    :param hash_attr: Unit attribute name containing a hash.
    :param method: Duplicate classification label.
    :return: Exact duplicate pairs for the selected hash field.
    """
    by_hash: dict[tuple[str, str, str], list[CodeUnit]] = defaultdict(list)

    for unit in units:
        value = getattr(unit, hash_attr, None)
        if value:
            # Exact structural/token equality is intentionally same-language and
            # same-blocking-kind. A C function and Rust function cannot become an
            # "exact duplicate" merely because their canonical token streams
            # align, but a function copied into a class as a method can.
            by_hash[(unit.language, _block_kind(unit.unit_type), value)].append(unit)

    duplicates = []
    for group in by_hash.values():
        if len(group) <= 1:
            continue
        for a, b in combinations(group, 2):
            if a.overlaps(b):
                continue
            duplicates.append(DuplicatePair(unit_a=a, unit_b=b, similarity=1.0, method=method))

    return duplicates


def find_exact_pair_keys(units: list[CodeUnit]) -> set[tuple[str, str]]:
    """Return ordered uid pair keys for every exact-duplicate pair.

    Uses the same predicate as :func:`run_traditional_analysis` exact detection:
    two same-language units of the same blocking kind are exact duplicates when
    they share a structural or token fingerprint.

    :param units: Candidate units to compare.
    :return: Ordered uid pair keys covering all exact-duplicate pairs.
    """
    pairs = _find_exact_duplicates(units, "structural_hash", "ast_hash") + _find_exact_duplicates(
        units, "token_hash", "token_hash"
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
    return intersection / union if union > 0 else 0.0


def collect_identifiers(node: ast.AST) -> set[str]:
    """Collect normalized identifier names bound or referenced under one AST subtree.

    :param node: AST subtree to scan.
    :return: Identifier names excluding Python keywords, builtins, and digits.
    """
    identifiers = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            identifiers.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            identifiers.add(child.name)
        elif isinstance(child, ast.arg):
            identifiers.add(child.arg)
    return _normalize_identifiers(identifiers)


def extract_identifiers(source: str) -> set[str]:
    """Extract all identifiers from source code.

    :param source: Source text.
    :return: Identifier names found in the AST.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return collect_identifiers(tree)


def unit_identifier_set(unit: CodeUnit) -> set[str]:
    """Return backend identifiers without reparsing non-Python source as Python.

    :param unit: Code unit whose identifiers are needed.
    :return: Identifier names for the unit.
    """
    if unit.identifiers or unit.language != "python":
        return set(unit.identifiers)
    return extract_identifiers(unit.source)


def _normalize_identifiers(identifiers: set[str]) -> set[str]:
    """Normalize identifier sets for stable near-duplicate matching.

    :param identifiers: Raw identifier names.
    :return: Normalized filtered identifiers.
    """
    normalized = set()
    for ident in identifiers:
        if not ident:
            continue
        if ident in _IGNORED_IDENTIFIERS:
            continue
        if ident.isdigit():
            continue
        normalized.add(ident)
    return normalized


# The prefix bound and the exact score must agree at the cutoff. Without a
# tolerance a float product landing a hair above an integer would shorten a
# prefix by one token and drop a pair that verification would have accepted.
_PREFIX_LENGTH_TOLERANCE = 1e-9


def _block_jaccard_pairs(
    group: list[CodeUnit],
    identifier_sets: dict[str, set[str]],
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Find every above-threshold Jaccard pair in one block via prefix-filtered candidates.

    Exact, not approximate: ``J(A, B) >= t`` forces ``|A & B| >= ceil(t * |X|)``
    for either side, and two sets sorted in one common token order that share
    that many elements must already share a token inside their
    ``len - ceil(t * len) + 1`` prefixes, so no verifiable pair goes unproposed.

    :param group: Units of one language/kind block, in report order.
    :param identifier_sets: Identifier sets keyed by unit uid.
    :param threshold: Jaccard cutoff.
    :return: ``(index_a, index_b, similarity)`` triples sorted by index pair.
    """
    sets = [identifier_sets[unit.uid] for unit in group]
    populated = [index for index, tokens in enumerate(sets) if tokens]

    if threshold <= 0.0:
        # At or below zero, disjoint sets clear the cutoff too, and prefix
        # filtering only ever proposes pairs that share a token.
        return [
            (index_a, index_b, jaccard_similarity(sets[index_a], sets[index_b]))
            for index_a, index_b in combinations(populated, 2)
            if not group[index_a].overlaps(group[index_b])
        ]

    # Rarest tokens first, so prefixes are the most selective probes available.
    # Only the order being common to the whole block matters for exactness.
    frequencies: Counter[str] = Counter()
    for index in populated:
        frequencies.update(sets[index])
    vocabulary = sorted(frequencies, key=lambda token: (frequencies[token], token))
    ranks = {token: rank for rank, token in enumerate(vocabulary)}

    postings: dict[str, list[int]] = {}
    pairs: list[tuple[int, int, float]] = []
    for index_b in populated:
        set_b = sets[index_b]
        size_b = len(set_b)
        ordered = sorted(set_b, key=ranks.__getitem__)
        prefix_length = size_b - math.ceil(threshold * size_b - _PREFIX_LENGTH_TOLERANCE) + 1
        prefix = ordered[: max(prefix_length, 0)]

        candidates: set[int] = set()
        for token in prefix:
            candidates.update(postings.get(token, ()))

        for index_a in candidates:
            set_a = sets[index_a]
            size_a = len(set_a)
            # J is bounded by the size ratio; expressing the bound as the same
            # division the exact score uses keeps float rounding from pruning a
            # pair that verification would accept.
            if min(size_a, size_b) / max(size_a, size_b) < threshold:
                continue
            if group[index_a].overlaps(group[index_b]):
                continue
            similarity = jaccard_similarity(set_a, set_b)
            if similarity >= threshold:
                pairs.append((index_a, index_b, similarity))

        for token in prefix:
            postings.setdefault(token, []).append(index_b)

    pairs.sort(key=lambda pair: pair[:2])
    return pairs


def find_near_duplicates_jaccard(
    units: list[CodeUnit],
    threshold: float = 0.8,
) -> list[DuplicatePair]:
    """Find near-duplicates via Jaccard similarity on identifiers.

    :param units: Candidate units.
    :param threshold: Jaccard cutoff.
    :return: Near-duplicate pairs above threshold.
    """
    identifier_sets = {unit.uid: unit_identifier_set(unit) for unit in units}

    # Candidate blocking removes meaningless mixed-language/kind comparisons
    # before the similarity join; functions and methods share a block.
    groups: dict[tuple[str, str], list[CodeUnit]] = defaultdict(list)
    for unit in units:
        groups[(unit.language, _block_kind(unit.unit_type))].append(unit)

    duplicates = []
    for group in groups.values():
        for index_a, index_b, similarity in _block_jaccard_pairs(group, identifier_sets, threshold):
            duplicates.append(
                DuplicatePair(
                    unit_a=group[index_a],
                    unit_b=group[index_b],
                    similarity=similarity,
                    method="jaccard",
                )
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


def run_traditional_analysis(
    units: list[CodeUnit],
    jaccard_threshold: float = 0.85,
) -> tuple[list[DuplicatePair], list[DuplicatePair]]:
    """Run all traditional duplicate detection methods.

    :param units: Candidate code units.
    :param jaccard_threshold: Similarity threshold for near-duplicate detection.
    :return: Exact duplicates and near duplicates.
    """
    logger.info(f"Running traditional analysis on {len(units)} code units")

    ast_dupes = _find_exact_duplicates(units, "structural_hash", "ast_hash")
    token_dupes = _find_exact_duplicates(units, "token_hash", "token_hash")
    exact = _dedupe_duplicate_pairs(ast_dupes + token_dupes)
    logger.debug(f"Found {len(exact)} exact duplicates before caller filtering")

    near = find_near_duplicates_jaccard(units, threshold=jaccard_threshold)
    exact_pairs = {ordered_pair_key(d.unit_a, d.unit_b) for d in exact}
    near = [d for d in near if ordered_pair_key(d.unit_a, d.unit_b) not in exact_pairs]
    logger.debug(f"Found {len(near)} near duplicates before caller filtering (Jaccard)")

    return exact, _dedupe_duplicate_pairs(near)
