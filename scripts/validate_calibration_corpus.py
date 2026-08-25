"""Validate calibration-corpus contracts without loading any embedding model.

Checks that every label resolves, that each labeled category exhibits the hash
relations that define it (exact, reformat, doc_variant, renamed, near_*), that
negative controls share no fingerprint, and that corpus files obey the
authoring constraints (no test-glob filenames, unique unit names per file,
zero parse-error diagnostics). Runs extraction only, so it is cheap enough for
corpus authors to iterate against.
"""

from __future__ import annotations

import argparse
import json
import sys
from fnmatch import fnmatch
from itertools import combinations
from pathlib import Path
from typing import Any

from codedupes.analyzer import DEFAULT_SEMANTIC_UNIT_TYPES, AnalyzerConfig, CodeAnalyzer
from codedupes.constants import DEFAULT_MIN_SEMANTIC_STATEMENTS
from codedupes.extractor import DEFAULT_EXCLUDE_PATTERNS
from codedupes.models import CodeUnit, DuplicatePair, ExtractionDiagnostic
from codedupes.pairs import ordered_pair_key
from codedupes.semantic import get_code_unit_statement_count

try:
    from .sweep_common import build_positive_pairs, corpus_files, resolve_label_unit
except ImportError:
    from sweep_common import build_positive_pairs, corpus_files, resolve_label_unit

# Category names and the hash relations that define them. ``near_*`` categories
# must differ in BOTH hashes so the pair can only be caught semantically.
CATEGORY_NAMES = (
    "exact",
    "reformat",
    "doc_variant",
    "renamed",
    "near_rename",
    "near_translation",
    "near_restructure",
)
NEAR_CATEGORIES = ("near_rename", "near_translation", "near_restructure")
PARSE_FAILURE_DIAGNOSTIC_CODES = frozenset({"parse-error", "partial-parse", "unit-parse-error"})


def _group_key(group: list[str]) -> tuple[str, ...]:
    """Build an order-independent identity for one label group.

    :param list[str] group: Label specs forming one clone group.
    :return tuple[str, ...]: Sorted spec tuple usable as a set member.
    """
    return tuple(sorted(group))


def _pair_text(unit_a: CodeUnit, unit_b: CodeUnit) -> str:
    """Format a unit pair for failure messages.

    :param CodeUnit unit_a: First unit.
    :param CodeUnit unit_b: Second unit.
    :return str: Human-readable pair description.
    """
    return f"{unit_a.file_path.name}::{unit_a.name} <-> {unit_b.file_path.name}::{unit_b.name}"


def _check_category_pair(category: str, unit_a: CodeUnit, unit_b: CodeUnit) -> list[str]:
    """Verify one labeled pair exhibits its category's defining hash relations.

    :param str category: Category name from ``CATEGORY_NAMES``.
    :param CodeUnit unit_a: First unit of the labeled pair.
    :param CodeUnit unit_b: Second unit of the labeled pair.
    :return list[str]: Failure messages, empty when the pair is consistent.
    """
    failures: list[str] = []
    pair = _pair_text(unit_a, unit_b)
    structural_equal = (
        unit_a.structural_hash is not None and unit_a.structural_hash == unit_b.structural_hash
    )
    token_equal = unit_a.token_hash is not None and unit_a.token_hash == unit_b.token_hash
    source_equal = unit_a.source == unit_b.source

    if category == "exact":
        if not source_equal:
            failures.append(f"exact pair has differing source text: {pair}")
        if not (structural_equal and token_equal):
            failures.append(f"exact pair does not share both hashes: {pair}")
    elif category == "reformat":
        if source_equal:
            failures.append(f"reformat pair is byte-identical (belongs in exact): {pair}")
        if not token_equal:
            failures.append(f"reformat pair must be token-hash equal: {pair}")
        if not structural_equal:
            failures.append(f"reformat pair must be structural-hash equal: {pair}")
    elif category == "doc_variant":
        if source_equal:
            failures.append(f"doc_variant pair is byte-identical (docs not in span?): {pair}")
        if not structural_equal:
            failures.append(f"doc_variant pair must be structural-hash equal: {pair}")
        # Python docstrings are string tokens; other languages prune comments
        # from the token stream entirely.
        if unit_a.language == "python":
            if token_equal:
                failures.append(f"python doc_variant pair unexpectedly token-equal: {pair}")
        elif not token_equal:
            failures.append(f"{unit_a.language} doc_variant pair must be token-equal: {pair}")
    elif category == "renamed":
        if not structural_equal:
            failures.append(f"renamed pair must be structural-hash equal: {pair}")
        if token_equal:
            failures.append(f"renamed pair is token-equal (rename changed nothing?): {pair}")
    elif category in NEAR_CATEGORIES:
        if structural_equal:
            failures.append(f"{category} pair is structural-hash equal (deterministic): {pair}")
        if token_equal:
            failures.append(f"{category} pair is token-hash equal (deterministic): {pair}")
    else:
        failures.append(f"unknown category {category!r} for pair {pair}")
    return failures


def _validate_labels(
    units: list[CodeUnit],
    labels: dict[str, Any],
    failures: list[str],
) -> dict[str, int]:
    """Validate label groups, category partition, and negative controls.

    :param list[CodeUnit] units: Extracted corpus units.
    :param dict[str, Any] labels: Loaded labels JSON.
    :param list[str] failures: Mutable failure sink.
    :return dict[str, int]: Pair counts per category for the summary.
    """
    positive_groups = labels.get("positive_groups", [])
    if not positive_groups:
        failures.append("labels JSON has no positive_groups")
        return {}

    resolved_groups: dict[tuple[str, ...], list[CodeUnit]] = {}
    for group in positive_groups:
        try:
            resolved_groups[_group_key(group)] = [resolve_label_unit(units, spec) for spec in group]
        except ValueError as exc:
            failures.append(str(exc))

    categories = labels.get("categories")
    counts: dict[str, int] = {}
    if categories is None:
        failures.append("labels JSON has no 'categories' map partitioning positive_groups")
        return counts

    seen_keys: set[tuple[str, ...]] = set()
    for category, groups in categories.items():
        if category not in CATEGORY_NAMES:
            failures.append(f"unknown category name {category!r}")
            continue
        # An empty list is a labeling mistake, not a vacuously covered category:
        # left in place it crashes the sweep's per-category recall loop only
        # after the full corpus embed.
        if not groups:
            failures.append(f"category {category!r} lists no positive groups")
            continue
        for group in groups:
            key = _group_key(group)
            if key in seen_keys:
                failures.append(f"group appears in more than one category: {group}")
            seen_keys.add(key)
            resolved = resolved_groups.get(key)
            if resolved is None:
                failures.append(
                    f"category {category!r} group missing from positive_groups: {group}"
                )
                continue
            for unit_a, unit_b in combinations(resolved, 2):
                counts[category] = counts.get(category, 0) + 1
                failures.extend(_check_category_pair(category, unit_a, unit_b))

    uncategorized = set(resolved_groups) - seen_keys
    for key in sorted(uncategorized):
        failures.append(f"positive group has no category: {list(key)}")

    for group in labels.get("negative_controls", []):
        try:
            resolved = [resolve_label_unit(units, spec) for spec in group]
        except ValueError as exc:
            failures.append(str(exc))
            continue
        for unit_a, unit_b in combinations(resolved, 2):
            pair = _pair_text(unit_a, unit_b)
            if (
                unit_a.structural_hash is not None
                and unit_a.structural_hash == unit_b.structural_hash
            ):
                failures.append(f"negative control shares structural hash: {pair}")
            if unit_a.token_hash is not None and unit_a.token_hash == unit_b.token_hash:
                failures.append(f"negative control shares token hash: {pair}")
    return counts


def _validate_deterministic_coverage(
    traditional_duplicates: list[DuplicatePair],
    positive_pairs: set[tuple[str, str]],
    failures: list[str],
) -> None:
    """Require every deterministic duplicate the traditional tier finds to be labeled.

    The clone categories are meant to partition the corpus's deterministic
    relations, so an exact/near pair that no label claims is an unmeasured
    decision: it silently becomes a false positive in every sweep.

    :param list[DuplicatePair] traditional_duplicates: Exact and near pairs found deterministically.
    :param set[tuple[str, str]] positive_pairs: Ordered uid keys of all labeled positive pairs.
    :param list[str] failures: Mutable failure sink.
    :return None: ``None``.
    """
    for duplicate in traditional_duplicates:
        if ordered_pair_key(duplicate.unit_a, duplicate.unit_b) not in positive_pairs:
            failures.append(
                f"unlabeled deterministic pair ({duplicate.method}): "
                f"{_pair_text(duplicate.unit_a, duplicate.unit_b)}"
            )


def _validate_probes(
    units: list[CodeUnit],
    probes: list[dict[str, Any]],
    failures: list[str],
    *,
    min_statements: int,
) -> tuple[int, int]:
    """Validate search probes resolve to semantic-eligible units.

    :param list[CodeUnit] units: Extracted corpus units.
    :param list[dict[str, Any]] probes: Loaded probe list.
    :param list[str] failures: Mutable failure sink.
    :param int min_statements: Production candidate statement-count floor.
    :return tuple[int, int]: Total and production-scoreable expected targets.
    """
    if not probes:
        failures.append("search probes JSON has an empty 'probes' list")
    eligible = {unit_type.strip().lower() for unit_type in DEFAULT_SEMANTIC_UNIT_TYPES}
    total_targets = 0
    scoreable_targets = 0
    for index, probe in enumerate(probes):
        if not probe.get("query", "").strip():
            failures.append(f"probe {index} has an empty query")
        expected = probe.get("expected", [])
        if not expected:
            failures.append(f"probe {index} has no expected units")
        for spec in expected:
            total_targets += 1
            try:
                unit = resolve_label_unit(units, spec)
            except ValueError as exc:
                failures.append(f"probe {index}: {exc}")
                continue
            unit_type = unit.unit_type.name.lower()
            if unit_type not in eligible:
                failures.append(
                    f"probe {index} expects {spec!r} of type {unit_type!r}; "
                    f"semantic candidates only cover {sorted(eligible)}"
                )
            elif get_code_unit_statement_count(unit) >= min_statements:
                scoreable_targets += 1
    return total_targets, scoreable_targets


def _validate_files(corpus_path: Path, units: list[CodeUnit], failures: list[str]) -> None:
    """Validate corpus file naming and per-basename unit-name uniqueness.

    The walk shares the sweep manifest's debris exclusions, so filesystem
    noise (``.DS_Store``, ``__pycache__``) never trips the zero-unit check.

    :param Path corpus_path: Corpus root directory.
    :param list[CodeUnit] units: Extracted corpus units.
    :param list[str] failures: Mutable failure sink.
    """
    source_files = corpus_files(corpus_path)
    files_with_units = {unit.file_path.resolve() for unit in units}
    for path in source_files:
        relative = path.relative_to(corpus_path).as_posix()
        for pattern in DEFAULT_EXCLUDE_PATTERNS:
            if fnmatch(relative, pattern) or fnmatch(relative, pattern.removeprefix("**/")):
                failures.append(f"file matches default exclude pattern {pattern!r}: {relative}")
        if path.suffix != ".json" and path.resolve() not in files_with_units:
            failures.append(f"corpus file produced zero units: {relative}")

    # Label specs resolve by bare filename, so a unit name must be unique per
    # basename across the whole corpus, not merely within one file.
    per_file: dict[tuple[str, str], int] = {}
    for unit in units:
        key = (unit.file_path.name, unit.name)
        per_file[key] = per_file.get(key, 0) + 1
    for (filename, name), count in sorted(per_file.items()):
        if count > 1:
            failures.append(f"unit name not unique for basename: {filename}::{name} x{count}")


def _rejected_extraction_diagnostics(
    diagnostics: list[ExtractionDiagnostic],
) -> list[ExtractionDiagnostic]:
    """Return extraction diagnostics that invalidate a calibration corpus.

    :param list[ExtractionDiagnostic] diagnostics: Diagnostics emitted during extraction.
    :return list[ExtractionDiagnostic]: Errors and every parser-recovery diagnostic.
    """
    return [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.severity == "error" or diagnostic.code in PARSE_FAILURE_DIAGNOSTIC_CODES
    ]


def main() -> int:
    """Entry point.

    :return int: Process exit code, non-zero when any contract fails.
    """
    parser = argparse.ArgumentParser(description="Validate one calibration corpus (no model).")
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--search-probes-path", type=Path, default=None)
    parser.add_argument(
        "--min-statements",
        type=int,
        default=DEFAULT_MIN_SEMANTIC_STATEMENTS,
        help=(
            "Minimum recursive statement count used for candidate-coverage reporting "
            f"(default: production value {DEFAULT_MIN_SEMANTIC_STATEMENTS})."
        ),
    )
    parser.add_argument(
        "--language",
        action="append",
        default=None,
        help="Restrict extraction to a language (repeat for multiple).",
    )
    args = parser.parse_args()

    config = AnalyzerConfig(
        run_traditional=True,
        run_semantic=False,
        run_unused=False,
        include_private=True,
        languages=tuple(args.language) if args.language else None,
    )
    result = CodeAnalyzer(config).analyze(args.corpus_path)
    failures: list[str] = []

    bad_diagnostics = _rejected_extraction_diagnostics(result.extraction_diagnostics)
    for diagnostic in bad_diagnostics:
        failures.append(
            f"diagnostic {diagnostic.code} in {diagnostic.file_path}: {diagnostic.message}"
        )

    labels = json.loads(args.labels_path.read_text())
    counts = _validate_labels(result.units, labels, failures)
    _validate_files(args.corpus_path, result.units, failures)

    eligible_types = {unit_type.strip().lower() for unit_type in DEFAULT_SEMANTIC_UNIT_TYPES}
    candidate_uids = {
        unit.uid
        for unit in result.units
        if unit.unit_type.name.lower() in eligible_types
        and get_code_unit_statement_count(unit) >= args.min_statements
    }
    # ``_validate_labels`` already recorded any unresolvable spec; rebuilding the
    # pair set must not abort the run, or one renamed corpus symbol replaces the
    # whole failure report with a traceback.
    coverage: tuple[int, int] | None = None
    try:
        positive_pairs = build_positive_pairs(result.units, labels)
    except ValueError as exc:
        failures.append(f"cannot build positive pairs: {exc}")
    else:
        scoreable_pairs = {
            pair
            for pair in positive_pairs
            if pair[0] in candidate_uids and pair[1] in candidate_uids
        }
        coverage = (len(scoreable_pairs), len(positive_pairs))
        _validate_deterministic_coverage(result.traditional_duplicates, positive_pairs, failures)

    probe_coverage: tuple[int, int] | None = None
    if args.search_probes_path is not None:
        probes = json.loads(args.search_probes_path.read_text())["probes"]
        probe_coverage = _validate_probes(
            result.units,
            probes,
            failures,
            min_statements=args.min_statements,
        )

    print(f"Corpus: {args.corpus_path} ({len(result.units)} units)")
    for category in CATEGORY_NAMES:
        if category in counts:
            print(f"  {category}: {counts[category]} labeled pairs")
    negative_groups = labels.get("negative_controls", [])
    negative_pairs = sum(len(group) * (len(group) - 1) // 2 for group in negative_groups)
    print(f"  negative_controls: {len(negative_groups)} groups ({negative_pairs} pairs)")
    print(f"  deterministic pairs found: {len(result.traditional_duplicates)}")
    if coverage is not None:
        scoreable_count, positive_count = coverage
        print(
            f"  production candidate coverage (min statements {args.min_statements}): "
            f"{scoreable_count}/{positive_count} labeled pairs"
        )
    if probe_coverage is not None:
        total_targets, scoreable_targets = probe_coverage
        print(f"  production search-target coverage: {scoreable_targets}/{total_targets}")

    if failures:
        print(f"\nFAIL ({len(failures)} problems):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
