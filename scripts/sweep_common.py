"""Shared helpers for synthetic sweep scripts."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import Any

from codedupes.constants import DEFAULT_MIN_SEMANTIC_STATEMENTS
from codedupes.models import CodeUnit
from codedupes.pairs import ordered_pair_key


def add_common_sweep_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the corpus/labels/extraction options shared by all sweep scripts.

    :param argparse.ArgumentParser parser: Sweep script argument parser.
    :return None: ``None``.
    """
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("test_fixtures/hybrid_tuning/crab_visibility"),
        help="Root path of the synthetic corpus package/scripts.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("test_fixtures/hybrid_tuning/labels.json"),
        help="Path to labels.json with expected duplicate groups.",
    )
    parser.add_argument(
        "--language",
        action="append",
        dest="language",
        default=None,
        metavar="LANGUAGE",
        help="Restrict extraction to a language (repeat for multiple); omit to auto-detect.",
    )
    parser.add_argument(
        "--min-statements",
        type=int,
        default=DEFAULT_MIN_SEMANTIC_STATEMENTS,
        help=(
            "Minimum statement count for semantic candidate extraction "
            f"(default: production value {DEFAULT_MIN_SEMANTIC_STATEMENTS})."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Embedding batch size used for candidate extraction.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of best rows to print.",
    )


def rank_sweep_rows(
    rows: list[Any],
    *,
    extra_key: Callable[[Any], tuple[Any, ...]] | None = None,
) -> None:
    """Sort sweep rows in place, best first, by the shared recall-preferring policy.

    Rows rank by ``(f1, precision, recall, -fp)`` descending. ``extra_key`` appends
    trailing tiebreak terms, for example ``-threshold`` so equal-metric ties prefer
    the looser threshold (recall over precision).

    :param list[Any] rows: Sweep rows exposing ``f1``, ``precision``, ``recall``, ``fp``.
    :param Callable extra_key: Optional builder of trailing tiebreak terms.
    :return None: ``None``.
    """

    def sort_key(row: Any) -> tuple[Any, ...]:
        """Build one row's full ranking tuple.

        :param Any row: Sweep row to rank.
        :return tuple[Any, ...]: Ranking terms, highest-first under ``reverse=True``.
        """
        base = (row.f1, row.precision, row.recall, -row.fp)
        return base + (extra_key(row) if extra_key is not None else ())

    rows.sort(key=sort_key, reverse=True)


def parse_label_spec(spec: str) -> tuple[str, str]:
    """Parse a label selector string.

    :param str spec: Label selector in the form ``file.py::symbol_name``.
    :raises ValueError: If the label selector format is invalid.
    :return tuple[str, str]: Parsed ``(filename, symbol_name)`` tuple.
    """
    try:
        filename, symbol = spec.split("::", 1)
    except ValueError as exc:
        msg = f"Invalid label spec {spec!r}; expected 'file.py::symbol_name'."
        raise ValueError(msg) from exc
    return filename, symbol


def resolve_label_unit(units: list[CodeUnit], spec: str) -> CodeUnit:
    """Resolve a label selector to one extracted code unit.

    :param list[CodeUnit] units: Extracted units from the sweep corpus.
    :param str spec: Label selector in the form ``file.py::symbol_name``.
    :raises ValueError: If the selector does not match exactly one unit.
    :return CodeUnit: Matched code unit.
    """
    filename, symbol = parse_label_spec(spec)
    matches = [unit for unit in units if unit.file_path.name == filename and unit.name == symbol]
    if len(matches) != 1:
        msg = f"Label {spec!r} matched {len(matches)} units (expected exactly 1)."
        raise ValueError(msg)
    return matches[0]


def validate_labels_shape(labels: dict[str, Any]) -> None:
    """Fail fast on structurally malformed labels JSON, before any model work.

    Shape checks only - specs are not resolved against extracted units - so sweep
    entry points can reject a bad labels file in milliseconds instead of aborting
    after the corpus embed, where an empty category list surfaced as
    ``build_positive_pairs``'s misleading top-level ``positive_groups`` error.

    :param dict[str, Any] labels: Loaded labels JSON dictionary.
    :raises ValueError: If ``positive_groups`` or any ``categories`` entry is malformed.
    :return None: ``None``.
    """
    groups = labels.get("positive_groups")
    if not isinstance(groups, list) or not groups:
        msg = "labels.json must define a non-empty 'positive_groups' list."
        raise ValueError(msg)
    for group in groups:
        if not isinstance(group, list) or len(group) < 2:
            msg = f"Invalid positive group {group!r}; expected a list with at least two specs."
            raise ValueError(msg)

    categories = labels.get("categories")
    if categories is None:
        return
    if not isinstance(categories, dict):
        msg = "labels.json 'categories' must map category names to lists of positive groups."
        # Kept a ValueError so callers catch one shape-error type, matching
        # every other malformed-labels raise in this module.
        raise ValueError(msg)  # noqa: TRY004
    for category, category_groups in categories.items():
        if not isinstance(category_groups, list) or not category_groups:
            msg = f"labels.json category {category!r} must list at least one positive group."
            raise ValueError(msg)
        for group in category_groups:
            if not isinstance(group, list) or len(group) < 2:
                msg = (
                    f"Invalid group {group!r} in category {category!r}; "
                    "expected a list with at least two specs."
                )
                raise ValueError(msg)


def corpus_files(root: Path) -> list[Path]:
    """List corpus source files, excluding caches and hidden files.

    One walk shared by the calibration-manifest digest and the corpus
    validator: ``__pycache__`` entries, ``.pyc`` artifacts, and hidden files
    (``.DS_Store`` and friends) are filesystem debris, not corpus contract
    surface, so neither the digest nor the zero-unit-file check may see them.

    :param Path root: Corpus root directory.
    :return list[Path]: Sorted regular files under ``root``.
    """
    files: list[Path] = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(root)
        if "__pycache__" in relative.parts or relative.suffix == ".pyc":
            continue
        if any(part.startswith(".") for part in relative.parts):
            continue
        files.append(file_path)
    return files


def validate_probes_shape(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Fail fast on structurally malformed search-probes JSON, before any model work.

    Shape checks only, mirroring :func:`validate_labels_shape`: specs are not
    resolved against extracted units. Without this gate an empty or mis-keyed
    probes file still produced a complete search report - zero probes, every row
    scored 0.0, and the loosest-tie ranking selected the grid floor under a full
    calibration manifest.

    :param dict[str, Any] payload: Loaded search-probes JSON dictionary.
    :raises ValueError: If the payload or any probe entry is malformed.
    :return list[dict[str, Any]]: The validated probe list.
    """
    probes = payload.get("probes")
    if not isinstance(probes, list) or not probes:
        msg = "search probes JSON must define a non-empty 'probes' list."
        raise ValueError(msg)
    for index, probe in enumerate(probes):
        if not isinstance(probe, dict):
            msg = f"probe {index} must be an object; got {probe!r}."
            # Kept a ValueError so callers catch one shape-error type, matching
            # validate_labels_shape.
            raise ValueError(msg)  # noqa: TRY004
        query = probe.get("query")
        if not isinstance(query, str) or not query.strip():
            msg = f"probe {index} must define a non-empty string 'query'."
            raise ValueError(msg)
        expected = probe.get("expected")
        if not isinstance(expected, list) or not expected:
            msg = f"probe {index} must define a non-empty 'expected' spec list."
            raise ValueError(msg)
        for spec in expected:
            if not isinstance(spec, str) or not spec.strip():
                msg = f"probe {index} has an invalid expected spec: {spec!r}."
                raise ValueError(msg)
    return probes


def build_positive_pairs(units: list[CodeUnit], labels: dict[str, Any]) -> set[tuple[str, str]]:
    """Build expected-positive duplicate pairs from label groups.

    :param list[CodeUnit] units: Extracted units from the sweep corpus.
    :param dict[str, Any] labels: Loaded labels JSON dictionary.
    :raises ValueError: If label data is missing or malformed.
    :return set[tuple[str, str]]: Unordered positive pair keys.
    """
    groups = labels.get("positive_groups", [])
    if not isinstance(groups, list) or not groups:
        msg = "labels.json must define a non-empty 'positive_groups' list."
        raise ValueError(msg)

    positives: set[tuple[str, str]] = set()
    for group in groups:
        if not isinstance(group, list) or len(group) < 2:
            msg = f"Invalid positive group {group!r}; expected a list with at least two specs."
            raise ValueError(msg)
        resolved = [resolve_label_unit(units, spec) for spec in group]
        for unit_a, unit_b in combinations(resolved, 2):
            positives.add(ordered_pair_key(unit_a, unit_b))
    return positives


def metrics(
    predicted_pairs: set[tuple[str, str]],
    positive_pairs: set[tuple[str, str]],
) -> tuple[int, int, int, float, float, float]:
    """Compute precision/recall metrics for predicted pair sets.

    :param set[tuple[str, str]] predicted_pairs: Predicted positive pair keys.
    :param set[tuple[str, str]] positive_pairs: Ground-truth positive pair keys.
    :return tuple[int, int, int, float, float, float]: ``tp, fp, fn, precision, recall, f1``.
    """
    tp = len(predicted_pairs & positive_pairs)
    fp = len(predicted_pairs - positive_pairs)
    fn = len(positive_pairs - predicted_pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return tp, fp, fn, precision, recall, f1
