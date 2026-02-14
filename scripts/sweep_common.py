#!/usr/bin/env python
"""Shared helpers for synthetic sweep scripts."""

from __future__ import annotations

from itertools import combinations
from typing import Any

from codedupes.models import CodeUnit
from codedupes.pairs import ordered_pair_key


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
