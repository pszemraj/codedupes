#!/usr/bin/env python
"""Sweep semantic thresholds for built-in semantic model profiles."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from codedupes.models import CodeUnit, DuplicatePair
from codedupes.pairs import ordered_pair_key
from codedupes.semantic_profiles import list_supported_models, resolve_model_profile

THRESHOLD_START = 0.70
THRESHOLD_STOP = 0.96
THRESHOLD_STEP = 0.02


@dataclass(frozen=True)
class SweepRow:
    """Single threshold evaluation row."""

    threshold: float
    predicted: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class ModelSweep:
    """Sweep results for one model."""

    model_key: str
    canonical_name: str
    selected_threshold: float
    rows: list[SweepRow]


def _threshold_grid() -> list[float]:
    values: list[float] = []
    current = THRESHOLD_START
    while current <= THRESHOLD_STOP + 1e-9:
        values.append(round(current, 2))
        current += THRESHOLD_STEP
    return values


def _parse_label_spec(spec: str) -> tuple[str, str]:
    try:
        filename, symbol = spec.split("::", 1)
    except ValueError as exc:  # pragma: no cover - argument error path
        msg = f"Invalid label spec {spec!r}; expected 'file.py::symbol_name'."
        raise ValueError(msg) from exc
    return filename, symbol


def _resolve_label_unit(units: list[CodeUnit], spec: str) -> CodeUnit:
    filename, symbol = _parse_label_spec(spec)
    matches = [unit for unit in units if unit.file_path.name == filename and unit.name == symbol]
    if len(matches) != 1:
        msg = f"Label {spec!r} matched {len(matches)} units (expected exactly 1)."
        raise ValueError(msg)
    return matches[0]


def _build_positive_pairs(units: list[CodeUnit], labels: dict[str, Any]) -> set[tuple[str, str]]:
    groups = labels.get("positive_groups", [])
    if not isinstance(groups, list) or not groups:
        msg = "labels.json must define a non-empty 'positive_groups' list."
        raise ValueError(msg)

    positives: set[tuple[str, str]] = set()
    for group in groups:
        if not isinstance(group, list) or len(group) < 2:
            msg = f"Invalid positive group {group!r}; expected a list with at least two specs."
            raise ValueError(msg)
        resolved = [_resolve_label_unit(units, spec) for spec in group]
        for i, unit_a in enumerate(resolved):
            for unit_b in resolved[i + 1 :]:
                positives.add(ordered_pair_key(unit_a, unit_b))
    return positives


def _metrics(
    predicted_pairs: set[tuple[str, str]],
    positive_pairs: set[tuple[str, str]],
) -> tuple[int, int, int, float, float, float]:
    tp = len(predicted_pairs & positive_pairs)
    fp = len(predicted_pairs - positive_pairs)
    fn = len(positive_pairs - predicted_pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return tp, fp, fn, precision, recall, f1


def _evaluate_thresholds(
    duplicates: list[DuplicatePair],
    positive_pairs: set[tuple[str, str]],
    *,
    thresholds: list[float],
) -> list[SweepRow]:
    rows: list[SweepRow] = []
    for threshold in thresholds:
        predicted_pairs = {
            ordered_pair_key(duplicate.unit_a, duplicate.unit_b)
            for duplicate in duplicates
            if duplicate.similarity >= threshold
        }
        tp, fp, fn, precision, recall, f1 = _metrics(predicted_pairs, positive_pairs)
        rows.append(
            SweepRow(
                threshold=threshold,
                predicted=len(predicted_pairs),
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    rows.sort(
        key=lambda row: (
            row.f1,
            row.precision,
            row.recall,
            -row.fp,
            -row.threshold,
        ),
        reverse=True,
    )
    return rows


def _run_model_sweep(
    *,
    model_name: str,
    corpus_path: Path,
    labels: dict[str, Any],
    min_lines: int,
    batch_size: int,
) -> ModelSweep:
    profile = resolve_model_profile(model_name)
    config = AnalyzerConfig(
        run_traditional=False,
        run_semantic=True,
        run_unused=False,
        include_private=True,
        model_name=model_name,
        semantic_threshold=THRESHOLD_START,
        min_semantic_lines=min_lines,
        batch_size=batch_size,
    )
    analyzer = CodeAnalyzer(config)
    result = analyzer.analyze(corpus_path)

    positive_pairs = _build_positive_pairs(result.units, labels)
    rows = _evaluate_thresholds(
        result.semantic_duplicates,
        positive_pairs,
        thresholds=_threshold_grid(),
    )
    selected = rows[0]

    return ModelSweep(
        model_key=model_name,
        canonical_name=profile.canonical_name,
        selected_threshold=selected.threshold,
        rows=rows,
    )


def _print_sweep(model_sweep: ModelSweep, top_n: int) -> None:
    print(f"\nModel: {model_sweep.model_key}")
    print(f"Selected threshold: {model_sweep.selected_threshold:.2f}")
    print("Top rows:")
    for idx, row in enumerate(model_sweep.rows[:top_n], start=1):
        print(
            f"  {idx:02d}. threshold={row.threshold:.2f} f1={row.f1:.3f} "
            f"precision={row.precision:.3f} recall={row.recall:.3f} "
            f"tp={row.tp} fp={row.fp} fn={row.fn} pred={row.predicted}"
        )


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Sweep semantic thresholds across built-in model profiles."
    )
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
        "--models",
        nargs="*",
        default=[profile.key for profile in list_supported_models()],
        help="Model keys or IDs to sweep. Defaults to all built-in profiles.",
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=0,
        help="Minimum statement count for semantic candidate extraction.",
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
        help="Number of best rows to print per model.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("test_fixtures/hybrid_tuning/semantic_threshold_report.json"),
        help="Path to write full sweep output JSON.",
    )
    args = parser.parse_args()

    labels = json.loads(args.labels_path.read_text())

    results: list[ModelSweep] = []
    for model_name in args.models:
        results.append(
            _run_model_sweep(
                model_name=model_name,
                corpus_path=args.corpus_path,
                labels=labels,
                min_lines=args.min_lines,
                batch_size=args.batch_size,
            )
        )

    print("Semantic threshold sweep (synthetic corpus guardrail)")
    print(f"Corpus: {args.corpus_path}")
    print(f"Labels: {args.labels_path}")

    for item in results:
        _print_sweep(item, top_n=args.top_n)

    payload = {
        "corpus_path": str(args.corpus_path),
        "labels_path": str(args.labels_path),
        "grid": _threshold_grid(),
        "models": [
            {
                "model_key": item.model_key,
                "canonical_name": item.canonical_name,
                "selected_threshold": item.selected_threshold,
                "rows": [asdict(row) for row in item.rows],
            }
            for item in results
        ],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote sweep report: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
