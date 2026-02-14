#!/usr/bin/env python
"""Sweep hybrid semantic-only gate thresholds against a labeled synthetic corpus."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import codedupes.analyzer as analyzer_module
from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from codedupes.constants import (
    DEFAULT_MODEL,
    DEFAULT_SEMANTIC_THRESHOLD,
    DEFAULT_TRADITIONAL_THRESHOLD,
)
from codedupes.models import CodeUnit, HybridDuplicate
from codedupes.pairs import ordered_pair_key


@dataclass(frozen=True)
class GateConfig:
    """Threshold configuration for semantic-only hybrid gating."""

    semantic_min: float
    weak_identifier_jaccard_min: float
    statement_ratio_min: float


@dataclass(frozen=True)
class SweepRow:
    """One evaluated gate row."""

    config: GateConfig
    predicted: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def _parse_csv_floats(value: str) -> list[float]:
    out = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        msg = "Expected at least one float value."
        raise argparse.ArgumentTypeError(msg)
    return out


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
        for unit_a, unit_b in itertools.combinations(resolved, 2):
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


def _run_sweep(
    *,
    traditional_duplicates,
    semantic_duplicates,
    positive_pairs: set[tuple[str, str]],
    semantic_threshold: float,
    traditional_threshold: float,
    grid: list[GateConfig],
) -> tuple[list[SweepRow], dict[str, float]]:
    old_values = {
        "semantic_min": float(analyzer_module.HYBRID_SEMANTIC_ONLY_MIN),
        "weak_min": float(analyzer_module.HYBRID_WEAK_JACCARD_MIN),
        "ratio_min": float(analyzer_module.HYBRID_STATEMENT_RATIO_MIN),
    }

    rows: list[SweepRow] = []
    try:
        for config in grid:
            analyzer_module.HYBRID_SEMANTIC_ONLY_MIN = config.semantic_min
            analyzer_module.HYBRID_WEAK_JACCARD_MIN = config.weak_identifier_jaccard_min
            analyzer_module.HYBRID_STATEMENT_RATIO_MIN = config.statement_ratio_min

            hybrid: list[HybridDuplicate]
            hybrid, _ = analyzer_module._synthesize_hybrid_duplicates(
                traditional_duplicates,
                semantic_duplicates,
                semantic_threshold=semantic_threshold,
                jaccard_threshold=traditional_threshold,
            )
            predicted_pairs = {ordered_pair_key(item.unit_a, item.unit_b) for item in hybrid}
            tp, fp, fn, precision, recall, f1 = _metrics(predicted_pairs, positive_pairs)
            rows.append(
                SweepRow(
                    config=config,
                    predicted=len(predicted_pairs),
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                )
            )
    finally:
        analyzer_module.HYBRID_SEMANTIC_ONLY_MIN = old_values["semantic_min"]
        analyzer_module.HYBRID_WEAK_JACCARD_MIN = old_values["weak_min"]
        analyzer_module.HYBRID_STATEMENT_RATIO_MIN = old_values["ratio_min"]

    rows.sort(key=lambda row: (row.f1, row.precision, row.recall, -row.fp), reverse=True)
    return rows, old_values


def _print_rows(rows: list[SweepRow], *, top_n: int) -> None:
    print("\nTop sweep rows:\n")
    for idx, row in enumerate(rows[:top_n], start=1):
        print(
            f"{idx:02d}. f1={row.f1:.3f} precision={row.precision:.3f} "
            f"recall={row.recall:.3f} tp={row.tp} fp={row.fp} fn={row.fn} "
            f"pred={row.predicted} "
            f"semantic_min={row.config.semantic_min:.3f} "
            f"weak_id_jaccard_min={row.config.weak_identifier_jaccard_min:.3f} "
            f"statement_ratio_min={row.config.statement_ratio_min:.3f}"
        )


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Sweep hybrid semantic-only gate thresholds on a labeled synthetic corpus."
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
        "--semantic-threshold",
        type=float,
        default=0.70,
        help="Low semantic threshold used to collect raw semantic candidates for the sweep.",
    )
    parser.add_argument(
        "--hybrid-semantic-threshold",
        type=float,
        default=DEFAULT_SEMANTIC_THRESHOLD,
        help="Semantic threshold passed into hybrid synthesis for mixed-evidence pairs.",
    )
    parser.add_argument(
        "--traditional-threshold",
        type=float,
        default=DEFAULT_TRADITIONAL_THRESHOLD,
        help="Traditional jaccard threshold used by hybrid synthesis.",
    )
    parser.add_argument(
        "--semantic-grid",
        type=_parse_csv_floats,
        default=[0.85, 0.88, 0.90, 0.92, 0.94],
        help="Comma-separated semantic-only minimum values to sweep.",
    )
    parser.add_argument(
        "--weak-jaccard-grid",
        type=_parse_csv_floats,
        default=[0.10, 0.15, 0.20, 0.25, 0.30],
        help="Comma-separated weak identifier jaccard minimum values to sweep.",
    )
    parser.add_argument(
        "--statement-ratio-grid",
        type=_parse_csv_floats,
        default=[0.20, 0.25, 0.35, 0.45, 0.55],
        help="Comma-separated statement count ratio minimum values to sweep.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model name.")
    parser.add_argument(
        "--model-revision",
        default=None,
        help=(
            "Model revision / commit hash. If omitted, uses model-profile default "
            "(for example pinned for C2LLM 0.5B)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Embedding batch size used for the candidate extraction run.",
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=0,
        help="Minimum statement count for semantic candidate extraction.",
    )
    trust_group = parser.add_mutually_exclusive_group()
    trust_group.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        help="Enable model remote code execution during load.",
    )
    trust_group.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable model remote code execution during load.",
    )
    parser.set_defaults(trust_remote_code=None)
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of best rows to print.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full sweep output JSON.",
    )

    args = parser.parse_args()

    labels = json.loads(args.labels_path.read_text())
    config = AnalyzerConfig(
        run_traditional=True,
        run_semantic=True,
        run_unused=False,
        include_private=True,
        min_semantic_lines=args.min_lines,
        jaccard_threshold=args.traditional_threshold,
        semantic_threshold=args.semantic_threshold,
        model_name=args.model,
        model_revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        batch_size=args.batch_size,
    )
    analyzer = CodeAnalyzer(config)
    result = analyzer.analyze(args.corpus_path)

    positive_pairs = _build_positive_pairs(result.units, labels)
    grid = [
        GateConfig(semantic, weak, ratio)
        for semantic, weak, ratio in itertools.product(
            args.semantic_grid,
            args.weak_jaccard_grid,
            args.statement_ratio_grid,
        )
    ]

    rows, baseline = _run_sweep(
        traditional_duplicates=result.traditional_duplicates,
        semantic_duplicates=result.semantic_duplicates,
        positive_pairs=positive_pairs,
        semantic_threshold=args.hybrid_semantic_threshold,
        traditional_threshold=args.traditional_threshold,
        grid=grid,
    )

    print("Hybrid gate sweep (synthetic corpus guardrail)")
    print(f"Corpus: {args.corpus_path}")
    print(f"Labels: {args.labels_path}")
    print(f"Units extracted: {len(result.units)}")
    print(
        "Raw candidates: "
        f"traditional={len(result.traditional_duplicates)} "
        f"semantic={len(result.semantic_duplicates)}"
    )
    print(
        "Current defaults: "
        f"semantic_min={baseline['semantic_min']:.3f} "
        f"weak_id_jaccard_min={baseline['weak_min']:.3f} "
        f"statement_ratio_min={baseline['ratio_min']:.3f}"
    )
    _print_rows(rows, top_n=args.top_n)

    if args.json_out is not None:
        payload = {
            "corpus_path": str(args.corpus_path),
            "labels_path": str(args.labels_path),
            "units": len(result.units),
            "raw_candidates": {
                "traditional_duplicates": len(result.traditional_duplicates),
                "semantic_duplicates": len(result.semantic_duplicates),
            },
            "baseline_defaults": baseline,
            "rows": [
                {
                    **asdict(row),
                    "config": asdict(row.config),
                }
                for row in rows
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote sweep report: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
