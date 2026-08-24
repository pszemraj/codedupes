"""Sweep hybrid semantic-only confidence thresholds on a labeled corpus."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import codedupes.analyzer as analyzer_module
from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from codedupes.constants import (
    DEFAULT_MODEL,
    DEFAULT_TRADITIONAL_THRESHOLD,
)
from codedupes.models import HybridDuplicate
from codedupes.pairs import ordered_pair_key

try:
    from .sweep_common import (
        add_common_sweep_arguments,
        build_positive_pairs,
        metrics,
        rank_sweep_rows,
        validate_labels_shape,
    )
except ImportError:
    from sweep_common import (
        add_common_sweep_arguments,
        build_positive_pairs,
        metrics,
        rank_sweep_rows,
        validate_labels_shape,
    )


@dataclass(frozen=True)
class GateConfig:
    """Threshold configuration for semantic-only hybrid confidence tiering.

    ``semantic_min`` emulates the per-language duplicate gate the analyzer
    applies to semantic pairs before hybrid synthesis; the other two values are
    the synthesis-time corroboration guards.
    """

    semantic_min: float
    weak_identifier_jaccard_min: float
    statement_ratio_min: float


@dataclass(frozen=True)
class SweepRow:
    """One evaluated confidence-tier row."""

    config: GateConfig
    published: int
    review: int
    high_confidence: int
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


def _run_sweep(
    *,
    traditional_duplicates,
    semantic_duplicates,
    positive_pairs: set[tuple[str, str]],
    traditional_threshold: float,
    grid: list[GateConfig],
) -> tuple[list[SweepRow], dict[str, float]]:
    baseline = {
        "weak_min": float(analyzer_module.HYBRID_WEAK_JACCARD_MIN),
        "ratio_min": float(analyzer_module.HYBRID_STATEMENT_RATIO_MIN),
    }

    rows: list[SweepRow] = []
    for config in grid:
        gated_semantic = [
            duplicate
            for duplicate in semantic_duplicates
            if duplicate.similarity >= config.semantic_min
        ]
        hybrid: list[HybridDuplicate]
        hybrid = analyzer_module._synthesize_hybrid_duplicates(
            traditional_duplicates,
            gated_semantic,
            jaccard_threshold=traditional_threshold,
            weak_identifier_jaccard_min=config.weak_identifier_jaccard_min,
            statement_ratio_min=config.statement_ratio_min,
        )
        published_pairs = {ordered_pair_key(item.unit_a, item.unit_b) for item in hybrid}
        review_pairs = {
            ordered_pair_key(item.unit_a, item.unit_b)
            for item in hybrid
            if item.tier == "semantic_review"
        }
        high_confidence_pairs = published_pairs - review_pairs
        tp, fp, fn, precision, recall, f1 = metrics(high_confidence_pairs, positive_pairs)
        rows.append(
            SweepRow(
                config=config,
                published=len(published_pairs),
                review=len(review_pairs),
                high_confidence=len(high_confidence_pairs),
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    # Ties prefer the looser gate on every axis, matching the semantic sweep's
    # recall-first policy; without it equal-metric rows resolve by grid order.
    rank_sweep_rows(
        rows,
        extra_key=lambda row: (
            -row.config.semantic_min,
            -row.config.weak_identifier_jaccard_min,
            -row.config.statement_ratio_min,
        ),
    )
    return rows, baseline


def _print_rows(rows: list[SweepRow], *, top_n: int) -> None:
    print("\nTop sweep rows:\n")
    for idx, row in enumerate(rows[:top_n], start=1):
        print(
            f"{idx:02d}. f1={row.f1:.3f} precision={row.precision:.3f} "
            f"recall={row.recall:.3f} tp={row.tp} fp={row.fp} fn={row.fn} "
            f"high_conf={row.high_confidence} review={row.review} "
            f"published={row.published} "
            f"semantic_min={row.config.semantic_min:.3f} "
            f"weak_id_jaccard_min={row.config.weak_identifier_jaccard_min:.3f} "
            f"statement_ratio_min={row.config.statement_ratio_min:.3f}"
        )


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Sweep hybrid semantic-only confidence thresholds on a labeled synthetic corpus."
        )
    )
    add_common_sweep_arguments(parser)
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=None,
        help=(
            "Semantic threshold used to collect raw semantic candidates for the sweep "
            "(default: the lowest --semantic-grid value, so every grid row is reachable)."
        ),
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
        default=[0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92],
        help=(
            "Comma-separated semantic gate values to sweep; each emulates the "
            "per-language duplicate gate applied before hybrid synthesis."
        ),
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
        help=("Model revision / commit hash. If omitted, uses the model-profile default."),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Embedding device for the sweep. Defaults to cpu for reproducible float32.",
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
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full sweep output JSON.",
    )

    args = parser.parse_args()

    grid_floor = min(args.semantic_grid)
    collection_threshold = (
        args.semantic_threshold if args.semantic_threshold is not None else grid_floor
    )
    if collection_threshold > grid_floor:
        parser.error(
            f"--semantic-threshold {collection_threshold} is above the lowest --semantic-grid "
            f"value {grid_floor}; grid rows below the collection threshold would silently "
            "repeat the collection threshold's results."
        )

    labels = json.loads(args.labels_path.read_text())
    try:
        validate_labels_shape(labels)
    except ValueError as exc:
        parser.error(str(exc))
    config = AnalyzerConfig(
        run_traditional=True,
        run_semantic=True,
        run_unused=False,
        include_private=True,
        languages=tuple(args.language) if args.language else None,
        min_semantic_statements=args.min_statements,
        jaccard_threshold=args.traditional_threshold,
        semantic_threshold=collection_threshold,
        model_name=args.model,
        model_revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        batch_size=args.batch_size,
        device=args.device,
    )
    analyzer = CodeAnalyzer(config)
    result = analyzer.analyze(args.corpus_path)

    positive_pairs = build_positive_pairs(result.units, labels)
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
        traditional_threshold=args.traditional_threshold,
        grid=grid,
    )

    print("Hybrid confidence sweep (synthetic corpus guardrail)")
    print(f"Corpus: {args.corpus_path}")
    print(f"Labels: {args.labels_path}")
    print(f"Units extracted: {len(result.units)}")
    print(
        "Raw candidates: "
        f"traditional={len(result.traditional_duplicates)} "
        f"semantic={len(result.semantic_duplicates)}"
    )
    print(
        "Current confidence defaults: "
        f"weak_id_jaccard_min={baseline['weak_min']:.3f} "
        f"statement_ratio_min={baseline['ratio_min']:.3f} "
        "(semantic_min rows emulate the analyzer's per-language duplicate gate)"
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
