"""Sweep the hybrid tier split that decides which semantic pairs are shown by default.

``codedupes check`` withholds ``semantic_review`` pairs unless asked, so the
corroboration constants (identifier overlap, statement-count ratio) and the
per-language similarity promotion gate decide default visibility. This sweep
scores the visible subset (``output_policy`` ``hybrid_high_confidence``: the
published output minus ``semantic_review``) against the labeled positives at
each language's shipped admission gate, in two stages per model:

1. corroboration constants, promotion disabled, swept jointly over every corpus
   and selected once (they are embedding-independent, so they stay global);
2. the promotion gate, per language, at the stage-1 constants.

Selection maximizes visible precision subject to keeping at least
``--recall-retention-min`` of the all-published recall and never falling below
the all-published precision. Ties prefer F1, then the stricter setting: the
default view widens only on measured evidence, because admission already
carries the recall hedge. Every report records the calibration manifest per
corpus (pinned model commit, pipeline schema, effective embedding-space
identity, candidate policy, corpus and label digests) and the run refuses
models that cannot be pinned to an immutable 40-character commit.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import codedupes.analyzer as analyzer_module
from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from codedupes.constants import (
    DEFAULT_CHECK_SEMANTIC_TASK,
    DEFAULT_TRADITIONAL_THRESHOLD,
)
from codedupes.languages import normalize_languages
from codedupes.models import DuplicatePair, HybridDuplicate
from codedupes.pairs import ordered_pair_key
from codedupes.semantic_profiles import list_supported_models, resolve_model_profile

try:
    from .sweep_common import (
        add_common_sweep_arguments,
        build_positive_pairs,
        metrics,
        rank_sweep_rows,
        validate_labels_shape,
    )
    from .sweep_semantic_thresholds import _calibration_manifest, _require_immutable_revision
except ImportError:
    from sweep_common import (
        add_common_sweep_arguments,
        build_positive_pairs,
        metrics,
        rank_sweep_rows,
        validate_labels_shape,
    )
    from sweep_semantic_thresholds import _calibration_manifest, _require_immutable_revision

POLYGLOT_LANGUAGES = ("c", "rust", "javascript", "typescript", "python")
DEFAULT_WEAK_JACCARD_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
DEFAULT_STATEMENT_RATIO_GRID = [0.0, 0.20, 0.35, 0.50, 0.65, 0.80]
DEFAULT_HIGH_GATE_STOP = 0.96
DEFAULT_HIGH_GATE_STEP = 0.02
DEFAULT_RECALL_RETENTION_MIN = 0.85


@dataclass(frozen=True)
class GateConfig:
    """One tier-split configuration evaluated by the sweep.

    ``semantic_gate`` is the admission gate the semantic pairs were collected
    at (``None`` on pooled rows, where it differs per corpus); the other values
    are the synthesis-time promotion controls. ``high_gate`` ``None`` disables
    similarity promotion.
    """

    semantic_gate: float | None
    weak_identifier_jaccard_min: float
    statement_ratio_min: float
    high_gate: float | None = None


@dataclass(frozen=True)
class SweepRow:
    """One evaluated configuration.

    ``tp``/``fp``/``fn`` and ``precision``/``recall``/``f1`` score the visible
    subset (published minus ``semantic_review``); the ``published_*`` fields
    score every published pair and ``review_*`` the withheld tier alone.
    """

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
    published_tp: int
    published_fp: int
    published_fn: int
    published_precision: float
    published_recall: float
    published_f1: float
    review_tp: int
    review_fp: int


@dataclass(frozen=True)
class CorpusSpec:
    """One labeled, single-language corpus the sweep embeds and scores."""

    name: str
    corpus_path: Path
    labels_path: Path
    language: str


@dataclass(frozen=True)
class CorpusRun:
    """Analyzer output for one corpus, reused across every grid row."""

    spec: CorpusSpec
    semantic_gate: float
    manifest: dict[str, Any]
    units: int
    traditional_duplicates: list[DuplicatePair]
    semantic_duplicates: list[DuplicatePair]
    positive_pairs: set[tuple[str, str]]


def _parse_csv_floats(value: str) -> list[float]:
    out = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        msg = "Expected at least one float value."
        raise argparse.ArgumentTypeError(msg)
    if any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in out):
        msg = "Expected comma-separated finite values in [0.0, 1.0]."
        raise argparse.ArgumentTypeError(msg)
    return out


def _looseness(config: GateConfig) -> tuple[float, float, float]:
    """Rank terms where a larger value means a looser (more permissive) split.

    :param GateConfig config: Configuration to rank.
    :return tuple[float, float, float]: Negated constants; a disabled promotion gate counts as strictest.
    """
    high = config.high_gate if config.high_gate is not None else 1.0 + DEFAULT_HIGH_GATE_STEP
    return (-config.weak_identifier_jaccard_min, -config.statement_ratio_min, -high)


def _strictness(config: GateConfig) -> tuple[float, float, float]:
    """Rank terms where a larger value means a stricter (narrower default view) split.

    :param GateConfig config: Configuration to rank.
    :return tuple[float, float, float]: Negation of :func:`_looseness`.
    """
    return tuple(-term for term in _looseness(config))  # type: ignore[return-value]


def _run_sweep(
    *,
    traditional_duplicates: list[DuplicatePair],
    semantic_duplicates: list[DuplicatePair],
    positive_pairs: set[tuple[str, str]],
    traditional_threshold: float,
    grid: list[GateConfig],
) -> list[SweepRow]:
    """Evaluate every grid configuration on one corpus's collected candidates.

    :param list[DuplicatePair] traditional_duplicates: Traditional pairs from the analyzer run.
    :param list[DuplicatePair] semantic_duplicates: Semantic pairs collected at the admission gate.
    :param set[tuple[str, str]] positive_pairs: Labeled positive pair keys.
    :param float traditional_threshold: Jaccard threshold used by hybrid synthesis.
    :param list[GateConfig] grid: Configurations to evaluate.
    :return list[SweepRow]: Rows ranked best-first by the recall-preferring policy.
    """
    languages = {
        unit.language
        for duplicate in semantic_duplicates
        for unit in (duplicate.unit_a, duplicate.unit_b)
    }

    rows: list[SweepRow] = []
    for config in grid:
        gated_semantic = [
            duplicate
            for duplicate in semantic_duplicates
            if config.semantic_gate is None or duplicate.similarity >= config.semantic_gate
        ]
        high_gates = (
            dict.fromkeys(languages, config.high_gate) if config.high_gate is not None else None
        )
        hybrid: list[HybridDuplicate] = analyzer_module._synthesize_hybrid_duplicates(
            traditional_duplicates,
            gated_semantic,
            jaccard_threshold=traditional_threshold,
            weak_identifier_jaccard_min=config.weak_identifier_jaccard_min,
            statement_ratio_min=config.statement_ratio_min,
            semantic_high_gates=high_gates,
        )
        published_pairs = {ordered_pair_key(item.unit_a, item.unit_b) for item in hybrid}
        review_pairs = {
            ordered_pair_key(item.unit_a, item.unit_b)
            for item in hybrid
            if item.tier == "semantic_review"
        }
        visible_pairs = published_pairs - review_pairs
        tp, fp, fn, precision, recall, f1 = metrics(visible_pairs, positive_pairs)
        p_tp, p_fp, p_fn, p_precision, p_recall, p_f1 = metrics(published_pairs, positive_pairs)
        rows.append(
            SweepRow(
                config=config,
                published=len(published_pairs),
                review=len(review_pairs),
                high_confidence=len(visible_pairs),
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
                published_tp=p_tp,
                published_fp=p_fp,
                published_fn=p_fn,
                published_precision=p_precision,
                published_recall=p_recall,
                published_f1=p_f1,
                review_tp=len(review_pairs & positive_pairs),
                review_fp=len(review_pairs - positive_pairs),
            )
        )

    # Ties prefer the looser split on every axis, matching the semantic sweep's
    # recall-first policy; without it equal-metric rows resolve by grid order.
    rank_sweep_rows(rows, extra_key=lambda row: _looseness(row.config))
    return rows


def is_feasible(row: SweepRow, *, recall_retention_min: float) -> bool:
    """Return whether a split keeps enough recall and does not lower precision.

    :param SweepRow row: Row to check.
    :param float recall_retention_min: Minimum visible recall as a fraction of published recall.
    :return bool: ``True`` when hiding review pairs costs bounded recall and buys precision.
    """
    return (
        row.recall >= recall_retention_min * row.published_recall
        and row.precision >= row.published_precision
    )


def select_visible_row(
    rows: Iterable[SweepRow],
    *,
    recall_retention_min: float = DEFAULT_RECALL_RETENTION_MIN,
    feasible: Callable[[SweepRow], bool] | None = None,
) -> SweepRow | None:
    """Pick the split with the best visible precision among feasible rows.

    Ties prefer F1, then the stricter split: a looser split that measures the
    same on the corpus would widen the default view on no evidence.

    :param Iterable[SweepRow] rows: Candidate rows.
    :param float recall_retention_min: Retention floor used by the default feasibility check.
    :param Callable feasible: Optional feasibility override (pooled rows carry per-corpus constraints).
    :return SweepRow | None: Best feasible row, or ``None`` when nothing is feasible.
    """
    check = feasible or (lambda row: is_feasible(row, recall_retention_min=recall_retention_min))
    candidates = [row for row in rows if check(row)]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row.precision, row.f1, _strictness(row.config)))


def _split_key(config: GateConfig) -> tuple[float, float, float | None]:
    """Identify a split independently of the corpus admission gate.

    :param GateConfig config: Configuration to key.
    :return tuple: Corroboration constants and promotion gate.
    """
    return (config.weak_identifier_jaccard_min, config.statement_ratio_min, config.high_gate)


def pool_rows(rows_by_corpus: dict[str, list[SweepRow]]) -> list[SweepRow]:
    """Sum per-corpus counts for each split and rescore the pooled totals.

    :param dict[str, list[SweepRow]] rows_by_corpus: Ranked rows per corpus, over one shared grid.
    :return list[SweepRow]: One pooled row per split, ranked best-first.
    """
    grouped: dict[tuple[float, float, float | None], list[SweepRow]] = {}
    for rows in rows_by_corpus.values():
        for row in rows:
            grouped.setdefault(_split_key(row.config), []).append(row)

    pooled: list[SweepRow] = []
    for members in grouped.values():
        first = members[0]
        totals = {
            name: sum(getattr(row, name) for row in members)
            for name in (
                "published",
                "review",
                "high_confidence",
                "tp",
                "fp",
                "fn",
                "published_tp",
                "published_fp",
                "published_fn",
                "review_tp",
                "review_fp",
            )
        }
        precision, recall, f1 = _prf(totals["tp"], totals["fp"], totals["fn"])
        p_precision, p_recall, p_f1 = _prf(
            totals["published_tp"], totals["published_fp"], totals["published_fn"]
        )
        pooled.append(
            SweepRow(
                config=replace(first.config, semantic_gate=None),
                precision=precision,
                recall=recall,
                f1=f1,
                published_precision=p_precision,
                published_recall=p_recall,
                published_f1=p_f1,
                **totals,
            )
        )
    rank_sweep_rows(pooled, extra_key=lambda row: _looseness(row.config))
    return pooled


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 from counts.

    :param int tp: True positives.
    :param int fp: False positives.
    :param int fn: False negatives.
    :return tuple[float, float, float]: ``precision, recall, f1``.
    """
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return precision, recall, f1


def select_pooled_row(
    pooled: list[SweepRow],
    rows_by_corpus: dict[str, list[SweepRow]],
    *,
    recall_retention_min: float,
) -> SweepRow | None:
    """Select the global split: feasible in every corpus, best pooled precision.

    :param list[SweepRow] pooled: Pooled rows from :func:`pool_rows`.
    :param dict[str, list[SweepRow]] rows_by_corpus: The per-corpus rows they were pooled from.
    :param float recall_retention_min: Per-corpus retention floor.
    :return SweepRow | None: Selected pooled row, or ``None`` when no split is feasible everywhere.
    """
    feasible_keys: set[tuple[float, float, float | None]] | None = None
    for rows in rows_by_corpus.values():
        keys = {
            _split_key(row.config)
            for row in rows
            if is_feasible(row, recall_retention_min=recall_retention_min)
        }
        feasible_keys = keys if feasible_keys is None else feasible_keys & keys
    allowed = feasible_keys or set()
    return select_visible_row(
        pooled,
        recall_retention_min=recall_retention_min,
        feasible=lambda row: (
            _split_key(row.config) in allowed and row.precision >= row.published_precision
        ),
    )


def _high_gate_grid(start: float, stop: float, step: float) -> list[float | None]:
    """Promotion-gate grid from the admission gate upward, plus ``None`` (disabled).

    :param float start: Admission gate; the loosest promotion gate promotes every admitted pair.
    :param float stop: Inclusive upper bound.
    :param float step: Grid step.
    :return list[float | None]: Ascending gates followed by ``None``.
    :raises ValueError: If a bound is non-finite or outside ``[0.0, 1.0]``,
        or if ``step`` is not finite and positive.
    """
    for name, value in (("start", start), ("stop", stop)):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0.0, 1.0]")
    if not math.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    values: list[float | None] = []
    steps = 0
    while True:
        current = round(start + steps * step, 9)
        if current > stop + 1e-9:
            break
        values.append(current)
        steps += 1
    values.append(None)
    return values


def _run_corpus(
    spec: CorpusSpec,
    *,
    model: str,
    revision: str,
    semantic_gate: float,
    args: argparse.Namespace,
) -> CorpusRun:
    """Embed one corpus at its admission gate and record its calibration manifest.

    :param CorpusSpec spec: Corpus to analyze.
    :param str model: Model key or id.
    :param str revision: Immutable model commit.
    :param float semantic_gate: Admission gate the semantic pairs are collected at.
    :param argparse.Namespace args: Parsed command-line options.
    :return CorpusRun: Collected candidates, labels, and manifest.
    """
    labels = json.loads(spec.labels_path.read_text())
    validate_labels_shape(labels)
    config = AnalyzerConfig(
        run_traditional=True,
        run_semantic=True,
        run_unused=False,
        include_private=True,
        languages=(spec.language,),
        min_semantic_statements=args.min_statements,
        jaccard_threshold=args.traditional_threshold,
        semantic_threshold=semantic_gate,
        model_name=model,
        model_revision=revision,
        trust_remote_code=args.trust_remote_code,
        batch_size=args.batch_size,
        device=args.device,
    )
    analyzer = CodeAnalyzer(config)
    result = analyzer.analyze(spec.corpus_path)
    embeddings = analyzer._embeddings
    dimension = int(embeddings.shape[1]) if embeddings is not None and embeddings.size else 0
    identity = analyzer._embedding_space_identity
    assert identity is not None
    manifest = _calibration_manifest(
        profile=resolve_model_profile(model),
        resolved_revision=revision,
        mode="hybrid_gates",
        semantic_task=DEFAULT_CHECK_SEMANTIC_TASK,
        requested_device=args.device,
        identity=identity,
        dimension=dimension,
        min_statements=args.min_statements,
        batch_size=args.batch_size,
        languages=config.languages,
        corpus_path=spec.corpus_path,
        labels_path=spec.labels_path,
        traditional_config=config,
    )
    manifest["semantic_gate"] = {
        "language": spec.language,
        "value": semantic_gate,
        "source": "explicit" if args.semantic_gate is not None else "profile",
    }
    return CorpusRun(
        spec=spec,
        semantic_gate=semantic_gate,
        manifest=manifest,
        units=len(result.units),
        traditional_duplicates=list(result.traditional_duplicates),
        semantic_duplicates=list(result.semantic_duplicates),
        positive_pairs=build_positive_pairs(result.units, labels),
    )


def _row_payload(row: SweepRow) -> dict[str, Any]:
    """Serialize one row with its configuration expanded.

    :param SweepRow row: Row to serialize.
    :return dict[str, Any]: JSON-safe mapping.
    """
    return {**asdict(row), "config": asdict(row.config)}


def _optional_row(row: SweepRow | None) -> dict[str, Any] | None:
    """Serialize an optional selected row.

    :param SweepRow | None row: Row or ``None``.
    :return dict[str, Any] | None: Serialized row or ``None``.
    """
    return _row_payload(row) if row is not None else None


def _print_rows(title: str, rows: list[SweepRow], *, top_n: int) -> None:
    print(f"\n{title}\n")
    for idx, row in enumerate(rows[:top_n], start=1):
        high = f"{row.config.high_gate:.2f}" if row.config.high_gate is not None else "off"
        print(
            f"{idx:02d}. visible precision={row.precision:.3f} recall={row.recall:.3f} "
            f"f1={row.f1:.3f} (published precision={row.published_precision:.3f} "
            f"recall={row.published_recall:.3f}) tp={row.tp} fp={row.fp} fn={row.fn} "
            f"review tp={row.review_tp} fp={row.review_fp} "
            f"weak_id_jaccard_min={row.config.weak_identifier_jaccard_min:.2f} "
            f"statement_ratio_min={row.config.statement_ratio_min:.2f} high_gate={high}"
        )


def _describe_selected(label: str, row: SweepRow | None) -> None:
    if row is None:
        print(f"{label}: no feasible split (every row loses too much recall or precision)")
        return
    high = f"{row.config.high_gate:.2f}" if row.config.high_gate is not None else "off"
    print(
        f"{label}: weak_id_jaccard_min={row.config.weak_identifier_jaccard_min:.2f} "
        f"statement_ratio_min={row.config.statement_ratio_min:.2f} high_gate={high} -> "
        f"visible precision={row.precision:.3f} recall={row.recall:.3f} "
        f"(published precision={row.published_precision:.3f} recall={row.published_recall:.3f})"
    )


def _sweep_model(
    model: str,
    corpora: list[CorpusSpec],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run both stages for one model over every corpus.

    :param str model: Model key or id.
    :param list[CorpusSpec] corpora: Corpora to embed and score.
    :param argparse.Namespace args: Parsed command-line options.
    :return dict[str, Any]: Report entry for the model.
    """
    revision = _require_immutable_revision(model, args.model_revision)
    profile = resolve_model_profile(model)
    runs: list[CorpusRun] = []
    for spec in corpora:
        gate = (
            args.semantic_gate
            if args.semantic_gate is not None
            else profile.semantic_threshold_for_language(spec.language)
        )
        print(f"\n[{model}] embedding {spec.name} at gate {gate:.2f} ...")
        runs.append(
            _run_corpus(spec, model=model, revision=revision, semantic_gate=gate, args=args)
        )

    # Stage 1: corroboration constants with promotion disabled, one shared grid.
    stage1_rows: dict[str, list[SweepRow]] = {}
    baseline: dict[str, float] = {
        "weak_min": profile.hybrid_weak_identifier_jaccard_min,
        "ratio_min": profile.hybrid_statement_ratio_min,
    }
    for run in runs:
        grid = [
            GateConfig(run.semantic_gate, weak, ratio)
            for weak, ratio in itertools.product(args.weak_jaccard_grid, args.statement_ratio_grid)
        ]
        stage1_rows[run.spec.name] = _run_sweep(
            traditional_duplicates=run.traditional_duplicates,
            semantic_duplicates=run.semantic_duplicates,
            positive_pairs=run.positive_pairs,
            traditional_threshold=args.traditional_threshold,
            grid=grid,
        )
    pooled = pool_rows(stage1_rows)
    pooled_selected = select_pooled_row(
        pooled, stage1_rows, recall_retention_min=args.recall_retention_min
    )
    _describe_selected(f"[{model}] stage 1 pooled selection", pooled_selected)
    _print_rows(f"[{model}] stage 1 pooled rows", pooled, top_n=args.top_n)

    # Stage 2: per-corpus promotion gate at the stage-1 constants.
    stage2: dict[str, Any] | None = None
    if pooled_selected is None:
        print(f"[{model}] stage 2: not run because stage 1 has no feasible pooled selection")
    else:
        constants = (
            pooled_selected.config.weak_identifier_jaccard_min,
            pooled_selected.config.statement_ratio_min,
        )
        stage2_corpora: dict[str, dict[str, Any]] = {}
        for run in runs:
            grid = [
                GateConfig(run.semantic_gate, constants[0], constants[1], high)
                for high in _high_gate_grid(
                    run.semantic_gate, args.high_gate_stop, args.high_gate_step
                )
            ]
            rows = _run_sweep(
                traditional_duplicates=run.traditional_duplicates,
                semantic_duplicates=run.semantic_duplicates,
                positive_pairs=run.positive_pairs,
                traditional_threshold=args.traditional_threshold,
                grid=grid,
            )
            selected = select_visible_row(rows, recall_retention_min=args.recall_retention_min)
            _describe_selected(f"[{model}] stage 2 {run.spec.name}", selected)
            stage2_corpora[run.spec.name] = {
                "semantic_gate": run.semantic_gate,
                "selected": _optional_row(selected),
                "rows": [_row_payload(row) for row in rows],
            }
        stage2 = {
            "constants": {
                "weak_identifier_jaccard_min": constants[0],
                "statement_ratio_min": constants[1],
            },
            "corpora": stage2_corpora,
        }

    return {
        "model_key": model,
        "canonical_name": profile.canonical_name,
        "resolved_revision": revision,
        # The profile's shipped split; tests compare it against the selections below.
        "baseline_defaults": {
            **baseline,
            "high_gates": dict(profile.language_high_confidence_thresholds),
        },
        "corpora": {
            run.spec.name: {
                "language": run.spec.language,
                "corpus_path": str(run.spec.corpus_path),
                "labels_path": str(run.spec.labels_path),
                "units": run.units,
                "raw_candidates": {
                    "traditional_duplicates": len(run.traditional_duplicates),
                    "semantic_duplicates": len(run.semantic_duplicates),
                },
                "calibration": run.manifest,
            }
            for run in runs
        },
        "stage1": {
            "corpora": {
                name: {
                    "selected": _optional_row(
                        select_visible_row(rows, recall_retention_min=args.recall_retention_min)
                    ),
                    "rows": [_row_payload(row) for row in rows],
                }
                for name, rows in stage1_rows.items()
            },
            "pooled": {
                "selected": _optional_row(pooled_selected),
                "rows": [_row_payload(row) for row in pooled],
            },
        },
        "stage2": stage2,
    }


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the hybrid tier split (corroboration constants and similarity promotion "
            "gate) that decides which semantic pairs codedupes shows by default. "
            "Without --corpus-root, specify exactly one --language for the single corpus."
        )
    )
    add_common_sweep_arguments(
        parser,
        language_help=(
            "Single-corpus language; required exactly once without --corpus-root. "
            "Use --languages with --corpus-root."
        ),
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help=(
            "Polyglot calibration root (<root>/<language>/ and <root>/labels/<language>.json); "
            "sweeps every --languages entry. Overrides --corpus-path/--labels-path."
        ),
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(POLYGLOT_LANGUAGES),
        help="Languages to sweep under --corpus-root.",
    )
    parser.add_argument(
        "--semantic-gate",
        type=float,
        default=None,
        help=(
            "Flat admission gate to collect semantic candidates at "
            "(default: each corpus language's shipped profile gate)."
        ),
    )
    parser.add_argument(
        "--traditional-threshold",
        type=float,
        default=DEFAULT_TRADITIONAL_THRESHOLD,
        help="Traditional jaccard threshold used by hybrid synthesis.",
    )
    parser.add_argument(
        "--weak-jaccard-grid",
        type=_parse_csv_floats,
        default=DEFAULT_WEAK_JACCARD_GRID,
        help="Comma-separated weak identifier jaccard minimum values to sweep.",
    )
    parser.add_argument(
        "--statement-ratio-grid",
        type=_parse_csv_floats,
        default=DEFAULT_STATEMENT_RATIO_GRID,
        help="Comma-separated statement count ratio minimum values to sweep.",
    )
    parser.add_argument(
        "--high-gate-stop",
        type=float,
        default=DEFAULT_HIGH_GATE_STOP,
        help="Inclusive upper bound of the promotion-gate grid (starts at the admission gate).",
    )
    parser.add_argument(
        "--high-gate-step",
        type=float,
        default=DEFAULT_HIGH_GATE_STEP,
        help="Promotion-gate grid step.",
    )
    parser.add_argument(
        "--recall-retention-min",
        type=float,
        default=DEFAULT_RECALL_RETENTION_MIN,
        help="Minimum visible recall as a fraction of all-published recall, per corpus.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[profile.key for profile in list_supported_models()],
        help="Model keys or IDs to sweep. Defaults to all built-in profiles.",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help=(
            "Immutable 40-character commit to calibrate against (single --models entry only). "
            "Defaults to each profile's pinned default_revision."
        ),
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
    if args.model_revision is not None and len(args.models) != 1:
        parser.error("--model-revision applies to exactly one --models entry.")
    if not 0.0 < args.recall_retention_min <= 1.0:
        parser.error("--recall-retention-min must be in (0, 1].")
    if not math.isfinite(args.high_gate_step) or args.high_gate_step <= 0:
        parser.error("--high-gate-step must be positive.")
    if not math.isfinite(args.high_gate_stop) or not 0.0 <= args.high_gate_stop <= 1.0:
        parser.error("--high-gate-stop must be finite and in [0.0, 1.0].")
    if args.semantic_gate is not None and (
        not math.isfinite(args.semantic_gate) or not 0.0 <= args.semantic_gate <= 1.0
    ):
        parser.error("--semantic-gate must be finite and in [0.0, 1.0].")

    if args.corpus_root is not None:
        try:
            languages = normalize_languages(args.languages)
        except ValueError as exc:
            parser.error(str(exc))
        assert languages is not None
        corpora = [
            CorpusSpec(
                name=language,
                corpus_path=args.corpus_root / language,
                labels_path=args.corpus_root / "labels" / f"{language}.json",
                language=language,
            )
            for language in languages
        ]
    else:
        if not args.language or len(args.language) != 1:
            parser.error(
                "--language must be specified exactly once without --corpus-root "
                "(for example, --language python)."
            )
        try:
            languages = normalize_languages(args.language)
        except ValueError as exc:
            parser.error(str(exc))
        assert languages is not None
        language = languages[0]
        corpora = [
            CorpusSpec(
                name=language,
                corpus_path=args.corpus_path,
                labels_path=args.labels_path,
                language=language,
            )
        ]
    for spec in corpora:
        try:
            validate_labels_shape(json.loads(spec.labels_path.read_text()))
        except (OSError, ValueError) as exc:
            parser.error(f"{spec.labels_path}: {exc}")

    print("Hybrid tier-split sweep (visible subset = published minus semantic_review)")
    print(
        f"Selection: max visible precision, subject to visible recall >= "
        f"{args.recall_retention_min:.2f} x published recall and visible precision >= "
        "published precision in every corpus; ties prefer f1, then the stricter split."
    )
    results = [_sweep_model(model, corpora, args) for model in args.models]

    if args.json_out is not None:
        payload = {
            "output_policy": "hybrid_high_confidence",
            "selection_policy": {
                "objective": ["precision", "f1", "strictest"],
                "recall_retention_min": args.recall_retention_min,
                "precision_floor": "published",
                "stage1": "global corroboration constants, promotion disabled, pooled over corpora",
                "stage2": (
                    "per-language promotion gate at the stage-1 constants; null when stage 1 "
                    "has no feasible pooled selection"
                ),
            },
            "grid": {
                "weak_identifier_jaccard_min": args.weak_jaccard_grid,
                "statement_ratio_min": args.statement_ratio_grid,
                "high_gate_stop": args.high_gate_stop,
                "high_gate_step": args.high_gate_step,
            },
            "models": results,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote sweep report: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
