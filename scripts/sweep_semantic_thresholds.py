"""Sweep duplicate and search thresholds for built-in semantic model profiles.

Every sweep records a full calibration manifest (pinned model commit, embedding
pipeline schema and runtime identity, encode plan, the effective embedding-space
identity the analyzer actually produced (covering dtype and Metal math policy
even when an accelerator request fell back to CPU mid-run), dimension, candidate
policy, and corpus/label digests) so a selected threshold is always tied to a
reproducible model and pipeline identity. Calibration refuses to run when the
model cannot be pinned to an immutable 40-character commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import codedupes.analyzer as analyzer_module
from codedupes import __version__
from codedupes.analyzer import DEFAULT_SEMANTIC_UNIT_TYPES, AnalyzerConfig, CodeAnalyzer
from codedupes.constants import (
    DEFAULT_CHECK_SEMANTIC_TASK,
    DEFAULT_SEARCH_SEMANTIC_TASK,
    DEFAULT_TRADITIONAL_THRESHOLD,
)
from codedupes.pairs import ordered_pair_key
from codedupes.semantic import (
    EMBEDDING_PIPELINE_SCHEMA,
    EmbeddingSpaceIdentity,
    _embedding_runtime_fingerprint,
    get_semantic_runtime_versions,
    resolve_encode_plan,
)
from codedupes.semantic_profiles import (
    SemanticModelProfile,
    list_supported_models,
    resolve_model_profile,
)

try:
    from .sweep_common import (
        add_common_sweep_arguments,
        build_positive_pairs,
        metrics,
        rank_sweep_rows,
        resolve_label_unit,
    )
except ImportError:
    from sweep_common import (
        add_common_sweep_arguments,
        build_positive_pairs,
        metrics,
        rank_sweep_rows,
        resolve_label_unit,
    )

DUPLICATE_THRESHOLD_START = 0.70
DUPLICATE_THRESHOLD_STOP = 0.96
SEARCH_THRESHOLD_START = 0.20
SEARCH_THRESHOLD_STOP = 0.70
THRESHOLD_STEP = 0.02
SEARCH_SWEEP_FLOOR = 0.01


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
    """Sweep results and calibration identity for one model."""

    model_key: str
    canonical_name: str
    selected_threshold: float
    manifest: dict[str, Any]
    rows: list[SweepRow]


def _threshold_grid(start: float, stop: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 2))
        current += THRESHOLD_STEP
    return values


def _sha256_of_tree(root: Path) -> str:
    """Digest a fixture tree by sorted relative path and file contents.

    Every regular source file participates so non-Python corpora do not digest
    identically; caches and hidden files are excluded. For the historical
    all-Python corpus this matches the previous ``*.py``-only digest.
    """
    digest = hashlib.sha256()
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(root).as_posix()
        if "__pycache__" in relative or relative.endswith(".pyc"):
            continue
        if any(part.startswith(".") for part in file_path.relative_to(root).parts):
            continue
        digest.update(relative.encode())
        digest.update(b"\x00")
        digest.update(file_path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_immutable_revision(model_name: str, explicit_revision: str | None) -> str:
    """Resolve the pinned commit for calibration, refusing mutable identities."""
    profile = resolve_model_profile(model_name)
    revision = explicit_revision or profile.default_revision
    is_commit = (
        revision is not None
        and len(revision) == 40
        and all(character in "0123456789abcdefABCDEF" for character in revision)
    )
    if not is_commit:
        raise SystemExit(
            f"Refusing to calibrate {model_name!r}: no immutable 40-character commit. "
            "Pass --model-revision <commit> or pin the profile's default_revision. "
            f"(resolved revision: {revision!r})"
        )
    assert revision is not None
    return revision


def _calibration_manifest(
    *,
    profile: SemanticModelProfile,
    resolved_revision: str,
    mode: str,
    semantic_task: str,
    requested_device: str,
    identity: EmbeddingSpaceIdentity,
    dimension: int,
    min_statements: int,
    batch_size: int,
    corpus_path: Path,
    labels_path: Path,
) -> dict[str, Any]:
    """Assemble the reproducible identity under which one threshold was swept.

    ``identity`` is the analyzer's effective embedding-space identity, recorded
    verbatim: it reflects the policy that produced the swept matrix (dtype and
    Metal math policy included) even when the requested accelerator fell back
    and the run restarted on CPU, so thresholds are never labeled with a device
    or dtype that did not produce them.
    """
    code_plan = resolve_encode_plan(profile.canonical_name, "code", None, semantic_task)
    manifest: dict[str, Any] = {
        "model": profile.canonical_name,
        "resolved_revision": resolved_revision,
        "embedding_pipeline_schema": EMBEDDING_PIPELINE_SCHEMA,
        "embedding_runtime_identity": _embedding_runtime_fingerprint(),
        "runtime_versions": get_semantic_runtime_versions(),
        "codedupes_version": __version__,
        "mode": mode,
        "semantic_task": semantic_task,
        "encode_plan": {"code": {"route": code_plan.route, "prompt": code_plan.prompt}},
        "requested_device": requested_device,
        "embedding_space": asdict(identity),
        "dimension": dimension,
        "normalized": True,
        "candidate_policy": {
            "unit_types": list(DEFAULT_SEMANTIC_UNIT_TYPES),
            "min_recursive_statements": min_statements,
            "include_private": True,
        },
        "batch_size": batch_size,
        "corpus_path": str(corpus_path),
        "corpus_sha256": _sha256_of_tree(corpus_path),
        "labels_path": str(labels_path),
        "labels_sha256": _sha256_of_file(labels_path),
    }
    if mode == "search":
        query_plan = resolve_encode_plan(profile.canonical_name, "query", None, semantic_task)
        manifest["encode_plan"]["query"] = {
            "route": query_plan.route,
            "prompt": query_plan.prompt,
        }
    return manifest


def _evaluate_thresholds(
    scored_pairs: list[tuple[tuple[str, str], float]],
    positive_pairs: set[tuple[str, str]],
    *,
    thresholds: list[float],
) -> list[SweepRow]:
    rows: list[SweepRow] = []
    for threshold in thresholds:
        predicted_pairs = {pair for pair, score in scored_pairs if score >= threshold}
        tp, fp, fn, precision, recall, f1 = metrics(predicted_pairs, positive_pairs)
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
    # Ties prefer the looser threshold: recall over precision at equal F1.
    rank_sweep_rows(rows, extra_key=lambda row: (-row.threshold,))
    return rows


def _analyzer_config(
    *,
    model_name: str,
    revision: str,
    semantic_task: str,
    semantic_threshold: float,
    min_statements: int,
    batch_size: int,
    device: str,
    languages: tuple[str, ...] | None = None,
    run_traditional: bool = False,
) -> AnalyzerConfig:
    return AnalyzerConfig(
        run_traditional=run_traditional,
        run_semantic=True,
        run_unused=False,
        include_private=True,
        languages=languages,
        model_name=model_name,
        model_revision=revision,
        semantic_task=semantic_task,
        semantic_threshold=semantic_threshold,
        min_semantic_statements=min_statements,
        batch_size=batch_size,
        device=device,
    )


def _run_duplicate_sweep(
    *,
    model_name: str,
    revision: str,
    languages: tuple[str, ...] | None = None,
    corpus_path: Path,
    labels_path: Path,
    labels: dict[str, Any],
    min_statements: int,
    batch_size: int,
    device: str,
    duplicate_start: float = DUPLICATE_THRESHOLD_START,
    duplicate_stop: float = DUPLICATE_THRESHOLD_STOP,
) -> ModelSweep:
    profile = resolve_model_profile(model_name)
    analyzer = CodeAnalyzer(
        _analyzer_config(
            model_name=model_name,
            revision=revision,
            semantic_task=DEFAULT_CHECK_SEMANTIC_TASK,
            semantic_threshold=duplicate_start,
            min_statements=min_statements,
            batch_size=batch_size,
            device=device,
            languages=languages,
            run_traditional=True,
        )
    )
    result = analyzer.analyze(corpus_path)
    embeddings = analyzer._embeddings
    dimension = int(embeddings.shape[1]) if embeddings is not None and embeddings.size else 0
    identity = analyzer._embedding_space_identity
    assert identity is not None

    positive_pairs = build_positive_pairs(result.units, labels)
    embedded_uids = {unit.uid for unit in analyzer._semantic_units or []}
    scoreable_pairs = {
        pair for pair in positive_pairs if pair[0] in embedded_uids and pair[1] in embedded_uids
    }
    excluded_pairs = len(positive_pairs) - len(scoreable_pairs)
    if excluded_pairs:
        print(
            f"Excluding {excluded_pairs} labeled pairs outside the semantic candidate pool "
            "(class-level or below-min-statement units); the semantic tier can never "
            "predict them."
        )
    thresholds = _threshold_grid(duplicate_start, duplicate_stop)
    rows: list[SweepRow] = []
    predicted_by_threshold: dict[float, set[tuple[str, str]]] = {}
    for threshold in thresholds:
        gated_semantic = [
            duplicate
            for duplicate in result.semantic_duplicates
            if duplicate.similarity >= threshold
        ]
        hybrid, _ = analyzer_module._synthesize_hybrid_duplicates(
            result.traditional_duplicates,
            gated_semantic,
            jaccard_threshold=DEFAULT_TRADITIONAL_THRESHOLD,
        )
        predicted_pairs = {ordered_pair_key(item.unit_a, item.unit_b) for item in hybrid}
        predicted_by_threshold[threshold] = predicted_pairs
        tp, fp, fn, precision, recall, f1 = metrics(predicted_pairs, positive_pairs)
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
    rank_sweep_rows(rows, extra_key=lambda row: (-row.threshold,))
    selected = rows[0]

    manifest = _calibration_manifest(
        profile=profile,
        resolved_revision=revision,
        mode="duplicate",
        semantic_task=DEFAULT_CHECK_SEMANTIC_TASK,
        requested_device=device,
        identity=identity,
        dimension=dimension,
        min_statements=min_statements,
        batch_size=batch_size,
        corpus_path=corpus_path,
        labels_path=labels_path,
    )
    manifest["output_policy"] = "hybrid_duplicates"
    manifest["candidate_coverage"] = {
        "labeled_positive_pairs": len(positive_pairs),
        "scoreable_positive_pairs": len(scoreable_pairs),
        "excluded_positive_pairs": excluded_pairs,
        "recall_ceiling": (len(scoreable_pairs) / len(positive_pairs) if positive_pairs else 0.0),
    }
    selected_pairs = predicted_by_threshold[selected.threshold]
    manifest["selected_category_recall"] = {}
    for category, groups in labels.get("categories", {}).items():
        category_pairs = build_positive_pairs(result.units, {"positive_groups": groups})
        detected = len(selected_pairs & category_pairs)
        manifest["selected_category_recall"][category] = {
            "labeled": len(category_pairs),
            "detected": detected,
            "recall": detected / len(category_pairs) if category_pairs else 0.0,
        }

    return ModelSweep(
        model_key=model_name,
        canonical_name=profile.canonical_name,
        selected_threshold=selected.threshold,
        manifest=manifest,
        rows=rows,
    )


def _run_search_sweep(
    *,
    model_name: str,
    revision: str,
    languages: tuple[str, ...] | None = None,
    corpus_path: Path,
    probes_path: Path,
    probes: list[dict[str, Any]],
    min_statements: int,
    batch_size: int,
    device: str,
) -> ModelSweep:
    profile = resolve_model_profile(model_name)
    analyzer = CodeAnalyzer(
        _analyzer_config(
            model_name=model_name,
            revision=revision,
            semantic_task=DEFAULT_SEARCH_SEMANTIC_TASK,
            semantic_threshold=SEARCH_SWEEP_FLOOR,
            min_statements=min_statements,
            batch_size=batch_size,
            device=device,
            languages=languages,
        )
    )
    indexed = analyzer.index(corpus_path)
    embeddings = analyzer._embeddings
    dimension = int(embeddings.shape[1]) if embeddings is not None and embeddings.size else 0
    identity = analyzer._embedding_space_identity
    assert identity is not None
    assert analyzer._semantic_units is not None
    assert analyzer._units is not None

    scored_pairs: list[tuple[tuple[str, str], float]] = []
    positive_pairs: set[tuple[str, str]] = set()
    for probe_index, probe in enumerate(probes):
        query = probe["query"]
        query_key = f"probe-{probe_index}"
        expected_units = {
            resolve_label_unit(analyzer._units, spec).uid for spec in probe["expected"]
        }
        positive_pairs.update((query_key, uid) for uid in expected_units)
        for unit, score in analyzer.search(query, top_k=indexed):
            scored_pairs.append(((query_key, unit.uid), score))

    rows = _evaluate_thresholds(
        scored_pairs,
        positive_pairs,
        thresholds=_threshold_grid(SEARCH_THRESHOLD_START, SEARCH_THRESHOLD_STOP),
    )
    selected = rows[0]

    manifest = _calibration_manifest(
        profile=profile,
        resolved_revision=revision,
        mode="search",
        semantic_task=DEFAULT_SEARCH_SEMANTIC_TASK,
        requested_device=device,
        identity=identity,
        dimension=dimension,
        min_statements=min_statements,
        batch_size=batch_size,
        corpus_path=corpus_path,
        labels_path=probes_path,
    )
    manifest["probe_count"] = len(probes)
    embedded_uids = {unit.uid for unit in analyzer._semantic_units}
    scoreable_targets = {pair for pair in positive_pairs if pair[1] in embedded_uids}
    manifest["candidate_coverage"] = {
        "labeled_positive_targets": len(positive_pairs),
        "scoreable_positive_targets": len(scoreable_targets),
        "excluded_positive_targets": len(positive_pairs) - len(scoreable_targets),
        "recall_ceiling": (len(scoreable_targets) / len(positive_pairs) if positive_pairs else 0.0),
    }

    return ModelSweep(
        model_key=model_name,
        canonical_name=profile.canonical_name,
        selected_threshold=selected.threshold,
        manifest=manifest,
        rows=rows,
    )


def _print_sweep(model_sweep: ModelSweep, top_n: int) -> None:
    print(f"\nModel: {model_sweep.model_key} ({model_sweep.manifest['mode']})")
    print(f"Revision: {model_sweep.manifest['resolved_revision']}")
    print(f"Selected threshold: {model_sweep.selected_threshold:.2f}")
    print("Top rows:")
    for idx, row in enumerate(model_sweep.rows[:top_n], start=1):
        print(
            f"  {idx:02d}. threshold={row.threshold:.2f} f1={row.f1:.3f} "
            f"precision={row.precision:.3f} recall={row.recall:.3f} "
            f"tp={row.tp} fp={row.fp} fn={row.fn} pred={row.predicted}"
        )


def _report_payload(results: list[ModelSweep], grid: list[float]) -> dict[str, Any]:
    return {
        "grid": grid,
        "models": [
            {
                "model_key": item.model_key,
                "canonical_name": item.canonical_name,
                "selected_threshold": item.selected_threshold,
                "selected_metrics": asdict(item.rows[0]),
                "calibration": item.manifest,
                "rows": [asdict(row) for row in item.rows],
            }
            for item in results
        ],
    }


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Sweep duplicate and search thresholds across built-in model profiles."
    )
    add_common_sweep_arguments(parser)
    parser.add_argument(
        "--search-probes-path",
        type=Path,
        default=Path("test_fixtures/hybrid_tuning/search_probes.json"),
        help="Path to search_probes.json with labeled query probes.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[profile.key for profile in list_supported_models()],
        help="Model keys or IDs to sweep. Defaults to all built-in profiles.",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Immutable 40-character commit to calibrate against. Defaults to the "
        "profile's pinned default_revision; calibration refuses to run without one.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Embedding device for the sweep. Defaults to cpu for reproducible float32.",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Only sweep duplicate thresholds.",
    )
    parser.add_argument(
        "--duplicate-start",
        type=float,
        default=DUPLICATE_THRESHOLD_START,
        help="Duplicate-threshold grid floor; also the analyzer floor, so pairs "
        "below it are never scored. Lower it for corpora whose positive pairs "
        f"sit below the default {DUPLICATE_THRESHOLD_START:.2f}.",
    )
    parser.add_argument(
        "--duplicate-stop",
        type=float,
        default=DUPLICATE_THRESHOLD_STOP,
        help="Duplicate-threshold grid ceiling.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("test_fixtures/hybrid_tuning/semantic_threshold_report.json"),
        help="Path to write the duplicate-threshold sweep report JSON.",
    )
    parser.add_argument(
        "--search-json-out",
        type=Path,
        default=Path("test_fixtures/hybrid_tuning/search_threshold_report.json"),
        help="Path to write the search-threshold sweep report JSON.",
    )
    args = parser.parse_args()

    if args.duplicate_start > args.duplicate_stop:
        parser.error(
            f"--duplicate-start {args.duplicate_start} must not exceed "
            f"--duplicate-stop {args.duplicate_stop}; the sweep grid would be empty."
        )

    labels = json.loads(args.labels_path.read_text())
    probes: list[dict[str, Any]] = []
    if not args.skip_search:
        probes = json.loads(args.search_probes_path.read_text())["probes"]

    duplicate_results: list[ModelSweep] = []
    search_results: list[ModelSweep] = []
    for model_name in args.models:
        revision = _require_immutable_revision(model_name, args.model_revision)
        duplicate_results.append(
            _run_duplicate_sweep(
                model_name=model_name,
                revision=revision,
                languages=tuple(args.language) if args.language else None,
                corpus_path=args.corpus_path,
                labels_path=args.labels_path,
                labels=labels,
                min_statements=args.min_statements,
                batch_size=args.batch_size,
                device=args.device,
                duplicate_start=args.duplicate_start,
                duplicate_stop=args.duplicate_stop,
            )
        )
        if not args.skip_search:
            search_results.append(
                _run_search_sweep(
                    model_name=model_name,
                    revision=revision,
                    languages=tuple(args.language) if args.language else None,
                    corpus_path=args.corpus_path,
                    probes_path=args.search_probes_path,
                    probes=probes,
                    min_statements=args.min_statements,
                    batch_size=args.batch_size,
                    device=args.device,
                )
            )

    print("Semantic threshold sweep (synthetic corpus guardrail)")
    print(f"Corpus: {args.corpus_path}")
    print(f"Labels: {args.labels_path}")

    for item in duplicate_results + search_results:
        _print_sweep(item, top_n=args.top_n)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(
            _report_payload(
                duplicate_results,
                _threshold_grid(args.duplicate_start, args.duplicate_stop),
            ),
            indent=2,
        )
    )
    print(f"\nWrote duplicate sweep report: {args.json_out}")

    if not args.skip_search:
        args.search_json_out.parent.mkdir(parents=True, exist_ok=True)
        args.search_json_out.write_text(
            json.dumps(
                _report_payload(
                    search_results,
                    _threshold_grid(SEARCH_THRESHOLD_START, SEARCH_THRESHOLD_STOP),
                ),
                indent=2,
            )
        )
        print(f"Wrote search sweep report: {args.search_json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
