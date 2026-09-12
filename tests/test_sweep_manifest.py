"""Regression coverage for sweep-script manifests and row ranking."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from codedupes.analyzer import CodeAnalyzer
from codedupes.constants import DEFAULT_CHECK_SEMANTIC_TASK, DEFAULT_MIN_SEMANTIC_STATEMENTS
from codedupes.models import HYBRID_TIERS, CodeUnit, CodeUnitType, DuplicatePair
from codedupes.semantic import EmbeddingSpaceIdentity
from codedupes.semantic_profiles import resolve_model_profile
from scripts import sweep_hybrid_gates
from scripts.report_calibration_distributions import _analyze_language
from scripts.sweep_common import (
    add_common_sweep_arguments,
    validate_labels_shape,
    validate_probes_shape,
)
from scripts.sweep_hybrid_gates import (
    GateConfig,
    SweepRow,
    _high_gate_grid,
    is_feasible,
    pool_rows,
    select_pooled_row,
    select_visible_row,
)
from scripts.sweep_hybrid_gates import _run_sweep as _run_hybrid_gate_sweep
from scripts.sweep_hybrid_gates import main as _hybrid_gates_main
from scripts.sweep_semantic_thresholds import (
    THRESHOLD_STEP,
    DuplicateSweepRow,
    _calibration_manifest,
    _grid_edge,
    _report_payload,
    _require_immutable_revision,
    _run_duplicate_sweep,
    _run_search_sweep,
    _threshold_grid,
)
from scripts.sweep_semantic_thresholds import main as _semantic_sweep_main

PINNED_COMMIT = "a" * 40


def _unit(name: str, file_path: Path, lineno: int) -> CodeUnit:
    return CodeUnit(
        name=name,
        qualified_name=name,
        unit_type=CodeUnitType.FUNCTION,
        file_path=file_path,
        lineno=lineno,
        end_lineno=lineno + 1,
        source=f"def {name}():\n    return {lineno}\n",
    )


def _patch_analyze(
    monkeypatch: pytest.MonkeyPatch,
    *,
    units: list[CodeUnit],
    identity: EmbeddingSpaceIdentity,
    embeddings: np.ndarray,
    semantic_units: list[CodeUnit] | None = None,
    traditional_duplicates: list[DuplicatePair] | None = None,
    semantic_duplicates: list[DuplicatePair] | None = None,
) -> None:
    """Replace ``CodeAnalyzer.analyze`` with a stub that plants one embedding state.

    ``semantic_units`` defaults to ``units``; pass ``[]`` for a corpus whose units
    were dropped from the semantic matrix after traditional analysis.
    """
    matrix_units = units if semantic_units is None else semantic_units

    def fake_analyze(self: CodeAnalyzer, path: Path) -> SimpleNamespace:
        self._embeddings = embeddings
        self._embedding_space_identity = identity
        self._semantic_units = matrix_units
        return SimpleNamespace(
            units=units,
            traditional_duplicates=list(traditional_duplicates or []),
            semantic_duplicates=list(semantic_duplicates or []),
        )

    monkeypatch.setattr(CodeAnalyzer, "analyze", fake_analyze)


def test_manifest_records_effective_embedding_space_not_the_request(
    tmp_path: Path, monkeypatch
) -> None:
    """A sweep whose accelerator request fell back to CPU must record the CPU identity.

    The analyzer's effective ``EmbeddingSpaceIdentity`` already reflects an
    OOM/invalid-output restart on CPU (and drops an active fast-math policy
    with it); the manifest must copy that identity verbatim instead of
    re-deriving device and dtype from the request, or CPU-float32 calibration
    results get labeled as accelerator results.
    """
    corpus_path = tmp_path / "corpus"
    corpus_path.mkdir()
    corpus_file = corpus_path / "alpha.py"
    corpus_file.write_text("def first():\n    return 1\n\n\ndef second():\n    return 2\n")
    labels = {"positive_groups": [["alpha.py::first", "alpha.py::second"]]}
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))

    profile = resolve_model_profile("gte-modernbert-base")
    effective_identity = EmbeddingSpaceIdentity(
        model_name=profile.canonical_name,
        resolved_revision=PINNED_COMMIT,
        runtime_variant="cpu-faithful-after-fallback",
    )
    units = [_unit("first", corpus_file, 1), _unit("second", corpus_file, 5)]

    _patch_analyze(
        monkeypatch,
        units=units,
        identity=effective_identity,
        embeddings=np.zeros((2, 4), dtype=np.float32),
    )
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")

    sweep = _run_duplicate_sweep(
        model_name="gte-modernbert-base",
        revision=PINNED_COMMIT,
        corpus_path=corpus_path,
        labels_path=labels_path,
        labels=labels,
        min_statements=0,
        batch_size=4,
        device="mps",
    )

    manifest = sweep.manifest
    assert manifest["requested_device"] == "mps"
    assert manifest["embedding_space"] == {
        "model_name": profile.canonical_name,
        "resolved_revision": PINNED_COMMIT,
        "runtime_variant": "cpu-faithful-after-fallback",
        # Provenance metadata: None for the pinned commits sweeps require.
        "source_commit": None,
        "search_document": "source",
    }
    assert "device" not in manifest
    assert "dtype_variant" not in manifest
    assert manifest["output_policy"] == "hybrid_duplicates"
    assert manifest["traditional_candidate_policy"] == {
        "jaccard_threshold": 0.85,
        "filter_tiny_traditional": True,
        "tiny_unit_statement_cutoff": 3,
    }
    assert manifest["visible_policy"] == {
        "excluded_tiers": ["semantic_review"],
        "metrics_field": "visible",
    }
    assert manifest["corroboration"] == {
        "weak_identifier_jaccard_min": profile.hybrid_weak_identifier_jaccard_min,
        "statement_ratio_min": profile.hybrid_statement_ratio_min,
        "high_confidence_gates": {
            "python": profile.high_confidence_threshold_for_language("python")
        },
    }
    # Zero candidates tie every row at f1=0, so the loosest-tie policy selects
    # the grid floor - a boundary selection the manifest must record.
    assert manifest["selected_at_grid_edge"] == "start"
    assert manifest["candidate_coverage"] == {
        "labeled_positive_pairs": 1,
        "embedded_positive_pairs": 1,
        "scoreable_positive_pairs": 1,
        "traditional_recovered_pairs": 0,
        "reachable_positive_pairs": 1,
        "unreachable_positive_pairs": 0,
        "recall_ceiling": 1.0,
    }
    row = sweep.rows[0]
    assert isinstance(row, DuplicateSweepRow)
    assert set(row.tiers) == set(HYBRID_TIERS)
    assert all(counts.predicted == 0 and counts.precision is None for counts in row.tiers.values())
    assert (row.visible.predicted, row.visible.tp, row.visible.fp, row.visible.fn) == (0, 0, 0, 1)


def test_manifest_recall_ceiling_includes_traditional_overflow_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    corpus_path = tmp_path / "corpus"
    corpus_path.mkdir()
    first_path = corpus_path / "alpha.py"
    second_path = corpus_path / "beta.py"
    first_path.write_text("def first():\n    return 1\n")
    second_path.write_text("def second():\n    return 1\n")
    labels = {"positive_groups": [["alpha.py::first", "beta.py::second"]]}
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))

    profile = resolve_model_profile("gte-modernbert-base")
    identity = EmbeddingSpaceIdentity(
        model_name=profile.canonical_name,
        resolved_revision=PINNED_COMMIT,
        runtime_variant="cpu-faithful",
    )
    first = _unit("first", first_path, 1)
    second = _unit("second", second_path, 1)
    first.structural_hash = "shared-exact-fingerprint"
    second.structural_hash = "shared-exact-fingerprint"
    traditional = DuplicatePair(first, second, 1.0, "ast_hash")

    # Both endpoints are embedded, but the analyzer suppresses exact-hash pairs
    # from semantic output. The traditional result must still make it reachable.
    _patch_analyze(
        monkeypatch,
        units=[first, second],
        identity=identity,
        embeddings=np.zeros((2, 4), dtype=np.float32),
        traditional_duplicates=[traditional],
    )

    sweep = _run_duplicate_sweep(
        model_name="gte-modernbert-base",
        revision=PINNED_COMMIT,
        corpus_path=corpus_path,
        labels_path=labels_path,
        labels=labels,
        min_statements=0,
        batch_size=4,
        device="cpu",
    )

    # The recovered pair is counted as recovered, not "excluded": the old
    # field names read "1 of 1 excluded, ceiling 1.0" for exactly this case.
    assert sweep.manifest["candidate_coverage"] == {
        "labeled_positive_pairs": 1,
        "embedded_positive_pairs": 1,
        "scoreable_positive_pairs": 0,
        "traditional_recovered_pairs": 1,
        "reachable_positive_pairs": 1,
        "unreachable_positive_pairs": 0,
        "recall_ceiling": 1.0,
    }
    assert {row.recall for row in sweep.rows} == {1.0}


@pytest.mark.parametrize("ineligible_reason", ["cross-language", "overlap", "kind"])
def test_manifest_excludes_pairs_the_semantic_scanner_will_not_compare(
    tmp_path: Path, monkeypatch, ineligible_reason: str
) -> None:
    corpus_path = tmp_path / "corpus"
    corpus_path.mkdir()
    first_path = corpus_path / "alpha.py"
    second_path = corpus_path / "beta.py"
    first_path.write_text("def first():\n    return 1\n")
    second_path.write_text("def second():\n    return 2\n")
    first = _unit("first", first_path, 1)
    second = _unit("second", second_path, 1)
    if ineligible_reason == "cross-language":
        second.language = "rust"
    elif ineligible_reason == "overlap":
        second.file_path = first.file_path
    else:
        second.unit_type = CodeUnitType.CLASS

    labels = {
        "positive_groups": [
            [
                "alpha.py::first",
                f"{second.file_path.name}::second",
            ]
        ]
    }
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))
    profile = resolve_model_profile("gte-modernbert-base")
    identity = EmbeddingSpaceIdentity(
        model_name=profile.canonical_name,
        resolved_revision=PINNED_COMMIT,
        runtime_variant="cpu-faithful",
    )
    _patch_analyze(
        monkeypatch,
        units=[first, second],
        identity=identity,
        embeddings=np.zeros((2, 4), dtype=np.float32),
    )

    sweep = _run_duplicate_sweep(
        model_name="gte-modernbert-base",
        revision=PINNED_COMMIT,
        corpus_path=corpus_path,
        labels_path=labels_path,
        labels=labels,
        min_statements=0,
        batch_size=4,
        device="cpu",
    )

    assert sweep.manifest["candidate_coverage"] == {
        "labeled_positive_pairs": 1,
        "embedded_positive_pairs": 1,
        "scoreable_positive_pairs": 0,
        "traditional_recovered_pairs": 0,
        "reachable_positive_pairs": 0,
        "unreachable_positive_pairs": 1,
        "recall_ceiling": 0.0,
    }


def test_duplicate_rows_split_published_pairs_by_tier(tmp_path: Path, monkeypatch) -> None:
    """Rows must attribute tp/fp to tiers and score the default-visible subset separately.

    One labeled exact pair and one unlabeled lopsided semantic pair (three
    statements against one, below the profile's statement-ratio floor, so it
    lands in ``semantic_review``) must produce a row whose published metrics
    count both, whose ``visible`` metrics count only the exact pair, and whose
    per-tier sums reproduce the published totals.
    """
    corpus_path = tmp_path / "corpus"
    corpus_path.mkdir()
    alpha_path = corpus_path / "alpha.py"
    beta_path = corpus_path / "beta.py"
    alpha_path.write_text(
        "def first(a, b):\n    c = a + b\n    d = c * 2\n    return d\n\n\n"
        "def second(a, b):\n    c = a + b\n    d = c * 2\n    return d\n"
    )
    beta_path.write_text(
        "def left(x, y):\n    z = x + y\n    w = z * 2\n    return w\n\n\ndef right(p, q):\n    return p - q\n"
    )
    labels = {
        "positive_groups": [["alpha.py::first", "alpha.py::second"]],
        "categories": {"exact": [["alpha.py::first", "alpha.py::second"]]},
    }
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels))

    profile = resolve_model_profile("gte-modernbert-base")
    identity = EmbeddingSpaceIdentity(
        model_name=profile.canonical_name,
        resolved_revision=PINNED_COMMIT,
        runtime_variant="cpu-faithful",
    )
    first = CodeUnit(
        name="first",
        qualified_name="first",
        unit_type=CodeUnitType.FUNCTION,
        file_path=alpha_path,
        lineno=1,
        end_lineno=4,
        source="def first(a, b):\n    c = a + b\n    d = c * 2\n    return d\n",
    )
    second = CodeUnit(
        name="second",
        qualified_name="second",
        unit_type=CodeUnitType.FUNCTION,
        file_path=alpha_path,
        lineno=7,
        end_lineno=10,
        source="def second(a, b):\n    c = a + b\n    d = c * 2\n    return d\n",
        start_byte=60,
    )
    left = CodeUnit(
        name="left",
        qualified_name="left",
        unit_type=CodeUnitType.FUNCTION,
        file_path=beta_path,
        lineno=1,
        end_lineno=4,
        source="def left(x, y):\n    z = x + y\n    w = z * 2\n    return w\n",
    )
    right = CodeUnit(
        name="right",
        qualified_name="right",
        unit_type=CodeUnitType.FUNCTION,
        file_path=beta_path,
        lineno=7,
        end_lineno=8,
        source="def right(p, q):\n    return p - q\n",
        start_byte=60,
    )
    units = [first, second, left, right]

    _patch_analyze(
        monkeypatch,
        units=units,
        identity=identity,
        embeddings=np.zeros((4, 4), dtype=np.float32),
        traditional_duplicates=[DuplicatePair(first, second, 1.0, "ast_hash")],
        semantic_duplicates=[DuplicatePair(left, right, 0.91, "semantic")],
    )

    sweep = _run_duplicate_sweep(
        model_name="gte-modernbert-base",
        revision=PINNED_COMMIT,
        corpus_path=corpus_path,
        labels_path=labels_path,
        labels=labels,
        min_statements=0,
        batch_size=4,
        device="cpu",
        duplicate_start=0.90,
        duplicate_stop=0.90,
    )

    row = sweep.rows[0]
    assert isinstance(row, DuplicateSweepRow)
    assert (row.predicted, row.tp, row.fp, row.fn) == (2, 1, 1, 0)
    assert row.tiers["exact"].tp == 1
    assert row.tiers["exact"].precision == 1.0
    assert row.tiers["semantic_review"].fp == 1
    assert row.tiers["semantic_review"].precision == 0.0
    assert sum(counts.predicted for counts in row.tiers.values()) == row.predicted
    assert sum(counts.tp for counts in row.tiers.values()) == row.tp
    assert sum(counts.fp for counts in row.tiers.values()) == row.fp
    assert sum(counts.positive_share for counts in row.tiers.values()) == pytest.approx(row.recall)
    assert (row.visible.predicted, row.visible.tp, row.visible.fp, row.visible.fn) == (1, 1, 0, 0)
    assert row.visible.precision == 1.0
    assert sweep.manifest["selected_category_recall"]["exact"] == {
        "labeled": 1,
        "detected": 1,
        "recall": 1.0,
        "visible_detected": 1,
        "visible_recall": 1.0,
    }

    payload = json.loads(json.dumps(_report_payload([sweep], [0.90])))
    serialized_row = payload["models"][0]["rows"][0]
    assert serialized_row["tiers"]["hybrid_confirmed"]["precision"] is None
    assert serialized_row["visible"]["precision"] == 1.0
    assert payload["models"][0]["selected_metrics"]["tiers"]["exact"]["tp"] == 1


def test_common_sweep_defaults_match_production_candidate_policy() -> None:
    parser = argparse.ArgumentParser()
    add_common_sweep_arguments(parser)

    assert parser.parse_args([]).min_statements == DEFAULT_MIN_SEMANTIC_STATEMENTS


def test_distribution_report_carries_the_sweep_calibration_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    """Distribution stats are cited as gate evidence, so they need the same identity block.

    Without it the recorded JSON is a bare ``{model: {language: stats}}`` map: no
    resolved revision, pipeline schema, embedding space, or corpus digest to tie
    the numbers to a reproducible run.
    """
    corpus_root = tmp_path / "corpus_root"
    language_path = corpus_root / "python"
    language_path.mkdir(parents=True)
    source_path = language_path / "alpha.py"
    source_path.write_text("def first():\n    return 1\n\n\ndef second():\n    return 2\n")
    labels_path = corpus_root / "labels" / "python.json"
    labels_path.parent.mkdir(parents=True)
    labels_path.write_text(
        json.dumps(
            {
                "positive_groups": [["alpha.py::first", "alpha.py::second"]],
                "categories": {"exact": [["alpha.py::first", "alpha.py::second"]]},
            }
        )
    )

    profile = resolve_model_profile("gte-modernbert-base")
    units = [_unit("first", source_path, 1), _unit("second", source_path, 5)]
    identity = EmbeddingSpaceIdentity(
        model_name=profile.canonical_name,
        resolved_revision=profile.default_revision or PINNED_COMMIT,
        runtime_variant="cpu-faithful",
    )

    _patch_analyze(
        monkeypatch, units=units, identity=identity, embeddings=np.eye(2, 4, dtype=np.float32)
    )

    report = _analyze_language(
        language="python",
        model_name="gte-modernbert-base",
        corpus_root=corpus_root,
        device="cpu",
        batch_size=4,
        min_statements=0,
    )

    manifest = report["calibration"]
    assert manifest["model"] == profile.canonical_name
    assert manifest["resolved_revision"] == profile.default_revision
    assert manifest["embedding_space"]["runtime_variant"] == "cpu-faithful"
    assert manifest["requested_device"] == "cpu"
    assert manifest["mode"] == "distribution"
    assert manifest["candidate_policy"]["min_recursive_statements"] == 0
    assert manifest["corpus_path"] == str(language_path)
    assert manifest["labels_path"] == str(labels_path)
    # The digests must cover this corpus, not the sweep's default fixture tree.
    expected = _calibration_manifest(
        profile=profile,
        resolved_revision=profile.default_revision or PINNED_COMMIT,
        mode="distribution",
        semantic_task=DEFAULT_CHECK_SEMANTIC_TASK,
        requested_device="cpu",
        identity=identity,
        dimension=4,
        min_statements=0,
        batch_size=4,
        corpus_path=language_path,
        labels_path=labels_path,
    )
    assert manifest == expected


def test_hybrid_gate_ties_resolve_to_the_loosest_split_not_grid_order() -> None:
    """Equal-metric hybrid rows must rank recall-first, like the semantic sweep.

    Without an explicit tiebreak the winner is whichever configuration
    ``itertools.product`` happened to emit first, so a grid reordering silently
    changes the recommended split. A disabled promotion gate is the strictest
    setting on that axis.
    """
    # Deliberately ordered strictest-first so grid order and the policy disagree.
    grid = [
        GateConfig(0.80, 0.30, 0.55, None),
        GateConfig(0.80, 0.30, 0.55, 0.90),
        GateConfig(0.80, 0.10, 0.20, None),
        GateConfig(0.80, 0.10, 0.20, 0.90),
    ]

    rows = _run_hybrid_gate_sweep(
        traditional_duplicates=[],
        semantic_duplicates=[],
        positive_pairs=set(),
        traditional_threshold=0.8,
        grid=grid,
    )

    assert {(row.f1, row.precision, row.recall, row.fp) for row in rows} == {(0.0, 0.0, 0.0, 0)}
    assert [row.config for row in rows] == [
        GateConfig(0.80, 0.10, 0.20, 0.90),
        GateConfig(0.80, 0.10, 0.20, None),
        GateConfig(0.80, 0.30, 0.55, 0.90),
        GateConfig(0.80, 0.30, 0.55, None),
    ]


def _row(
    weak: float,
    ratio: float,
    high: float | None,
    *,
    tp: int,
    fp: int,
    fn: int,
    published: tuple[int, int, int],
) -> SweepRow:
    """Build one synthetic sweep row from visible and published counts."""
    p_tp, p_fp, p_fn = published

    def prf(a: int, b: int, c: int) -> tuple[float, float, float]:
        precision = a / (a + b) if a + b else 0.0
        recall = a / (a + c) if a + c else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    precision, recall, f1 = prf(tp, fp, fn)
    p_precision, p_recall, p_f1 = prf(p_tp, p_fp, p_fn)
    return SweepRow(
        config=GateConfig(0.80, weak, ratio, high),
        published=p_tp + p_fp,
        review=(p_tp + p_fp) - (tp + fp),
        high_confidence=tp + fp,
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
        review_tp=p_tp - tp,
        review_fp=p_fp - fp,
    )


def test_select_visible_row_maximizes_precision_within_the_retention_floor() -> None:
    """The split must buy precision, keep 85% of published recall, and prefer stricter ties."""
    published = (20, 10, 4)  # precision 0.667, recall 0.833
    everything = _row(0.0, 0.0, None, tp=20, fp=10, fn=4, published=published)
    too_lossy = _row(0.30, 0.50, None, tp=15, fp=1, fn=9, published=published)  # recall 0.625
    good = _row(0.20, 0.35, None, tp=18, fp=3, fn=6, published=published)  # recall 0.75
    good_looser = _row(0.10, 0.35, None, tp=18, fp=3, fn=6, published=published)
    worse_precision = _row(0.05, 0.20, None, tp=19, fp=10, fn=5, published=published)

    assert is_feasible(everything, recall_retention_min=0.85)
    assert not is_feasible(too_lossy, recall_retention_min=0.85)
    assert is_feasible(good, recall_retention_min=0.85)
    assert not is_feasible(worse_precision, recall_retention_min=0.85)

    selected = select_visible_row(
        [too_lossy, everything, good_looser, worse_precision, good], recall_retention_min=0.85
    )
    # Equal on the corpus: the looser row would only widen the default view on no evidence.
    assert selected is good
    assert select_visible_row([too_lossy, worse_precision], recall_retention_min=0.85) is None


def test_pooled_selection_requires_feasibility_in_every_corpus() -> None:
    """A split that guts one language's default view must not win on pooled precision."""
    strict = (0.30, 0.50)
    mild = (0.10, 0.35)
    python_rows = [
        _row(*strict, None, tp=5, fp=0, fn=15, published=(20, 10, 0)),  # recall 0.25: infeasible
        _row(*mild, None, tp=18, fp=4, fn=2, published=(20, 10, 0)),
    ]
    rust_rows = [
        _row(*strict, None, tp=19, fp=0, fn=1, published=(20, 10, 0)),
        _row(*mild, None, tp=18, fp=5, fn=2, published=(20, 10, 0)),
    ]
    rows_by_corpus = {"python": python_rows, "rust": rust_rows}

    pooled = pool_rows(rows_by_corpus)
    by_key = {
        (row.config.weak_identifier_jaccard_min, row.config.statement_ratio_min): row
        for row in pooled
    }
    assert by_key[strict].tp == 24 and by_key[strict].published_tp == 40
    assert by_key[strict].config.semantic_gate is None
    assert by_key[strict].precision > by_key[mild].precision

    selected = select_pooled_row(pooled, rows_by_corpus, recall_retention_min=0.85)
    assert selected is not None
    assert (
        selected.config.weak_identifier_jaccard_min,
        selected.config.statement_ratio_min,
    ) == mild


def test_high_gate_grid_starts_at_the_admission_gate_and_ends_disabled() -> None:
    assert _high_gate_grid(0.90, 0.96, 0.02) == [0.90, 0.92, 0.94, 0.96, None]
    assert _high_gate_grid(0.97, 0.96, 0.02) == [None]


@pytest.mark.parametrize("step", [0.0, -0.02, float("nan"), float("inf")])
def test_high_gate_grid_rejects_non_positive_step(step: float) -> None:
    assert _high_gate_grid(0.68, 0.72, 0.02) == [0.68, 0.70, 0.72, None]
    with pytest.raises(ValueError, match="step must be finite and positive"):
        _high_gate_grid(0.68, 0.96, step)


@pytest.mark.parametrize(
    ("start", "stop"),
    [
        (float("nan"), 0.96),
        (float("inf"), 0.96),
        (-0.01, 0.96),
        (0.68, float("nan")),
        (0.68, float("inf")),
        (0.68, 1.01),
    ],
)
def test_high_gate_grid_rejects_invalid_bounds(start: float, stop: float) -> None:
    with pytest.raises(ValueError, match=r"must be finite and in \[0.0, 1.0\]"):
        _high_gate_grid(start, stop, 0.02)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--weak-jaccard-grid", "0.0,nan"),
        ("--weak-jaccard-grid", "-0.1,0.2"),
        ("--statement-ratio-grid", "0.2,inf"),
        ("--statement-ratio-grid", "0.2,1.1"),
        ("--high-gate-stop", "nan"),
        ("--high-gate-stop", "1.1"),
        ("--semantic-gate", "inf"),
        ("--semantic-gate", "-0.1"),
    ],
)
def test_hybrid_gate_sweep_rejects_invalid_grid_values_before_model_work(
    monkeypatch, capsys, flag: str, value: str
) -> None:
    def fail_sweep(*args, **kwargs):
        raise AssertionError("model work must not begin with an invalid grid")

    monkeypatch.setattr(sweep_hybrid_gates, "_sweep_model", fail_sweep)
    option_args = [f"{flag}={value}"] if value.startswith("-") else [flag, value]
    monkeypatch.setattr(
        "sys.argv",
        ["sweep_hybrid_gates.py", "--language", "python", *option_args],
    )

    with pytest.raises(SystemExit) as exc:
        _hybrid_gates_main()

    assert exc.value.code == 2
    assert "finite" in capsys.readouterr().err


@pytest.mark.parametrize(
    "language_args", [[], ["--language", "python", "--language", "typescript"]]
)
def test_hybrid_gate_sweep_requires_one_language_before_model_work(
    monkeypatch, capsys, language_args: list[str]
) -> None:
    """A single-corpus sweep must never silently select the fallback admission gate."""

    def fail_sweep(*args, **kwargs):
        raise AssertionError("model work must not begin without one corpus language")

    monkeypatch.setattr(sweep_hybrid_gates, "_sweep_model", fail_sweep)
    monkeypatch.setattr("sys.argv", ["sweep_hybrid_gates.py", *language_args])

    with pytest.raises(SystemExit) as exc:
        _hybrid_gates_main()

    assert exc.value.code == 2
    assert (
        "--language must be specified exactly once without --corpus-root" in capsys.readouterr().err
    )


def test_hybrid_gate_sweep_normalizes_and_deduplicates_polyglot_languages(
    tmp_path: Path, monkeypatch
) -> None:
    corpus_root = tmp_path / "polyglot"
    (corpus_root / "python").mkdir(parents=True)
    labels_path = corpus_root / "labels" / "python.json"
    labels_path.parent.mkdir()
    labels_path.write_text(
        json.dumps({"positive_groups": [["alpha.py::first", "alpha.py::second"]]})
    )
    captured = []

    def capture_sweep(model, corpora, args):
        captured.extend(corpora)
        return {"model_key": model}

    monkeypatch.setattr(sweep_hybrid_gates, "_sweep_model", capture_sweep)
    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep_hybrid_gates.py",
            "--corpus-root",
            str(corpus_root),
            "--languages",
            "py",
            "Python",
            "--models",
            "gte-modernbert-base",
        ],
    )

    assert _hybrid_gates_main() == 0
    assert len(captured) == 1
    assert captured[0].name == "python"
    assert captured[0].language == "python"
    assert captured[0].corpus_path == corpus_root / "python"
    assert captured[0].labels_path == labels_path


@pytest.mark.parametrize(
    ("main", "argv"),
    [
        (
            _hybrid_gates_main,
            [
                "sweep_hybrid_gates.py",
                "--corpus-root",
                "unused",
                "--languages",
                "--models",
                "gte-modernbert-base",
            ],
        ),
        (
            _hybrid_gates_main,
            ["sweep_hybrid_gates.py", "--language", "python", "--models"],
        ),
        (
            _semantic_sweep_main,
            ["sweep_semantic_thresholds.py", "--models", "--skip-search"],
        ),
    ],
)
def test_sweeps_reject_explicitly_empty_selectors_before_work(
    monkeypatch, main, argv: list[str]
) -> None:
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 2


@pytest.mark.parametrize("model_name", ["gte-modernbert-base", "embeddinggemma-300m"])
@pytest.mark.parametrize("semantic_gate", [None, 0.90])
@pytest.mark.parametrize("language", ["python", "py"])
def test_hybrid_gate_sweep_records_calibration_provenance(
    tmp_path: Path, monkeypatch, model_name: str, semantic_gate: float | None, language: str
) -> None:
    """The hybrid report needs the same identity block the semantic sweep records.

    Without it a report carried no model, revision, device, or embedding-space
    identity at all, and nothing marked that its precision/recall fields score
    only the high-confidence tiers - the semantic sweep's same-named fields
    score all published pairs.
    """
    corpus_path = tmp_path / "corpus"
    corpus_path.mkdir()
    corpus_file = corpus_path / "alpha.py"
    corpus_file.write_text("def first():\n    return 1\n\n\ndef second():\n    return 2\n")
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps({"positive_groups": [["alpha.py::first", "alpha.py::second"]]})
    )
    json_out = tmp_path / "hybrid_report.json"

    profile = resolve_model_profile(model_name)
    identity = EmbeddingSpaceIdentity(
        model_name=profile.canonical_name,
        resolved_revision=profile.default_revision or PINNED_COMMIT,
        runtime_variant="cpu-faithful",
    )
    units = [_unit("first", corpus_file, 1), _unit("second", corpus_file, 5)]

    _patch_analyze(
        monkeypatch, units=units, identity=identity, embeddings=np.zeros((2, 4), dtype=np.float32)
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep_hybrid_gates.py",
            "--corpus-path",
            str(corpus_path),
            "--labels-path",
            str(labels_path),
            "--language",
            language,
            "--models",
            model_name,
            "--json-out",
            str(json_out),
            "--traditional-threshold",
            "0.91",
            *(["--semantic-gate", str(semantic_gate)] if semantic_gate is not None else []),
        ],
    )

    assert _hybrid_gates_main() == 0

    payload = json.loads(json_out.read_text())
    assert payload["output_policy"] == "hybrid_high_confidence"
    assert payload["selection_policy"]["recall_retention_min"] == 0.85
    model_entry = payload["models"][0]
    assert model_entry["resolved_revision"] == profile.default_revision
    corpus_entry = model_entry["corpora"]["python"]
    manifest = corpus_entry["calibration"]
    assert manifest["model"] == profile.canonical_name
    assert manifest["resolved_revision"] == profile.default_revision
    assert manifest["mode"] == "hybrid_gates"
    assert manifest["requested_device"] == "cpu"
    assert manifest["traditional_candidate_policy"] == {
        "jaccard_threshold": 0.91,
        "filter_tiny_traditional": True,
        "tiny_unit_statement_cutoff": 3,
    }
    assert manifest["embedding_space"]["runtime_variant"] == "cpu-faithful"
    assert manifest["semantic_gate"] == {
        "language": "python",
        "value": (
            semantic_gate
            if semantic_gate is not None
            else profile.semantic_threshold_for_language("python")
        ),
        "source": "explicit" if semantic_gate is not None else "profile",
    }
    # No candidates: every split ties at zero and equality with the published
    # precision floor is allowed, so the strictest row wins the tie.
    stage1 = model_entry["stage1"]
    assert stage1["pooled"]["selected"]["config"]["weak_identifier_jaccard_min"] == 0.40
    assert stage1["pooled"]["selected"]["config"]["statement_ratio_min"] == 0.80
    assert stage1["pooled"]["selected"]["config"]["semantic_gate"] is None
    stage2 = model_entry["stage2"]["corpora"]["python"]
    assert stage2["selected"]["config"]["high_gate"] is None
    assert stage2["rows"][0]["config"]["high_gate"] == stage2["semantic_gate"]
    assert stage2["rows"][-1]["config"]["high_gate"] is None


def test_hybrid_gate_sweep_refuses_a_mutable_model_revision(tmp_path: Path, monkeypatch) -> None:
    """Calibrating corroboration constants against a movable label is not calibration."""
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps({"positive_groups": [["alpha.py::first", "alpha.py::second"]]})
    )

    def fail_analyze(self: CodeAnalyzer, path: Path) -> SimpleNamespace:
        raise AssertionError("analyze() must not run for a mutable revision")

    monkeypatch.setattr(CodeAnalyzer, "analyze", fail_analyze)
    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep_hybrid_gates.py",
            "--labels-path",
            str(labels_path),
            "--language",
            "python",
            "--models",
            "gte-modernbert-base",
            "--model-revision",
            "main",
        ],
    )

    with pytest.raises(SystemExit, match="immutable 40-character commit"):
        _hybrid_gates_main()


def test_calibration_refuses_a_local_model_even_with_a_commit_revision(tmp_path: Path) -> None:
    """A caller-supplied commit cannot pin local weights whose loader ignores it."""
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()

    with pytest.raises(SystemExit, match="local model directory.*ignore --model-revision"):
        _require_immutable_revision(str(model_dir), PINNED_COMMIT)


def test_threshold_grid_rows_are_the_exact_gates_evaluated() -> None:
    """An off-grid ``--duplicate-start`` must not label rows below the collection floor.

    ``round(current, 2)`` labeled a 0.705-floor sweep's first row ``0.70`` while
    no pair below 0.705 was ever collected, and the loosest-tie ranking then
    preferred exactly that mislabeled row into ``selected_threshold``.
    """
    grid = _threshold_grid(0.705, 0.75)

    assert grid == [0.705, 0.725, 0.745]


def test_threshold_grid_default_bounds_are_unchanged() -> None:
    """The shipped 2-decimal grids must survive the exact-gate rewrite verbatim.

    The search ceiling is 0.90: the old 0.70 ceiling censored two boundary
    selections with F1 still rising into them.
    """
    duplicate_grid = _threshold_grid(0.70, 0.96)
    search_grid = _threshold_grid(0.20, 0.90)

    assert duplicate_grid[0] == 0.70
    assert duplicate_grid[-1] == 0.96
    assert len(duplicate_grid) == 14
    assert len(search_grid) == 36
    assert all(
        round(after - before, 9) == THRESHOLD_STEP for before, after in pairwise(duplicate_grid)
    )


@pytest.mark.parametrize(
    ("start", "stop", "message"),
    [
        (float("nan"), 0.9, "finite"),
        (0.2, float("inf"), "finite"),
        (-0.01, 0.9, r"\[0\.0, 1\.0\]"),
        (0.2, 1.01, r"\[0\.0, 1\.0\]"),
        (0.8, 0.4, "must not exceed"),
    ],
)
def test_threshold_grid_rejects_invalid_bounds(start: float, stop: float, message: str) -> None:
    """Invalid bounds must fail instead of hanging or bypassing analyzer validation."""
    with pytest.raises(ValueError, match=message):
        _threshold_grid(start, stop)


def test_search_sweep_collects_at_its_chosen_grid_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A custom search floor must also be the analyzer's score-collection floor."""
    corpus_path = tmp_path / "corpus"
    corpus_path.mkdir()
    unit_path = corpus_path / "alpha.py"
    unit_path.write_text("def target():\n    return 1\n")
    probes_path = tmp_path / "probes.json"
    probes_path.write_text("{}")
    unit = _unit("target", unit_path, 1)
    identity = EmbeddingSpaceIdentity(
        model_name=resolve_model_profile("gte-modernbert-base").canonical_name,
        resolved_revision=PINNED_COMMIT,
        runtime_variant="cpu-faithful",
    )
    captured_thresholds: list[float | None] = []

    class FakeAnalyzer:
        def __init__(self, config) -> None:
            self.config = config
            captured_thresholds.append(config.semantic_threshold)
            self._embeddings = np.zeros((1, 4), dtype=np.float32)
            self._embedding_space_identity = identity
            self._semantic_units = [unit]
            self._units = [unit]

        def index(self, path: Path) -> int:
            return 1

        def search(self, query: str, top_k: int) -> list[tuple[CodeUnit, float]]:
            return [(unit, 0.05)]

    monkeypatch.setattr("scripts.sweep_semantic_thresholds.CodeAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(
        "scripts.sweep_semantic_thresholds._calibration_manifest", lambda **kwargs: {}
    )

    sweep = _run_search_sweep(
        model_name="gte-modernbert-base",
        revision=PINNED_COMMIT,
        corpus_path=corpus_path,
        probes_path=probes_path,
        probes=[{"query": "target", "expected": ["alpha.py::target"]}],
        min_statements=0,
        batch_size=4,
        device="cpu",
        search_start=0.04,
        search_stop=0.08,
    )

    assert captured_thresholds == [0.04]
    assert {row.threshold for row in sweep.rows} == {0.04, 0.06, 0.08}


def test_grid_edge_labels_boundary_selections() -> None:
    """A boundary selection is censored evidence and must be recorded, not hidden."""
    grid = [0.2, 0.22, 0.24]

    assert _grid_edge(0.2, grid) == "start"
    assert _grid_edge(0.24, grid) == "stop"
    assert _grid_edge(0.22, grid) is None


def test_semantic_sweep_rejects_an_empty_search_grid(monkeypatch, capsys) -> None:
    """``--search-start`` above ``--search-stop`` must abort like the duplicate bounds."""
    monkeypatch.setattr(
        "sys.argv",
        ["sweep_semantic_thresholds.py", "--search-start", "0.8", "--search-stop", "0.4"],
    )

    with pytest.raises(SystemExit) as excinfo:
        _semantic_sweep_main()

    assert excinfo.value.code == 2
    assert "--search-start" in capsys.readouterr().err


def test_labels_shape_validation_rejects_an_empty_category() -> None:
    """An empty category list must fail by name, not as a bogus positive_groups error."""
    labels = {
        "positive_groups": [["alpha.py::first", "alpha.py::second"]],
        "categories": {
            "exact": [["alpha.py::first", "alpha.py::second"]],
            "near_translation": [],
        },
    }

    with pytest.raises(ValueError, match="near_translation"):
        validate_labels_shape(labels)


@pytest.mark.parametrize("payload", [[], ["alpha.py::first"], "labels"])
def test_labels_shape_validation_requires_a_top_level_object(payload: object) -> None:
    with pytest.raises(ValueError, match="must contain a JSON object"):
        validate_labels_shape(payload)


@pytest.mark.parametrize(
    "spec",
    [None, 4, "", "alpha.py", "::first", "alpha.py::", "alpha.py::first::extra"],
)
def test_labels_shape_validation_rejects_malformed_selectors(spec: object) -> None:
    labels = {"positive_groups": [["alpha.py::first", spec]]}

    with pytest.raises(ValueError, match="Invalid label spec"):
        validate_labels_shape(labels)


def test_labels_shape_validation_checks_category_selectors() -> None:
    labels = {
        "positive_groups": [["alpha.py::first", "alpha.py::second"]],
        "categories": {"near": [["alpha.py::first", None]]},
    }

    with pytest.raises(ValueError, match="Invalid label spec"):
        validate_labels_shape(labels)


def test_probes_shape_validation_rejects_empty_and_malformed_probes() -> None:
    """Probes get the same fail-fast shape contract the labels already have."""
    with pytest.raises(ValueError, match="must contain a JSON object"):
        validate_probes_shape([])
    with pytest.raises(ValueError, match="non-empty 'probes' list"):
        validate_probes_shape({"probes": []})
    with pytest.raises(ValueError, match="non-empty 'probes' list"):
        validate_probes_shape({"queries": [{"query": "q", "expected": ["a.py::f"]}]})
    with pytest.raises(ValueError, match="probe 0 must define a non-empty string 'query'"):
        validate_probes_shape({"probes": [{"query": "  ", "expected": ["a.py::f"]}]})
    with pytest.raises(ValueError, match="probe 1 must define a non-empty 'expected'"):
        validate_probes_shape(
            {"probes": [{"query": "q", "expected": ["a.py::f"]}, {"query": "r", "expected": []}]}
        )
    with pytest.raises(ValueError, match="probe 0 has an invalid expected spec"):
        validate_probes_shape({"probes": [{"query": "q", "expected": ["a.py"]}]})

    probes = [{"query": "q", "expected": ["a.py::f", "b.py::g"]}]
    assert validate_probes_shape({"probes": probes}) == probes


def test_semantic_sweep_rejects_malformed_probes_before_any_analysis(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A zero-probe file must abort at argument time, not become a search report.

    Before the shape gate, ``{"probes": []}`` swept every threshold over zero
    scored pairs and wrote a full search report selecting the grid floor.
    """
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps({"positive_groups": [["alpha.py::a", "alpha.py::b"]]}))
    probes_path = tmp_path / "search_probes.json"
    probes_path.write_text(json.dumps({"probes": []}))

    def fail_analyze(self: CodeAnalyzer, path: Path) -> SimpleNamespace:
        raise AssertionError("analyze() must not run for malformed probes")

    monkeypatch.setattr(CodeAnalyzer, "analyze", fail_analyze)
    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep_semantic_thresholds.py",
            "--labels-path",
            str(labels_path),
            "--search-probes-path",
            str(probes_path),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        _semantic_sweep_main()

    assert excinfo.value.code == 2
    assert "non-empty 'probes' list" in capsys.readouterr().err


def test_semantic_sweep_rejects_malformed_labels_before_any_analysis(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A bad labels file must abort at argument time, not after the corpus embed."""
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "positive_groups": [["alpha.py::first", "alpha.py::second"]],
                "categories": {"exact": []},
            }
        )
    )

    def fail_analyze(self: CodeAnalyzer, path: Path) -> SimpleNamespace:
        raise AssertionError("analyze() must not run for malformed labels")

    monkeypatch.setattr(CodeAnalyzer, "analyze", fail_analyze)
    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep_semantic_thresholds.py",
            "--labels-path",
            str(labels_path),
            "--skip-search",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        _semantic_sweep_main()

    assert excinfo.value.code == 2
    assert "'exact'" in capsys.readouterr().err
