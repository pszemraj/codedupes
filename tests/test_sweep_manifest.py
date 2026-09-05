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
from codedupes.models import CodeUnit, CodeUnitType, DuplicatePair
from codedupes.semantic import EmbeddingSpaceIdentity
from codedupes.semantic_profiles import resolve_model_profile
from scripts.report_calibration_distributions import _analyze_language
from scripts.sweep_common import (
    add_common_sweep_arguments,
    validate_labels_shape,
    validate_probes_shape,
)
from scripts.sweep_hybrid_gates import GateConfig
from scripts.sweep_hybrid_gates import _run_sweep as _run_hybrid_gate_sweep
from scripts.sweep_hybrid_gates import main as _hybrid_gates_main
from scripts.sweep_semantic_thresholds import (
    THRESHOLD_STEP,
    _calibration_manifest,
    _grid_edge,
    _run_duplicate_sweep,
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

    def fake_analyze(self: CodeAnalyzer, path: Path) -> SimpleNamespace:
        self._embeddings = np.zeros((2, 4), dtype=np.float32)
        self._embedding_space_identity = effective_identity
        self._semantic_units = units
        return SimpleNamespace(
            units=units,
            traditional_duplicates=[],
            semantic_duplicates=[],
        )

    monkeypatch.setattr(CodeAnalyzer, "analyze", fake_analyze)
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
    # Zero candidates tie every row at f1=0, so the loosest-tie policy selects
    # the grid floor - a boundary selection the manifest must record.
    assert manifest["selected_at_grid_edge"] == "start"
    assert manifest["candidate_coverage"] == {
        "labeled_positive_pairs": 1,
        "scoreable_positive_pairs": 1,
        "traditional_recovered_pairs": 0,
        "unreachable_positive_pairs": 0,
        "recall_ceiling": 1.0,
    }


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
    traditional = DuplicatePair(first, second, 1.0, "ast_hash")

    def fake_analyze(self: CodeAnalyzer, path: Path) -> SimpleNamespace:
        self._embeddings = np.zeros((0, 0), dtype=np.float32)
        self._embedding_space_identity = identity
        # Both units passed the initial candidate policy but were dropped from
        # the semantic matrix after traditional analysis, as context overflows are.
        self._semantic_units = []
        return SimpleNamespace(
            units=[first, second],
            traditional_duplicates=[traditional],
            semantic_duplicates=[],
        )

    monkeypatch.setattr(CodeAnalyzer, "analyze", fake_analyze)

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
        "scoreable_positive_pairs": 0,
        "traditional_recovered_pairs": 1,
        "unreachable_positive_pairs": 0,
        "recall_ceiling": 1.0,
    }
    assert {row.recall for row in sweep.rows} == {1.0}


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

    def fake_analyze(self: CodeAnalyzer, path: Path) -> SimpleNamespace:
        self._embeddings = np.eye(2, 4, dtype=np.float32)
        self._embedding_space_identity = identity
        self._semantic_units = units
        return SimpleNamespace(units=units, traditional_duplicates=[], semantic_duplicates=[])

    monkeypatch.setattr(CodeAnalyzer, "analyze", fake_analyze)

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


def test_hybrid_gate_ties_resolve_to_the_loosest_gate_not_grid_order() -> None:
    """Equal-metric hybrid rows must rank recall-first, like the semantic sweep.

    Without an explicit tiebreak the winner is whichever configuration
    ``itertools.product`` happened to emit first, so a grid reordering silently
    changes the recommended gate.
    """
    # Deliberately ordered strictest-first so grid order and the policy disagree.
    grid = [
        GateConfig(0.92, 0.30, 0.55),
        GateConfig(0.92, 0.10, 0.20),
        GateConfig(0.68, 0.30, 0.55),
        GateConfig(0.68, 0.10, 0.20),
    ]

    rows, _ = _run_hybrid_gate_sweep(
        traditional_duplicates=[],
        semantic_duplicates=[],
        positive_pairs=set(),
        traditional_threshold=0.8,
        grid=grid,
    )

    assert {(row.f1, row.precision, row.recall, row.fp) for row in rows} == {(0.0, 0.0, 0.0, 0)}
    assert [row.config for row in rows] == [
        GateConfig(0.68, 0.10, 0.20),
        GateConfig(0.68, 0.30, 0.55),
        GateConfig(0.92, 0.10, 0.20),
        GateConfig(0.92, 0.30, 0.55),
    ]


def test_hybrid_gate_sweep_records_calibration_provenance(tmp_path: Path, monkeypatch) -> None:
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

    profile = resolve_model_profile("gte-modernbert-base")
    identity = EmbeddingSpaceIdentity(
        model_name=profile.canonical_name,
        resolved_revision=profile.default_revision or PINNED_COMMIT,
        runtime_variant="cpu-faithful",
    )
    units = [_unit("first", corpus_file, 1), _unit("second", corpus_file, 5)]

    def fake_analyze(self: CodeAnalyzer, path: Path) -> SimpleNamespace:
        self._embeddings = np.zeros((2, 4), dtype=np.float32)
        self._embedding_space_identity = identity
        self._semantic_units = units
        return SimpleNamespace(
            units=units,
            traditional_duplicates=[],
            semantic_duplicates=[],
        )

    monkeypatch.setattr(CodeAnalyzer, "analyze", fake_analyze)
    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep_hybrid_gates.py",
            "--corpus-path",
            str(corpus_path),
            "--labels-path",
            str(labels_path),
            "--json-out",
            str(json_out),
        ],
    )

    assert _hybrid_gates_main() == 0

    payload = json.loads(json_out.read_text())
    assert payload["output_policy"] == "hybrid_high_confidence"
    manifest = payload["calibration"]
    assert manifest["model"] == profile.canonical_name
    assert manifest["resolved_revision"] == profile.default_revision
    assert manifest["mode"] == "hybrid_gates"
    assert manifest["requested_device"] == "cpu"
    assert manifest["embedding_space"]["runtime_variant"] == "cpu-faithful"


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
            "--model-revision",
            "main",
        ],
    )

    with pytest.raises(SystemExit, match="immutable 40-character commit"):
        _hybrid_gates_main()


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


def test_probes_shape_validation_rejects_empty_and_malformed_probes() -> None:
    """Probes get the same fail-fast shape contract the labels already have."""
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
