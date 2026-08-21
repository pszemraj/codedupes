"""Calibration-manifest regression coverage for the semantic threshold sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from codedupes.analyzer import CodeAnalyzer
from codedupes.constants import DEFAULT_MIN_SEMANTIC_STATEMENTS
from codedupes.models import CodeUnit, CodeUnitType
from codedupes.semantic import EmbeddingSpaceIdentity
from codedupes.semantic_profiles import resolve_model_profile
from scripts.sweep_common import add_common_sweep_arguments
from scripts.sweep_semantic_thresholds import _run_duplicate_sweep

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
    }
    assert "device" not in manifest
    assert "dtype_variant" not in manifest
    assert manifest["output_policy"] == "hybrid_duplicates"
    assert manifest["candidate_coverage"] == {
        "labeled_positive_pairs": 1,
        "scoreable_positive_pairs": 1,
        "excluded_positive_pairs": 0,
        "recall_ceiling": 1.0,
    }


def test_common_sweep_defaults_match_production_candidate_policy() -> None:
    parser = argparse.ArgumentParser()
    add_common_sweep_arguments(parser)

    assert parser.parse_args([]).min_statements == DEFAULT_MIN_SEMANTIC_STATEMENTS
