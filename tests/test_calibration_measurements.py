"""Replay and artifact-lifecycle checks for the replacement calibration pilot."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from codedupes.semantic_profiles import resolve_model_profile
from scripts.calibration_contract import digest, load_projects, pair_key, read_json, write_json
from scripts.calibration_evaluation import metrics, replay, replay_parity
from scripts.calibration_measurements import (
    artifact_path,
    file_sha256,
    frozen_defaults,
    load_measurement,
    query_identity,
    source_identity,
    write_rows,
)
from scripts.sweep_semantic_thresholds import duplicate_rows


def synthetic_measurement() -> dict:
    """Build a small raw table covering deterministic and semantic replay paths."""
    model = resolve_model_profile("gte-modernbert-base")
    units = [{"id": name, "language": "python"} for name in ("a", "b", "c", "d")]
    pairs = [
        {
            "a": "a",
            "b": "b",
            "cosine": 1.0,
            "comparable": False,
            "identifier_jaccard": 1.0,
            "statement_ratio": 1.0,
            "traditional": [{"method": "structural_hash", "similarity": 1.0}],
        },
        {
            "a": "a",
            "b": "c",
            "cosine": 0.81,
            "comparable": True,
            "identifier_jaccard": 0.0,
            "statement_ratio": 0.9,
            "traditional": [],
        },
        {
            "a": "b",
            "b": "c",
            "cosine": 0.81,
            "comparable": True,
            "identifier_jaccard": 0.0,
            "statement_ratio": 0.5,
            "traditional": [],
        },
        {
            "a": "c",
            "b": "d",
            "cosine": 0.79,
            "comparable": True,
            "identifier_jaccard": 0.0,
            "statement_ratio": 1.0,
            "traditional": [],
        },
    ]
    measurement = {
        "metadata": {"identity": {"model": model.canonical_name}},
        "units": units,
        "pairs": pairs,
    }
    default = replay(measurement)
    explicit = replay(
        measurement,
        semantic_threshold=model.default_semantic_threshold,
        high_gate=None,
    )
    measurement["metadata"].update(
        {
            "live_default": [
                {"a": row["a"], "b": row["b"], "tier": row["tier"]} for row in default
            ],
            "live_explicit_override": {
                "threshold": model.default_semantic_threshold,
                "findings": [
                    {"a": row["a"], "b": row["b"], "tier": row["tier"]} for row in explicit
                ],
            },
        }
    )
    return measurement


def test_frozen_default_snapshot_matches_profiles():
    expected = read_json(Path("test_fixtures/calibration/frozen_defaults.json"))
    assert frozen_defaults() == expected


def test_replay_distinguishes_profile_and_explicit_threshold_paths():
    measurement = synthetic_measurement()
    default = {(pair_key(row["a"], row["b"]), row["tier"]) for row in replay(measurement)}
    explicit = {
        (pair_key(row["a"], row["b"]), row["tier"])
        for row in replay(measurement, semantic_threshold=0.82, high_gate=None)
    }
    assert default == {
        (("a", "b"), "exact"),
        (("a", "c"), "semantic_high_confidence"),
        (("b", "c"), "semantic_review"),
    }
    assert explicit == {(("a", "b"), "exact")}
    assert replay_parity(measurement) == {"default": True, "explicit_override": True}


def test_metrics_keep_ambiguity_and_unjudged_output_separate():
    labels = {
        ("a", "b"): {"judgment": "positive"},
        ("a", "c"): {"judgment": "negative"},
        ("b", "c"): {"judgment": "ambiguous"},
    }
    report = metrics({("a", "b"), ("a", "c"), ("b", "c"), ("c", "d")}, labels)
    assert report["tp"] == 1
    assert report["fp"] == 1
    assert report["reviewed_ambiguities"] == 1
    assert report["unjudged_predictions"] == 1
    assert report["judged_only_precision"] == 0.5
    assert report["selection_eligible"] is False


def test_threshold_rows_do_not_expose_internal_signatures():
    project = copy.copy(load_projects(project_ids=["ledger"])[0])
    project.annotations = {
        "pairs": [
            {
                "a": "a",
                "b": "b",
                "judgment": "positive",
            }
        ]
    }
    rows, plateaus = duplicate_rows(project, synthetic_measurement(), [0.8, 0.82])
    assert all("_signature" not in row for row in rows)
    assert plateaus


def test_artifact_reuse_and_staleness_are_separate(tmp_path: Path, monkeypatch):
    project = load_projects(project_ids=["ledger"])[0]
    profile = resolve_model_profile("gte-modernbert-base")
    directory = artifact_path(tmp_path, project, profile.key, "cpu")
    directory.mkdir(parents=True)
    for filename in ("units.jsonl", "pairs.jsonl", "query_scores.jsonl"):
        write_rows(directory / filename, [])
    identity = {
        "source": source_identity(project),
        "queries": query_identity(project),
        "model": profile.canonical_name,
        "revision": profile.default_revision,
        "requested_device": "cpu",
        "tasks": {},
        "batch_size": 4,
    }
    metadata = {
        "schema_version": 1,
        "identity": identity,
        "measurement_id": digest(identity),
        "tables": {
            filename: file_sha256(directory / filename)
            for filename in ("units.jsonl", "pairs.jsonl", "query_scores.jsonl")
        },
    }
    write_json(directory / "metadata.json", metadata)
    assert load_measurement(directory, project)["pairs"] == []

    project.annotations["pairs"][0]["rationale"] += " Label-only clarification."
    assert load_measurement(directory, project)["pairs"] == []

    original_query = project.annotations["probes"][0]["query"]
    project.annotations["probes"][0]["query"] = original_query + " changed"
    with pytest.raises(ValueError, match="stale query"):
        load_measurement(directory, project)
    project.annotations["probes"][0]["query"] = original_query

    monkeypatch.setattr(
        "scripts.calibration_measurements.source_identity",
        lambda ignored: {"changed": True},
    )
    with pytest.raises(ValueError, match="stale source"):
        load_measurement(directory, project)
