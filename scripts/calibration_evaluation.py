"""Evaluate calibration scores against explicit pair and search judgments."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from codedupes.constants import DEFAULT_TOP_K, DEFAULT_TRADITIONAL_THRESHOLD
from codedupes.semantic_profiles import resolve_model_profile

try:
    from .calibration_contract import REPO, Project, pair_key, write_json
    from .calibration_measurements import artifact_path, load_measurement, measurement_fingerprint
except ImportError:
    from calibration_contract import REPO, Project, pair_key, write_json
    from calibration_measurements import artifact_path, load_measurement, measurement_fingerprint


F1_RECALL_TOLERANCE = 0.005
MINIMUM_SELECTION_PRECISION = 0.5


def near_best_f1(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return precision-safe rows within the allowed F1 loss for recall preference."""
    safe = [row for row in rows if row["precision"] >= MINIMUM_SELECTION_PRECISION]
    if not safe:
        raise ValueError(
            "no calibration candidate satisfies the minimum precision "
            f"{MINIMUM_SELECTION_PRECISION:.2f}"
        )
    floor = max(row["f1"] for row in safe) - F1_RECALL_TOLERANCE
    return [row for row in safe if row["f1"] >= floor - 1e-12]


def recall_preference(row: dict[str, Any]) -> tuple[int, int, float, float]:
    """Prefer resolved judgments, then recall and precision within the F1 bound."""
    return (
        -row.get("ambiguous_predictions", 0),
        -row.get("unjudged_predictions", 0),
        row["recall"],
        row["precision"],
    )


def selection_digest(payload: Any) -> str:
    """Identify a derived selection or its annotation inputs."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def development_projects(projects: list[Project]) -> list[Project]:
    """Return the reviewed development split used to select shipped defaults."""
    selected = [project for project in projects if project.spec["split"] == "development"]
    if not selected:
        raise ValueError("calibration selection requires at least one development project")
    return selected


def selection_context(projects: list[Project], models: list[str]) -> dict[str, Any]:
    """Bind selections to their policy, corpus scope, judgments, and measured inputs."""
    return {
        "selection_policy": selection_digest(
            {
                path: (REPO / path).read_text()
                for path in (
                    "scripts/calibration_evaluation.py",
                    "scripts/sweep_semantic_thresholds.py",
                    "scripts/sweep_hybrid_gates.py",
                    "src/codedupes/semantic_profiles.py",
                )
            }
        ),
        "projects": {
            project.id: {
                "annotations": selection_digest(project.annotations),
                "project": selection_digest(project.spec),
                "policy": project.policy_name,
                "measurements": {
                    resolve_model_profile(model).key: measurement_fingerprint(project, model)
                    for model in models
                },
            }
            for project in projects
        },
    }


def validate_selection_context(
    payload: dict[str, Any], projects: list[Project], models: list[str]
) -> None:
    """Reject selections from another source, review state, model, or corpus scope."""
    if payload.get("input_context") != selection_context(projects, models):
        raise ValueError("stale or mismatched selection inputs; rerun the selection sweeps")


def judgments(project: Project) -> dict[tuple[str, str], dict[str, Any]]:
    """Index explicit pair judgments by unordered unit IDs."""
    return {pair_key(item["a"], item["b"]): item for item in project.annotations["pairs"]}


def metrics(
    predicted: set[tuple[str, str]], labels: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    """Score predictions against reviewed positives and negatives."""
    positives = {key for key, item in labels.items() if item["judgment"] == "positive"}
    negatives = {key for key, item in labels.items() if item["judgment"] == "negative"}
    ambiguities = {key for key, item in labels.items() if item["judgment"] == "ambiguous"}
    tp = len(predicted & positives)
    fp = len(predicted & negatives)
    fn = len(positives - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "judged_only_precision": precision,
        "recall": recall,
        "f1": f1,
        "ambiguous_predictions": len(predicted & ambiguities),
        "unjudged_predictions": len(predicted - labels.keys()),
    }


def replay_pair(
    row: dict[str, Any],
    units: dict[str, dict[str, Any]],
    profile: Any,
    *,
    semantic_threshold: float | None = None,
    weak_identifier_jaccard_min: float | None = None,
    statement_ratio_min: float | None = None,
    high_gate: float | None | object = ...,
) -> dict[str, Any] | None:
    """Apply production duplicate synthesis to one measured pair."""
    unit_a = units[row["a"]]
    language = unit_a["language"]
    gate = (
        profile.semantic_threshold_for_language(language)
        if semantic_threshold is None
        else semantic_threshold
    )
    exact = any(item["method"] in {"structural_hash", "token_hash"} for item in row["traditional"])
    jaccard = max(
        (item["similarity"] for item in row["traditional"] if item["method"] == "jaccard"),
        default=None,
    )
    semantic = (
        row["cosine"]
        if row["comparable"] and row["cosine"] is not None and row["cosine"] >= gate
        else None
    )
    if exact:
        tier, confidence = "exact", 1.0
    elif jaccard is not None and jaccard >= DEFAULT_TRADITIONAL_THRESHOLD:
        if semantic is None:
            tier, confidence = "traditional_near", 0.55 + 0.45 * jaccard
        else:
            tier, confidence = "hybrid_confirmed", 0.5 * semantic + 0.5 * jaccard
    elif semantic is not None:
        weak = (
            profile.hybrid_weak_identifier_jaccard_min
            if weak_identifier_jaccard_min is None
            else weak_identifier_jaccard_min
        )
        ratio = (
            profile.hybrid_statement_ratio_min
            if statement_ratio_min is None
            else statement_ratio_min
        )
        if high_gate is ...:
            promotion = profile.high_confidence_threshold_for_language(language)
        else:
            promotion = high_gate
        corroborated = row["identifier_jaccard"] >= weak and row["statement_ratio"] >= ratio
        strong = promotion is not None and semantic >= promotion
        if corroborated or strong:
            tier, confidence = "semantic_high_confidence", 0.45 + 0.55 * semantic
        else:
            tier, confidence = "semantic_review", 0.40 + 0.45 * semantic
    else:
        return None
    return {
        "a": row["a"],
        "b": row["b"],
        "tier": tier,
        "confidence": confidence,
        "semantic_similarity": semantic,
        "jaccard_similarity": jaccard,
    }


def replay(
    measurement: dict[str, Any],
    *,
    semantic_threshold: float | None = None,
    weak_identifier_jaccard_min: float | None = None,
    statement_ratio_min: float | None = None,
    high_gate: float | None | object = ...,
) -> list[dict[str, Any]]:
    """Replay measured pair scores through production duplicate policy."""
    if semantic_threshold is not None and high_gate is ...:
        high_gate = None
    profile = resolve_model_profile(measurement["metadata"]["model"])
    units = {item["id"]: item for item in measurement["units"]}
    findings = []
    for row in measurement["pairs"]:
        finding = replay_pair(
            row,
            units,
            profile,
            semantic_threshold=semantic_threshold,
            weak_identifier_jaccard_min=weak_identifier_jaccard_min,
            statement_ratio_min=statement_ratio_min,
            high_gate=high_gate,
        )
        if finding is not None:
            findings.append(finding)
    findings.sort(key=lambda item: (-item["confidence"], item["a"], item["b"]))
    return findings


def replay_parity(measurement: dict[str, Any]) -> bool:
    """Verify the local replay produces the analyzer's captured shipped findings."""
    captured = measurement["metadata"]["captured_profile"]
    replayed = replay(
        measurement,
        semantic_threshold=captured["semantic_threshold"],
        weak_identifier_jaccard_min=captured["weak_identifier_jaccard_min"],
        statement_ratio_min=captured["statement_ratio_min"],
        high_gate=captured["high_gate"],
    )
    signature = {
        (pair_key(item["a"], item["b"]), item["tier"])
        for item in measurement["metadata"]["live_default"]
    }
    replayed_signature = {(pair_key(item["a"], item["b"]), item["tier"]) for item in replayed}
    if signature != replayed_signature:
        raise ValueError("measurement replay diverges from captured analyzer findings")
    return True


def duplicate_report(
    project: Project, measurement: dict[str, Any], *, semantic_threshold: float | None = None
) -> dict[str, Any]:
    """Report complete and default-visible duplicate performance."""
    labels = judgments(project)
    findings = replay(measurement, semantic_threshold=semantic_threshold)
    published = {pair_key(item["a"], item["b"]) for item in findings}
    visible = {
        pair_key(item["a"], item["b"]) for item in findings if item["tier"] != "semantic_review"
    }
    deterministic = {
        pair_key(row["a"], row["b"]) for row in measurement["pairs"] if row["traditional"]
    }
    comparable = {
        pair_key(row["a"], row["b"])
        for row in measurement["pairs"]
        if row["comparable"] and not row["traditional"]
    }
    semantic_predictions = {
        pair_key(item["a"], item["b"])
        for item in findings
        if item["semantic_similarity"] is not None
        and pair_key(item["a"], item["b"]) not in deterministic
    }
    return {
        "published": metrics(published, labels),
        "visible": metrics(visible, labels),
        "deterministic": metrics(deterministic, labels),
        "semantic_eligible": metrics(
            semantic_predictions,
            {key: label for key, label in labels.items() if key in comparable},
        ),
        "tiers": dict(sorted(Counter(item["tier"] for item in findings).items())),
    }


def search_report(
    project: Project, measurement: dict[str, Any], *, threshold: float | None = None
) -> dict[str, Any]:
    """Report search precision and recall at the production top-k limit."""
    profile = resolve_model_profile(measurement["metadata"]["model"])
    gate = profile.default_search_threshold if threshold is None else threshold
    expected = {
        (probe["id"], unit) for probe in project.annotations["probes"] for unit in probe["expected"]
    }
    output = {
        (row["probe"], row["unit"])
        for row in measurement["query_scores"]
        if row["cosine"] is not None and row["cosine"] >= gate and row["rank"] <= DEFAULT_TOP_K
    }
    tp = len(output & expected)
    fp = len(output - expected)
    fn = len(expected - output)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    no_result = {probe["id"] for probe in project.annotations["probes"] if not probe["expected"]}
    violated = sorted(
        probe for probe in no_result if any(output_probe == probe for output_probe, _ in output)
    )
    return {
        "threshold": gate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "no_result": {
            "clean": len(no_result) - len(violated),
            "total": len(no_result),
            "violations": violated,
        },
    }


def compare_devices(cpu: dict[str, Any], mps: dict[str, Any], project: Project) -> dict[str, Any]:
    """Summarize CPU/MPS score drift and shipped-decision differences."""
    cpu_pairs = {
        pair_key(row["a"], row["b"]): row["cosine"]
        for row in cpu["pairs"]
        if row["cosine"] is not None
    }
    mps_pairs = {
        pair_key(row["a"], row["b"]): row["cosine"]
        for row in mps["pairs"]
        if row["cosine"] is not None
    }
    pair_drifts = [abs(cpu_pairs[key] - mps_pairs[key]) for key in cpu_pairs.keys() & mps_pairs]
    cpu_queries = {
        (row["probe"], row["unit"]): row["cosine"]
        for row in cpu["query_scores"]
        if row["cosine"] is not None
    }
    mps_queries = {
        (row["probe"], row["unit"]): row["cosine"]
        for row in mps["query_scores"]
        if row["cosine"] is not None
    }
    query_drifts = [
        abs(cpu_queries[key] - mps_queries[key]) for key in cpu_queries.keys() & mps_queries
    ]

    def summary(values: list[float]) -> dict[str, float | int | None]:
        return {
            "count": len(values),
            "p95": float(np.quantile(values, 0.95)) if values else None,
            "max": max(values, default=None),
        }

    cpu_output = {pair_key(item["a"], item["b"]): item["tier"] for item in replay(cpu)}
    mps_output = {pair_key(item["a"], item["b"]): item["tier"] for item in replay(mps)}
    search_gate = resolve_model_profile(cpu["metadata"]["model"]).default_search_threshold

    def search_decisions(measurement: dict[str, Any]) -> set[tuple[str, str]]:
        return {
            (row["probe"], row["unit"])
            for row in measurement["query_scores"]
            if row["cosine"] is not None
            and row["cosine"] >= search_gate
            and row["rank"] <= DEFAULT_TOP_K
        }

    cpu_search = search_decisions(cpu)
    mps_search = search_decisions(mps)
    return {
        "pair_score_abs_drift": summary(pair_drifts),
        "query_score_abs_drift": summary(query_drifts),
        "duplicate_decision_changes": [
            list(key)
            for key in sorted(cpu_output.keys() | mps_output.keys())
            if cpu_output.get(key) != mps_output.get(key)
        ],
        "search_decision_changes": [list(key) for key in sorted(cpu_search ^ mps_search)],
    }


def full_report(project: Project, measurement: dict[str, Any]) -> dict[str, Any]:
    """Build a compact report for shipped settings."""
    execution = {
        task: {
            "device": stats["execution_device"],
            "encoded_inputs": stats["encoded_inputs"],
            "cache_hit_rows": stats["cache_hit_rows"],
        }
        for task, stats in measurement["metadata"]["execution"].items()
    }
    return {
        "project": project.id,
        "model": measurement["metadata"]["model"],
        "device": measurement["metadata"]["requested_device"],
        "inference_dtype": measurement["metadata"]["inference_dtype"],
        "runtime_versions": measurement["metadata"]["runtime_versions"],
        "timing_seconds": measurement["metadata"]["timing_seconds"],
        "execution": execution,
        "replay_parity": replay_parity(measurement),
        "duplicate": duplicate_report(project, measurement),
        "search": search_report(project, measurement),
    }


def load_all(
    project: Project, root: Path, models: list[str], devices: list[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load requested raw measurements."""
    return {
        (resolve_model_profile(model).key, device): load_measurement(
            artifact_path(root, project, model, device),
            project,
            expected_model=resolve_model_profile(model).key,
            expected_device=device,
        )
        for model in models
        for device in devices
    }


def write_full_report(path: Path, report: dict[str, Any]) -> None:
    """Write one compact report."""
    write_json(path, report)
