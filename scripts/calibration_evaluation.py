"""Replay production decisions and report explicitly judged calibration populations."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from codedupes.constants import DEFAULT_TOP_K, DEFAULT_TRADITIONAL_THRESHOLD
from codedupes.semantic_profiles import resolve_model_profile

try:
    from .calibration_contract import Project, digest, pair_key, write_json
    from .calibration_measurements import artifact_path, load_measurement
except ImportError:
    from calibration_contract import Project, digest, pair_key, write_json
    from calibration_measurements import artifact_path, load_measurement


def judgments(project: Project) -> dict[tuple[str, str], dict[str, Any]]:
    """Index explicit maintenance judgments by unordered stable unit IDs."""
    return {pair_key(item["a"], item["b"]): item for item in project.annotations["pairs"]}


def _semantic_gate(profile: Any, row: dict[str, Any], override: float | None) -> float:
    """Resolve a same-language admission gate from a measured unit pair."""
    if override is not None:
        return override
    # Pilot projects are same-language; the unit prefix is deliberately irrelevant.
    languages = {item["language"] for item in row["unit_records"]}
    return min(profile.semantic_threshold_for_language(language) for language in languages)


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
    """Replay the production hybrid tier from raw pair measurements."""
    row = dict(row)
    row["unit_records"] = (units[row["a"]], units[row["b"]])
    traditional = row["traditional"]
    exact = any(item["method"] in {"structural_hash", "token_hash"} for item in traditional)
    jaccard = max(
        (item["similarity"] for item in traditional if item["method"] == "jaccard"),
        default=None,
    )
    semantic = None
    if row["comparable"] and row["cosine"] >= _semantic_gate(profile, row, semantic_threshold):
        semantic = row["cosine"]
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
        corroborated = row["identifier_jaccard"] >= weak and row["statement_ratio"] >= ratio
        if high_gate is ...:
            language = row["unit_records"][0]["language"]
            promotion = profile.high_confidence_threshold_for_language(language)
        else:
            promotion = high_gate
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
    """Replay all candidate decisions, ordered like production reports."""
    profile = resolve_model_profile(measurement["metadata"]["identity"]["model"])
    units = {item["id"]: item for item in measurement["units"]}
    output = [
        finding
        for row in measurement["pairs"]
        if (
            finding := replay_pair(
                row,
                units,
                profile,
                semantic_threshold=semantic_threshold,
                weak_identifier_jaccard_min=weak_identifier_jaccard_min,
                statement_ratio_min=statement_ratio_min,
                high_gate=high_gate,
            )
        )
        is not None
    ]
    output.sort(key=lambda item: (-item["confidence"], item["a"], item["b"]))
    return output


def replay_parity(measurement: dict[str, Any]) -> dict[str, bool]:
    """Verify raw-table replay against live default and explicit analyzer output."""

    def signature(rows: list[dict[str, Any]]) -> set[tuple[tuple[str, str], str]]:
        return {(pair_key(row["a"], row["b"]), row["tier"]) for row in rows}

    metadata = measurement["metadata"]
    default_matches = signature(replay(measurement)) == signature(metadata["live_default"])
    explicit = metadata["live_explicit_override"]
    explicit_matches = signature(
        replay(measurement, semantic_threshold=explicit["threshold"], high_gate=None)
    ) == signature(explicit["findings"])
    if not default_matches or not explicit_matches:
        raise ValueError(
            "measurement replay diverges from live analyzer output: "
            f"default={default_matches}, explicit_override={explicit_matches}"
        )
    return {"default": default_matches, "explicit_override": explicit_matches}


def metrics(
    predicted: set[tuple[str, str]], labels: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    """Score reviewed positives/negatives while exposing ambiguity and unjudged output."""
    positives = {key for key, item in labels.items() if item["judgment"] == "positive"}
    negatives = {key for key, item in labels.items() if item["judgment"] == "negative"}
    ambiguities = {key for key, item in labels.items() if item["judgment"] == "ambiguous"}
    tp = len(predicted & positives)
    fp = len(predicted & negatives)
    fn = len(positives - predicted)
    ambiguous = len(predicted & ambiguities)
    unjudged = len(predicted - labels.keys())
    unresolved = ambiguous + unjudged
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    denominator = tp + fp + unresolved
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "reviewed_ambiguities": ambiguous,
        "unjudged_predictions": unjudged,
        "judged_only_precision": precision,
        "recall": recall,
        "precision_bounds": {
            "all_unresolved_negative": tp / denominator if denominator else None,
            "all_unresolved_positive": (tp + unresolved) / denominator if denominator else None,
        },
        "selection_eligible": unresolved == 0,
    }


def duplicate_report(project: Project, measurement: dict[str, Any]) -> dict[str, Any]:
    """Report full-pipeline, incremental semantic, and eligibility-conditional populations."""
    labels = judgments(project)
    findings = replay(measurement)
    published = {pair_key(item["a"], item["b"]) for item in findings}
    visible = {
        pair_key(item["a"], item["b"]) for item in findings if item["tier"] != "semantic_review"
    }
    traditional = {
        pair_key(row["a"], row["b"]) for row in measurement["pairs"] if row["traditional"]
    }
    semantic = {
        pair_key(item["a"], item["b"])
        for item in findings
        if item["semantic_similarity"] is not None
    }
    comparable = {pair_key(row["a"], row["b"]) for row in measurement["pairs"] if row["comparable"]}
    semantic_labels = {
        key: value for key, value in labels.items() if key not in traditional and key in comparable
    }
    by_tier = Counter(item["tier"] for item in findings)
    return {
        "default_visible": metrics(visible, labels),
        "complete_published": metrics(published, labels),
        "deterministic_only": metrics(traditional, labels),
        "semantic_incremental": metrics(semantic - traditional, semantic_labels),
        "semantic_eligibility_conditional": metrics(semantic & comparable, semantic_labels),
        "tiers": dict(sorted(by_tier.items())),
        "findings": [
            item
            | {
                "judgment": labels.get(pair_key(item["a"], item["b"]), {}).get(
                    "judgment", "unjudged"
                )
            }
            for item in findings
        ],
    }


def search_report(project: Project, measurement: dict[str, Any]) -> dict[str, Any]:
    """Score threshold retrieval, production top-k retrieval, and no-result behavior."""
    profile = resolve_model_profile(measurement["metadata"]["identity"]["model"])
    probes = {item["id"]: item for item in project.annotations["probes"]}
    by_probe = defaultdict(list)
    for row in measurement["query_scores"]:
        if row["cosine"] is not None:
            by_probe[row["probe"]].append(row)
    expected = {(probe["id"], unit) for probe in probes.values() for unit in probe["expected"]}
    threshold = profile.default_search_threshold
    threshold_output = {
        (row["probe"], row["unit"])
        for rows in by_probe.values()
        for row in rows
        if row["cosine"] >= threshold
    }
    top_output = {
        (row["probe"], row["unit"])
        for rows in by_probe.values()
        for row in rows
        if row["cosine"] >= threshold and row["rank"] <= DEFAULT_TOP_K
    }

    def relevance(result: set[tuple[str, str]]) -> dict[str, Any]:
        tp = len(result & expected)
        fp = len(result - expected)
        fn = len(expected - result)
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }

    no_result = {}
    for identifier, probe in probes.items():
        if probe["kind"] == "no_result":
            no_result[identifier] = sum(row["cosine"] >= threshold for row in by_probe[identifier])
    return {
        "threshold": threshold,
        "threshold_level": relevance(threshold_output),
        "production_top_k": DEFAULT_TOP_K,
        "top_k": relevance(top_output),
        "no_result_returned": no_result,
    }


def distribution_report(project: Project, measurement: dict[str, Any]) -> dict[str, Any]:
    """Summarize raw score populations without threshold selection."""
    labels = judgments(project)
    groups: dict[str, list[float]] = defaultdict(list)
    for row in measurement["pairs"]:
        if row["cosine"] is None:
            continue
        label = labels.get(pair_key(row["a"], row["b"]))
        groups[label["judgment"] if label else "unjudged"].append(row["cosine"])
        if label:
            for tag in label["tags"]:
                groups[f"tag:{tag}"].append(row["cosine"])
    return {
        key: {
            "count": len(values),
            "min": min(values),
            "median": float(np.median(values)),
            "max": max(values),
        }
        for key, values in sorted(groups.items())
    }


def review_queue(
    project: Project,
    measurements: list[dict[str, Any]],
    *,
    sample_size: int = 20,
    seed: int = 20,
) -> dict[str, Any]:
    """Build the required union of authored, deterministic, published, and sampled pairs."""
    authored = set(judgments(project))
    deterministic = set()
    published = set()
    all_pairs = set()
    for measurement in measurements:
        for row in measurement["pairs"]:
            key = pair_key(row["a"], row["b"])
            all_pairs.add(key)
            if row["traditional"]:
                deterministic.add(key)
        published.update(pair_key(item["a"], item["b"]) for item in replay(measurement))
    candidates = sorted(all_pairs - authored - deterministic - published)
    sample = sorted(random.Random(seed).sample(candidates, min(sample_size, len(candidates))))
    required = authored | deterministic | published | set(sample)
    labels = judgments(project)
    return {
        "seed": seed,
        "sample_size": len(sample),
        "authored": [list(key) for key in sorted(authored)],
        "deterministic": [list(key) for key in sorted(deterministic)],
        "published": [list(key) for key in sorted(published)],
        "background_sample": [list(key) for key in sample],
        "pending": [list(key) for key in sorted(required - labels.keys())],
        "queue_id": digest([list(key) for key in sorted(required)]),
    }


def compare_devices(cpu: dict[str, Any], mps: dict[str, Any], project: Project) -> dict[str, Any]:
    """Compare scores and production decisions from independent CPU and MPS runs."""
    cpu_pairs = {
        pair_key(row["a"], row["b"]): row for row in cpu["pairs"] if row["cosine"] is not None
    }
    mps_pairs = {
        pair_key(row["a"], row["b"]): row for row in mps["pairs"] if row["cosine"] is not None
    }
    pair_drifts = [
        abs(cpu_pairs[key]["cosine"] - mps_pairs[key]["cosine"])
        for key in cpu_pairs.keys() & mps_pairs.keys()
    ]
    cpu_queries = {
        (row["probe"], row["unit"]): row for row in cpu["query_scores"] if row["cosine"] is not None
    }
    mps_queries = {
        (row["probe"], row["unit"]): row for row in mps["query_scores"] if row["cosine"] is not None
    }
    query_drifts = [
        abs(cpu_queries[key]["cosine"] - mps_queries[key]["cosine"])
        for key in cpu_queries.keys() & mps_queries.keys()
    ]
    cpu_output = {pair_key(item["a"], item["b"]) for item in replay(cpu)}
    mps_output = {pair_key(item["a"], item["b"]) for item in replay(mps)}
    cpu_search = search_report(project, cpu)
    mps_search = search_report(project, mps)
    return {
        "pair_score_max_abs_drift": max(pair_drifts, default=0.0),
        "query_score_max_abs_drift": max(query_drifts, default=0.0),
        "duplicate_decision_changes": [list(key) for key in sorted(cpu_output ^ mps_output)],
        "search_top_k_metrics_changed": cpu_search["top_k"] != mps_search["top_k"],
    }


def full_report(project: Project, measurement: dict[str, Any]) -> dict[str, Any]:
    """Build a stateless report for one fixed policy measurement."""
    return {
        "schema_version": 1,
        "project": project.id,
        "measurement_id": measurement["metadata"]["measurement_id"],
        "annotation_sha256": digest(project.annotations),
        "replay_parity": replay_parity(measurement),
        "duplicate": duplicate_report(project, measurement),
        "search": search_report(project, measurement),
        "distributions": distribution_report(project, measurement),
        "selection": None,
        "selection_note": "Development pilot only; no replacement defaults are selected.",
    }


def load_all(
    project: Project, root: Path, models: list[str], devices: list[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load requested artifacts with source and pipeline validation."""
    return {
        (resolve_model_profile(model).key, device): load_measurement(
            artifact_path(root, project, model, device), project
        )
        for model in models
        for device in devices
    }


def write_full_report(path: Path, report: dict[str, Any]) -> None:
    """Write one evaluated report."""
    write_json(path, report)
