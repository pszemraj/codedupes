"""Evaluate calibration scores against explicit pair and search judgments."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from codedupes import semantic
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
SELECTION_SCHEMA_VERSION = 4
CHECKED_REPORT_SCHEMA_VERSION = 4
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_JUDGMENT_METRIC_KEYS = {
    "tp",
    "fp",
    "fn",
    "precision",
    "judged_only_precision",
    "recall",
    "f1",
    "ambiguous_predictions",
    "unjudged_predictions",
}
_DUPLICATE_TIERS = {
    "exact",
    "traditional_near",
    "hybrid_confirmed",
    "semantic_high_confidence",
    "semantic_review",
}


def _is_finite_number(value: Any) -> bool:
    """Return whether one JSON scalar is a finite non-boolean number."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _score_ratios(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Recompute precision, recall, and F1 from nonnegative confusion counts."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _validate_judgment_metrics(
    payload: Any,
    label: str,
    *,
    extra_keys: set[str] | None = None,
) -> None:
    """Validate one TP/FP/FN metric block and recompute every stored ratio."""
    extras = extra_keys or set()
    if not isinstance(payload, dict) or set(payload) != _JUDGMENT_METRIC_KEYS | extras:
        raise ValueError(f"{label} has an invalid metric schema")
    for field in ("tp", "fp", "fn", "ambiguous_predictions", "unjudged_predictions"):
        if type(payload[field]) is not int or payload[field] < 0:
            raise ValueError(f"{label} has invalid metric counts")
    precision, recall, f1 = _score_ratios(payload["tp"], payload["fp"], payload["fn"])
    expected = {
        "precision": precision,
        "judged_only_precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if any(
        not _is_finite_number(payload[field])
        or not 0.0 <= payload[field] <= 1.0
        or not math.isclose(payload[field], value, rel_tol=0.0, abs_tol=1e-12)
        for field, value in expected.items()
    ):
        raise ValueError(f"{label} has inconsistent derived metrics")


def _prediction_count(payload: dict[str, Any]) -> int:
    """Return the number of predictions represented by one judgment metric block."""
    return (
        payload["tp"]
        + payload["fp"]
        + payload["ambiguous_predictions"]
        + payload["unjudged_predictions"]
    )


def _validate_checked_duplicate_report(payload: Any, label: str) -> None:
    """Validate the complete duplicate-report schema and tier accounting."""
    metric_fields = {"published", "visible", "deterministic", "semantic_eligible"}
    if not isinstance(payload, dict) or set(payload) != metric_fields | {"tiers"}:
        raise ValueError(f"{label} has an invalid duplicate report schema")
    for field in metric_fields:
        _validate_judgment_metrics(payload[field], f"{label} {field}")

    tiers = payload["tiers"]
    if (
        not isinstance(tiers, dict)
        or not set(tiers) <= _DUPLICATE_TIERS
        or any(type(count) is not int or count < 0 for count in tiers.values())
    ):
        raise ValueError(f"{label} has invalid duplicate tier counts")
    if sum(tiers.values()) != _prediction_count(payload["published"]):
        raise ValueError(f"{label} duplicate tiers do not match published findings")
    if sum(
        count for tier, count in tiers.items() if tier != "semantic_review"
    ) != _prediction_count(payload["visible"]):
        raise ValueError(f"{label} duplicate tiers do not match visible findings")
    if tiers.get("semantic_high_confidence", 0) + tiers.get(
        "semantic_review", 0
    ) != _prediction_count(payload["semantic_eligible"]):
        raise ValueError(f"{label} duplicate tiers do not match semantic findings")


def _validate_checked_search_report(payload: Any, label: str, expected_threshold: float) -> None:
    """Validate one checked search summary and recompute its derived ratios."""
    expected_keys = {"threshold", "tp", "fp", "fn", "precision", "recall", "f1", "no_result"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError(f"{label} has an invalid search report schema")
    if (
        not _is_finite_number(payload["threshold"])
        or payload["threshold"] != expected_threshold
        or any(
            type(payload[field]) is not int or payload[field] < 0 for field in ("tp", "fp", "fn")
        )
    ):
        raise ValueError(f"{label} has invalid search report values")
    precision, recall, f1 = _score_ratios(payload["tp"], payload["fp"], payload["fn"])
    if any(
        not _is_finite_number(payload[field])
        or not 0.0 <= payload[field] <= 1.0
        or not math.isclose(payload[field], value, rel_tol=0.0, abs_tol=1e-12)
        for field, value in {"precision": precision, "recall": recall, "f1": f1}.items()
    ):
        raise ValueError(f"{label} has inconsistent search metrics")

    no_result = payload["no_result"]
    if (
        not isinstance(no_result, dict)
        or set(no_result) != {"clean", "total", "violations"}
        or type(no_result["clean"]) is not int
        or type(no_result["total"]) is not int
        or not 0 <= no_result["clean"] <= no_result["total"]
        or not isinstance(no_result["violations"], list)
        or any(not isinstance(item, str) or not item for item in no_result["violations"])
        or len(set(no_result["violations"])) != len(no_result["violations"])
        or no_result["total"] - no_result["clean"] != len(no_result["violations"])
    ):
        raise ValueError(f"{label} has invalid no-result search evidence")


def _validate_search_selection_metrics(payload: Any, label: str) -> None:
    """Validate a threshold sweep's compact search metric row."""
    expected_keys = {
        "threshold",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "no_result_clean",
        "no_result_total",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError(f"{label} has an invalid search metric schema")
    if (
        not _is_finite_number(payload["threshold"])
        or any(
            type(payload[field]) is not int or payload[field] < 0 for field in ("tp", "fp", "fn")
        )
        or type(payload["no_result_clean"]) is not int
        or type(payload["no_result_total"]) is not int
        or not 0 <= payload["no_result_clean"] <= payload["no_result_total"]
    ):
        raise ValueError(f"{label} has invalid search metric values")
    precision, recall, f1 = _score_ratios(payload["tp"], payload["fp"], payload["fn"])
    if any(
        not _is_finite_number(payload[field])
        or not 0.0 <= payload[field] <= 1.0
        or not math.isclose(payload[field], value, rel_tol=0.0, abs_tol=1e-12)
        for field, value in {"precision": precision, "recall": recall, "f1": f1}.items()
    ):
        raise ValueError(f"{label} has inconsistent search metrics")


def _judgment_metric_core(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the common judgment fields from a validated metric payload."""
    return {key: payload[key] for key in _JUDGMENT_METRIC_KEYS}


def _combine_judgment_metrics(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate validated judgment metrics and recompute their ratios."""
    counts = {
        field: sum(payload[field] for payload in payloads)
        for field in ("tp", "fp", "fn", "ambiguous_predictions", "unjudged_predictions")
    }
    precision, recall, f1 = _score_ratios(counts["tp"], counts["fp"], counts["fn"])
    return {
        **counts,
        "precision": precision,
        "judged_only_precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _semantic_visible_metrics(duplicate_report: dict[str, Any]) -> dict[str, Any]:
    """Remove traditionally recovered predictions from a visible report block."""
    visible = duplicate_report["visible"]
    deterministic = duplicate_report["deterministic"]
    counts = {
        "tp": visible["tp"] - deterministic["tp"],
        "fp": visible["fp"] - deterministic["fp"],
        "fn": visible["fn"],
        "ambiguous_predictions": visible["ambiguous_predictions"]
        - deterministic["ambiguous_predictions"],
        "unjudged_predictions": visible["unjudged_predictions"]
        - deterministic["unjudged_predictions"],
    }
    if any(value < 0 for value in counts.values()):
        raise ValueError("checked deterministic findings are not a subset of visible findings")
    precision, recall, f1 = _score_ratios(counts["tp"], counts["fp"], counts["fn"])
    return {
        **counts,
        "precision": precision,
        "judged_only_precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _combine_search_metrics(payloads: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Aggregate validated checked search reports into a selection metric row."""
    tp = sum(payload["tp"] for payload in payloads)
    fp = sum(payload["fp"] for payload in payloads)
    fn = sum(payload["fn"] for payload in payloads)
    precision, recall, f1 = _score_ratios(tp, fp, fn)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "no_result_clean": sum(payload["no_result"]["clean"] for payload in payloads),
        "no_result_total": sum(payload["no_result"]["total"] for payload in payloads),
    }


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


def selection_objective() -> dict[str, float | str]:
    """Return the exact, versioned objective used by both selection sweeps."""
    return {
        "primary": "f1",
        "minimum_precision": MINIMUM_SELECTION_PRECISION,
        "recall_preference_max_f1_loss": F1_RECALL_TOLERANCE,
    }


def validate_selection_contract(payload: dict[str, Any]) -> None:
    """Reject a derived selection with a foreign schema or objective contract."""
    if not isinstance(payload, dict):
        raise ValueError("selection must be a JSON object")  # noqa: TRY004
    if payload.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError("selection has an unsupported schema version")
    if payload.get("objective") != selection_objective():
        raise ValueError("selection has a mismatched objective contract")


def support_files_digest(project: Project) -> str:
    """Fingerprint declared support files and behavior-test trees."""
    files: dict[str, str] = {}
    for pattern in [*project.spec["support_files"], *project.spec["test_roots"]]:
        relative = Path(pattern)
        if not pattern or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{project.id}: invalid support file pattern {pattern!r}")
        matches = sorted(project.root.glob(pattern))
        matched_files = [
            file
            for match in matches
            for file in ([match] if match.is_file() else sorted(match.rglob("*")))
            if file.is_file()
            and not ({"__pycache__", "node_modules", "target"} & set(file.parts))
            and file.name != ".DS_Store"
            and file.suffix not in {".pyc", ".pyo"}
        ]
        if not matched_files:
            raise ValueError(f"{project.id}: support file pattern matched nothing: {pattern}")
        for file in matched_files:
            files[file.relative_to(project.root).as_posix()] = hashlib.sha256(
                file.read_bytes()
            ).hexdigest()
    return selection_digest(files)


def measurement_digest(measurement: dict[str, Any]) -> str:
    """Bind derived selections to the score payload and its inference provenance."""
    metadata = measurement["metadata"]
    return selection_digest(
        {
            "schema_version": measurement["schema_version"],
            "metadata": {
                key: metadata[key]
                for key in (
                    "project",
                    "model",
                    "canonical_model",
                    "revision",
                    "requested_device",
                    "batch_size",
                    "inference_dtype",
                    "runtime_versions",
                    "input_fingerprint",
                    "captured_profile",
                )
            },
            "units": measurement["units"],
            "pairs": measurement["pairs"],
            "query_scores": measurement["query_scores"],
        }
    )


def measurement_digests(measurements: list[dict[str, Any]]) -> dict[str, str]:
    """Index stable raw-score digests by project, model, and device."""
    return {
        "/".join(
            (
                measurement["metadata"]["project"],
                measurement["metadata"]["model"],
                measurement["metadata"]["requested_device"],
            )
        ): measurement_digest(measurement)
        for measurement in measurements
    }


def validate_measurement_digests(
    payload: dict[str, Any], measurements: list[dict[str, Any]]
) -> None:
    """Reject a derived selection produced from different raw score payloads."""
    expected = payload.get("measurement_digests")
    actual = measurement_digests(measurements)
    if not isinstance(expected, dict) or not expected or expected != actual:
        raise ValueError("selection used different raw measurements; rerun the selection sweeps")


def validate_measurement_provenance(project: Project, measurement: dict[str, Any]) -> None:
    """Verify report metadata against the current model, runtime, and fresh-run policy."""
    metadata = measurement["metadata"]
    profile = resolve_model_profile(metadata["model"])
    device = metadata["requested_device"]
    expected_dtype = str(semantic._resolve_model_dtype(profile.family, device)).removeprefix(
        "torch."
    )
    expected_profile = {
        "semantic_threshold": profile.semantic_threshold_for_language(project.spec["languages"][0]),
        "weak_identifier_jaccard_min": profile.hybrid_weak_identifier_jaccard_min,
        "statement_ratio_min": profile.hybrid_statement_ratio_min,
        "high_gate": profile.high_confidence_threshold_for_language(project.spec["languages"][0]),
    }
    if (
        metadata["canonical_model"] != profile.canonical_name
        or metadata["revision"] != profile.default_revision
        or metadata["runtime_versions"] != semantic.get_semantic_runtime_versions()
        or metadata["inference_dtype"] != expected_dtype
        or metadata["captured_profile"] != expected_profile
    ):
        raise ValueError(f"{project.id}: measurement provenance does not match current policy")
    if set(metadata["timing_seconds"]) != {"duplicate", "search"} or any(
        not _is_finite_number(value) or value < 0 for value in metadata["timing_seconds"].values()
    ):
        raise ValueError(f"{project.id}: invalid measurement timing provenance")
    encoded_inputs = sum(unit["embedded"] for unit in measurement["units"])
    if set(metadata["execution"]) != {"duplicate", "search"}:
        raise ValueError(f"{project.id}: incomplete measurement execution provenance")
    for task, stats in metadata["execution"].items():
        if (
            stats["execution_device"] != device
            or stats["cache_hit_rows"] != 0
            or stats["cache_enabled"] is not False
            or stats["model_loaded"] is not True
            or stats["requested_rows"] != encoded_inputs
            or not 0 < stats["unique_inputs"] <= encoded_inputs
            or stats["encoded_inputs"] != encoded_inputs
        ):
            raise ValueError(f"{project.id}: invalid {task} execution provenance")


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
                "support_files": support_files_digest(project),
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


def _selection_models(payload: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    """Index one selection's model entries while rejecting malformed duplicates."""
    entries = payload.get("models")
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError(f"{label} selection models must be objects")
    mapped = {entry.get("model"): entry for entry in entries}
    if len(mapped) != len(entries) or any(not isinstance(model, str) for model in mapped):
        raise ValueError(f"{label} selection has invalid model entries")
    return mapped


def _selection_gate_map(entries: Any, field: str, label: str) -> dict[str, float | None]:
    """Index per-language selected gates with one unambiguous value per language."""
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError(f"{label} selection gates must be objects")
    gates = {entry.get("language"): entry.get(field) for entry in entries}
    if len(gates) != len(entries) or any(not isinstance(language, str) for language in gates):
        raise ValueError(f"{label} selection has invalid language gates")
    return gates


def _same_gate(actual: Any, expected: float | None) -> bool:
    """Compare optional gate values without accepting boolean JSON scalars."""
    return actual is None if expected is None else _is_finite_number(actual) and actual == expected


def validate_shipped_selection_profiles(
    threshold_selection: dict[str, Any],
    hybrid_selection: dict[str, Any],
    models: list[str],
    languages: set[str],
) -> None:
    """Require checked threshold and hybrid selections to match shipped profile gates."""
    validate_selection_contract(threshold_selection)
    validate_selection_contract(hybrid_selection)
    expected_models = {resolve_model_profile(model).key for model in models}
    thresholds = _selection_models(threshold_selection, "threshold")
    hybrids = _selection_models(hybrid_selection, "hybrid")
    if thresholds.keys() != expected_models or hybrids.keys() != expected_models:
        raise ValueError("calibration selections do not cover the requested shipped profiles")

    for model in expected_models:
        profile = resolve_model_profile(model)
        threshold = thresholds[model]
        if set(threshold) != {"model", "duplicate_by_language", "search"}:
            raise ValueError(f"{model}: threshold selection has an invalid schema")
        duplicate_entries = threshold.get("duplicate_by_language")
        duplicate_gates = _selection_gate_map(duplicate_entries, "selected_threshold", "threshold")
        expected_duplicate_gates = {
            language: profile.semantic_threshold_for_language(language) for language in languages
        }
        if (
            duplicate_gates.keys() != languages
            or any(
                not _same_gate(duplicate_gates[language], expected)
                for language, expected in expected_duplicate_gates.items()
            )
            or any(item.get("selection_ready") is not True for item in duplicate_entries)
        ):
            raise ValueError(f"{model}: selected calibration gates do not match shipped defaults")
        for item in duplicate_entries:
            language = item["language"]
            expected_gate = expected_duplicate_gates[language]
            if set(item) != {
                "language",
                "project",
                "current_threshold",
                "current_metrics",
                "current_difficulty_recall",
                "selected_threshold",
                "selected_metrics",
                "selection_ready",
                "selected_difficulty_recall",
                "positive_scores",
                "negative_scores",
                "unjudged_above_selected",
            } or not _same_gate(item.get("current_threshold"), expected_gate):
                raise ValueError(f"{model}: threshold selection has an invalid schema")
            for field in ("current_metrics", "selected_metrics"):
                metrics_payload = item.get(field)
                _validate_judgment_metrics(
                    metrics_payload,
                    f"{model} {language} {field}",
                    extra_keys={"threshold", "predicted"},
                )
                if (
                    not _same_gate(metrics_payload["threshold"], expected_gate)
                    or type(metrics_payload["predicted"]) is not int
                    or metrics_payload["predicted"] != _prediction_count(metrics_payload)
                ):
                    raise ValueError(f"{model} {language} {field} is inconsistent")

        search = threshold.get("search")
        if (
            not isinstance(search, dict)
            or set(search)
            != {
                "current_threshold",
                "current_metrics",
                "selected_threshold",
                "selected_metrics",
                "selection_window",
            }
            or not _same_gate(search.get("selected_threshold"), profile.default_search_threshold)
            or not _same_gate(search.get("current_threshold"), profile.default_search_threshold)
        ):
            raise ValueError(f"{model}: selected calibration gates do not match shipped defaults")
        for field in ("current_metrics", "selected_metrics"):
            _validate_search_selection_metrics(search.get(field), f"{model} search {field}")
            if search[field]["threshold"] != profile.default_search_threshold:
                raise ValueError(f"{model}: search selection metrics use another threshold")
        window = search.get("selection_window")
        if not isinstance(window, list):
            raise ValueError(  # noqa: TRY004 -- checked JSON is one validation failure type
                f"{model}: search selection window must be a list"
            )
        for index, row in enumerate(window):
            _validate_search_selection_metrics(row, f"{model} search window row {index}")

        hybrid = hybrids[model]
        if set(hybrid) != {
            "model",
            "admission_thresholds",
            "current",
            "selected",
            "promotion_by_language",
        }:
            raise ValueError(f"{model}: hybrid selection has an invalid schema")
        admissions = hybrid.get("admission_thresholds")
        current = hybrid.get("current")
        selected = hybrid.get("selected")
        promotion_entries = hybrid.get("promotion_by_language")
        promotion_gates = _selection_gate_map(promotion_entries, "selected_gate", "hybrid")
        expected_promotion_gates = {
            language: profile.high_confidence_threshold_for_language(language)
            for language in languages
        }
        if (
            admissions != expected_duplicate_gates
            or not isinstance(current, dict)
            or set(current)
            != {
                "admission_thresholds",
                "weak_identifier_jaccard_min",
                "statement_ratio_min",
                "metrics",
            }
            or current.get("admission_thresholds") != expected_duplicate_gates
            or not _same_gate(
                current.get("weak_identifier_jaccard_min"),
                profile.hybrid_weak_identifier_jaccard_min,
            )
            or not _same_gate(
                current.get("statement_ratio_min"), profile.hybrid_statement_ratio_min
            )
            or not isinstance(selected, dict)
            or set(selected)
            != {
                "weak_identifier_jaccard_min",
                "statement_ratio_min",
                "metrics",
                "corroboration_only_metrics",
                "selection_ready",
            }
            or not _same_gate(
                selected.get("weak_identifier_jaccard_min"),
                profile.hybrid_weak_identifier_jaccard_min,
            )
            or not _same_gate(
                selected.get("statement_ratio_min"), profile.hybrid_statement_ratio_min
            )
            or selected.get("selection_ready") is not True
            or promotion_gates.keys() != languages
            or any(
                not _same_gate(promotion_gates[language], expected)
                for language, expected in expected_promotion_gates.items()
            )
            or any(item.get("selection_ready") is not True for item in promotion_entries)
        ):
            raise ValueError(f"{model}: selected calibration gates do not match shipped defaults")
        _validate_judgment_metrics(current.get("metrics"), f"{model} current hybrid metrics")
        _validate_judgment_metrics(selected.get("metrics"), f"{model} selected hybrid metrics")
        _validate_judgment_metrics(
            selected.get("corroboration_only_metrics"),
            f"{model} corroboration-only hybrid metrics",
            extra_keys={"weak_identifier_jaccard_min", "statement_ratio_min"},
        )
        corroboration = selected["corroboration_only_metrics"]
        if not _same_gate(
            corroboration["weak_identifier_jaccard_min"],
            profile.hybrid_weak_identifier_jaccard_min,
        ) or not _same_gate(
            corroboration["statement_ratio_min"], profile.hybrid_statement_ratio_min
        ):
            raise ValueError(f"{model}: corroboration-only metrics use another hybrid policy")
        for item in promotion_entries:
            language = item["language"]
            if set(item) != {
                "language",
                "current_gate",
                "selected_gate",
                "selected_metrics",
                "selection_ready",
            } or not _same_gate(item.get("current_gate"), expected_promotion_gates[language]):
                raise ValueError(f"{model}: hybrid promotion selection has an invalid schema")
            _validate_judgment_metrics(
                item.get("selected_metrics"), f"{model} {language} promotion metrics"
            )


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
    if cpu_pairs.keys() != mps_pairs.keys():
        raise ValueError(f"{project.id}: CPU/MPS pair score coverage differs")
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
    if cpu_queries.keys() != mps_queries.keys():
        raise ValueError(f"{project.id}: CPU/MPS query score coverage differs")
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
        "batch_size": measurement["metadata"]["batch_size"],
        "inference_dtype": measurement["metadata"]["inference_dtype"],
        "runtime_versions": measurement["metadata"]["runtime_versions"],
        "timing_seconds": measurement["metadata"]["timing_seconds"],
        "execution": execution,
        "replay_parity": replay_parity(measurement),
        "duplicate": duplicate_report(project, measurement),
        "search": search_report(project, measurement),
    }


def _validate_checked_selection_outcomes(
    threshold_selection: dict[str, Any],
    hybrid_selection: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
    profile_keys: tuple[str, ...],
) -> None:
    """Bind embedded selection metrics to the checked CPU report summaries."""
    threshold_models = _selection_models(threshold_selection, "threshold")
    hybrid_models = _selection_models(hybrid_selection, "hybrid")
    records_by_language: dict[str, list[dict[str, Any]]] = {}
    for record in records_by_id.values():
        records_by_language.setdefault(record["language"], []).append(record)

    for model in profile_keys:
        profile = resolve_model_profile(model)
        threshold = threshold_models[model]
        for entry in threshold["duplicate_by_language"]:
            project_id = entry["project"]
            record = records_by_id.get(project_id)
            if record is None or record["language"] != entry["language"]:
                raise ValueError(f"{model}: threshold selection names an unknown project")
            expected = record["reports"][f"{model}/cpu"]["duplicate"]["semantic_eligible"]
            for field in ("current_metrics", "selected_metrics"):
                if _judgment_metric_core(entry[field]) != expected:
                    raise ValueError(
                        f"{model}: threshold selection metrics differ from checked CPU reports"
                    )

        search_reports = [
            record["reports"][f"{model}/cpu"]["search"] for record in records_by_id.values()
        ]
        expected_search = _combine_search_metrics(search_reports, profile.default_search_threshold)
        for field in ("current_metrics", "selected_metrics"):
            if threshold["search"][field] != expected_search:
                raise ValueError(
                    f"{model}: search selection metrics differ from checked CPU reports"
                )

        hybrid = hybrid_models[model]
        visible_by_language = {
            language: _combine_judgment_metrics(
                [
                    _semantic_visible_metrics(record["reports"][f"{model}/cpu"]["duplicate"])
                    for record in records
                ]
            )
            for language, records in records_by_language.items()
        }
        for entry in hybrid["promotion_by_language"]:
            if entry["selected_metrics"] != visible_by_language.get(entry["language"]):
                raise ValueError(f"{model}: promotion metrics differ from checked CPU reports")
        expected_visible = _combine_judgment_metrics(list(visible_by_language.values()))
        if (
            hybrid["current"]["metrics"] != expected_visible
            or hybrid["selected"]["metrics"] != expected_visible
        ):
            raise ValueError(f"{model}: hybrid metrics differ from checked CPU reports")


def validate_checked_report(
    payload: dict[str, Any], projects: list[Project], models: list[str]
) -> None:
    """Validate the committed report shape and provenance invariants without raw scores.

    Raw score matrices are intentionally local-only, so this verifies every
    invariant available in the checked summary: its selection schema, objective,
    context, linkage, and shipped gates; its report inventory; raw digest
    references; report identities; execution provenance; and shared runtime
    summary. It cannot establish that a syntactically valid digest names a
    particular uncommitted raw artifact.

    :param dict payload: Parsed committed calibration report.
    :param list projects: Manifest projects expected in the report.
    :param list models: Built-in profiles expected for every project/device.
    :raises ValueError: If the checked report is malformed or internally inconsistent.
    :return: None
    """
    if payload.get("schema_version") != CHECKED_REPORT_SCHEMA_VERSION:
        raise ValueError("checked calibration report has an unsupported schema")

    profile_keys = tuple(dict.fromkeys(resolve_model_profile(model).key for model in models))
    project_by_id = {project.id: project for project in projects}
    if len(project_by_id) != len(projects):
        raise ValueError("checked calibration report has duplicate expected projects")
    selection_projects = list(project_by_id.values())
    selection_languages = {project.spec["languages"][0] for project in selection_projects}
    threshold_selection = payload.get("threshold_selection")
    hybrid_selection = payload.get("hybrid_selection")
    if not isinstance(threshold_selection, dict) or not isinstance(hybrid_selection, dict):
        raise ValueError(  # noqa: TRY004 -- checked JSON is one validation failure type
            "checked calibration report selections must be objects"
        )
    if set(threshold_selection) != {
        "schema_version",
        "objective",
        "input_context",
        "grids",
        "models",
        "measurement_digests",
    } or set(hybrid_selection) != {
        "schema_version",
        "objective",
        "input_context",
        "threshold_selection_digest",
        "models",
        "measurement_digests",
    }:
        raise ValueError("checked calibration report selections have an invalid schema")
    validate_selection_contract(threshold_selection)
    validate_selection_contract(hybrid_selection)
    validate_selection_context(threshold_selection, selection_projects, list(profile_keys))
    validate_selection_context(hybrid_selection, selection_projects, list(profile_keys))
    if hybrid_selection.get("threshold_selection_digest") != selection_digest(threshold_selection):
        raise ValueError("checked calibration report hybrid selection has another threshold input")
    validate_shipped_selection_profiles(
        threshold_selection,
        hybrid_selection,
        list(profile_keys),
        selection_languages,
    )
    expected_report_keys = {
        f"{model}/{device}" for model in profile_keys for device in ("cpu", "mps")
    }
    expected_digest_keys = {
        f"{project.id}/{model}/{device}"
        for project in projects
        for model in profile_keys
        for device in ("cpu", "mps")
    }

    digests = payload.get("measurement_digests")
    if (
        not isinstance(digests, dict)
        or set(digests) != expected_digest_keys
        or any(
            not isinstance(value, str) or _SHA256_HEX.fullmatch(value) is None
            for value in digests.values()
        )
    ):
        raise ValueError("checked calibration report has invalid measurement digests")
    cpu_digests = {key: value for key, value in digests.items() if key.endswith("/cpu")}
    for label, selection in (
        ("threshold", threshold_selection),
        ("hybrid", hybrid_selection),
    ):
        if selection.get("measurement_digests") != cpu_digests:
            raise ValueError(
                f"checked calibration report {label} selection has mismatched measurement digests"
            )

    records = payload.get("projects")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError("checked calibration report projects must be objects")
    record_ids = [record.get("project") for record in records]
    if any(not isinstance(project_id, str) for project_id in record_ids):
        raise ValueError("checked calibration report project IDs must be strings")
    records_by_id = dict(zip(record_ids, records, strict=True))
    if len(records_by_id) != len(records) or set(records_by_id) != set(project_by_id):
        raise ValueError("checked calibration report project inventory differs from the manifest")

    report_runtimes: list[dict[str, str]] = []
    report_devices: set[str] = set()
    for project_id, project in project_by_id.items():
        record = records_by_id[project_id]
        expected_corpus = {
            "annotated_units": len(project.annotations["units"]),
            "positive_pairs": sum(
                pair["judgment"] == "positive" for pair in project.annotations["pairs"]
            ),
            "negative_pairs": sum(
                pair["judgment"] == "negative" for pair in project.annotations["pairs"]
            ),
            "probes": len(project.annotations["probes"]),
        }
        if (
            set(record)
            != {"project", "split", "language", "corpus", "reports", "device_comparisons"}
            or record.get("split") != project.spec["split"]
            or record.get("language") != project.spec["languages"][0]
            or record.get("corpus") != expected_corpus
        ):
            raise ValueError(
                f"{project_id}: checked report project identity differs from the manifest"
            )
        reports = record.get("reports")
        if not isinstance(reports, dict) or set(reports) != expected_report_keys:
            raise ValueError(f"{project_id}: checked report inventory is incomplete")
        for model in profile_keys:
            profile = resolve_model_profile(model)
            for device in ("cpu", "mps"):
                report = reports[f"{model}/{device}"]
                expected_dtype = str(
                    semantic._resolve_model_dtype(profile.family, device)
                ).removeprefix("torch.")
                if not isinstance(report, dict):
                    raise ValueError(  # noqa: TRY004 -- checked JSON is one validation failure type
                        f"{project_id}: checked report entry is not an object"
                    )
                if (
                    set(report)
                    != {
                        "project",
                        "model",
                        "device",
                        "batch_size",
                        "inference_dtype",
                        "runtime_versions",
                        "timing_seconds",
                        "execution",
                        "replay_parity",
                        "duplicate",
                        "search",
                    }
                    or report.get("project") != project_id
                    or report.get("model") != model
                    or report.get("device") != device
                    or type(report.get("batch_size")) is not int
                    or report["batch_size"] <= 0
                    or report.get("inference_dtype") != expected_dtype
                    or report.get("replay_parity") is not True
                    or not isinstance(report.get("duplicate"), dict)
                    or not isinstance(report.get("search"), dict)
                ):
                    raise ValueError(f"{project_id}: checked report identity is invalid")
                _validate_checked_duplicate_report(
                    report["duplicate"], f"{project_id}/{model}/{device}"
                )
                _validate_checked_search_report(
                    report["search"],
                    f"{project_id}/{model}/{device}",
                    profile.default_search_threshold,
                )

                runtime = report.get("runtime_versions")
                if (
                    not isinstance(runtime, dict)
                    or not runtime
                    or any(
                        not isinstance(key, str) or not isinstance(value, str) or not value
                        for key, value in runtime.items()
                    )
                ):
                    raise ValueError(f"{project_id}: checked report runtime provenance is invalid")
                report_runtimes.append(runtime)
                report_devices.add(device)

                timings = report.get("timing_seconds")
                if (
                    not isinstance(timings, dict)
                    or set(timings) != {"duplicate", "search"}
                    or any(not _is_finite_number(value) or value < 0 for value in timings.values())
                ):
                    raise ValueError(f"{project_id}: checked report timings are invalid")

                execution = report.get("execution")
                if not isinstance(execution, dict) or set(execution) != {"duplicate", "search"}:
                    raise ValueError(
                        f"{project_id}: checked report execution provenance is incomplete"
                    )
                for task, stats in execution.items():
                    if (
                        not isinstance(stats, dict)
                        or set(stats) != {"device", "encoded_inputs", "cache_hit_rows"}
                        or stats["device"] != device
                        or type(stats["encoded_inputs"]) is not int
                        or stats["encoded_inputs"] <= 0
                        or type(stats["cache_hit_rows"]) is not int
                        or stats["cache_hit_rows"] != 0
                    ):
                        raise ValueError(
                            f"{project_id}: invalid checked {task} execution provenance"
                        )

        comparisons = record["device_comparisons"]
        if not isinstance(comparisons, dict) or set(comparisons) != set(profile_keys):
            raise ValueError(f"{project_id}: checked report device comparisons are incomplete")
        for model, comparison in comparisons.items():
            if (
                not isinstance(comparison, dict)
                or set(comparison)
                != {
                    "pair_score_abs_drift",
                    "query_score_abs_drift",
                    "duplicate_decision_changes",
                    "search_decision_changes",
                }
                or not isinstance(comparison["duplicate_decision_changes"], list)
                or not isinstance(comparison["search_decision_changes"], list)
            ):
                raise ValueError(f"{project_id}: invalid checked {model} device comparison")
            for field in ("pair_score_abs_drift", "query_score_abs_drift"):
                summary = comparison[field]
                if (
                    not isinstance(summary, dict)
                    or set(summary) != {"count", "p95", "max"}
                    or type(summary["count"]) is not int
                    or summary["count"] < 0
                    or any(
                        value is not None and (not _is_finite_number(value) or value < 0)
                        for value in (summary["p95"], summary["max"])
                    )
                    or (summary["count"] == 0) != (summary["p95"] is None)
                    or (summary["count"] == 0) != (summary["max"] is None)
                    or (
                        summary["p95"] is not None
                        and summary["max"] is not None
                        and summary["p95"] > summary["max"]
                    )
                ):
                    raise ValueError(f"{project_id}: invalid checked {model} {field}")
            for field in ("duplicate_decision_changes", "search_decision_changes"):
                changes = comparison[field]
                serialized = [tuple(item) for item in changes if isinstance(item, list)]
                if (
                    len(serialized) != len(changes)
                    or any(
                        len(item) != 2
                        or any(not isinstance(value, str) or not value for value in item)
                        for item in serialized
                    )
                    or len(set(serialized)) != len(serialized)
                ):
                    raise ValueError(f"{project_id}: invalid checked {model} {field}")

    _validate_checked_selection_outcomes(
        threshold_selection,
        hybrid_selection,
        records_by_id,
        profile_keys,
    )

    if not report_runtimes or any(runtime != report_runtimes[0] for runtime in report_runtimes[1:]):
        raise ValueError("checked calibration report has mixed runtime provenance")
    runtime_summary = payload.get("measurement_runtime")
    expected_scope = (
        f"all checked {' and '.join(sorted(device.upper() for device in report_devices))} reports"
    )
    if (
        not isinstance(runtime_summary, dict)
        or set(runtime_summary) != {"torch", "scope"}
        or runtime_summary["torch"] != report_runtimes[0].get("torch")
        or runtime_summary["scope"] != expected_scope
    ):
        raise ValueError("checked calibration report runtime summary is invalid")


def load_all(
    project: Project, root: Path, models: list[str], devices: list[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load requested raw measurements."""
    measurements = {
        (resolve_model_profile(model).key, device): load_measurement(
            artifact_path(root, project, model, device),
            project,
            expected_model=resolve_model_profile(model).key,
            expected_device=device,
        )
        for model in models
        for device in devices
    }
    for measurement in measurements.values():
        validate_measurement_provenance(project, measurement)
    return measurements


def write_full_report(path: Path, report: dict[str, Any]) -> None:
    """Write one compact report."""
    write_json(path, report)
