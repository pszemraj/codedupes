"""Select hybrid visibility gates after semantic admission calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from codedupes.semantic_profiles import list_supported_models, resolve_model_profile

try:
    from .calibration_contract import (
        add_contract_arguments,
        load_projects,
        pair_key,
        read_json,
        write_json,
    )
    from .calibration_evaluation import judgments, load_all, metrics, replay
    from .calibration_measurements import DEFAULT_MEASUREMENTS
    from .sweep_semantic_thresholds import threshold_grid
except ImportError:
    from calibration_contract import (
        add_contract_arguments,
        load_projects,
        pair_key,
        read_json,
        write_json,
    )
    from calibration_evaluation import judgments, load_all, metrics, replay
    from calibration_measurements import DEFAULT_MEASUREMENTS
    from sweep_semantic_thresholds import threshold_grid

WEAK_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
RATIO_GRID = (0.0, 0.20, 0.35, 0.50, 0.65, 0.80)


def _best(rows: list[dict[str, Any]], *, prefer_off: bool = False) -> dict[str, Any]:
    """Maximize visible F1, recall, and precision with a stable simple tie break."""
    key = max(
        (
            row["f1"],
            -row.get("ambiguous_predictions", 0),
            -row.get("unjudged_predictions", 0),
            row["recall"],
            row["precision"],
        )
        for row in rows
    )
    tied = [
        row
        for row in rows
        if (
            row["f1"],
            -row.get("ambiguous_predictions", 0),
            -row.get("unjudged_predictions", 0),
            row["recall"],
            row["precision"],
        )
        == key
    ]
    if prefer_off:
        off = next((row for row in tied if row.get("high_gate") is None), None)
        if off is not None:
            return off
    return min(
        tied,
        key=lambda row: (
            row.get("weak_identifier_jaccard_min", 0),
            row.get("statement_ratio_min", 0),
            row.get("high_gate") or 0,
        ),
    )


def _selection_map(payload: dict[str, Any]) -> dict[tuple[str, str], float]:
    """Index selected semantic gates by model and language."""
    unready = [
        (model["model"], item["language"])
        for model in payload["models"]
        for item in model["duplicate_by_language"]
        if not item["selection_ready"]
    ]
    if unready:
        raise ValueError(f"review unjudged admission findings before hybrid selection: {unready}")
    return {
        (model["model"], item["language"]): item["selected_threshold"]
        for model in payload["models"]
        for item in model["duplicate_by_language"]
    }


def _semantic_labels(project: Any, measurement: dict[str, Any]) -> dict[tuple[str, str], dict]:
    """Keep labels whose visibility is actually controlled by hybrid gates."""
    measured = {
        pair_key(row["a"], row["b"]): row
        for row in measurement["pairs"]
        if row["comparable"] and not row["traditional"]
    }
    return {
        key: value
        for key, value in judgments(project).items()
        if key in measured and value["judgment"] in {"positive", "negative", "ambiguous"}
    }


def _visible_semantic(
    measurement: dict[str, Any],
    admission: float,
    weak: float,
    ratio: float,
    high_gate: float | None,
) -> set[tuple[str, str]]:
    """Return semantic-only pairs visible under one hybrid setting."""
    return {
        pair_key(item["a"], item["b"])
        for item in replay(
            measurement,
            semantic_threshold=admission,
            weak_identifier_jaccard_min=weak,
            statement_ratio_min=ratio,
            high_gate=high_gate,
        )
        if item["tier"] == "semantic_high_confidence"
    }


def _pooled_metrics(
    projects: list[Any],
    measurements: dict[str, dict[str, Any]],
    admissions: dict[str, float],
    weak: float,
    ratio: float,
    high_gates: dict[str, float | None],
) -> dict[str, Any]:
    """Pool semantic-only judgments across projects without ID collisions."""
    predicted = set()
    labels = {}
    for project in projects:
        language = project.spec["languages"][0]
        measurement = measurements[project.id]
        predicted.update(
            (project.id, *key)
            for key in _visible_semantic(
                measurement,
                admissions[language],
                weak,
                ratio,
                high_gates.get(language),
            )
        )
        labels.update(
            {
                (project.id, *key): value
                for key, value in _semantic_labels(project, measurement).items()
            }
        )
    return metrics(predicted, labels)


def main() -> int:
    """Select global corroboration and per-language promotion gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_contract_arguments(parser)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument(
        "--threshold-selection",
        type=Path,
        default=DEFAULT_MEASUREMENTS / "threshold-selection.json",
    )
    parser.add_argument("--models", nargs="+", default=[p.key for p in list_supported_models()])
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    projects = load_projects(args.manifest, args.projects, args.policy)
    selected_admissions = _selection_map(read_json(args.threshold_selection))
    payload: dict[str, Any] = {"schema_version": 3, "models": []}

    for model in args.models:
        profile = resolve_model_profile(model)
        measurements = {
            project.id: load_all(project, args.measurements, [model], ["cpu"])[(profile.key, "cpu")]
            for project in projects
        }
        admissions = {
            project.spec["languages"][0]: selected_admissions[
                (profile.key, project.spec["languages"][0])
            ]
            for project in projects
        }
        shipped_admissions = {
            language: profile.semantic_threshold_for_language(language) for language in admissions
        }
        stage1 = []
        for weak in WEAK_GRID:
            for ratio in RATIO_GRID:
                stage1.append(
                    {
                        "weak_identifier_jaccard_min": weak,
                        "statement_ratio_min": ratio,
                        **_pooled_metrics(
                            projects,
                            measurements,
                            admissions,
                            weak,
                            ratio,
                            {language: None for language in admissions},
                        ),
                    }
                )
        corroboration = _best(stage1)
        weak = corroboration["weak_identifier_jaccard_min"]
        ratio = corroboration["statement_ratio_min"]

        promotion = []
        selected_gates: dict[str, float | None] = {}
        for project in projects:
            language = project.spec["languages"][0]
            measurement = measurements[project.id]
            labels = _semantic_labels(project, measurement)
            candidates: list[float | None] = [
                None,
                *threshold_grid(admissions[language], 0.98, 0.01),
            ]
            rows = []
            for high_gate in candidates:
                predicted = _visible_semantic(
                    measurement, admissions[language], weak, ratio, high_gate
                )
                rows.append({"high_gate": high_gate, **metrics(predicted, labels)})
            selected = _best(rows, prefer_off=True)
            selected_gates[language] = selected["high_gate"]
            promotion.append(
                {
                    "language": language,
                    "current_gate": profile.high_confidence_threshold_for_language(language),
                    "selected_gate": selected["high_gate"],
                    "selected_metrics": selected,
                    "selection_ready": (
                        selected["ambiguous_predictions"] == 0
                        and selected["unjudged_predictions"] == 0
                    ),
                }
            )

        current_metrics = _pooled_metrics(
            projects,
            measurements,
            shipped_admissions,
            profile.hybrid_weak_identifier_jaccard_min,
            profile.hybrid_statement_ratio_min,
            {
                language: profile.high_confidence_threshold_for_language(language)
                for language in admissions
            },
        )
        final_metrics = _pooled_metrics(
            projects,
            measurements,
            admissions,
            weak,
            ratio,
            selected_gates,
        )
        payload["models"].append(
            {
                "model": profile.key,
                "admission_thresholds": admissions,
                "current": {
                    "admission_thresholds": shipped_admissions,
                    "weak_identifier_jaccard_min": profile.hybrid_weak_identifier_jaccard_min,
                    "statement_ratio_min": profile.hybrid_statement_ratio_min,
                    "metrics": current_metrics,
                },
                "selected": {
                    "weak_identifier_jaccard_min": weak,
                    "statement_ratio_min": ratio,
                    "metrics": final_metrics,
                    "corroboration_only_metrics": corroboration,
                    "selection_ready": (
                        final_metrics["ambiguous_predictions"] == 0
                        and final_metrics["unjudged_predictions"] == 0
                    ),
                },
                "promotion_by_language": promotion,
            }
        )

    if args.json_out:
        write_json(args.json_out, payload)
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
