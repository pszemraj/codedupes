"""Select hybrid visibility gates after semantic admission calibration."""

from __future__ import annotations

import argparse
import json
from itertools import product
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
    from .calibration_evaluation import (
        judgments,
        load_all,
        metrics,
        replay,
        selection_context,
        selection_digest,
        validate_selection_context,
    )
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
    from calibration_evaluation import (
        judgments,
        load_all,
        metrics,
        replay,
        selection_context,
        selection_digest,
        validate_selection_context,
    )
    from calibration_measurements import DEFAULT_MEASUREMENTS
    from sweep_semantic_thresholds import threshold_grid

WEAK_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
RATIO_GRID = (0.0, 0.20, 0.35, 0.50, 0.65, 0.80)


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


def _promotion_options(
    projects: list[Any],
    measurements: dict[str, dict[str, Any]],
    admissions: dict[str, float],
    weak: float,
    ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    """Return distinct promotion outcomes for every language at one corroboration setting.

    Metric totals are sufficient for pooling because each project/language
    namespace is disjoint. Keeping one representative gate for equal totals
    makes the later Cartesian product small without changing its objective.
    """
    by_language: dict[str, list[Any]] = {}
    for project in projects:
        by_language.setdefault(project.spec["languages"][0], []).append(project)

    options: dict[str, list[dict[str, Any]]] = {}
    for language, language_projects in by_language.items():
        labels = {
            (project.id, *key): value
            for project in language_projects
            for key, value in _semantic_labels(project, measurements[project.id]).items()
        }
        outcomes: dict[tuple[int, int, int, int], dict[str, Any]] = {}
        for high_gate in [None, *threshold_grid(admissions[language], 1.0, 0.01)]:
            predicted = {
                (project.id, *key)
                for project in language_projects
                for key in _visible_semantic(
                    measurements[project.id], admissions[language], weak, ratio, high_gate
                )
            }
            outcome = {"high_gate": high_gate, **metrics(predicted, labels)}
            signature = (
                outcome["tp"],
                outcome["fp"],
                outcome["ambiguous_predictions"],
                outcome["unjudged_predictions"],
            )
            previous = outcomes.get(signature)
            if previous is None or (high_gate is not None, high_gate or 0.0) < (
                previous["high_gate"] is not None,
                previous["high_gate"] or 0.0,
            ):
                outcomes[signature] = outcome
        options[language] = list(outcomes.values())
    return options


def _combined_metrics(options: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Pool independent promotion outcomes without materializing their pair sets."""
    tp = sum(option["tp"] for option in options)
    fp = sum(option["fp"] for option in options)
    fn = sum(option["fn"] for option in options)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "judged_only_precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "ambiguous_predictions": sum(option["ambiguous_predictions"] for option in options),
        "unjudged_predictions": sum(option["unjudged_predictions"] for option in options),
    }


def _selection_key(row: dict[str, Any]) -> tuple[float, int, int, float, float]:
    """Return the existing F1-first policy key for one selection row."""
    return (
        row["f1"],
        -row["ambiguous_predictions"],
        -row["unjudged_predictions"],
        row["recall"],
        row["precision"],
    )


def _joint_tiebreak(row: dict[str, Any], languages: list[str]) -> tuple[Any, ...]:
    """Choose a stable, simple setting among exactly equal pooled outcomes."""
    return (
        row["weak_identifier_jaccard_min"],
        row["statement_ratio_min"],
        tuple(
            (row["high_gates"][language] is not None, row["high_gates"][language] or 0.0)
            for language in languages
        ),
    )


def _select_joint(
    projects: list[Any],
    measurements: dict[str, dict[str, Any]],
    admissions: dict[str, float],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Jointly select corroboration and promotion gates by pooled judged F1.

    Promotion and corroboration interact: a similarity gate can make a strict
    corroboration setting recover pairs that a promotion-disabled sweep misses.
    Search the complete product of distinct per-language outcomes for each
    corroboration row, retaining the existing F1/unresolved/recall/precision
    priorities.
    """
    languages = sorted(admissions)
    selected: dict[str, Any] | None = None
    selected_options: dict[str, dict[str, Any]] | None = None
    for weak in WEAK_GRID:
        for ratio in RATIO_GRID:
            options_by_language = _promotion_options(
                projects, measurements, admissions, weak, ratio
            )
            for combination in product(*(options_by_language[language] for language in languages)):
                high_gates = {
                    language: option["high_gate"]
                    for language, option in zip(languages, combination)
                }
                row = {
                    "weak_identifier_jaccard_min": weak,
                    "statement_ratio_min": ratio,
                    "high_gates": high_gates,
                    **_combined_metrics(combination),
                }
                if (
                    selected is None
                    or _selection_key(row) > _selection_key(selected)
                    or (
                        _selection_key(row) == _selection_key(selected)
                        and _joint_tiebreak(row, languages) < _joint_tiebreak(selected, languages)
                    )
                ):
                    selected = row
                    selected_options = {
                        language: option for language, option in zip(languages, combination)
                    }
    assert selected is not None and selected_options is not None
    return selected, selected_options


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
    threshold_selection = read_json(args.threshold_selection)
    validate_selection_context(threshold_selection, projects, args.models)
    selected_admissions = _selection_map(threshold_selection)
    payload: dict[str, Any] = {
        "schema_version": 3,
        "input_context": selection_context(projects, args.models),
        "threshold_selection_digest": selection_digest(threshold_selection),
        "models": [],
    }

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
        selected, selected_options = _select_joint(projects, measurements, admissions)
        weak = selected["weak_identifier_jaccard_min"]
        ratio = selected["statement_ratio_min"]
        selected_gates = selected["high_gates"]
        corroboration = {
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

        promotion = []
        for language in dict.fromkeys(project.spec["languages"][0] for project in projects):
            option = selected_options[language]
            promotion.append(
                {
                    "language": language,
                    "current_gate": profile.high_confidence_threshold_for_language(language),
                    "selected_gate": option["high_gate"],
                    "selected_metrics": {
                        key: value for key, value in option.items() if key != "high_gate"
                    },
                    "selection_ready": (
                        option["ambiguous_predictions"] == 0 and option["unjudged_predictions"] == 0
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
