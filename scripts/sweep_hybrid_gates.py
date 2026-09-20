"""Select hybrid visibility gates after semantic admission calibration."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

from codedupes.semantic_profiles import list_supported_models, resolve_model_profile

try:
    from . import calibration_evaluation
    from .calibration_contract import (
        add_contract_arguments,
        load_projects,
        pair_key,
        read_json,
        write_json,
    )
    from .calibration_evaluation import (
        MINIMUM_SELECTION_PRECISION,
        SELECTION_SCHEMA_VERSION,
        canonical_model_keys,
        development_projects,
        hybrid_candidate_grids,
        judgments,
        load_all,
        measurement_digests,
        metrics,
        near_best_f1,
        recall_preference,
        replay,
        selection_context,
        selection_digest,
        selection_objective,
        validate_hybrid_candidate_grids,
        validate_measurement_digests,
        validate_selection_context,
        validate_selection_contract,
    )
    from .calibration_measurements import DEFAULT_MEASUREMENTS
    from .sweep_semantic_thresholds import threshold_grid, validate_threshold_selection
except ImportError:
    import calibration_evaluation
    from calibration_contract import (
        add_contract_arguments,
        load_projects,
        pair_key,
        read_json,
        write_json,
    )
    from calibration_evaluation import (
        MINIMUM_SELECTION_PRECISION,
        SELECTION_SCHEMA_VERSION,
        canonical_model_keys,
        development_projects,
        hybrid_candidate_grids,
        judgments,
        load_all,
        measurement_digests,
        metrics,
        near_best_f1,
        recall_preference,
        replay,
        selection_context,
        selection_digest,
        selection_objective,
        validate_hybrid_candidate_grids,
        validate_measurement_digests,
        validate_selection_context,
        validate_selection_contract,
    )
    from calibration_measurements import DEFAULT_MEASUREMENTS
    from sweep_semantic_thresholds import threshold_grid, validate_threshold_selection


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


_JOINT_METRIC_FIELDS = (
    "tp",
    "fp",
    "fn",
    "precision",
    "judged_only_precision",
    "recall",
    "f1",
    "ambiguous_predictions",
    "unjudged_predictions",
)


def _promotion_options(
    projects: list[Any],
    measurements: dict[str, dict[str, Any]],
    admissions: dict[str, float],
    weak: float,
    ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    """Return distinct promotion outcomes for every language at one corroboration setting.

    Each retained option carries the exact predicted-pair signature as private
    sweep state. Aggregate metric totals determine the objective, but they are
    not sufficient to distinguish policies that make different mistakes.
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
        outcomes: dict[tuple[tuple[str, str, str], ...], dict[str, Any]] = {}
        for high_gate in [
            None,
            *threshold_grid(
                admissions[language], 1.0, calibration_evaluation.HYBRID_PROMOTION_GATE_STEP
            ),
        ]:
            predicted = {
                (project.id, *key)
                for project in language_projects
                for key in _visible_semantic(
                    measurements[project.id], admissions[language], weak, ratio, high_gate
                )
            }
            signature = tuple(sorted(predicted))
            outcome = outcomes.get(signature)
            if outcome is None:
                outcome = {
                    "_candidate_gates": [],
                    "outcome_digest": selection_digest(signature),
                    "_outcome_signature": signature,
                    **metrics(predicted, labels),
                }
                outcomes[signature] = outcome
            outcome["_candidate_gates"].append(high_gate)

        options[language] = []
        for outcome in outcomes.values():
            candidate_gates = outcome.pop("_candidate_gates")
            numeric_gates = [gate for gate in candidate_gates if gate is not None]
            # Promotion-off is not a threshold on the numeric lattice. When
            # corroboration already produces this outcome, retain that strict
            # policy; otherwise center the equivalent numeric plateau using
            # the same right-of-middle rule as admission selection.
            outcome["high_gate"] = (
                None
                if len(numeric_gates) != len(candidate_gates)
                else numeric_gates[len(numeric_gates) // 2]
            )
            options[language].append(outcome)
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


def _compact_joint_candidate(
    row: dict[str, Any], options: dict[str, dict[str, Any]], languages: list[str]
) -> dict[str, Any]:
    """Serialize one joint candidate with evidence for every precision constraint."""
    return {
        "weak_identifier_jaccard_min": row["weak_identifier_jaccard_min"],
        "statement_ratio_min": row["statement_ratio_min"],
        "high_gates": dict(row["high_gates"]),
        "metrics": {field: row[field] for field in _JOINT_METRIC_FIELDS},
        "per_language": [
            {
                "language": language,
                "high_gate": options[language]["high_gate"],
                "outcome_digest": options[language]["outcome_digest"],
                "metrics": _combined_metrics((options[language],)),
            }
            for language in languages
        ],
    }


def _rank_joint_candidates(
    candidates: list[tuple[dict[str, Any], dict[str, dict[str, Any]]]], languages: list[str]
) -> list[tuple[dict[str, Any], dict[str, dict[str, Any]]]]:
    """Order F1-eligible candidates by the selection rule's final preferences."""
    ranked = sorted(candidates, key=lambda candidate: _joint_tiebreak(candidate[0], languages))
    ranked.sort(key=lambda candidate: recall_preference(candidate[0]), reverse=True)
    return ranked


def _joint_outcome_signature(
    options: dict[str, dict[str, Any]], languages: list[str]
) -> tuple[tuple[tuple[str, str, str], ...], ...]:
    """Identify one joint prediction outcome independently of no-op policy variants."""
    return tuple(options[language]["_outcome_signature"] for language in languages)


def _select_joint(
    projects: list[Any],
    measurements: dict[str, dict[str, Any]],
    admissions: dict[str, float],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """Jointly select corroboration and promotion gates by pooled judged F1.

    Promotion and corroboration interact: a similarity gate can make a strict
    corroboration setting recover pairs that a promotion-disabled sweep misses.
    Search the complete product of distinct per-language outcomes for each
    corroboration row, requiring every language and the pooled result to clear
    the precision floor before applying the shared recall/F1 policy.
    """
    languages = sorted(admissions)
    # Two distinct outcomes per F1 are sufficient to preserve the global winner
    # and an informative runner-up without retaining the full product in memory.
    by_f1: dict[float, list[tuple[dict[str, Any], dict[str, dict[str, Any]]]]] = {}
    for weak in calibration_evaluation.HYBRID_WEAK_GRID:
        for ratio in calibration_evaluation.HYBRID_RATIO_GRID:
            options_by_language = _promotion_options(
                projects, measurements, admissions, weak, ratio
            )
            for combination in product(*(options_by_language[language] for language in languages)):
                if any(option["precision"] < MINIMUM_SELECTION_PRECISION for option in combination):
                    continue
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
                if row["precision"] < MINIMUM_SELECTION_PRECISION:
                    continue
                candidate = (
                    row,
                    {language: option for language, option in zip(languages, combination)},
                )
                bucket = by_f1.setdefault(row["f1"], [])
                signature = _joint_outcome_signature(candidate[1], languages)
                equivalent = next(
                    (
                        previous
                        for previous in bucket
                        if _joint_outcome_signature(previous[1], languages) == signature
                    ),
                    None,
                )
                if equivalent is not None:
                    candidate = _rank_joint_candidates([equivalent, candidate], languages)[0]
                bucket[:] = [
                    previous
                    for previous in bucket
                    if _joint_outcome_signature(previous[1], languages) != signature
                ]
                bucket.append(candidate)
                bucket[:] = _rank_joint_candidates(bucket, languages)[:2]
    if not by_f1:
        raise ValueError(
            "no joint calibration candidate satisfies the minimum precision "
            f"{MINIMUM_SELECTION_PRECISION:.2f} in every language"
        )
    representatives = [candidate for bucket in by_f1.values() for candidate in bucket]
    eligible_rows = near_best_f1([row for row, _ in representatives])
    eligible = [(row, options) for row, options in representatives if row in eligible_rows]
    ranked = _rank_joint_candidates(eligible, languages)
    selected, selected_options = ranked[0]
    best_f1, best_f1_options = by_f1[max(by_f1)][0]
    audit = {
        "best_f1_candidate": _compact_joint_candidate(best_f1, best_f1_options, languages),
        "runner_up": (_compact_joint_candidate(*ranked[1], languages) if len(ranked) > 1 else None),
    }
    return selected, selected_options, audit


def _hybrid_models(
    projects: list[Any],
    models: list[str],
    all_measurements: dict[tuple[str, str], dict[str, Any]],
    selected_admissions: dict[tuple[str, str], float],
) -> list[dict[str, Any]]:
    """Derive all hybrid decisions from validated admission gates and measurements."""
    results = []
    for model in models:
        profile = resolve_model_profile(model)
        measurements = {
            project.id: all_measurements[(project.id, profile.key)] for project in projects
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
        selected, selected_options, selection_audit = _select_joint(
            projects, measurements, admissions
        )
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
                dict.fromkeys(admissions),
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
                    "selected_outcome_digest": option["outcome_digest"],
                    "selected_metrics": {key: option[key] for key in _JOINT_METRIC_FIELDS},
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
        results.append(
            {
                "model": profile.key,
                "admission_thresholds": admissions,
                "current": {
                    "admission_thresholds": shipped_admissions,
                    "weak_identifier_jaccard_min": profile.hybrid_weak_identifier_jaccard_min,
                    "statement_ratio_min": profile.hybrid_statement_ratio_min,
                    "metrics": current_metrics,
                },
                "selection_audit": selection_audit,
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
    return results


def validate_hybrid_selection(
    payload: dict[str, Any],
    threshold_selection: dict[str, Any],
    projects: list[Any],
    models: list[str],
    measurements: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """Reject hybrid decisions not reproducible from bound admissions and scores."""
    validate_selection_contract(threshold_selection)
    validate_selection_contract(payload)
    validate_hybrid_candidate_grids(payload.get("candidate_grids"))
    models = canonical_model_keys(models)
    expected = _hybrid_models(projects, models, measurements, _selection_map(threshold_selection))
    if payload.get("models") != expected:
        raise ValueError("hybrid selection does not match its threshold selection and raw data")


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
    try:
        args.models = canonical_model_keys(args.models)
    except ValueError as exc:
        parser.error(str(exc))
    projects = development_projects(load_projects(args.manifest, args.projects, args.policy))
    threshold_selection = read_json(args.threshold_selection)
    validate_selection_context(threshold_selection, projects, args.models)
    measurements = {
        (project.id, resolve_model_profile(model).key): load_all(
            project, args.measurements, [model], ["cpu"]
        )[(resolve_model_profile(model).key, "cpu")]
        for model in args.models
        for project in projects
    }
    raw_measurements = list(measurements.values())
    validate_measurement_digests(threshold_selection, raw_measurements)
    validate_threshold_selection(threshold_selection, projects, args.models, measurements)
    selected_admissions = _selection_map(threshold_selection)
    payload: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "objective": selection_objective(),
        "input_context": selection_context(projects, args.models),
        "candidate_grids": hybrid_candidate_grids(),
        "threshold_selection_digest": selection_digest(threshold_selection),
        "models": _hybrid_models(projects, args.models, measurements, selected_admissions),
    }
    payload["measurement_digests"] = measurement_digests(raw_measurements)

    if args.json_out:
        write_json(args.json_out, payload)
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
