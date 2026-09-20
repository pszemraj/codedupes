"""Select duplicate and search thresholds from reviewed calibration scores."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from codedupes.constants import DEFAULT_TOP_K
from codedupes.semantic_profiles import list_supported_models, resolve_model_profile

try:
    from .calibration_contract import (
        add_contract_arguments,
        load_projects,
        pair_key,
        write_json,
    )
    from .calibration_evaluation import (
        SEARCH_SELECTION_WINDOW_RADIUS,
        SELECTION_SCHEMA_VERSION,
        canonical_model_keys,
        development_projects,
        judgments,
        load_all,
        measurement_digests,
        metrics,
        near_best_f1,
        recall_preference,
        selection_context,
        selection_objective,
        threshold_candidate_grids,
        validate_selection_contract,
        validate_threshold_candidate_grids,
    )
    from .calibration_measurements import DEFAULT_MEASUREMENTS
except ImportError:
    from calibration_contract import (
        add_contract_arguments,
        load_projects,
        pair_key,
        write_json,
    )
    from calibration_evaluation import (
        SEARCH_SELECTION_WINDOW_RADIUS,
        SELECTION_SCHEMA_VERSION,
        canonical_model_keys,
        development_projects,
        judgments,
        load_all,
        measurement_digests,
        metrics,
        near_best_f1,
        recall_preference,
        selection_context,
        selection_objective,
        threshold_candidate_grids,
        validate_selection_contract,
        validate_threshold_candidate_grids,
    )
    from calibration_measurements import DEFAULT_MEASUREMENTS


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    """Build an inclusive threshold grid."""
    if (
        not all(math.isfinite(value) for value in (start, stop, step))
        or not 0 <= start <= stop <= 1
        or step <= 0
    ):
        raise ValueError(
            "threshold grid must use finite values satisfying 0 <= start <= stop <= 1 and step > 0"
        )
    values = []
    value = start
    while value <= stop + 1e-12:
        values.append(round(value, 6))
        value += step
    if values[-1] != round(stop, 6):
        values.append(round(stop, 6))
    return values


def _select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Favor recall within the precision-safe F1 bound, centering exact tie plateaus."""
    eligible = near_best_f1(rows)
    best_key = max(recall_preference(row) for row in eligible)
    tied = [row for row in eligible if recall_preference(row) == best_key]
    return tied[len(tied) // 2]


def _select_search(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select only among thresholds that keep every no-result probe empty."""
    clean = [row for row in rows if row["no_result_clean"] == row["no_result_total"]]
    if not clean:
        raise ValueError("no search candidate keeps all no-result probes empty")
    return _select(clean)


def _score_summary(values: list[float]) -> dict[str, float | int | None]:
    """Summarize one labeled score population."""
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0] if ordered else None,
        "median": median(ordered) if ordered else None,
        "max": ordered[-1] if ordered else None,
    }


def _selection_window(rows: list[dict[str, Any]], selected: dict[str, Any]) -> list[dict[str, Any]]:
    """Return nearby sweep rows so a checked selection remains auditable."""
    selected_index = rows.index(selected)
    start = max(0, selected_index - SEARCH_SELECTION_WINDOW_RADIUS)
    stop = selected_index + SEARCH_SELECTION_WINDOW_RADIUS + 1
    return rows[start:stop]


def _difficulty_recall(
    project: Any, measurement: dict[str, Any], threshold: float
) -> dict[str, dict[str, float | int]]:
    """Report positive recall by authored challenge level."""
    labels = judgments(project)
    measured = {
        pair_key(row["a"], row["b"]): row
        for row in measurement["pairs"]
        if row["comparable"] and not row.get("traditional") and row["cosine"] is not None
    }
    result = {}
    for difficulty in ("easy", "medium", "hard"):
        keys = {
            key
            for key, label in labels.items()
            if key in measured
            and label["judgment"] == "positive"
            and label.get("difficulty") == difficulty
        }
        detected = sum(measured[key]["cosine"] >= threshold for key in keys)
        result[difficulty] = {
            "detected": detected,
            "total": len(keys),
            "recall": detected / len(keys) if keys else 0.0,
        }
    return result


def duplicate_rows(
    project: Any, measurement: dict[str, Any], grid: list[float]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sweep the semantic admission gate over comparable reviewed pairs."""
    labels = judgments(project)
    measured = {
        pair_key(row["a"], row["b"]): row
        for row in measurement["pairs"]
        if row["comparable"] and not row.get("traditional") and row["cosine"] is not None
    }
    eligible_labels = {
        key: label
        for key, label in labels.items()
        if key in measured and label["judgment"] in {"positive", "negative", "ambiguous"}
    }
    if not any(label["judgment"] == "positive" for label in eligible_labels.values()):
        raise ValueError(f"{project.id}: no comparable positive labels")
    rows = []
    for threshold in grid:
        predicted = {key for key, row in measured.items() if row["cosine"] >= threshold}
        rows.append(
            {
                "threshold": threshold,
                "predicted": len(predicted),
                **metrics(predicted, eligible_labels),
            }
        )
    selected = _select(rows)
    positive_scores = [
        measured[key]["cosine"]
        for key, label in eligible_labels.items()
        if label["judgment"] == "positive"
    ]
    negative_scores = [
        measured[key]["cosine"]
        for key, label in eligible_labels.items()
        if label["judgment"] == "negative"
    ]
    predicted = {key for key, row in measured.items() if row["cosine"] >= selected["threshold"]}
    detail = {
        "selected": selected,
        "selection_ready": (
            selected["ambiguous_predictions"] == 0 and selected["unjudged_predictions"] == 0
        ),
        "selected_difficulty_recall": _difficulty_recall(
            project, measurement, selected["threshold"]
        ),
        "positive_scores": _score_summary(positive_scores),
        "negative_scores": _score_summary(negative_scores),
        "unjudged_above_selected": [
            [*key, measured[key]["cosine"]]
            for key in sorted(predicted - labels.keys(), key=lambda key: -measured[key]["cosine"])
        ],
    }
    return rows, detail


def _search_records(project: Any, measurement: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {
        (probe["id"], unit) for probe in project.annotations["probes"] for unit in probe["expected"]
    }
    scored = {
        (row["probe"], row["unit"])
        for row in measurement["query_scores"]
        if row["cosine"] is not None
    }
    if missing := expected - scored:
        raise ValueError(
            f"{project.id}: expected search targets were not embedded: {sorted(missing)}"
        )
    no_result = {
        probe["id"] for probe in project.annotations["probes"] if probe["kind"] == "no_result"
    }
    return [
        {
            "key": (project.id, row["probe"], row["unit"]),
            "score": row["cosine"],
            "rank": row["rank"],
            "expected": (row["probe"], row["unit"]) in expected,
            "no_result": row["probe"] in no_result,
        }
        for row in measurement["query_scores"]
        if row["cosine"] is not None
    ]


def search_rows(records: list[dict[str, Any]], grid: list[float]) -> list[dict[str, Any]]:
    """Sweep one global search threshold at the production top-k limit."""
    expected = {row["key"] for row in records if row["expected"]}
    no_result_queries = {(row["key"][0], row["key"][1]) for row in records if row["no_result"]}
    rows = []
    for threshold in grid:
        output = {
            row["key"]
            for row in records
            if row["score"] >= threshold and row["rank"] <= DEFAULT_TOP_K
        }
        tp = len(output & expected)
        fp = len(output - expected)
        fn = len(expected - output)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        violated = {(project, probe) for project, probe, _ in output} & no_result_queries
        rows.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "no_result_clean": len(no_result_queries) - len(violated),
                "no_result_total": len(no_result_queries),
            }
        )
    return rows


def _selection_models(
    projects: list[Any],
    models: list[str],
    measurements: dict[tuple[str, str], dict[str, Any]],
    duplicate_grid: list[float],
    search_grid: list[float],
) -> list[dict[str, Any]]:
    """Derive all threshold decisions from already validated CPU measurements."""
    results = []
    for model in models:
        profile = resolve_model_profile(model)
        model_result: dict[str, Any] = {
            "model": profile.key,
            "duplicate_by_language": [],
        }
        search_records = []
        seen_languages = set()
        for project in projects:
            language = project.spec["languages"][0]
            if language in seen_languages:
                raise ValueError(f"multiple projects for {language}; pooling is not yet explicit")
            seen_languages.add(language)
            measurement = measurements[(project.id, profile.key)]
            duplicate, detail = duplicate_rows(project, measurement, duplicate_grid)
            current = profile.semantic_threshold_for_language(language)
            current_metrics = duplicate_rows(project, measurement, [current])[0][0]
            model_result["duplicate_by_language"].append(
                {
                    "language": language,
                    "project": project.id,
                    "current_threshold": current,
                    "current_metrics": current_metrics,
                    "current_difficulty_recall": _difficulty_recall(
                        project, measurement, current_metrics["threshold"]
                    ),
                    "selected_threshold": detail["selected"]["threshold"],
                    "selected_metrics": detail["selected"],
                    "selection_window": _selection_window(duplicate, detail["selected"]),
                    "selection_ready": detail["selection_ready"],
                    "selected_difficulty_recall": detail["selected_difficulty_recall"],
                    "positive_scores": detail["positive_scores"],
                    "negative_scores": detail["negative_scores"],
                    "unjudged_above_selected": detail["unjudged_above_selected"],
                }
            )
            search_records.extend(_search_records(project, measurement))
        search = search_rows(search_records, search_grid)
        selected_search = _select_search(search)
        model_result["search"] = {
            "current_threshold": profile.default_search_threshold,
            "current_metrics": search_rows(search_records, [profile.default_search_threshold])[0],
            "selected_threshold": selected_search["threshold"],
            "selected_metrics": selected_search,
            "selection_window": _selection_window(search, selected_search),
        }
        results.append(model_result)
    return results


def validate_threshold_selection(
    payload: dict[str, Any],
    projects: list[Any],
    models: list[str],
    measurements: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """Reject threshold decisions not reproducible from their bound measurements."""
    validate_selection_contract(payload)
    validate_threshold_candidate_grids(payload.get("grids"))
    models = canonical_model_keys(models)
    expected = _selection_models(
        projects,
        models,
        measurements,
        payload["grids"]["duplicate"],
        payload["grids"]["search"],
    )
    if payload.get("models") != expected:
        raise ValueError("threshold selection does not match its raw measurements")


def main() -> int:
    """Select CPU-reference admission and search thresholds for both model profiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_contract_arguments(parser)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--models", nargs="+", default=[p.key for p in list_supported_models()])
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        args.models = canonical_model_keys(args.models)
    except ValueError as exc:
        parser.error(str(exc))
    projects = development_projects(load_projects(args.manifest, args.projects, args.policy))
    grids = threshold_candidate_grids()
    duplicate_grid = grids["duplicate"]
    search_grid = grids["search"]
    payload: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "objective": selection_objective(),
        "input_context": selection_context(projects, args.models),
        "grids": {"duplicate": duplicate_grid, "search": search_grid},
        "models": [],
    }
    measurements = {
        (project.id, resolve_model_profile(model).key): load_all(
            project, args.measurements, [model], ["cpu"]
        )[(resolve_model_profile(model).key, "cpu")]
        for model in args.models
        for project in projects
    }
    raw_measurements = list(measurements.values())
    payload["models"] = _selection_models(
        projects, args.models, measurements, duplicate_grid, search_grid
    )
    payload["measurement_digests"] = measurement_digests(raw_measurements)

    if args.json_out:
        write_json(args.json_out, payload)
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
