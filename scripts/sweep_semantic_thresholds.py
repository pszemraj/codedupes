"""Select duplicate and search thresholds from reviewed calibration scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from codedupes.constants import DEFAULT_TOP_K
from codedupes.semantic_profiles import list_supported_models, resolve_model_profile

try:
    from .calibration_contract import add_contract_arguments, load_projects, pair_key, write_json
    from .calibration_evaluation import judgments, load_all, metrics, selection_context
    from .calibration_measurements import DEFAULT_MEASUREMENTS
except ImportError:
    from calibration_contract import add_contract_arguments, load_projects, pair_key, write_json
    from calibration_evaluation import judgments, load_all, metrics, selection_context
    from calibration_measurements import DEFAULT_MEASUREMENTS


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    """Build an inclusive threshold grid."""
    if not 0 <= start <= stop <= 1 or step <= 0:
        raise ValueError("threshold grid must satisfy 0 <= start <= stop <= 1 and step > 0")
    values = []
    value = start
    while value <= stop + 1e-12:
        values.append(round(value, 6))
        value += step
    return values


def _select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Maximize F1, then recall and precision; use the center of an exact tie plateau."""
    best_key = max(
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
        == best_key
    ]
    return tied[len(tied) // 2]


def _at_threshold(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Return the grid row nearest a production threshold."""
    return min(rows, key=lambda row: abs(row["threshold"] - threshold))


def _score_summary(values: list[float]) -> dict[str, float | int | None]:
    """Summarize one labeled score population."""
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0] if ordered else None,
        "median": median(ordered) if ordered else None,
        "max": ordered[-1] if ordered else None,
    }


def _difficulty_recall(
    project: Any, measurement: dict[str, Any], threshold: float
) -> dict[str, dict[str, float | int]]:
    """Report positive recall by authored challenge level."""
    labels = judgments(project)
    measured = {
        pair_key(row["a"], row["b"]): row
        for row in measurement["pairs"]
        if row["comparable"] and row["cosine"] is not None
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
        if row["comparable"] and row["cosine"] is not None
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
    no_result = {probe["id"] for probe in project.annotations["probes"] if not probe["expected"]}
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


def main() -> int:
    """Select CPU-reference admission and search thresholds for both model profiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_contract_arguments(parser)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--models", nargs="+", default=[p.key for p in list_supported_models()])
    parser.add_argument("--duplicate-start", type=float, default=0.0)
    parser.add_argument("--duplicate-stop", type=float, default=1.0)
    parser.add_argument("--search-start", type=float, default=0.0)
    parser.add_argument("--search-stop", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    projects = load_projects(args.manifest, args.projects, args.policy)
    duplicate_grid = threshold_grid(args.duplicate_start, args.duplicate_stop, args.step)
    search_grid = threshold_grid(args.search_start, args.search_stop, args.step)
    payload: dict[str, Any] = {
        "schema_version": 3,
        "input_context": selection_context(projects, args.models),
        "models": [],
    }

    for model in args.models:
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
            measurement = load_all(project, args.measurements, [model], ["cpu"])[
                (profile.key, "cpu")
            ]
            rows, detail = duplicate_rows(project, measurement, duplicate_grid)
            current = profile.semantic_threshold_for_language(language)
            current_metrics = _at_threshold(rows, current)
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
                    "selection_ready": detail["selection_ready"],
                    "selected_difficulty_recall": detail["selected_difficulty_recall"],
                    "positive_scores": detail["positive_scores"],
                    "negative_scores": detail["negative_scores"],
                    "unjudged_above_selected": detail["unjudged_above_selected"],
                }
            )
            search_records.extend(_search_records(project, measurement))
        search = search_rows(search_records, search_grid)
        selected_search = _select(search)
        model_result["search"] = {
            "current_threshold": profile.default_search_threshold,
            "current_metrics": _at_threshold(search, profile.default_search_threshold),
            "selected_threshold": selected_search["threshold"],
            "selected_metrics": selected_search,
        }
        payload["models"].append(model_result)

    if args.json_out:
        write_json(args.json_out, payload)
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
