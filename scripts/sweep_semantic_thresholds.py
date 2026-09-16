"""Replay duplicate and search threshold grids from reusable measurement tables."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from codedupes.constants import DEFAULT_TOP_K
from codedupes.semantic_profiles import list_supported_models, resolve_model_profile

try:
    from .calibration_contract import add_contract_arguments, load_projects, pair_key, write_json
    from .calibration_evaluation import judgments, load_all, metrics, replay
    from .calibration_measurements import DEFAULT_MEASUREMENTS
except ImportError:
    from calibration_contract import add_contract_arguments, load_projects, pair_key, write_json
    from calibration_evaluation import judgments, load_all, metrics, replay
    from calibration_measurements import DEFAULT_MEASUREMENTS


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    """Build an inclusive finite grid with stable decimal values."""
    if not 0 <= start <= stop <= 1 or step <= 0:
        raise ValueError("threshold grid must satisfy 0 <= start <= stop <= 1 and step > 0")
    values = []
    value = start
    while value <= stop + 1e-12:
        values.append(round(value, 6))
        value += step
    return values


def decision_plateaus(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse adjacent settings producing identical decisions."""
    plateaus = []
    for row in rows:
        signature = row.pop("_signature")
        if plateaus and plateaus[-1]["signature"] == signature:
            plateaus[-1]["stop"] = row["threshold"]
        else:
            plateaus.append(
                {"start": row["threshold"], "stop": row["threshold"], "signature": signature}
            )
    return plateaus


def duplicate_rows(
    project: Any, measurement: dict[str, Any], grid: list[float]
) -> tuple[list, list]:
    """Replay one duplicate-admission grid without selecting an optimum."""
    labels = judgments(project)
    rows = []
    for threshold in grid:
        findings = replay(measurement, semantic_threshold=threshold, high_gate=None)
        predicted = {pair_key(item["a"], item["b"]) for item in findings}
        row = {
            "threshold": threshold,
            **metrics(predicted, labels),
            "predicted": len(predicted),
            "_signature": [[item["a"], item["b"], item["tier"]] for item in findings],
        }
        rows.append(row)
    plateaus = decision_plateaus([dict(row) for row in rows])
    for row in rows:
        row.pop("_signature")
    return rows, plateaus


def search_rows(project: Any, measurement: dict[str, Any], grid: list[float]) -> tuple[list, list]:
    """Replay threshold and production top-k search metrics."""
    expected = {
        (probe["id"], unit) for probe in project.annotations["probes"] for unit in probe["expected"]
    }
    by_probe = defaultdict(list)
    for row in measurement["query_scores"]:
        if row["cosine"] is not None:
            by_probe[row["probe"]].append(row)
    rows = []
    for threshold in grid:
        output = {
            (row["probe"], row["unit"])
            for values in by_probe.values()
            for row in values
            if row["cosine"] >= threshold and row["rank"] <= DEFAULT_TOP_K
        }
        tp, fp, fn = len(output & expected), len(output - expected), len(expected - output)
        rows.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "_signature": [list(key) for key in sorted(output)],
            }
        )
    plateaus = decision_plateaus([dict(row) for row in rows])
    for row in rows:
        row.pop("_signature")
    return rows, plateaus


def main() -> int:
    """Replay grids for selected CPU reference artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_contract_arguments(parser)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--models", nargs="+", default=[p.key for p in list_supported_models()])
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--duplicate-start", type=float, default=0.0)
    parser.add_argument("--duplicate-stop", type=float, default=1.0)
    parser.add_argument("--search-start", type=float, default=0.0)
    parser.add_argument("--search-stop", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    duplicate_grid = threshold_grid(args.duplicate_start, args.duplicate_stop, args.step)
    search_grid = threshold_grid(args.search_start, args.search_stop, args.step)
    payload = {
        "schema_version": 2,
        "selection": None,
        "selection_note": "Development pilot: grids expose plateaus and select no defaults.",
        "projects": [],
    }
    for project in load_projects(args.manifest, args.projects, args.policy):
        loaded = load_all(project, args.measurements, args.models, [args.device])
        for (model, device), measurement in loaded.items():
            drows, dplateaus = duplicate_rows(project, measurement, duplicate_grid)
            srows, splateaus = search_rows(project, measurement, search_grid)
            payload["projects"].append(
                {
                    "project": project.id,
                    "model": model,
                    "device": device,
                    "shipped_duplicate_threshold": resolve_model_profile(
                        model
                    ).semantic_threshold_for_language(project.spec["languages"][0]),
                    "shipped_search_threshold": resolve_model_profile(
                        model
                    ).default_search_threshold,
                    "duplicate_rows": drows,
                    "duplicate_indifference_plateaus": dplateaus,
                    "search_rows": srows,
                    "search_indifference_plateaus": splateaus,
                }
            )
    if args.json_out:
        write_json(args.json_out, payload)
    else:
        import json

        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
