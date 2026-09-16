"""Evaluate fixed-policy measurements, raw score distributions, and CPU/MPS drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from codedupes.semantic_profiles import list_supported_models

try:
    from .calibration_contract import add_contract_arguments, load_projects, write_json
    from .calibration_evaluation import compare_devices, full_report, load_all, review_queue
    from .calibration_measurements import DEFAULT_MEASUREMENTS
except ImportError:
    from calibration_contract import add_contract_arguments, load_projects, write_json
    from calibration_evaluation import compare_devices, full_report, load_all, review_queue
    from calibration_measurements import DEFAULT_MEASUREMENTS


def main() -> int:
    """Create the complete pilot report from validated measurement artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_contract_arguments(parser)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--models", nargs="+", default=[p.key for p in list_supported_models()])
    parser.add_argument("--devices", nargs="+", choices=["cpu", "mps"], default=["cpu", "mps"])
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    payload = {
        "schema_version": 2,
        "selection": None,
        "selection_note": "Development pilot only; no replacement defaults are selected.",
        "projects": [],
    }
    for project in load_projects(args.manifest, args.projects, args.policy):
        loaded = load_all(project, args.measurements, args.models, args.devices)
        reports = {
            f"{model}/{device}": full_report(project, measurement)
            for (model, device), measurement in loaded.items()
        }
        comparisons = {}
        if {"cpu", "mps"} <= set(args.devices):
            for model in args.models:
                comparisons[model] = compare_devices(
                    loaded[(model, "cpu")], loaded[(model, "mps")], project
                )
        queue = review_queue(project, list(loaded.values()))
        payload["projects"].append(
            {
                "project": project.id,
                "reports": reports,
                "device_comparisons": comparisons,
                "review_queue": queue,
            }
        )
    if args.json_out:
        write_json(args.json_out, payload)
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
