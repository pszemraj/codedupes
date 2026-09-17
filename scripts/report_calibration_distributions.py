"""Summarize shipped-policy behavior and CPU/MPS drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from codedupes.semantic_profiles import list_supported_models, resolve_model_profile

try:
    from .calibration_contract import (
        add_contract_arguments,
        load_projects,
        read_json,
        write_json,
    )
    from .calibration_evaluation import (
        compare_devices,
        development_projects,
        full_report,
        load_all,
        selection_digest,
        validate_selection_context,
    )
    from .calibration_measurements import DEFAULT_MEASUREMENTS
except ImportError:
    from calibration_contract import (
        add_contract_arguments,
        load_projects,
        read_json,
        write_json,
    )
    from calibration_evaluation import (
        compare_devices,
        development_projects,
        full_report,
        load_all,
        selection_digest,
        validate_selection_context,
    )
    from calibration_measurements import DEFAULT_MEASUREMENTS


def main() -> int:
    """Create a compact report from raw local measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_contract_arguments(parser)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--models", nargs="+", default=[p.key for p in list_supported_models()])
    parser.add_argument("--devices", nargs="+", choices=["cpu", "mps"], default=["cpu", "mps"])
    parser.add_argument(
        "--threshold-selection",
        type=Path,
        default=DEFAULT_MEASUREMENTS / "threshold-selection.json",
    )
    parser.add_argument(
        "--hybrid-selection",
        type=Path,
        default=DEFAULT_MEASUREMENTS / "hybrid-selection.json",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    args.models = [resolve_model_profile(model).key for model in args.models]
    projects = load_projects(args.manifest, args.projects, args.policy)
    selection_projects = development_projects(projects)
    payload = {
        "schema_version": 3,
        "threshold_selection": read_json(args.threshold_selection),
        "hybrid_selection": read_json(args.hybrid_selection),
        "projects": [],
    }
    for field in ("threshold_selection", "hybrid_selection"):
        validate_selection_context(payload[field], selection_projects, args.models)
    if payload["hybrid_selection"].get("threshold_selection_digest") != selection_digest(
        payload["threshold_selection"]
    ):
        raise ValueError(
            "hybrid selection used another threshold selection; rerun the hybrid sweep"
        )
    for project in projects:
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
        payload["projects"].append(
            {
                "project": project.id,
                "split": project.spec["split"],
                "language": project.spec["languages"][0],
                "corpus": {
                    "annotated_units": len(project.annotations["units"]),
                    "positive_pairs": sum(
                        pair["judgment"] == "positive" for pair in project.annotations["pairs"]
                    ),
                    "negative_pairs": sum(
                        pair["judgment"] == "negative" for pair in project.annotations["pairs"]
                    ),
                    "probes": len(project.annotations["probes"]),
                },
                "reports": reports,
                "device_comparisons": comparisons,
            }
        )
    reports = [report for project in payload["projects"] for report in project["reports"].values()]
    torch_versions = {report["runtime_versions"]["torch"] for report in reports}
    if len(torch_versions) != 1:
        raise ValueError("runtime summary requires one PyTorch version across all reports")
    devices = " and ".join(sorted({report["device"].upper() for report in reports}))
    payload["measurement_runtime"] = {
        "torch": torch_versions.pop(),
        "scope": f"all checked {devices} reports",
    }
    if args.json_out:
        write_json(args.json_out, payload)
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
