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
        measurement_digests,
        selection_digest,
        validate_measurement_digests,
        validate_selection_context,
        validate_selection_contract,
        validate_shipped_selection_profiles,
    )
    from .calibration_measurements import DEFAULT_MEASUREMENTS
    from .sweep_hybrid_gates import validate_hybrid_selection
    from .sweep_semantic_thresholds import validate_threshold_selection
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
        measurement_digests,
        selection_digest,
        validate_measurement_digests,
        validate_selection_context,
        validate_selection_contract,
        validate_shipped_selection_profiles,
    )
    from calibration_measurements import DEFAULT_MEASUREMENTS
    from sweep_hybrid_gates import validate_hybrid_selection
    from sweep_semantic_thresholds import validate_threshold_selection


def _validate_shipped_selections(
    threshold_selection: dict, hybrid_selection: dict, models: list[str]
) -> None:
    """Require the checked report's selected gates to equal the shipped profiles."""
    entries = threshold_selection.get("models")
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError("threshold selection models must be objects")
    languages = {
        item.get("language")
        for entry in entries
        for item in entry.get("duplicate_by_language", [])
        if isinstance(item, dict)
    }
    if not languages or any(not isinstance(language, str) for language in languages):
        raise ValueError("threshold selection has invalid language gates")
    validate_shipped_selection_profiles(threshold_selection, hybrid_selection, models, languages)


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
        "schema_version": 4,
        "threshold_selection": read_json(args.threshold_selection),
        "hybrid_selection": read_json(args.hybrid_selection),
        "projects": [],
    }
    cpu_measurements = []
    all_measurements = []
    development_cpu = {}
    for field in ("threshold_selection", "hybrid_selection"):
        validate_selection_contract(payload[field])
        validate_selection_context(payload[field], selection_projects, args.models)
    if payload["hybrid_selection"].get("threshold_selection_digest") != selection_digest(
        payload["threshold_selection"]
    ):
        raise ValueError(
            "hybrid selection used another threshold selection; rerun the hybrid sweep"
        )
    for project in projects:
        loaded = load_all(project, args.measurements, args.models, args.devices)
        all_measurements.extend(loaded.values())
        cpu_measurements.extend(
            measurement
            for (_model, device), measurement in loaded.items()
            if device == "cpu" and project.spec["split"] == "development"
        )
        if project.spec["split"] == "development":
            development_cpu.update(
                {
                    (project.id, model): measurement
                    for (model, device), measurement in loaded.items()
                    if device == "cpu"
                }
            )
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
    for field in ("threshold_selection", "hybrid_selection"):
        validate_measurement_digests(payload[field], cpu_measurements)
    validate_threshold_selection(
        payload["threshold_selection"], selection_projects, args.models, development_cpu
    )
    validate_hybrid_selection(
        payload["hybrid_selection"],
        payload["threshold_selection"],
        selection_projects,
        args.models,
        development_cpu,
    )
    _validate_shipped_selections(
        payload["threshold_selection"], payload["hybrid_selection"], args.models
    )
    payload["measurement_digests"] = measurement_digests(all_measurements)
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
