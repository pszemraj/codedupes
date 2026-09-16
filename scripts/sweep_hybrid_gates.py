"""Replay identifier/size corroboration grids from fixed admission measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from codedupes.semantic_profiles import list_supported_models

try:
    from .calibration_contract import add_contract_arguments, load_projects, pair_key, write_json
    from .calibration_evaluation import judgments, load_all, metrics, replay
    from .calibration_measurements import DEFAULT_MEASUREMENTS
except ImportError:
    from calibration_contract import add_contract_arguments, load_projects, pair_key, write_json
    from calibration_evaluation import judgments, load_all, metrics, replay
    from calibration_measurements import DEFAULT_MEASUREMENTS

WEAK_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
RATIO_GRID = (0.0, 0.20, 0.35, 0.50, 0.65, 0.80)


def main() -> int:
    """Replay the pilot grid without selecting replacement constants."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_contract_arguments(parser)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--models", nargs="+", default=[p.key for p in list_supported_models()])
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    payload = {
        "schema_version": 2,
        "selection": None,
        "selection_note": "Development pilot; shipped constants remain frozen.",
        "projects": [],
    }
    for project in load_projects(args.manifest, args.projects, args.policy):
        labels = judgments(project)
        for (model, device), measurement in load_all(
            project, args.measurements, args.models, [args.device]
        ).items():
            rows = []
            for weak in WEAK_GRID:
                for ratio in RATIO_GRID:
                    findings = replay(
                        measurement, weak_identifier_jaccard_min=weak, statement_ratio_min=ratio
                    )
                    published = {pair_key(item["a"], item["b"]) for item in findings}
                    visible = {
                        pair_key(item["a"], item["b"])
                        for item in findings
                        if item["tier"] != "semantic_review"
                    }
                    rows.append(
                        {
                            "weak_identifier_jaccard_min": weak,
                            "statement_ratio_min": ratio,
                            "withheld": len(published - visible),
                            "visible": metrics(visible, labels),
                            "published": metrics(published, labels),
                        }
                    )
            payload["projects"].append(
                {
                    "project": project.id,
                    "model": model,
                    "device": device,
                    "rows": rows,
                    "promotion_identifiable": any(row["withheld"] for row in rows),
                    "inactive_promotion_note": None
                    if any(row["withheld"] for row in rows)
                    else "No pairs were withheld; promotion selection is skipped.",
                }
            )
    if args.json_out:
        write_json(args.json_out, payload)
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
