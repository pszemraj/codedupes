"""Validate the versioned calibration contract and optionally run behavior evidence."""

from __future__ import annotations

import argparse
import json

try:
    from .calibration_contract import (
        add_contract_arguments,
        load_projects,
        run_behavior,
        validate_project,
        write_json,
    )
except ImportError:
    from calibration_contract import (
        add_contract_arguments,
        load_projects,
        run_behavior,
        validate_project,
        write_json,
    )


def main() -> int:
    """Validate selected projects without loading an embedding model."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_contract_arguments(parser)
    parser.add_argument("--run-behavior", action="store_true")
    parser.add_argument("--json-out", type=str)
    args = parser.parse_args()
    try:
        reports = []
        for project in load_projects(args.manifest, args.projects, args.policy):
            report = validate_project(project)
            if args.run_behavior:
                report["behavior"] = run_behavior(project)
            reports.append(report)
    except (KeyError, TypeError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    payload = {"schema_version": 2, "projects": reports}
    if args.json_out:
        from pathlib import Path

        write_json(Path(args.json_out), payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
