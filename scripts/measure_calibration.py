"""Generate one uncached, threshold-independent calibration measurement."""

from __future__ import annotations

import argparse

from codedupes.semantic_profiles import list_supported_models

try:
    from .calibration_contract import DEFAULT_MANIFEST, load_projects
    from .calibration_measurements import DEFAULT_MEASUREMENTS, capture
except ImportError:
    from calibration_contract import DEFAULT_MANIFEST, load_projects
    from calibration_measurements import DEFAULT_MEASUREMENTS, capture


def main() -> int:
    """Measure exactly one project/model/device so devices run in fresh processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--project", required=True)
    parser.add_argument("--policy", default="default")
    parser.add_argument(
        "--model", choices=[item.key for item in list_supported_models()], required=True
    )
    parser.add_argument("--device", choices=["cpu", "mps"], required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=str, default=str(DEFAULT_MEASUREMENTS))
    args = parser.parse_args()
    from pathlib import Path

    project = load_projects(Path(args.manifest), [args.project], args.policy)[0]
    directory = capture(project, args.model, args.device, Path(args.output), args.batch_size)
    print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
