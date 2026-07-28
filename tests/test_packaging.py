"""Release-metadata regression tests."""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict[str, object]:
    """Load the repository's PEP 621 metadata."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_torch_213_is_the_semantic_runtime_floor() -> None:
    """Keep the release contract aligned with the MPS implementation."""
    project = _pyproject()["project"]
    dependencies = project["dependencies"]  # type: ignore[index]
    torch_requirement = next(
        Requirement(item) for item in dependencies if Requirement(item).name == "torch"
    )

    assert Version("2.12.9") not in torch_requirement.specifier
    assert Version("2.13.0") in torch_requirement.specifier
    assert Version("3.0.0") not in torch_requirement.specifier


def test_vcs_less_source_archives_have_a_build_version_fallback() -> None:
    """GitHub/source snapshots must remain buildable without a .git directory."""
    hatch = _pyproject()["tool"]["hatch"]  # type: ignore[index]
    assert hatch["version"]["fallback-version"] == "0.0.0+unknown"


def test_sdist_uses_an_explicit_release_file_allowlist() -> None:
    """Local ignored worktrees must never leak into source distributions."""
    hatch = _pyproject()["tool"]["hatch"]  # type: ignore[index]
    sdist = hatch["build"]["targets"]["sdist"]

    assert sdist["only-include"] == [
        "src/codedupes",
        "tests",
        "test_fixtures",
        "scripts",
        "docs",
        "README.md",
        "LICENSE",
        "pyproject.toml",
    ]
    assert "include" not in sdist
