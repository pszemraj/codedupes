"""Manifest, corpus contract, and behavior-probe requirement tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from shutil import copyfile, copytree
from types import SimpleNamespace

import pytest

from scripts import (
    calibration_contract,
)
from scripts.calibration_contract import (
    DEFAULT_MANIFEST,
    extract_project,
    load_projects,
    missing_behavior_executables,
    missing_behavior_requirements,
    read_json,
    resolve_annotations,
    run_behavior,
    validate_project,
    write_json,
)
from scripts.calibration_evaluation import (
    development_projects,
    load_all,
    validate_checked_report,
)
from scripts.calibration_measurements import (
    capture,
)

pytestmark = pytest.mark.grammar


def test_manifest_has_substantive_five_language_corpus():
    projects = load_projects()
    assert {project.spec["languages"][0] for project in projects} == {
        "python",
        "c",
        "rust",
        "javascript",
        "typescript",
    }
    for project in projects:
        report = validate_project(project)
        positives = [
            pair for pair in project.annotations["pairs"] if pair["judgment"] == "positive"
        ]
        negatives = [
            pair for pair in project.annotations["pairs"] if pair["judgment"] == "negative"
        ]
        ineligible = {item["pair"] for item in report["ineligible_judgments"]}
        comparable_positives = [pair for pair in positives if pair["id"] not in ineligible]
        for difficulty in ("easy", "medium"):
            assert sum(pair["difficulty"] == difficulty for pair in positives) >= 5, project.id
            assert sum(pair["difficulty"] == difficulty for pair in comparable_positives) >= 5, (
                project.id
            )
        assert len(negatives) >= 10, project.id
        assert len(project.annotations["probes"]) >= 8
        assert report["pending_deterministic"] == []


@pytest.mark.toolchain
def test_manifest_behavior_contracts_execute():
    projects = load_projects()
    if missing := missing_behavior_requirements(projects):
        pytest.skip(f"missing external toolchain requirements: {', '.join(missing)}")
    for project in projects:
        report = run_behavior(project)
        assert report["project"] == project.id
        assert [run["id"] for run in report["runs"]] == [
            command["id"] for command in project.spec["behavior_tests"]
        ]
        assert all(run["returncode"] == 0 for run in report["runs"])


def test_behavior_executable_probe_reports_only_missing_tools(monkeypatch, tmp_path):
    project = SimpleNamespace(
        spec={
            "behavior_tests": [
                {"argv": ["present-tool"]},
                {"argv": ["missing-tool"]},
                {"argv": ["{python}"]},
            ]
        }
    )
    monkeypatch.setattr(
        calibration_contract.shutil,
        "which",
        lambda executable, path=None: (
            None if executable == "missing-tool" else str(tmp_path / "tool")
        ),
    )

    assert missing_behavior_executables([project]) == ["missing-tool"]


def test_behavior_executable_probe_uses_command_path(monkeypatch, tmp_path):
    command_path = str(tmp_path / "bin")
    project = SimpleNamespace(
        spec={
            "behavior_tests": [
                {"argv": ["fixture-tool"], "env": {"PATH": command_path}},
            ]
        }
    )
    observed_paths = []

    def available_on_command_path(executable, path=None):
        observed_paths.append(path)
        return str(tmp_path / executable) if path == command_path else None

    monkeypatch.setattr(calibration_contract.shutil, "which", available_on_command_path)

    assert missing_behavior_executables([project]) == []
    assert observed_paths == [command_path]


@pytest.mark.parametrize(
    ("global_cc", "expected"),
    [
        pytest.param("missing-cc --target=wasm32", "missing-cc", id="first-word-is-the-compiler"),
        pytest.param("", "CC=<empty>", id="empty-value-is-labelled"),
    ],
)
def test_behavior_requirement_probe_checks_c_compiler(
    monkeypatch, tmp_path, global_cc: str, expected: str
):
    project = SimpleNamespace(
        spec={
            "languages": ["c"],
            "behavior_tests": [{"argv": ["make", "test"]}],
        }
    )
    monkeypatch.setenv("CC", global_cc)
    monkeypatch.setattr(
        calibration_contract.shutil,
        "which",
        lambda executable, path=None: (
            None if executable == expected else str(tmp_path / executable)
        ),
    )

    assert missing_behavior_requirements([project]) == [expected]


@pytest.mark.parametrize(
    ("global_cc", "command_cc", "expected"),
    [
        ("present-global", "missing-command", ["missing-command"]),
        ("missing-global", "present-command", []),
    ],
)
def test_behavior_requirement_probe_uses_command_c_compiler(
    monkeypatch, tmp_path, global_cc: str, command_cc: str, expected: list[str]
):
    project = SimpleNamespace(
        spec={
            "languages": ["c"],
            "behavior_tests": [{"argv": ["make", "test"], "env": {"CC": f"{command_cc} --flag"}}],
        }
    )
    monkeypatch.setenv("CC", global_cc)
    monkeypatch.setattr(
        calibration_contract.shutil,
        "which",
        lambda executable, path=None: (
            None if executable.startswith("missing-") else str(tmp_path / executable)
        ),
    )

    assert missing_behavior_requirements([project]) == expected


def test_behavior_requirement_probe_checks_toolchain_variants(monkeypatch, tmp_path):
    projects = [
        SimpleNamespace(
            spec={
                "languages": ["rust", "typescript"],
                "behavior_tests": [
                    {"argv": ["cargo", "test"]},
                    {"argv": ["node", "--experimental-strip-types", "script.ts"]},
                ],
            }
        )
    ]
    monkeypatch.setattr(
        calibration_contract.shutil,
        "which",
        lambda executable, path=None: (
            None if executable == "rustup" else str(tmp_path / executable)
        ),
    )

    def unavailable_variants(argv, **_kwargs):
        assert argv[0].endswith("node")
        return SimpleNamespace(returncode=9, stdout="")

    monkeypatch.setattr(calibration_contract.subprocess, "run", unavailable_variants)

    assert missing_behavior_requirements(projects) == [
        "node --experimental-strip-types",
    ]


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (FileNotFoundError("missing"), "executable not found: missing-tool"),
        (
            calibration_contract.subprocess.TimeoutExpired(["missing-tool"], 300),
            "timed out after 300 seconds",
        ),
    ],
)
def test_behavior_launch_failures_are_validation_errors(monkeypatch, tmp_path, failure, message):
    project = SimpleNamespace(
        id="sample",
        root=tmp_path,
        spec={"behavior_tests": [{"id": "smoke", "argv": ["missing-tool"]}]},
    )

    def fail_run(*args, **kwargs):
        raise failure

    monkeypatch.setattr(calibration_contract.subprocess, "run", fail_run)
    with pytest.raises(ValueError, match=message):
        run_behavior(project)


def test_selection_uses_only_development_projects():
    development = SimpleNamespace(spec={"split": "development"})
    evaluation = SimpleNamespace(spec={"split": "evaluation"})
    assert development_projects([evaluation, development]) == [development]
    with pytest.raises(ValueError, match="at least one development project"):
        development_projects([evaluation])


def test_annotation_provenance_must_match_manifest_split(tmp_path: Path):
    manifest = read_json(DEFAULT_MANIFEST)
    for spec in manifest["projects"]:
        spec["root"] = str((DEFAULT_MANIFEST.parent / spec["root"]).resolve())
        spec["annotations"] = str((DEFAULT_MANIFEST.parent / spec["annotations"]).resolve())

    annotation = read_json(Path(manifest["projects"][0]["annotations"]))
    annotation["provenance"]["split"] = "evaluation"
    annotation["provenance"]["split_group"] = "foreign-evaluation"
    annotation_path = tmp_path / "forged-annotation.json"
    write_json(annotation_path, annotation)
    manifest["projects"][0]["annotations"] = str(annotation_path)
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="annotation provenance must match"):
        load_projects(manifest_path)


def test_calibration_manifest_rejects_multiple_languages(tmp_path: Path):
    manifest = read_json(DEFAULT_MANIFEST)
    for spec in manifest["projects"]:
        spec["root"] = str((DEFAULT_MANIFEST.parent / spec["root"]).resolve())
        spec["annotations"] = str((DEFAULT_MANIFEST.parent / spec["annotations"]).resolve())
    manifest["projects"][0]["languages"] = ["python", "javascript"]
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="require exactly one language"):
        load_projects(manifest_path)


@pytest.mark.parametrize("kind", ["behavioral", "partial_symbol", "exact_symbol"])
def test_non_no_result_probes_require_expected_targets(kind: str):
    project = load_projects(project_ids=["ledger"])[0]
    project.annotations["probes"][0]["kind"] = kind
    project.annotations["probes"][0]["expected"] = []
    inventory, _ = extract_project(project, inventory=True)

    with pytest.raises(ValueError, match="requires expected targets"):
        resolve_annotations(project, inventory)


def test_calibration_outputs_reject_unjudged_deterministic_findings(tmp_path: Path, monkeypatch):
    project = load_projects(project_ids=["ledger"])[0]
    copied_root = tmp_path / "ledger"
    copytree(project.root, copied_root)
    copyfile(
        copied_root / "src/ledger/audit.py",
        copied_root / "src/ledger/audit_clone.py",
    )
    copied_project = replace(
        project,
        manifest_path=tmp_path / "manifest.json",
        spec={**project.spec, "root": "ledger"},
    )
    monkeypatch.delenv("CODEDUPES_CPU_BF16", raising=False)
    monkeypatch.delenv("PYTORCH_MPS_FAST_MATH", raising=False)

    with pytest.raises(ValueError, match="unjudged deterministic findings"):
        validate_project(copied_project)
    with pytest.raises(ValueError, match="unjudged deterministic findings"):
        capture(copied_project, "gte-modernbert-base", "cpu", tmp_path / "raw")
    with pytest.raises(ValueError, match="unjudged deterministic findings"):
        load_all(
            copied_project,
            tmp_path / "raw",
            ["gte-modernbert-base"],
            ["cpu"],
        )
    with pytest.raises(ValueError, match="unjudged deterministic findings"):
        validate_checked_report({}, [copied_project], ["gte-modernbert-base"])
