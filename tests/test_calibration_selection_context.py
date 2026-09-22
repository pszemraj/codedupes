"""Selection context, support-file identity, and report-writer validation tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import (
    calibration_evaluation,
    report_calibration_distributions,
)
from scripts.calibration_contract import (
    DEFAULT_MANIFEST,
    load_projects,
    read_json,
    write_json,
)
from scripts.calibration_evaluation import (
    SELECTION_SCHEMA_VERSION,
    hybrid_candidate_grids,
    selection_context,
    selection_digest,
    selection_objective,
    selection_policy_identity,
    support_files_digest,
    validate_selection_context,
)
from scripts.sweep_semantic_thresholds import (
    _search_records,
)

pytestmark = pytest.mark.grammar


def test_search_selection_rejects_unembedded_expected_target():
    project = SimpleNamespace(
        id="sample",
        annotations={
            "probes": [{"id": "find", "expected": ["missing"]}],
        },
    )
    measurement = {
        "query_scores": [{"probe": "find", "unit": "missing", "cosine": None, "rank": None}]
    }
    with pytest.raises(ValueError, match="not embedded"):
        _search_records(project, measurement)


@pytest.mark.parametrize("change", ["judgment", "relevance", "policy", "model", "scope"])
def test_selections_reject_changed_review_or_scope(change: str):
    projects = load_projects()
    models = ["gte-modernbert-base"]
    payload = {"input_context": selection_context(projects, models)}
    validate_selection_context(payload, projects, models)
    if change == "judgment":
        projects[0].annotations["pairs"][0]["judgment"] = "ambiguous"
    elif change == "relevance":
        projects[0].annotations["probes"][0]["expected"] = []
    elif change == "policy":
        projects[0].policy = projects[0].policy | {"min_semantic_statements": 4}
    elif change == "model":
        models = ["embeddinggemma-300m"]
    else:
        projects = projects[:1]
    with pytest.raises(ValueError, match="stale or mismatched selection"):
        validate_selection_context(payload, projects, models)


def test_support_file_identity_tracks_declared_behavior_evidence(tmp_path: Path):
    readme = tmp_path / "README.md"
    contract = tmp_path / "fixture.toml"
    tests = tmp_path / "tests"
    tests.mkdir()
    behavior = tests / "test_behavior.txt"
    readme.write_text("fixture documentation v1\n")
    contract.write_text("behavior = 'v1'\n")
    behavior.write_text("behavior v1\n")
    project = SimpleNamespace(
        id="sample",
        root=tmp_path,
        spec={"support_files": ["fixture.toml"], "test_roots": ["tests"]},
    )
    original = support_files_digest(project)
    cache = tests / "__pycache__"
    cache.mkdir()
    (cache / "test_behavior.cpython-312.pyc").write_bytes(b"generated")
    assert support_files_digest(project) == original
    readme.write_text("fixture documentation reflowed\n")
    assert support_files_digest(project) == original
    contract.write_text("behavior = 'v2'\n")
    assert support_files_digest(project) != original
    contract.write_text("behavior = 'v1'\n")
    behavior.write_text("behavior v2\n")
    assert support_files_digest(project) != original
    project.spec["support_files"] = ["missing.*"]
    with pytest.raises(ValueError, match="matched nothing"):
        support_files_digest(project)


def test_selection_context_uses_structured_policy_identity(tmp_path: Path, monkeypatch):
    """Selections bind policy values without hashing incidental source text."""
    projects, models = load_projects(project_ids=["ledger"]), ["gte-modernbert-base"]
    original = selection_context(projects, models)
    assert original["selection_policy"] == selection_policy_identity()
    assert isinstance(original["selection_policy"], dict)
    monkeypatch.setattr(calibration_evaluation, "REPO", tmp_path, raising=False)
    updated = selection_context(projects, models)
    assert updated == original
    validate_selection_context({"input_context": original}, projects, models)


def test_selection_context_ignores_raw_pipeline_source_fingerprints(monkeypatch):
    """Checked evidence must survive implementation-only source edits."""
    projects, models = load_projects(project_ids=["ledger"]), ["gte-modernbert-base"]
    original = selection_context(projects, models)

    def fail_raw_fingerprint(*_args, **_kwargs):
        raise AssertionError("checked context must not use raw source-byte fingerprints")

    monkeypatch.setattr(
        calibration_evaluation,
        "measurement_fingerprint",
        fail_raw_fingerprint,
        raising=False,
    )

    assert selection_context(projects, models) == original


def test_checked_project_validation_is_memoized_by_context(monkeypatch):
    project = load_projects(project_ids=["ledger"])[0]
    context = calibration_evaluation._selection_project_context(project, ["gte-modernbert-base"])
    calls = []
    calibration_evaluation._VALIDATED_CHECKED_PROJECTS.clear()
    monkeypatch.setattr(
        calibration_evaluation,
        "validate_project",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    calibration_evaluation._validate_checked_project_once(project, context)
    calibration_evaluation._validate_checked_project_once(project, context)
    assert len(calls) == 1

    project.annotations["pairs"][0]["rationale"] += " changed"
    calibration_evaluation._validate_checked_project_once(project, context)
    assert len(calls) == 2
    calibration_evaluation._VALIDATED_CHECKED_PROJECTS.clear()


@pytest.mark.parametrize(
    "field",
    [
        "MINIMUM_SELECTION_PRECISION",
        "SELECTION_ALGORITHM_VERSION",
        "DEFAULT_TOP_K",
        "THRESHOLD_GRID_STEP",
        "HYBRID_WEAK_GRID",
    ],
)
def test_selection_context_rejects_changed_policy_values(monkeypatch, field: str):
    """Every recorded selection-policy value must invalidate stale selections."""
    projects, models = load_projects(project_ids=["ledger"]), ["gte-modernbert-base"]
    original = selection_context(projects, models)
    current = getattr(calibration_evaluation, field)
    replacement = {
        "MINIMUM_SELECTION_PRECISION": 0.6,
        "SELECTION_ALGORITHM_VERSION": 9,
        "DEFAULT_TOP_K": 11,
        "THRESHOLD_GRID_STEP": 0.02,
        "HYBRID_WEAK_GRID": (0.0, 0.5),
    }[field]
    assert replacement != current
    monkeypatch.setattr(calibration_evaluation, field, replacement)
    with pytest.raises(ValueError, match="stale or mismatched selection"):
        validate_selection_context({"input_context": original}, projects, models)


def test_report_rejects_hybrid_from_another_threshold_selection(tmp_path: Path, monkeypatch):
    project = load_projects(project_ids=["ledger"])[0]
    context = selection_context([project], ["gte-modernbert-base", "embeddinggemma-300m"])
    threshold = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "objective": selection_objective(),
        "input_context": context,
        "models": [],
    }
    hybrid = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "objective": selection_objective(),
        "input_context": context,
        "candidate_grids": hybrid_candidate_grids(),
        "threshold_selection_digest": selection_digest(threshold),
    }
    threshold["models"].append({"model": "changed selection"})
    threshold_path, hybrid_path = tmp_path / "threshold.json", tmp_path / "hybrid.json"
    write_json(threshold_path, threshold)
    write_json(hybrid_path, hybrid)
    monkeypatch.setattr(report_calibration_distributions, "load_projects", lambda *args: [project])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report",
            "--threshold-selection",
            str(threshold_path),
            "--hybrid-selection",
            str(hybrid_path),
        ],
    )
    with pytest.raises(ValueError, match="another threshold selection"):
        report_calibration_distributions.main()


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("schema", "unsupported schema version"),
        ("objective", "mismatched objective contract"),
    ],
)
def test_report_rejects_tampered_selection_contract_before_loading_raw(
    tmp_path: Path, monkeypatch, tamper: str, message: str
):
    """A linked hybrid file cannot legitimize forged selection provenance."""
    project = load_projects(project_ids=["ledger"])[0]
    models = ["gte-modernbert-base", "embeddinggemma-300m"]
    context = selection_context([project], models)
    threshold = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "objective": selection_objective(),
        "input_context": context,
        "models": [],
    }
    hybrid = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "objective": selection_objective(),
        "input_context": context,
        "candidate_grids": hybrid_candidate_grids(),
        "models": [],
    }
    if tamper == "schema":
        threshold["schema_version"] = hybrid["schema_version"] = 999
    else:
        threshold["objective"]["minimum_precision"] = 0.0
        hybrid["objective"]["minimum_precision"] = 0.0
    hybrid["threshold_selection_digest"] = selection_digest(threshold)
    threshold_path, hybrid_path = tmp_path / "threshold.json", tmp_path / "hybrid.json"
    write_json(threshold_path, threshold)
    write_json(hybrid_path, hybrid)
    monkeypatch.setattr(report_calibration_distributions, "load_projects", lambda *args: [project])
    monkeypatch.setattr(
        report_calibration_distributions,
        "load_all",
        lambda *args: pytest.fail("report loaded raw measurements before validating selections"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report",
            "--threshold-selection",
            str(threshold_path),
            "--hybrid-selection",
            str(hybrid_path),
        ],
    )
    with pytest.raises(ValueError, match=message):
        report_calibration_distributions.main()


def test_report_rejects_selection_that_is_not_shipped():
    checked = read_json(DEFAULT_MANIFEST.parent / "calibration-results.json")
    models = [item["model"] for item in checked["threshold_selection"]["models"]]
    report_calibration_distributions._validate_shipped_selections(
        checked["threshold_selection"], checked["hybrid_selection"], models
    )
    checked["threshold_selection"]["models"][0]["search"]["selected_threshold"] = 0.64
    with pytest.raises(ValueError, match="do not match shipped defaults"):
        report_calibration_distributions._validate_shipped_selections(
            checked["threshold_selection"], checked["hybrid_selection"], models
        )


@pytest.mark.parametrize("second_version", ["2.14.0", "2.13.0"])
def test_report_writer_derives_measurement_runtime(tmp_path: Path, monkeypatch, second_version):
    checked = read_json(DEFAULT_MANIFEST.parent / "calibration-results.json")
    project = load_projects(project_ids=["ledger"])[0]
    # Reuse recorded compact reports to exercise serialization without model inference.
    reports = {
        (report["model"], report["device"]): report
        for report in checked["projects"][0]["reports"].values()
    }
    for index, report in enumerate(reports.values()):
        version = "2.14.0" if index == 0 else second_version
        report["runtime_versions"] = {"torch": version}
    monkeypatch.setattr(report_calibration_distributions, "load_all", lambda *args: reports)
    monkeypatch.setattr(report_calibration_distributions, "full_report", lambda p, report: report)
    monkeypatch.setattr(
        report_calibration_distributions,
        "compare_devices",
        lambda *args: {},
    )
    monkeypatch.setattr(
        report_calibration_distributions, "validate_measurement_digests", lambda *args: None
    )
    monkeypatch.setattr(
        report_calibration_distributions, "validate_threshold_selection", lambda *args: None
    )
    monkeypatch.setattr(
        report_calibration_distributions, "validate_hybrid_selection", lambda *args: None
    )
    monkeypatch.setattr(report_calibration_distributions, "measurement_digests", lambda *args: {})
    monkeypatch.setattr(report_calibration_distributions, "load_projects", lambda *args: [project])
    validated = []
    monkeypatch.setattr(
        report_calibration_distributions,
        "validate_checked_report",
        lambda payload, *args: validated.append(payload),
    )
    threshold_path = tmp_path / "threshold-selection.json"
    hybrid_path = tmp_path / "hybrid-selection.json"
    output = tmp_path / "report.json"
    context = selection_context([project], ["gte-modernbert-base", "embeddinggemma-300m"])
    checked["threshold_selection"]["input_context"] = context
    checked["hybrid_selection"]["input_context"] = context
    checked["hybrid_selection"]["threshold_selection_digest"] = selection_digest(
        checked["threshold_selection"]
    )
    write_json(threshold_path, checked["threshold_selection"])
    write_json(hybrid_path, checked["hybrid_selection"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report_calibration_distributions.py",
            "--threshold-selection",
            str(threshold_path),
            "--hybrid-selection",
            str(hybrid_path),
            "--json-out",
            str(output),
        ],
    )
    if second_version != "2.14.0":
        with pytest.raises(ValueError, match="one PyTorch version"):
            report_calibration_distributions.main()
        assert not output.exists()
        return
    assert report_calibration_distributions.main() == 0
    result = read_json(output)
    assert result["measurement_runtime"] == {
        "torch": "2.14.0",
        "scope": "all checked CPU and MPS reports",
    }
    assert validated == [result]
    assert result["projects"][0]["split"] == "development"
    assert all(
        report["runtime_versions"]["torch"] == result["measurement_runtime"]["torch"]
        for report in result["projects"][0]["reports"].values()
    )
