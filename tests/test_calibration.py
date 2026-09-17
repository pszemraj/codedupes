"""Focused checks for the calibration corpus and threshold selection."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from codedupes.semantic_profiles import resolve_model_profile
from scripts import (
    calibration_evaluation,
    report_calibration_distributions,
    sweep_semantic_thresholds,
)
from scripts.calibration_contract import (
    DEFAULT_MANIFEST,
    load_projects,
    read_json,
    validate_project,
    write_json,
)
from scripts.calibration_evaluation import (
    replay,
    replay_parity,
    selection_context,
    selection_digest,
    validate_selection_context,
)
from scripts.calibration_measurements import (
    ARTIFACT_VERSION,
    load_measurement,
    measurement_fingerprint,
)
from scripts.sweep_hybrid_gates import _selection_map
from scripts.sweep_semantic_thresholds import (
    _search_records,
    duplicate_rows,
    search_rows,
    threshold_grid,
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
        assert len(positives) >= 6, project.id
        assert len(negatives) >= 10, project.id
        assert {pair["difficulty"] for pair in positives} == {"easy", "medium", "hard"}
        assert len(project.annotations["probes"]) >= 8
        assert report["pending_deterministic"] == []


def test_duplicate_sweep_uses_reviewed_comparable_pairs():
    project = SimpleNamespace(
        id="sample",
        annotations={
            "pairs": [
                {"a": "a", "b": "b", "judgment": "positive"},
                {"a": "a", "b": "c", "judgment": "positive"},
                {"a": "a", "b": "d", "judgment": "negative"},
            ]
        },
    )
    measurement = {
        "pairs": [
            {"a": "a", "b": "b", "cosine": 0.91, "comparable": True},
            {"a": "a", "b": "c", "cosine": 0.78, "comparable": True},
            {"a": "a", "b": "d", "cosine": 0.74, "comparable": True},
            {"a": "b", "b": "d", "cosine": 0.88, "comparable": True},
            {"a": "c", "b": "d", "cosine": 0.99, "comparable": False},
        ]
    }
    rows, detail = duplicate_rows(project, measurement, threshold_grid(0.70, 0.95, 0.01))
    assert detail["selected"]["f1"] == 1.0
    assert detail["selection_ready"] is False
    assert 0.75 <= detail["selected"]["threshold"] <= 0.78
    assert detail["unjudged_above_selected"] == [["b", "d", 0.88]]
    assert rows


def test_search_sweep_scores_complete_relevance_sets():
    records = [
        {
            "key": ("p", "q", "a"),
            "score": 0.82,
            "rank": 1,
            "expected": True,
            "no_result": False,
        },
        {
            "key": ("p", "q", "b"),
            "score": 0.66,
            "rank": 2,
            "expected": True,
            "no_result": False,
        },
        {
            "key": ("p", "q", "c"),
            "score": 0.40,
            "rank": 3,
            "expected": False,
            "no_result": False,
        },
        {
            "key": ("p", "none", "d"),
            "score": 0.90,
            "rank": 11,
            "expected": False,
            "no_result": True,
        },
    ]
    rows = search_rows(records, [0.4, 0.6, 0.7])
    assert rows[1]["f1"] == 1.0
    assert rows[2]["fn"] == 1
    assert rows[0]["no_result_clean"] == 1


def test_threshold_grid_includes_a_stop_between_steps():
    assert threshold_grid(0.70, 0.85, 0.10) == [0.70, 0.80, 0.85]
    assert threshold_grid(0.70, 0.70, 0.10) == [0.70]


def test_coarse_sweep_measures_shipped_thresholds_exactly(tmp_path: Path, monkeypatch):
    project = SimpleNamespace(
        id="sample",
        spec={"languages": ["python"]},
        annotations={
            "pairs": [{"a": "a", "b": "b", "judgment": "positive", "difficulty": "easy"}],
            "probes": [{"id": "q", "expected": ["a"]}],
        },
    )
    measurement = {
        "pairs": [{"a": "a", "b": "b", "cosine": 0.85, "comparable": True}],
        "query_scores": [
            {"probe": "q", "unit": "a", "cosine": 0.69, "rank": 1},
            {"probe": "q", "unit": "b", "cosine": 0.65, "rank": 2},
        ],
    }
    monkeypatch.setattr(sweep_semantic_thresholds, "load_projects", lambda *args: [project])
    monkeypatch.setattr(sweep_semantic_thresholds, "selection_context", lambda *args: {})
    monkeypatch.setattr(
        sweep_semantic_thresholds,
        "load_all",
        lambda *args: {("gte-modernbert-base", "cpu"): measurement},
    )
    output = tmp_path / "selection.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["sweep", "--models", "gte-modernbert-base", "--step", "0.3", "--json-out", str(output)],
    )
    assert sweep_semantic_thresholds.main() == 0
    result = read_json(output)
    duplicate = result["models"][0]["duplicate_by_language"][0]
    assert duplicate["current_threshold"] == duplicate["current_metrics"]["threshold"] == 0.80
    assert duplicate["current_metrics"]["tp"] == 1
    assert duplicate["current_difficulty_recall"]["easy"]["detected"] == 1
    search = result["models"][0]["search"]
    assert search["current_threshold"] == search["current_metrics"]["threshold"] == 0.68
    assert (search["current_metrics"]["tp"], search["current_metrics"]["fp"]) == (1, 0)
    assert result["grids"]["search"] == [0.0, 0.3, 0.6, 0.9, 1.0]


def test_replay_matches_production_tier_rules():
    measurement = {
        "metadata": {
            "model": "gte-modernbert-base",
            "captured_profile": {
                "semantic_threshold": 0.80,
                "weak_identifier_jaccard_min": 0.0,
                "statement_ratio_min": 0.80,
                "high_gate": None,
            },
        },
        "units": [
            {"id": "a", "language": "python"},
            {"id": "b", "language": "python"},
            {"id": "c", "language": "python"},
        ],
        "pairs": [
            {
                "a": "a",
                "b": "b",
                "cosine": 1.0,
                "comparable": False,
                "identifier_jaccard": 1.0,
                "statement_ratio": 1.0,
                "traditional": [{"method": "structural_hash", "similarity": 1.0}],
            },
            {
                "a": "a",
                "b": "c",
                "cosine": 0.84,
                "comparable": True,
                "identifier_jaccard": 0.0,
                "statement_ratio": 0.5,
                "traditional": [],
            },
            {
                "a": "b",
                "b": "c",
                "cosine": 0.70,
                "comparable": True,
                "identifier_jaccard": 0.0,
                "statement_ratio": 1.0,
                "traditional": [],
            },
        ],
    }
    findings = replay(measurement)
    assert [item["tier"] for item in findings] == ["exact", "semantic_review"]
    measurement["metadata"]["live_default"] = [
        {"a": item["a"], "b": item["b"], "tier": item["tier"]} for item in findings
    ]
    assert replay_parity(measurement) is True


def test_explicit_semantic_override_does_not_reuse_profile_promotion_gate():
    measurement = {
        "metadata": {"model": "gte-modernbert-base"},
        "units": [
            {"id": "a", "language": "typescript"},
            {"id": "b", "language": "typescript"},
        ],
        "pairs": [
            {
                "a": "a",
                "b": "b",
                "cosine": 0.90,
                "comparable": True,
                "identifier_jaccard": 0.0,
                "statement_ratio": 0.5,
                "traditional": [],
            }
        ],
    }
    assert replay(measurement, semantic_threshold=0.80)[0]["tier"] == "semantic_review"
    assert (
        replay(measurement, semantic_threshold=0.80, high_gate=0.88)[0]["tier"]
        == "semantic_high_confidence"
    )


def test_ambiguity_blocks_threshold_selection():
    project = SimpleNamespace(
        id="sample",
        annotations={
            "pairs": [
                {"a": "a", "b": "b", "judgment": "positive", "difficulty": "easy"},
                {"a": "a", "b": "c", "judgment": "ambiguous"},
            ]
        },
    )
    measurement = {
        "pairs": [
            {"a": "a", "b": "b", "cosine": 0.90, "comparable": True},
            {"a": "a", "b": "c", "cosine": 0.89, "comparable": True},
        ]
    }
    rows, detail = duplicate_rows(project, measurement, [0.80])
    assert rows[0]["ambiguous_predictions"] == 1
    assert rows[0]["unjudged_predictions"] == 0
    assert detail["selection_ready"] is False


def test_hybrid_selection_rejects_unready_semantic_admissions():
    payload = {
        "models": [
            {
                "model": "gte-modernbert-base",
                "duplicate_by_language": [
                    {
                        "language": "python",
                        "selected_threshold": 0.82,
                        "selection_ready": False,
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="review unjudged admission findings"):
        _selection_map(payload)


def test_measurements_reject_changed_source_queries_or_runtime(tmp_path: Path, monkeypatch):
    project = load_projects(project_ids=["ledger"])[0]
    path = tmp_path / "measurement.json"
    write_json(
        path,
        {
            "schema_version": ARTIFACT_VERSION,
            "metadata": {
                "project": project.id,
                "model": "gte-modernbert-base",
                "requested_device": "cpu",
                "execution": {
                    "duplicate": {"execution_device": "cpu", "cache_hit_rows": 0},
                    "search": {"execution_device": "cpu", "cache_hit_rows": 0},
                },
                "input_fingerprint": measurement_fingerprint(project, "gte-modernbert-base"),
            },
        },
    )
    assert (
        load_measurement(
            path,
            project,
            expected_model="gte-modernbert-base",
            expected_device="cpu",
        )["metadata"]["project"]
        == "ledger"
    )
    fingerprint = measurement_fingerprint(project, "gte-modernbert-base")
    project.annotations["pairs"][0]["rationale"] += " label-only edit"
    assert measurement_fingerprint(project, "gte-modernbert-base") == fingerprint
    project.annotations["units"].reverse()
    assert measurement_fingerprint(project, "gte-modernbert-base") == fingerprint
    with pytest.raises(ValueError, match="another model"):
        load_measurement(path, project, expected_model="embeddinggemma-300m")
    with pytest.raises(ValueError, match="did not execute on mps"):
        load_measurement(path, project, expected_device="mps")
    project.annotations["probes"][0]["query"] += " changed"
    with pytest.raises(ValueError, match="stale measurement"):
        load_measurement(path, project)

    project.annotations["probes"][0]["query"] = project.annotations["probes"][0][
        "query"
    ].removesuffix(" changed")
    monkeypatch.setattr(
        "scripts.calibration_measurements.semantic.get_semantic_runtime_versions",
        lambda: {
            "python": "3.15.0",
            "torch": "9.9.9",
            "transformers": "9.9.9",
            "sentence-transformers": "9.9.9",
        },
    )
    with pytest.raises(ValueError, match="stale measurement"):
        load_measurement(path, project)


def test_measurement_identity_ignores_generated_files_but_tracks_source(tmp_path: Path):
    project = load_projects(project_ids=["ledger"])[0]
    source = tmp_path / "src"
    source.mkdir()
    module = source / "example.py"
    module.write_text("def example(value):\n    return value + 1\n")
    project = replace(
        project,
        spec=project.spec | {"root": str(tmp_path)},
        annotations={"units": [], "probes": []},
    )
    fingerprint = measurement_fingerprint(project, "gte-modernbert-base")
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "example.cpython-312.pyc").write_bytes(b"generated bytecode")
    (source / ".DS_Store").write_bytes(b"finder metadata")
    assert measurement_fingerprint(project, "gte-modernbert-base") == fingerprint
    module.write_text("def example(value):\n    return value + 2\n")
    assert measurement_fingerprint(project, "gte-modernbert-base") != fingerprint
    module.write_text("def example(value):\n    return value + 1\n")
    (source / "additional.py").write_text("def additional(value):\n    return value * 2\n")
    assert measurement_fingerprint(project, "gte-modernbert-base") != fingerprint


@pytest.mark.parametrize("change", ["rename", "retarget", "remove"])
def test_measurements_reject_changed_annotation_identities(tmp_path: Path, change: str):
    project = load_projects(project_ids=["ledger"])[0]
    path = tmp_path / "measurement.json"
    write_json(
        path,
        {
            "schema_version": ARTIFACT_VERSION,
            "metadata": {
                "project": project.id,
                "model": "gte-modernbert-base",
                "input_fingerprint": measurement_fingerprint(project, "gte-modernbert-base"),
            },
        },
    )
    load_measurement(path, project)
    units = project.annotations["units"]
    if change == "rename":
        units[0]["id"] += "-renamed"
    elif change == "retarget":
        units[0]["selector"], units[1]["selector"] = units[1]["selector"], units[0]["selector"]
    else:
        units.pop()
    with pytest.raises(ValueError, match="stale measurement"):
        load_measurement(path, project)


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


@pytest.mark.parametrize(
    "changed_file",
    [
        "scripts/sweep_semantic_thresholds.py",
        "scripts/sweep_hybrid_gates.py",
        "scripts/calibration_evaluation.py",
        "src/codedupes/semantic_profiles.py",
    ],
)
def test_selection_policy_edits_reuse_measurements_but_reject_selections(
    tmp_path: Path, monkeypatch, changed_file: str
):
    # Isolate policy files without editing the real extraction/measurement inputs.
    for relative in (
        "scripts/sweep_semantic_thresholds.py",
        "scripts/sweep_hybrid_gates.py",
        "scripts/calibration_evaluation.py",
        "src/codedupes/semantic_profiles.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((calibration_evaluation.REPO / relative).read_text())
    monkeypatch.setattr(calibration_evaluation, "REPO", tmp_path)
    projects, models = load_projects(project_ids=["ledger"]), ["gte-modernbert-base"]
    original = selection_context(projects, models)
    policy_file = tmp_path / changed_file
    policy_file.write_text(policy_file.read_text() + "\n# changed selection policy\n")
    updated = selection_context(projects, models)
    assert updated["projects"] == original["projects"]
    assert updated["selection_policy"] != original["selection_policy"]
    with pytest.raises(ValueError, match="stale or mismatched selection"):
        validate_selection_context({"input_context": original}, projects, models)


def test_report_rejects_hybrid_from_another_threshold_selection(tmp_path: Path, monkeypatch):
    project = load_projects(project_ids=["ledger"])[0]
    context = selection_context([project], ["gte-modernbert-base", "embeddinggemma-300m"])
    threshold = {"input_context": context, "models": []}
    hybrid = {"input_context": context, "threshold_selection_digest": selection_digest(threshold)}
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


@pytest.mark.parametrize("second_version", ["2.14.0", "2.13.0"])
def test_report_writer_derives_measurement_runtime(tmp_path: Path, monkeypatch, second_version):
    checked = read_json(DEFAULT_MANIFEST.parent / "calibration-results.json")
    project = load_projects(project_ids=["ledger"])[0]
    # Reuse recorded CPU reports to exercise serialization without model inference.
    reports = {
        (report["model"], "cpu"): report
        for report in checked["projects"][0]["reports"].values()
        if report["device"] == "cpu"
    }
    for report, version in zip(reports.values(), ["2.14.0", second_version], strict=True):
        report["runtime_versions"] = {"torch": version}
    monkeypatch.setattr(report_calibration_distributions, "load_all", lambda *args: reports)
    monkeypatch.setattr(report_calibration_distributions, "full_report", lambda p, report: report)
    monkeypatch.setattr(report_calibration_distributions, "load_projects", lambda *args: [project])
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
            "--devices",
            "cpu",
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
        "scope": "all checked CPU reports",
    }
    assert all(
        report["runtime_versions"]["torch"] == result["measurement_runtime"]["torch"]
        for report in result["projects"][0]["reports"].values()
    )


def test_checked_calibration_result_matches_shipped_profiles():
    result = read_json(DEFAULT_MANIFEST.parent / "calibration-results.json")
    context = result["threshold_selection"]["input_context"]
    assert result["hybrid_selection"]["input_context"] == context
    assert result["hybrid_selection"]["threshold_selection_digest"] == selection_digest(
        result["threshold_selection"]
    )
    for project in load_projects():
        # Runtime identity is machine-specific; checked labels and scope are not.
        assert context["projects"][project.id]["annotations"] == selection_digest(
            project.annotations
        )
        assert context["projects"][project.id]["project"] == selection_digest(project.spec)
        assert context["projects"][project.id]["policy"] == project.policy_name
    assert context["selection_policy"] == selection_context([], [])["selection_policy"]
    assert result["measurement_runtime"]["scope"] == "all checked CPU and MPS reports"
    assert {
        report["runtime_versions"]["torch"]
        for project in result["projects"]
        for report in project["reports"].values()
    } == {result["measurement_runtime"]["torch"]}
    threshold_models = {item["model"]: item for item in result["threshold_selection"]["models"]}
    hybrid_models = {item["model"]: item for item in result["hybrid_selection"]["models"]}

    for model, selection in threshold_models.items():
        profile = resolve_model_profile(model)
        assert selection["search"]["current_threshold"] == profile.default_search_threshold
        assert (
            selection["search"]["current_metrics"]["threshold"] == profile.default_search_threshold
        )
        for language in selection["duplicate_by_language"]:
            assert language["selection_ready"] is True
            assert language["selected_threshold"] == profile.semantic_threshold_for_language(
                language["language"]
            )

        hybrid = hybrid_models[model]
        assert hybrid["selected"]["selection_ready"] is True
        assert (
            hybrid["selected"]["weak_identifier_jaccard_min"]
            == profile.hybrid_weak_identifier_jaccard_min
        )
        assert hybrid["selected"]["statement_ratio_min"] == profile.hybrid_statement_ratio_min
        assert {
            item["language"]: item["selected_gate"] for item in hybrid["promotion_by_language"]
        } == dict(profile.language_high_confidence_thresholds)

    for project in result["projects"]:
        assert all(report["replay_parity"] for report in project["reports"].values())
        assert {report["inference_dtype"] for report in project["reports"].values()} == {"float32"}
        for comparison in project["device_comparisons"].values():
            assert comparison["duplicate_decision_changes"] == []
            assert comparison["search_decision_changes"] == []
