"""Checked calibration result and report-schema validation tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from codedupes import semantic
from codedupes.semantic_profiles import resolve_model_profile
from scripts import (
    calibration_evaluation,
)
from scripts.calibration_contract import (
    DEFAULT_MANIFEST,
    load_projects,
    read_json,
)
from scripts.calibration_evaluation import (
    MINIMUM_SELECTION_PRECISION,
    development_projects,
    selection_context,
    selection_digest,
    support_files_digest,
    validate_checked_report,
)

pytestmark = pytest.mark.grammar


def test_checked_calibration_result_matches_shipped_profiles():
    result = read_json(DEFAULT_MANIFEST.parent / "calibration-results.json")
    assert len(result["measurement_digests"]) == 20
    assert all(key.endswith(("/cpu", "/mps")) for key in result["measurement_digests"])
    context = result["threshold_selection"]["input_context"]
    assert result["hybrid_selection"]["input_context"] == context
    assert result["hybrid_selection"]["threshold_selection_digest"] == selection_digest(
        result["threshold_selection"]
    )
    projects = load_projects()
    for project in projects:
        # Runtime identity is machine-specific; checked labels and scope are not.
        assert context["projects"][project.id]["annotations"] == selection_digest(
            project.annotations
        )
        assert context["projects"][project.id]["project"] == selection_digest(project.spec)
        assert context["projects"][project.id]["policy"] == project.policy_name
        assert context["projects"][project.id]["support_files"] == support_files_digest(project)
    assert context["selection_policy"] == selection_context([], [])["selection_policy"]
    recorded_runtimes = {
        tuple(sorted(report["runtime_versions"].items()))
        for project in result["projects"]
        for report in project["reports"].values()
    }
    assert len(recorded_runtimes) == 1
    models = [item["model"] for item in result["threshold_selection"]["models"]]
    validate_checked_report(result, projects, models)
    expected = selection_context(projects, models)
    assert {
        project_id: project_context["measurement_behavior"]
        for project_id, project_context in context["projects"].items()
    } == {
        project_id: project_context["measurement_behavior"]
        for project_id, project_context in expected["projects"].items()
    }
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
        assert selection["search"]["selected_threshold"] == profile.default_search_threshold
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
        assert all(
            item["selected_metrics"]["precision"] >= MINIMUM_SELECTION_PRECISION
            for item in hybrid["promotion_by_language"]
        )
        assert all(
            item["precision"] >= MINIMUM_SELECTION_PRECISION
            for item in selection["search"]["selected_per_language"]
        )

    for project in result["projects"]:
        assert all(report["replay_parity"] for report in project["reports"].values())
        assert {report["inference_dtype"] for report in project["reports"].values()} == {"float32"}
        for comparison in project["device_comparisons"].values():
            assert comparison["duplicate_decision_changes"] == []
            assert comparison["search_decision_changes"] == []


def test_checked_calibration_result_validation_is_runtime_independent(monkeypatch):
    """Validate recorded provenance without requiring the measurement runtime locally."""
    result = read_json(DEFAULT_MANIFEST.parent / "calibration-results.json")
    models = [item["model"] for item in result["threshold_selection"]["models"]]
    for name in ("get_semantic_runtime_versions", "_resolve_model_dtype", "_mps_fast_math_variant"):
        monkeypatch.setattr(
            semantic,
            name,
            lambda *_args, **_kwargs: pytest.fail(
                "checked-only validation inspected the installed runtime"
            ),
        )
    monkeypatch.setenv("CODEDUPES_CPU_BF16", "1")
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")

    validate_checked_report(result, load_projects(), models)


def test_checked_calibration_report_schema_accepts_committed_result():
    result = read_json(DEFAULT_MANIFEST.parent / "calibration-results.json")
    models = [item["model"] for item in result["threshold_selection"]["models"]]
    validate_checked_report(result, load_projects(), models)


def test_checked_selection_validation_scopes_context_to_development_projects(monkeypatch):
    result = read_json(DEFAULT_MANIFEST.parent / "calibration-results.json")
    projects = load_projects()
    projects[-1] = replace(projects[-1], spec=projects[-1].spec | {"split": "evaluation"})
    models = [item["model"] for item in result["threshold_selection"]["models"]]

    def assert_development_scope(_payload, scoped_projects, _models):
        assert scoped_projects == development_projects(projects)
        raise RuntimeError("development scope verified")

    monkeypatch.setattr(
        calibration_evaluation, "validate_selection_context", assert_development_scope
    )
    with pytest.raises(RuntimeError, match="development scope verified"):
        validate_checked_report(result, projects, models)


def test_checked_selection_metrics_ignore_evaluation_report_records():
    result = read_json(DEFAULT_MANIFEST.parent / "calibration-results.json")
    records = {record["project"]: record for record in result["projects"]}
    evaluation = deepcopy(result["projects"][0])
    evaluation["project"] = "held-out-evaluation"
    evaluation["split"] = "evaluation"
    evaluation["reports"]["gte-modernbert-base/cpu"]["search"]["tp"] = 999
    records[evaluation["project"]] = evaluation
    models = tuple(item["model"] for item in result["threshold_selection"]["models"])

    calibration_evaluation._validate_checked_selection_outcomes(
        result["threshold_selection"], result["hybrid_selection"], records, models
    )


@pytest.mark.parametrize(
    "tamper",
    ["weak_identifier_jaccard_min", "statement_ratio_min", "promotion"],
)
def test_checked_hybrid_selected_gates_belong_to_candidate_grids(tamper: str):
    result = read_json(DEFAULT_MANIFEST.parent / "calibration-results.json")
    hybrid = deepcopy(result["hybrid_selection"]["models"][0])
    selected = hybrid["selected"]
    promotions = hybrid["promotion_by_language"]
    if tamper == "weak_identifier_jaccard_min":
        selected[tamper] = 0.39
    elif tamper == "statement_ratio_min":
        selected[tamper] = 0.33
    else:
        promotions[0]["selected_gate"] = 0.875

    with pytest.raises(ValueError, match="selected .* gate is outside the candidate grid"):
        calibration_evaluation._validate_hybrid_selection_audit(
            hybrid["selection_audit"],
            selected,
            promotions,
            set(hybrid["admission_thresholds"]),
            hybrid["admission_thresholds"],
            "hybrid audit",
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("schema", "unsupported schema version"),
        ("objective", "mismatched objective contract"),
        ("gate", "selected calibration gates do not match shipped defaults"),
        ("metric", "inconsistent derived metrics"),
        ("readiness", "selection evidence is inconsistent"),
        ("grids", "candidate grids"),
        ("grids_policy", "mismatched candidate grids"),
        ("scores", "score summary"),
        ("difficulty", "difficulty schema"),
        ("unjudged", "invalid unresolved pair"),
        ("window", "search selection window"),
        ("search_language", "inconsistent language evidence"),
        ("admission_window", "selection window"),
        ("candidate_grids", "mismatched candidate grids"),
        ("candidate_grids_bool", "mismatched candidate grids"),
        ("hybrid_audit", "inconsistent derived metrics"),
        ("hybrid_audit_bool", "outside the candidate grid"),
        ("hybrid_audit_grid", "outside the candidate grid"),
        ("hybrid_audit_promotion_grid", "outside the candidate grid"),
        ("hybrid_audit_digest", "inconsistent language evidence"),
        ("hybrid_audit_denominator", "positive-pair denominator"),
        ("hybrid_audit_order", "outranks the selected candidate"),
        ("hybrid_audit_repeated_outcome", "repeats the selected outcome"),
        ("hybrid_audit_repeated_selection", "repeats the selected candidate"),
        ("corroboration", "positive-pair denominator"),
    ],
)
def test_checked_calibration_result_rejects_tampered_embedded_selections(tamper: str, message: str):
    result = read_json(DEFAULT_MANIFEST.parent / "calibration-results.json")
    threshold = result["threshold_selection"]
    hybrid = result["hybrid_selection"]
    if tamper == "schema":
        threshold["schema_version"] = hybrid["schema_version"] = 999
    elif tamper == "objective":
        threshold["objective"]["minimum_precision"] = 0.0
        hybrid["objective"]["minimum_precision"] = 0.0
    elif tamper == "gate":
        threshold["models"][0]["duplicate_by_language"][0]["selected_threshold"] = 0.0
    elif tamper == "metric":
        threshold["models"][0]["duplicate_by_language"][0]["selected_metrics"]["precision"] = 0.0
    elif tamper == "readiness":
        entry = threshold["models"][0]["duplicate_by_language"][0]
        metrics = entry["selected_metrics"]
        metrics["unjudged_predictions"] = 1
        metrics["predicted"] += 1
        selected_window_row = next(
            row
            for row in entry["selection_window"]
            if row["threshold"] == entry["selected_threshold"]
        )
        selected_window_row["unjudged_predictions"] = 1
        selected_window_row["predicted"] += 1
    elif tamper == "grids":
        threshold["grids"] = "forged"
    elif tamper == "grids_policy":
        threshold["grids"]["duplicate"][1] = 0.005
    elif tamper == "scores":
        threshold["models"][0]["duplicate_by_language"][0]["positive_scores"] = "forged"
    elif tamper == "difficulty":
        threshold["models"][0]["duplicate_by_language"][0]["selected_difficulty_recall"] = "forged"
    elif tamper == "unjudged":
        threshold["models"][0]["duplicate_by_language"][0]["unjudged_above_selected"] = [
            ["a", "b", 2.0]
        ]
    elif tamper == "window":
        threshold["models"][0]["search"]["selection_window"] = []
    elif tamper == "search_language":
        threshold["models"][0]["search"]["selected_per_language"][0]["language"] = "go"
    elif tamper == "admission_window":
        threshold["models"][0]["duplicate_by_language"][0]["selection_window"] = []
    elif tamper == "candidate_grids":
        hybrid["candidate_grids"] = {}
    elif tamper == "candidate_grids_bool":
        hybrid["candidate_grids"]["weak_identifier_jaccard_min"][0] = False
    elif tamper == "hybrid_audit":
        hybrid["models"][0]["selection_audit"]["best_f1_candidate"]["metrics"]["precision"] = 0.0
    elif tamper == "hybrid_audit_bool":
        hybrid["models"][0]["selection_audit"]["best_f1_candidate"]["statement_ratio_min"] = False
    elif tamper == "hybrid_audit_grid":
        hybrid["models"][0]["selection_audit"]["best_f1_candidate"][
            "weak_identifier_jaccard_min"
        ] = 0.39
    elif tamper == "hybrid_audit_promotion_grid":
        hybrid["models"][0]["selection_audit"]["best_f1_candidate"]["high_gates"]["c"] = 0.845
    elif tamper == "hybrid_audit_digest":
        hybrid["models"][0]["selection_audit"]["best_f1_candidate"]["per_language"][0][
            "outcome_digest"
        ] = "forged"
    elif tamper == "hybrid_audit_denominator":
        rows = hybrid["models"][0]["selection_audit"]["best_f1_candidate"]["per_language"]
        rows[0]["metrics"]["tp"] += 1
        rows[1]["metrics"]["tp"] -= 1
        for row in rows[:2]:
            metrics = row["metrics"]
            precision, recall, f1 = calibration_evaluation._score_ratios(
                metrics["tp"], metrics["fp"], metrics["fn"]
            )
            metrics.update(
                {
                    "precision": precision,
                    "judged_only_precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    elif tamper == "hybrid_audit_order":
        hybrid["models"][0]["selection_audit"]["best_f1_candidate"][
            "weak_identifier_jaccard_min"
        ] = 0.0
    elif tamper in {"hybrid_audit_repeated_outcome", "hybrid_audit_repeated_selection"}:
        model = hybrid["models"][1]
        candidate = {
            "weak_identifier_jaccard_min": model["selected"]["weak_identifier_jaccard_min"],
            "statement_ratio_min": model["selected"]["statement_ratio_min"],
            "high_gates": {
                item["language"]: item["selected_gate"] for item in model["promotion_by_language"]
            },
            "metrics": deepcopy(model["selected"]["metrics"]),
            "per_language": [
                {
                    "language": item["language"],
                    "high_gate": item["selected_gate"],
                    "outcome_digest": item["selected_outcome_digest"],
                    "metrics": deepcopy(item["selected_metrics"]),
                }
                for item in model["promotion_by_language"]
            ],
        }
        if tamper == "hybrid_audit_repeated_outcome":
            candidate["statement_ratio_min"] = 0.2
        model["selection_audit"] = {
            "best_f1_candidate": deepcopy(candidate),
            "runner_up": candidate,
        }
    else:
        corroboration = hybrid["models"][0]["selected"]["corroboration_only_metrics"]
        corroboration.update(
            {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "judged_only_precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ambiguous_predictions": 0,
                "unjudged_predictions": 0,
            }
        )
    hybrid["threshold_selection_digest"] = selection_digest(threshold)
    models = [item["model"] for item in threshold["models"]]
    with pytest.raises(ValueError, match=message):
        validate_checked_report(result, load_projects(), models)


@pytest.mark.parametrize(
    "tamper",
    [
        "digest",
        "selection_digest",
        "timing",
        "execution",
        "query_execution",
        "identity",
        "dtype",
        "runtime",
        "batch",
        "math",
        "fallback",
        "encoded_inputs",
    ],
)
def test_checked_calibration_result_rejects_tampered_provenance(tamper: str):
    result = read_json(DEFAULT_MANIFEST.parent / "calibration-results.json")
    projects = load_projects()
    models = [item["model"] for item in result["threshold_selection"]["models"]]
    first_project = result["projects"][0]
    first_report = first_project["reports"]["gte-modernbert-base/cpu"]
    if tamper == "digest":
        result["measurement_digests"]["ledger/gte-modernbert-base/cpu"] = "forged-digest"
    elif tamper == "selection_digest":
        result["threshold_selection"]["measurement_digests"]["ledger/gte-modernbert-base/cpu"] = (
            "0" * 64
        )
        result["hybrid_selection"]["threshold_selection_digest"] = selection_digest(
            result["threshold_selection"]
        )
    elif tamper == "timing":
        first_report["timing_seconds"]["duplicate"] = True
    elif tamper == "execution":
        first_report["execution"]["duplicate"]["cache_hit_rows"] = 1
    elif tamper == "query_execution":
        first_report["query_execution"]["device"] = "mps"
    elif tamper == "identity":
        first_report["device"] = "mps"
    elif tamper == "dtype":
        first_report["inference_dtype"] = "float16"
    elif tamper == "runtime":
        for project in result["projects"]:
            for report in project["reports"].values():
                report["runtime_versions"] = {
                    "python": "forged-python",
                    "torch": "0.0.0-forged",
                }
        result["measurement_runtime"]["torch"] = "0.0.0-forged"
    elif tamper == "batch":
        first_report["batch_size"] = 999_999
    elif tamper == "math":
        first_report["math_policy"] = "mpsfm=1"
    elif tamper == "fallback":
        first_report["mps_operator_fallback"] = True
    else:
        first_report["execution"]["duplicate"]["encoded_inputs"] = 1
    with pytest.raises(ValueError, match="checked"):
        validate_checked_report(result, projects, models)


@pytest.mark.parametrize(
    "tamper",
    [
        "schema",
        "arithmetic",
        "denominator",
        "no_result",
        "deterministic",
        "drift",
        "decision_changes",
        "foreign_decision_change",
    ],
)
def test_checked_calibration_result_rejects_forged_report_metrics(tamper: str):
    result = read_json(DEFAULT_MANIFEST.parent / "calibration-results.json")
    models = [item["model"] for item in result["threshold_selection"]["models"]]
    duplicate = result["projects"][0]["reports"]["gte-modernbert-base/cpu"]["duplicate"]
    if tamper == "schema":
        result["projects"][0]["reports"]["gte-modernbert-base/cpu"]["duplicate"] = {"forged": True}
    elif tamper == "arithmetic":
        duplicate["published"]["precision"] = 0.0
    elif tamper == "denominator":
        metrics = duplicate["published"]
        metrics["fn"] = 999
        metrics["recall"] = metrics["tp"] / (metrics["tp"] + metrics["fn"])
        metrics["f1"] = (
            2
            * metrics["precision"]
            * metrics["recall"]
            / (metrics["precision"] + metrics["recall"])
        )
    elif tamper == "no_result":
        result["projects"][0]["reports"]["gte-modernbert-base/mps"]["search"]["no_result"] = {
            "clean": 0,
            "total": 0,
            "violations": [],
        }
    elif tamper == "deterministic":
        for device in ("cpu", "mps"):
            report = result["projects"][0]["reports"][f"gte-modernbert-base/{device}"]
            for field in ("published", "visible", "deterministic"):
                metrics = report["duplicate"][field]
                metrics["fp"] += 1
                metrics["precision"] = metrics["judged_only_precision"] = metrics["tp"] / (
                    metrics["tp"] + metrics["fp"]
                )
                metrics["f1"] = (
                    2
                    * metrics["precision"]
                    * metrics["recall"]
                    / (metrics["precision"] + metrics["recall"])
                    if metrics["precision"] + metrics["recall"]
                    else 0.0
                )
            report["duplicate"]["tiers"]["traditional_near"] = 1
    elif tamper == "drift":
        summary = result["projects"][0]["device_comparisons"]["gte-modernbert-base"][
            "pair_score_abs_drift"
        ]
        summary["p95"] = summary["max"] = 0.0
    elif tamper == "decision_changes":
        search = result["projects"][0]["reports"]["gte-modernbert-base/mps"]["search"]
        search.update({"tp": 0, "fn": 12, "precision": 0.0, "recall": 0.0, "f1": 0.0})
        project = load_projects(project_ids=["ledger"])[0]
        result["projects"][0]["device_comparisons"]["gte-modernbert-base"][
            "search_decision_changes"
        ] = [[project.annotations["probes"][0]["id"], project.annotations["units"][0]["id"]]]
    else:
        result["projects"][0]["device_comparisons"]["gte-modernbert-base"][
            "search_decision_changes"
        ] = [["not-a-probe", "not-a-unit"]]

    with pytest.raises(
        ValueError, match="checked|duplicate|metrics|corpus|search evidence|contradict"
    ):
        validate_checked_report(result, load_projects(), models)
