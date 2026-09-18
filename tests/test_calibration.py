"""Focused checks for the calibration corpus and threshold selection."""

from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import pytest

from codedupes import semantic
from codedupes.analyzer import _statement_count_ratio
from codedupes.pairs import ordered_pair_key
from codedupes.semantic_profiles import resolve_model_profile
from codedupes.traditional import find_exact_pair_keys, jaccard_similarity
from scripts import (
    calibration_contract,
    calibration_evaluation,
    report_calibration_distributions,
    sweep_semantic_thresholds,
)
from scripts.calibration_contract import (
    DEFAULT_MANIFEST,
    ProjectAnalyzer,
    analyzer_config,
    eligibility_reason,
    extract_project,
    load_projects,
    read_json,
    resolve_annotations,
    run_behavior,
    unit_ids,
    validate_project,
    write_json,
)
from scripts.calibration_evaluation import (
    F1_RECALL_TOLERANCE,
    MINIMUM_SELECTION_PRECISION,
    SELECTION_SCHEMA_VERSION,
    compare_devices,
    development_projects,
    measurement_digest,
    measurement_digests,
    replay,
    replay_parity,
    selection_context,
    selection_digest,
    selection_objective,
    support_files_digest,
    validate_checked_report,
    validate_measurement_digests,
    validate_measurement_provenance,
    validate_selection_context,
)
from scripts.calibration_measurements import (
    ARTIFACT_VERSION,
    capture,
    load_measurement,
    measurement_fingerprint,
)
from scripts.sweep_hybrid_gates import _selection_map
from scripts.sweep_semantic_thresholds import (
    _search_records,
    _select,
    _select_search,
    duplicate_rows,
    search_rows,
    threshold_grid,
    validate_threshold_selection,
)

pytestmark = pytest.mark.grammar


def _empty_measurement(project, model: str = "gte-modernbert-base") -> dict:
    """Build a complete payload whose non-model evidence matches the corpus."""
    inventory, _ = extract_project(project, inventory=True)
    resolved = resolve_annotations(project, inventory)
    source, _ = extract_project(project)
    measured = list({unit.uid: unit for unit in [*source, *resolved.values()]}.values())
    ids = unit_ids(project, measured, resolved)
    ordered = sorted(measured, key=lambda unit: ids[unit.uid])
    candidates = ProjectAnalyzer(project, analyzer_config(project))._select_semantic_candidates(
        source
    )
    candidate_uids = {unit.uid for unit in candidates}
    exact = find_exact_pair_keys(candidates)
    traditional_result = ProjectAnalyzer(project, analyzer_config(project, semantic=False)).analyze(
        project.root
    )
    traditional = {}
    for duplicate in traditional_result.traditional_duplicates:
        traditional.setdefault(ordered_pair_key(duplicate.unit_a, duplicate.unit_b), []).append(
            {"method": duplicate.method, "similarity": duplicate.similarity}
        )
    unit_ids_by_name = [ids[unit.uid] for unit in ordered]
    rank_by_id = {
        identifier: rank
        for rank, identifier in enumerate(sorted(ids[unit.uid] for unit in candidates), start=1)
    }
    probes = [probe["id"] for probe in project.annotations["probes"]]
    return {
        "schema_version": ARTIFACT_VERSION,
        "metadata": {
            "project": project.id,
            "model": model,
            "canonical_model": resolve_model_profile(model).canonical_name,
            "revision": resolve_model_profile(model).default_revision,
            "requested_device": "cpu",
            "batch_size": 4,
            "inference_dtype": "float32",
            "math_policy": "standard",
            "runtime_versions": semantic.get_semantic_runtime_versions(),
            "captured_profile": {},
            "timing_seconds": {"duplicate": 0.0, "search": 0.0},
            "execution": {
                "duplicate": {"execution_device": "cpu", "cache_hit_rows": 0},
                "search": {"execution_device": "cpu", "cache_hit_rows": 0},
            },
            "live_default": [],
            "input_fingerprint": measurement_fingerprint(project, model, "cpu"),
        },
        "units": [
            {
                "id": ids[unit.uid],
                "language": unit.language,
                "kind": unit.unit_type.name.lower(),
                "statement_count": semantic.get_code_unit_statement_count(unit),
                "embedded": unit.uid in candidate_uids,
            }
            for unit in ordered
        ],
        "pairs": [
            {
                "a": ids[a.uid],
                "b": ids[b.uid],
                "cosine": 0.1 if a.uid in candidate_uids and b.uid in candidate_uids else None,
                "comparable": (
                    eligibility_reason(
                        a,
                        b,
                        candidate_uids,
                        exact,
                        suppress_tests=project.policy.get("suppress_test_semantic_matches", False),
                    )
                    is None
                ),
                "exclusion_reason": eligibility_reason(
                    a,
                    b,
                    candidate_uids,
                    exact,
                    suppress_tests=project.policy.get("suppress_test_semantic_matches", False),
                ),
                "identifier_jaccard": jaccard_similarity(a.identifiers, b.identifiers),
                "statement_ratio": _statement_count_ratio(a, b),
                "traditional": traditional.get(ordered_pair_key(a, b), []),
            }
            for a, b in combinations(ordered, 2)
        ],
        "query_scores": [
            {
                "probe": probe,
                "unit": unit,
                "cosine": 1 - rank_by_id[unit] / 1000 if unit in rank_by_id else None,
                "rank": rank_by_id.get(unit),
            }
            for probe in probes
            for unit in unit_ids_by_name
        ],
    }


def test_recall_preference_stays_within_f1_bound_and_safe_precision():
    rows = []
    for threshold, tp, fp in [(0.80, 70, 5), (0.79, 72, 9), (0.40, 95, 70)]:
        precision, recall = tp / (tp + fp), tp / 100
        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall),
            }
        )
    selected = _select(rows)
    assert selected["threshold"] == 0.79
    assert max(row["f1"] for row in rows) - selected["f1"] <= F1_RECALL_TOLERANCE
    assert _select([rows[0], rows[2]])["threshold"] == 0.80

    tied = [
        {"threshold": 0.82, "precision": 3 / 7, "recall": 0.6, "f1": 0.5},
        {"threshold": 0.87, "precision": 2 / 3, "recall": 0.4, "f1": 0.5},
        {"threshold": 0.89, "precision": 1.0, "recall": 1 / 3, "f1": 0.5},
    ]
    assert _select(tied)["threshold"] == 0.87
    with pytest.raises(ValueError, match="minimum precision"):
        _select([{**tied[0], "precision": MINIMUM_SELECTION_PRECISION - 0.01}])


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
    for project in load_projects():
        report = run_behavior(project)
        assert report["project"] == project.id
        assert [run["id"] for run in report["runs"]] == [
            command["id"] for command in project.spec["behavior_tests"]
        ]
        assert all(run["returncode"] == 0 for run in report["runs"])


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
            {
                "a": "a",
                "b": "b",
                "cosine": 0.91,
                "comparable": True,
                "traditional": [{"method": "jaccard", "similarity": 0.9}],
            },
            {"a": "a", "b": "c", "cosine": 0.78, "comparable": True},
            {"a": "a", "b": "d", "cosine": 0.74, "comparable": True},
            {"a": "b", "b": "d", "cosine": 0.88, "comparable": True},
            {"a": "c", "b": "d", "cosine": 0.99, "comparable": False},
        ]
    }
    rows, detail = duplicate_rows(project, measurement, threshold_grid(0.70, 0.95, 0.01))
    assert detail["selected"]["f1"] == 1.0
    assert detail["selected"]["tp"] == 1
    assert detail["positive_scores"]["count"] == 1
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


def test_search_selection_requires_all_no_result_probes_to_stay_empty():
    records = [
        {
            "key": ("p", "query", "a"),
            "score": 0.82,
            "rank": 1,
            "expected": True,
            "no_result": False,
        },
        {
            "key": ("p", "query", "b"),
            "score": 0.60,
            "rank": 2,
            "expected": True,
            "no_result": False,
        },
        {
            "key": ("p", "none", "c"),
            "score": 0.65,
            "rank": 1,
            "expected": False,
            "no_result": True,
        },
    ]
    rows = search_rows(records, [0.60, 0.70])
    assert _select(rows)["threshold"] == 0.60
    assert _select_search(rows)["threshold"] == 0.70
    with pytest.raises(ValueError, match="keeps all no-result probes empty"):
        _select_search(rows[:1])


def test_threshold_grid_includes_a_stop_between_steps():
    assert threshold_grid(0.70, 0.85, 0.10) == [0.70, 0.80, 0.85]
    assert threshold_grid(0.70, 0.70, 0.10) == [0.70]


def test_coarse_sweep_measures_shipped_thresholds_exactly(tmp_path: Path, monkeypatch):
    project = SimpleNamespace(
        id="sample",
        spec={"languages": ["python"], "split": "development"},
        annotations={
            "pairs": [{"a": "a", "b": "b", "judgment": "positive", "difficulty": "easy"}],
            "probes": [{"id": "q", "expected": ["a"]}],
        },
    )
    measurement = {
        "pairs": [{"a": "a", "b": "b", "cosine": 0.90, "comparable": True}],
        "query_scores": [
            {"probe": "q", "unit": "a", "cosine": 0.69, "rank": 1},
            {"probe": "q", "unit": "b", "cosine": 0.65, "rank": 2},
        ],
    }
    monkeypatch.setattr(sweep_semantic_thresholds, "load_projects", lambda *args: [project])
    monkeypatch.setattr(sweep_semantic_thresholds, "selection_context", lambda *args: {})
    monkeypatch.setattr(sweep_semantic_thresholds, "measurement_digests", lambda *args: {})
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
    assert duplicate["current_threshold"] == duplicate["current_metrics"]["threshold"] == 0.87
    assert duplicate["current_metrics"]["tp"] == 1
    assert duplicate["current_difficulty_recall"]["easy"]["detected"] == 1
    search = result["models"][0]["search"]
    assert search["current_threshold"] == search["current_metrics"]["threshold"] == 0.68
    assert (search["current_metrics"]["tp"], search["current_metrics"]["fp"]) == (1, 0)
    assert [row["threshold"] for row in search["selection_window"]] == result["grids"]["search"]
    assert search["selected_metrics"] in search["selection_window"]
    assert result["grids"]["search"] == [0.0, 0.3, 0.6, 0.9, 1.0]
    measurements = {("sample", "gte-modernbert-base"): measurement}
    validate_threshold_selection(result, [project], ["gte-modernbert-base"], measurements)
    result["models"][0]["duplicate_by_language"][0]["selected_threshold"] = 0.0
    with pytest.raises(ValueError, match="does not match its raw measurements"):
        validate_threshold_selection(result, [project], ["gte-modernbert-base"], measurements)


def test_replay_matches_production_tier_rules():
    measurement = {
        "metadata": {
            "model": "gte-modernbert-base",
            "captured_profile": {
                "semantic_threshold": 0.87,
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
                "cosine": 0.875,
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
    write_json(path, _empty_measurement(project))
    assert (
        load_measurement(
            path,
            project,
            expected_model="gte-modernbert-base",
            expected_device="cpu",
        )["metadata"]["project"]
        == "ledger"
    )
    fingerprint = measurement_fingerprint(project, "gte-modernbert-base", "cpu")
    project.annotations["pairs"][0]["rationale"] += " label-only edit"
    assert measurement_fingerprint(project, "gte-modernbert-base", "cpu") == fingerprint
    project.annotations["units"].reverse()
    assert measurement_fingerprint(project, "gte-modernbert-base", "cpu") == fingerprint
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


def test_calibration_identity_and_capture_reject_mps_fast_math(tmp_path: Path, monkeypatch):
    project = load_projects(project_ids=["ledger"])[0]
    monkeypatch.delenv("PYTORCH_MPS_FAST_MATH", raising=False)
    faithful = measurement_fingerprint(project, "gte-modernbert-base", "mps")
    cpu = measurement_fingerprint(project, "gte-modernbert-base", "cpu")

    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")

    assert measurement_fingerprint(project, "gte-modernbert-base", "mps") != faithful
    assert measurement_fingerprint(project, "gte-modernbert-base", "cpu") == cpu
    with pytest.raises(ValueError, match="disable PYTORCH_MPS_FAST_MATH"):
        capture(project, "gte-modernbert-base", "mps", tmp_path)


@pytest.mark.parametrize("mutation", ["missing_pair", "duplicate_query", "bad_rank"])
def test_measurements_reject_incomplete_or_duplicate_score_matrices(tmp_path: Path, mutation: str):
    project = load_projects(project_ids=["ledger"])[0]
    measurement = _empty_measurement(project)
    if mutation == "missing_pair":
        measurement["pairs"].pop()
        message = "incomplete measurement pair matrix"
    elif mutation == "duplicate_query":
        measurement["query_scores"].append(measurement["query_scores"][0].copy())
        message = "duplicate measurement query row"
    else:
        measurement["query_scores"][0]["rank"] = 0
        message = "inconsistent score state"
    path = tmp_path / "measurement.json"
    write_json(path, measurement)
    with pytest.raises(ValueError, match=message):
        load_measurement(path, project)


@pytest.mark.parametrize("field", ["pairs", "query_scores"])
@pytest.mark.parametrize("score", [True, 1.1, -1.1])
def test_measurements_reject_boolean_or_out_of_range_cosines(
    tmp_path: Path, field: str, score: float | bool
):
    project = load_projects(project_ids=["ledger"])[0]
    measurement = _empty_measurement(project)
    row = next(row for row in measurement[field] if row["cosine"] is not None)
    row["cosine"] = score
    path = tmp_path / "measurement.json"
    write_json(path, measurement)
    with pytest.raises(ValueError, match="inconsistent score state"):
        load_measurement(path, project)


@pytest.mark.parametrize(
    "field",
    [
        "comparable",
        "exclusion_reason",
        "traditional",
        "identifier_jaccard",
        "statement_ratio",
    ],
)
def test_measurements_reject_tampered_corpus_evidence(tmp_path: Path, field: str):
    project = load_projects(project_ids=["ledger"])[0]
    measurement = _empty_measurement(project)
    pair = measurement["pairs"][0]
    if field == "comparable":
        pair[field] = not pair[field]
    elif field == "exclusion_reason":
        pair[field] = "forged"
    elif field == "traditional":
        pair[field] = [*pair[field], {"method": "jaccard", "similarity": 1.0}]
    else:
        pair[field] = 0.0 if pair[field] != 0.0 else 1.0
    path = tmp_path / "measurement.json"
    write_json(path, measurement)
    with pytest.raises(ValueError, match="corpus evidence"):
        load_measurement(path, project)


def test_derived_selections_bind_exact_raw_scores():
    project = load_projects(project_ids=["ledger"])[0]
    measurement = _empty_measurement(project)
    payload = {"measurement_digests": measurement_digests([measurement])}
    validate_measurement_digests(payload, [measurement])
    measurement["pairs"][0]["cosine"] = 0.5
    assert payload["measurement_digests"] != measurement_digests([measurement])
    with pytest.raises(ValueError, match="different raw measurements"):
        validate_measurement_digests(payload, [measurement])
    assert measurement_digest(measurement)


@pytest.mark.parametrize("field", ["execution", "live_default", "timing_seconds"])
def test_derived_selections_bind_claimed_execution_provenance(field: str):
    project = load_projects(project_ids=["ledger"])[0]
    measurement = _empty_measurement(project)
    original = measurement_digest(measurement)
    if field == "execution":
        measurement["metadata"][field]["duplicate"]["cache_hit_rows"] = 17
    elif field == "live_default":
        measurement["metadata"][field].append({"a": "forged", "b": "pair", "tier": "exact"})
    else:
        measurement["metadata"][field]["duplicate"] = 99.0

    assert measurement_digest(measurement) != original


def test_device_comparison_rejects_score_coverage_mismatch():
    cpu = {
        "pairs": [{"a": "a", "b": "b", "cosine": 0.5}],
        "query_scores": [],
    }
    mps = {"pairs": [], "query_scores": []}
    with pytest.raises(ValueError, match="pair score coverage differs"):
        compare_devices(cpu, mps, SimpleNamespace(id="sample"))


def test_measurement_provenance_rejects_forged_runtime_and_fast_math(monkeypatch):
    project = load_projects(project_ids=["ledger"])[0]
    measurement = _empty_measurement(project)
    profile = resolve_model_profile("gte-modernbert-base")
    measurement["metadata"].update(
        {
            "captured_profile": {
                "semantic_threshold": profile.semantic_threshold_for_language("python"),
                "weak_identifier_jaccard_min": profile.hybrid_weak_identifier_jaccard_min,
                "statement_ratio_min": profile.hybrid_statement_ratio_min,
                "high_gate": profile.high_confidence_threshold_for_language("python"),
            },
            "timing_seconds": {"duplicate": 1.0, "search": 1.0},
        }
    )
    encoded_inputs = sum(unit["embedded"] for unit in measurement["units"])
    execution = {
        "execution_device": "cpu",
        "cache_hit_rows": 0,
        "cache_enabled": False,
        "model_loaded": True,
        "requested_rows": encoded_inputs,
        "unique_inputs": encoded_inputs,
        "encoded_inputs": encoded_inputs,
    }
    measurement["metadata"]["execution"] = {
        "duplicate": execution.copy(),
        "search": execution.copy(),
    }
    validate_measurement_provenance(project, measurement)
    recorded_runtime = measurement["metadata"]["runtime_versions"].copy()
    measurement["metadata"]["runtime_versions"]["torch"] = "forged"
    with pytest.raises(ValueError, match="provenance does not match"):
        validate_measurement_provenance(project, measurement)

    measurement["metadata"]["runtime_versions"] = recorded_runtime
    measurement["metadata"]["requested_device"] = "mps"
    measurement["metadata"]["input_fingerprint"] = measurement_fingerprint(
        project, "gte-modernbert-base", "mps"
    )
    for stats in measurement["metadata"]["execution"].values():
        stats["execution_device"] = "mps"
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")
    with pytest.raises(ValueError, match="provenance does not match"):
        validate_measurement_provenance(project, measurement)


def test_measurement_provenance_rejects_boolean_timings():
    project = load_projects(project_ids=["ledger"])[0]
    measurement = _empty_measurement(project)
    profile = resolve_model_profile("gte-modernbert-base")
    measurement["metadata"].update(
        {
            "captured_profile": {
                "semantic_threshold": profile.semantic_threshold_for_language("python"),
                "weak_identifier_jaccard_min": profile.hybrid_weak_identifier_jaccard_min,
                "statement_ratio_min": profile.hybrid_statement_ratio_min,
                "high_gate": profile.high_confidence_threshold_for_language("python"),
            },
            "timing_seconds": {"duplicate": True, "search": True},
        }
    )
    encoded_inputs = sum(unit["embedded"] for unit in measurement["units"])
    execution = {
        "execution_device": "cpu",
        "cache_hit_rows": 0,
        "cache_enabled": False,
        "model_loaded": True,
        "requested_rows": encoded_inputs,
        "unique_inputs": encoded_inputs,
        "encoded_inputs": encoded_inputs,
    }
    measurement["metadata"]["execution"] = {
        "duplicate": execution.copy(),
        "search": execution.copy(),
    }
    with pytest.raises(ValueError, match="invalid measurement timing provenance"):
        validate_measurement_provenance(project, measurement)


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
    fingerprint = measurement_fingerprint(project, "gte-modernbert-base", "cpu")
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "example.cpython-312.pyc").write_bytes(b"generated bytecode")
    (source / ".DS_Store").write_bytes(b"finder metadata")
    assert measurement_fingerprint(project, "gte-modernbert-base", "cpu") == fingerprint
    module.write_text("def example(value):\n    return value + 2\n")
    assert measurement_fingerprint(project, "gte-modernbert-base", "cpu") != fingerprint
    module.write_text("def example(value):\n    return value + 1\n")
    (source / "additional.py").write_text("def additional(value):\n    return value * 2\n")
    assert measurement_fingerprint(project, "gte-modernbert-base", "cpu") != fingerprint


@pytest.mark.parametrize("change", ["rename", "retarget", "remove"])
def test_measurements_reject_changed_annotation_identities(tmp_path: Path, change: str):
    project = load_projects(project_ids=["ledger"])[0]
    path = tmp_path / "measurement.json"
    write_json(path, _empty_measurement(project))
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


def test_support_file_identity_tracks_declared_behavior_evidence(tmp_path: Path):
    readme = tmp_path / "README.md"
    tests = tmp_path / "tests"
    tests.mkdir()
    behavior = tests / "test_behavior.txt"
    readme.write_text("behavior contract v1\n")
    behavior.write_text("behavior v1\n")
    project = SimpleNamespace(
        id="sample",
        root=tmp_path,
        spec={"support_files": ["README.md"], "test_roots": ["tests"]},
    )
    original = support_files_digest(project)
    cache = tests / "__pycache__"
    cache.mkdir()
    (cache / "test_behavior.cpython-312.pyc").write_bytes(b"generated")
    assert support_files_digest(project) == original
    readme.write_text("behavior contract v2\n")
    assert support_files_digest(project) != original
    readme.write_text("behavior contract v1\n")
    behavior.write_text("behavior v2\n")
    assert support_files_digest(project) != original
    project.spec["support_files"] = ["missing.*"]
    with pytest.raises(ValueError, match="matched nothing"):
        support_files_digest(project)


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
    assert result["projects"][0]["split"] == "development"
    assert all(
        report["runtime_versions"]["torch"] == result["measurement_runtime"]["torch"]
        for report in result["projects"][0]["reports"].values()
    )


def test_checked_calibration_result_matches_shipped_profiles(monkeypatch):
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
    recorded_runtime = dict(next(iter(recorded_runtimes)))
    monkeypatch.setattr(semantic, "get_semantic_runtime_versions", lambda: recorded_runtime)
    monkeypatch.setattr(
        "scripts.calibration_measurements.semantic.get_semantic_runtime_versions",
        lambda: recorded_runtime,
    )
    models = [item["model"] for item in result["threshold_selection"]["models"]]
    validate_checked_report(result, projects, models)
    expected = selection_context(projects, models)
    assert {
        project_id: project_context["measurements"]
        for project_id, project_context in context["projects"].items()
    } == {
        project_id: project_context["measurements"]
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

    for project in result["projects"]:
        assert all(report["replay_parity"] for report in project["reports"].values())
        assert {report["inference_dtype"] for report in project["reports"].values()} == {"float32"}
        for comparison in project["device_comparisons"].values():
            assert comparison["duplicate_decision_changes"] == []
            assert comparison["search_decision_changes"] == []


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
    ("tamper", "message"),
    [
        ("schema", "unsupported schema version"),
        ("objective", "mismatched objective contract"),
        ("gate", "selected calibration gates do not match shipped defaults"),
        ("metric", "inconsistent derived metrics"),
        ("readiness", "selection evidence is inconsistent"),
        ("grids", "candidate grids"),
        ("scores", "score summary"),
        ("difficulty", "difficulty schema"),
        ("unjudged", "invalid unresolved pair"),
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
        metrics = threshold["models"][0]["duplicate_by_language"][0]["selected_metrics"]
        metrics["unjudged_predictions"] = 1
        metrics["predicted"] += 1
    elif tamper == "grids":
        threshold["grids"] = "forged"
    elif tamper == "scores":
        threshold["models"][0]["duplicate_by_language"][0]["positive_scores"] = "forged"
    elif tamper == "difficulty":
        threshold["models"][0]["duplicate_by_language"][0]["selected_difficulty_recall"] = "forged"
    elif tamper == "unjudged":
        threshold["models"][0]["duplicate_by_language"][0]["unjudged_above_selected"] = [
            ["a", "b", 2.0]
        ]
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
        "identity",
        "dtype",
        "runtime",
        "batch",
        "math",
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
