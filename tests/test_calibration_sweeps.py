"""Threshold sweep, selection, and replay tests."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import (
    report_calibration_distributions,
    sweep_hybrid_gates,
    sweep_semantic_thresholds,
)
from scripts.calibration_contract import (
    read_json,
)
from scripts.calibration_evaluation import (
    MINIMUM_SELECTION_PRECISION,
    replay,
    replay_parity,
    threshold_candidate_grids,
)
from scripts.sweep_hybrid_gates import _selection_map
from scripts.sweep_semantic_thresholds import (
    _select,
    _select_search,
    duplicate_rows,
    search_rows,
    threshold_grid,
    validate_threshold_selection,
)

pytestmark = pytest.mark.grammar


def test_recall_preference_applies_only_to_exact_best_f1_ties():
    rows = []
    for threshold, tp, fp in [(0.80, 70, 5), (0.79, 72, 9), (0.40, 95, 70)]:
        precision, recall = tp / (tp + fp), tp / 100
        rows.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": 100 - tp,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall),
            }
        )
    selected = _select(rows)
    assert selected["threshold"] == 0.80
    assert selected["f1"] == max(row["f1"] for row in rows)
    assert _select([rows[0], rows[2]])["threshold"] == 0.80

    tied = [
        {
            "threshold": 0.82,
            "tp": 3,
            "fp": 4,
            "fn": 2,
            "precision": 3 / 7,
            "recall": 0.6,
            "f1": 0.5,
        },
        {
            "threshold": 0.87,
            "tp": 2,
            "fp": 1,
            "fn": 3,
            "precision": 2 / 3,
            "recall": 0.4,
            "f1": 0.5,
        },
        {
            "threshold": 0.89,
            "tp": 1,
            "fp": 0,
            "fn": 2,
            "precision": 1.0,
            "recall": 1 / 3,
            "f1": 0.5,
        },
    ]
    assert _select(tied)["threshold"] == 0.87

    rounded_tie = [
        {
            "threshold": 0.90,
            "tp": 1,
            "fp": 0,
            "fn": 4,
            "precision": 1.0,
            "recall": 0.2,
            "f1": 0.33333333333333337,
        },
        {
            "threshold": 0.80,
            "tp": 1,
            "fp": 1,
            "fn": 3,
            "precision": 0.5,
            "recall": 0.25,
            "f1": 0.3333333333333333,
        },
    ]
    assert _select(rounded_tie)["threshold"] == 0.80
    with pytest.raises(ValueError, match="minimum precision"):
        _select([{**tied[0], "precision": MINIMUM_SELECTION_PRECISION - 0.01}])


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
            "language": "python",
            "score": 0.82,
            "rank": 1,
            "expected": True,
            "no_result": False,
        },
        {
            "key": ("p", "q", "b"),
            "language": "python",
            "score": 0.66,
            "rank": 2,
            "expected": True,
            "no_result": False,
        },
        {
            "key": ("p", "q", "c"),
            "language": "python",
            "score": 0.40,
            "rank": 3,
            "expected": False,
            "no_result": False,
        },
        {
            "key": ("p", "none", "d"),
            "language": "python",
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
            "language": "python",
            "score": 0.82,
            "rank": 1,
            "expected": True,
            "no_result": False,
        },
        {
            "key": ("p", "query", "b"),
            "language": "python",
            "score": 0.60,
            "rank": 2,
            "expected": True,
            "no_result": False,
        },
        {
            "key": ("p", "none", "c"),
            "language": "python",
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


def test_search_selection_requires_safe_precision_in_every_language():
    records = [
        {
            "key": ("python-project", "query", f"py-{index}"),
            "language": "python",
            "score": 0.90 if index == 1 else 0.70,
            "rank": index,
            "expected": True,
            "no_result": False,
        }
        for index in range(1, 5)
    ]
    records.extend(
        [
            {
                "key": ("rust-project", "query", "rust-expected"),
                "language": "rust",
                "score": 0.90,
                "rank": 1,
                "expected": True,
                "no_result": False,
            },
            {
                "key": ("rust-project", "query", "rust-fp-1"),
                "language": "rust",
                "score": 0.65,
                "rank": 2,
                "expected": False,
                "no_result": False,
            },
            {
                "key": ("rust-project", "query", "rust-fp-2"),
                "language": "rust",
                "score": 0.64,
                "rank": 3,
                "expected": False,
                "no_result": False,
            },
        ]
    )
    rows = search_rows(records, [0.60, 0.80])
    assert rows[0]["precision"] > MINIMUM_SELECTION_PRECISION
    assert rows[0]["per_language"][1]["precision"] < MINIMUM_SELECTION_PRECISION
    assert _select_search(rows)["threshold"] == 0.80


def test_threshold_grid_includes_a_stop_between_steps():
    assert threshold_grid(0.70, 0.85, 0.10) == [0.70, 0.80, 0.85]
    assert threshold_grid(0.70, 0.70, 0.10) == [0.70]
    for step in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite values"):
            threshold_grid(0.0, 1.0, step)


def test_canonical_sweep_measures_shipped_thresholds_exactly(tmp_path: Path, monkeypatch):
    project = SimpleNamespace(
        id="sample",
        spec={"languages": ["python"], "split": "development"},
        annotations={
            "pairs": [{"a": "a", "b": "b", "judgment": "positive", "difficulty": "easy"}],
            "probes": [{"id": "q", "kind": "behavioral", "expected": ["a"]}],
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
        ["sweep", "--models", "gte-modernbert-base", "--json-out", str(output)],
    )
    assert sweep_semantic_thresholds.main() == 0
    result = read_json(output)
    duplicate = result["models"][0]["duplicate_by_language"][0]
    assert duplicate["current_threshold"] == duplicate["current_metrics"]["threshold"] == 0.87
    assert duplicate["current_metrics"]["tp"] == 1
    assert duplicate["current_difficulty_recall"]["easy"]["detected"] == 1
    duplicate_index = result["grids"]["duplicate"].index(duplicate["selected_threshold"])
    assert [row["threshold"] for row in duplicate["selection_window"]] == result["grids"][
        "duplicate"
    ][max(0, duplicate_index - 5) : duplicate_index + 6]
    assert duplicate["selected_metrics"] in duplicate["selection_window"]
    search = result["models"][0]["search"]
    assert search["current_threshold"] == search["current_metrics"]["threshold"] == 0.68
    assert (search["current_metrics"]["tp"], search["current_metrics"]["fp"]) == (1, 0)
    search_index = result["grids"]["search"].index(search["selected_threshold"])
    assert [row["threshold"] for row in search["selection_window"]] == result["grids"]["search"][
        max(0, search_index - 5) : search_index + 6
    ]
    assert search["selected_metrics"] in search["selection_window"]
    assert result["grids"] == threshold_candidate_grids()
    measurements = {("sample", "gte-modernbert-base"): measurement}
    validate_threshold_selection(result, [project], ["gte-modernbert-base"], measurements)
    custom_grid = deepcopy(result)
    custom_grid["grids"]["duplicate"][1] = 0.005
    with pytest.raises(ValueError, match="mismatched candidate grids"):
        validate_threshold_selection(
            custom_grid,
            [project],
            ["gte-modernbert-base"],
            measurements,
        )
    result["models"][0]["duplicate_by_language"][0]["selected_threshold"] = 0.0
    with pytest.raises(ValueError, match="does not match its raw measurements"):
        validate_threshold_selection(result, [project], ["gte-modernbert-base"], measurements)


@pytest.mark.parametrize(
    "entrypoint",
    [
        sweep_semantic_thresholds.main,
        sweep_hybrid_gates.main,
        report_calibration_distributions.main,
    ],
)
def test_calibration_clis_reject_duplicate_canonical_model_aliases(monkeypatch, capsys, entrypoint):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibration-command",
            "--models",
            "gte-modernbert-base",
            "Alibaba-NLP/gte-modernbert-base",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        entrypoint()

    assert "duplicate canonical profiles" in capsys.readouterr().err


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
