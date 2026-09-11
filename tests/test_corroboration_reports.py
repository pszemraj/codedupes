"""Tie the shipped hybrid tier split to the recorded corroboration sweep.

``codedupes check`` withholds ``semantic_review`` pairs by default, so the
corroboration constants and promotion gates on each built-in profile decide
what the default report contains. They come from
``test_fixtures/polyglot_calibration/reports/corroboration_report.json``; these
tests re-derive the documented selection policy from that report so neither the
profile nor the report can drift on its own.

The policy (``docs/hybrid-tuning.md``): at each language's shipped admission
gate, a split is feasible when the visible subset keeps at least
``recall_retention_min`` of the all-published recall and does not lower
precision; the corroboration constants are the pooled selection that is
feasible in every language, the promotion gate is selected per language at
those constants, and ties prefer F1 then the stricter split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from codedupes.semantic_profiles import (
    SemanticModelProfile,
    get_semantic_threshold_for_language,
    list_supported_models,
)

REPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "test_fixtures"
    / "polyglot_calibration"
    / "reports"
    / "corroboration_report.json"
)
LANGUAGES = ("c", "rust", "javascript", "typescript", "python")
MODEL_KEYS = tuple(profile.key for profile in list_supported_models())
FLOAT_TOLERANCE = 1e-9


def _report() -> dict[str, Any]:
    return json.loads(REPORT_PATH.read_text())


def _model_entry(model_key: str) -> dict[str, Any]:
    for entry in _report()["models"]:
        if entry["model_key"] == model_key:
            return entry
    pytest.fail(f"corroboration report has no sweep for {model_key!r}")


def _profile(model_key: str) -> SemanticModelProfile:
    return next(item for item in list_supported_models() if item.key == model_key)


def _split(row: dict[str, Any]) -> tuple[float, float]:
    config = row["config"]
    return (config["weak_identifier_jaccard_min"], config["statement_ratio_min"])


def _feasible(row: dict[str, Any], retention: float) -> bool:
    return (
        row["recall"] >= retention * row["published_recall"] - FLOAT_TOLERANCE
        and row["precision"] >= row["published_precision"] - FLOAT_TOLERANCE
    )


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_shipped_constants_are_the_pooled_selection(model_key: str) -> None:
    """The profile's corroboration constants must be the report's pooled, everywhere-feasible pick."""
    entry = _model_entry(model_key)
    profile = _profile(model_key)
    policy = _report()["selection_policy"]
    retention = policy["recall_retention_min"]
    shipped = (profile.hybrid_weak_identifier_jaccard_min, profile.hybrid_statement_ratio_min)

    selected = entry["stage1"]["pooled"]["selected"]
    assert selected is not None, f"{model_key}: no split is feasible in every language"
    assert _split(selected) == shipped

    # (a) The pooled pick is a real grid point that every language accepts.
    for language in LANGUAGES:
        rows = entry["stage1"]["corpora"][language]["rows"]
        row = next(item for item in rows if _split(item) == shipped)
        assert _feasible(row, retention), (
            f"{model_key}/{language}: shipped split keeps "
            f"{row['recall'] / row['published_recall']:.2f} of published recall at precision "
            f"{row['precision']:.3f} vs {row['published_precision']:.3f}"
        )

    # (b) No other everywhere-feasible split has strictly better pooled precision.
    pooled = entry["stage1"]["pooled"]["rows"]
    everywhere_feasible = {
        _split(item)
        for item in pooled
        if all(
            _feasible(
                next(
                    r
                    for r in entry["stage1"]["corpora"][language]["rows"]
                    if _split(r) == _split(item)
                ),
                retention,
            )
            for language in LANGUAGES
        )
    }
    best = max(item["precision"] for item in pooled if _split(item) in everywhere_feasible)
    assert selected["precision"] >= best - FLOAT_TOLERANCE


@pytest.mark.parametrize("model_key", MODEL_KEYS)
@pytest.mark.parametrize("language", LANGUAGES)
def test_shipped_promotion_gate_is_the_per_language_selection(
    language: str, model_key: str
) -> None:
    """Each promotion gate must be the row the report selected at the shipped constants."""
    entry = _model_entry(model_key)
    profile = _profile(model_key)
    corpus = entry["stage2"]["corpora"][language]
    stage_constants = entry["stage2"]["constants"]
    assert (
        stage_constants["weak_identifier_jaccard_min"],
        stage_constants["statement_ratio_min"],
    ) == (profile.hybrid_weak_identifier_jaccard_min, profile.hybrid_statement_ratio_min)

    shipped_gate = profile.high_confidence_threshold_for_language(language)
    selected = corpus["selected"]
    assert selected is not None
    assert selected["config"]["high_gate"] == shipped_gate
    # The gate was swept at the language's shipped admission gate.
    assert corpus["semantic_gate"] == get_semantic_threshold_for_language(model_key, language)

    # The selected row is feasible and no feasible row beats its precision.
    retention = _report()["selection_policy"]["recall_retention_min"]
    assert _feasible(selected, retention)
    best = max(item["precision"] for item in corpus["rows"] if _feasible(item, retention))
    assert selected["precision"] >= best - FLOAT_TOLERANCE


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_report_was_swept_on_the_checkpoint_the_profile_ships(model_key: str) -> None:
    """A split is only transferable from the exact checkpoint and candidate policy it was swept on."""
    entry = _model_entry(model_key)
    profile = _profile(model_key)

    assert entry["canonical_name"] == profile.canonical_name
    assert entry["resolved_revision"] == profile.default_revision
    for language in LANGUAGES:
        calibration = entry["corpora"][language]["calibration"]
        assert calibration["mode"] == "hybrid_gates"
        assert calibration["resolved_revision"] == profile.default_revision
        assert calibration["candidate_policy"]["min_recursive_statements"] == 3
        assert calibration["semantic_gate"]["source"] == "profile"


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_report_baseline_matches_the_profile(model_key: str) -> None:
    """Regenerating the report after a constant edit must record the new profile as its baseline."""
    entry = _model_entry(model_key)
    profile = _profile(model_key)
    baseline = entry["baseline_defaults"]

    assert baseline["weak_min"] == profile.hybrid_weak_identifier_jaccard_min
    assert baseline["ratio_min"] == profile.hybrid_statement_ratio_min
    assert baseline["high_gates"] == dict(profile.language_high_confidence_thresholds)
