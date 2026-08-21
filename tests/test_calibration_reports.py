"""Tie the shipped per-language semantic gates to the recorded calibration reports.

The gates in ``codedupes.semantic_profiles`` are the product of the sweeps in
``test_fixtures/polyglot_calibration/reports/``, but nothing executable connected
the two: a gate could be edited, or a report regenerated, without either noticing.
These tests re-derive the documented selection policy from the recorded grids.

The policy (``semantic_profiles`` header comment and the corpus README) is
recall-first: each gate is the loosest sweep threshold whose F1 stays near that
language's best while final combined-output precision remains workable. What that
makes checkable is the shape of the choice, not a specific number - the gate has
to be a real row of the grid it claims to come from, at or below the best-F1 row,
buying recall, and not giving up much F1 to do it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from codedupes.semantic_profiles import get_semantic_threshold_for_language, list_supported_models

REPORTS_PATH = Path(__file__).resolve().parents[1] / "test_fixtures" / "polyglot_calibration"
LANGUAGES = ("c", "rust", "javascript", "typescript", "python")
MODEL_KEYS = tuple(profile.key for profile in list_supported_models())

# Grid thresholds are rounded to two decimals when swept, so compare on that grid.
GRID_DECIMALS = 2
FLOAT_TOLERANCE = 1e-9

# "F1 stays near that language's best" as a bound. The shipped gates run from
# zero up to five grid steps looser than the best-F1 row (0.02 steps): the
# distance a gate may travel is governed by how flat the F1 curve is below the
# selection, not by a fixed step count, so this is the term worth pinning.
# Recorded spread at the shipped gates is 0.83-1.00 of the selected row's F1.
MIN_F1_RETENTION = 0.80


def _load_report(language: str) -> dict[str, Any]:
    """Load one language's recorded duplicate-threshold sweep report.

    :param str language: Corpus language key.
    :return dict[str, Any]: Parsed report payload.
    """
    path = REPORTS_PATH / "reports" / f"{language}_semantic_threshold_report.json"
    return json.loads(path.read_text())


def _model_entry(report: dict[str, Any], model_key: str) -> dict[str, Any]:
    """Find one model's block inside a sweep report.

    :param dict[str, Any] report: Parsed report payload.
    :param str model_key: Built-in model profile key.
    :raises AssertionError: If the report never swept that model.
    :return dict[str, Any]: The model's sweep block.
    """
    for entry in report["models"]:
        if entry["model_key"] == model_key:
            return entry
    swept = [entry["model_key"] for entry in report["models"]]
    pytest.fail(f"report has no sweep for {model_key!r}; it covers {swept}")


@pytest.mark.parametrize("model_key", MODEL_KEYS)
@pytest.mark.parametrize("language", LANGUAGES)
def test_shipped_gate_is_a_recall_first_pick_from_its_recorded_grid(
    language: str, model_key: str
) -> None:
    """Each shipped gate must be a real, recall-buying row of the sweep it came from."""
    report = _load_report(language)
    entry = _model_entry(report, model_key)
    rows = {round(row["threshold"], GRID_DECIMALS): row for row in entry["rows"]}
    gate = round(get_semantic_threshold_for_language(model_key, language), GRID_DECIMALS)
    selected = round(entry["selected_threshold"], GRID_DECIMALS)

    # (a) The gate is a measured grid point, not an interpolation or a guess.
    assert gate in rows, f"gate {gate} for {language}/{model_key} is not in the swept grid"
    assert gate in {round(value, GRID_DECIMALS) for value in report["grid"]}

    gate_row = rows[gate]
    selected_row = rows[selected]

    # (b) Recall-first means at or below the sweep's own best-F1 selection.
    assert gate <= selected + FLOAT_TOLERANCE, (
        f"gate {gate} for {language}/{model_key} is stricter than the selected "
        f"threshold {selected}; the shipped policy never tightens past the sweep"
    )

    # (c) Going looser has to actually buy recall, or it only costs precision.
    assert gate_row["recall"] >= selected_row["recall"] - FLOAT_TOLERANCE, (
        f"gate {gate} for {language}/{model_key} recalls "
        f"{gate_row['recall']:.3f} vs {selected_row['recall']:.3f} at the selected "
        f"threshold {selected}: a looser gate that loses recall is not recall-first"
    )

    # (d) ... while F1 stays near this language's best.
    assert gate_row["f1"] >= MIN_F1_RETENTION * selected_row["f1"] - FLOAT_TOLERANCE, (
        f"gate {gate} for {language}/{model_key} keeps only "
        f"{gate_row['f1'] / selected_row['f1']:.2f} of the best-F1 row's F1 "
        f"(floor {MIN_F1_RETENTION:.2f}); re-review the gate against the report"
    )


@pytest.mark.parametrize("model_key", MODEL_KEYS)
@pytest.mark.parametrize("language", LANGUAGES)
def test_report_was_swept_on_the_checkpoint_the_profile_ships(
    language: str, model_key: str
) -> None:
    """A calibrated gate is only meaningful against the exact checkpoint it was swept on."""
    entry = _model_entry(_load_report(language), model_key)
    profile = next(item for item in list_supported_models() if item.key == model_key)
    calibration = entry["calibration"]

    assert entry["canonical_name"] == profile.canonical_name
    assert calibration["resolved_revision"] == profile.default_revision
    assert calibration["mode"] == "duplicate"
    # The gates are only transferable if the sweep used the production candidate policy.
    assert calibration["candidate_policy"]["min_recursive_statements"] == 3
