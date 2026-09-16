"""Acceptance checks for the checked-in CPU and real-MPS pilot measurements."""

from __future__ import annotations

import pytest

from codedupes.semantic_profiles import list_supported_models, resolve_model_profile
from scripts.calibration_contract import evidence_identity, load_projects
from scripts.calibration_evaluation import (
    compare_devices,
    full_report,
    load_all,
    replay_parity,
    review_queue,
)
from scripts.calibration_measurements import DEFAULT_MEASUREMENTS

pytestmark = pytest.mark.grammar


@pytest.mark.parametrize("project", load_projects(), ids=lambda project: project.id)
def test_recorded_measurements_cover_both_models_and_real_devices(project):
    models = [profile.key for profile in list_supported_models()]
    measurements = load_all(project, DEFAULT_MEASUREMENTS, models, ["cpu", "mps"])
    assert set(measurements) == {(model, device) for model in models for device in ("cpu", "mps")}

    for (model, device), measurement in measurements.items():
        metadata = measurement["metadata"]
        assert metadata["evidence_sha256_at_capture"] == evidence_identity(project)
        assert replay_parity(measurement) == {"default": True, "explicit_override": True}
        assert all(
            execution["execution_device"] == device
            and execution["cache_enabled"] is False
            and execution["cache_hit_rows"] == 0
            for execution in metadata["execution"].values()
        )
        profile = resolve_model_profile(model)
        language = project.spec["languages"][0]
        duplicate_gate = profile.semantic_threshold_for_language(language)
        assert any(
            row["comparable"] and row["cosine"] is not None and row["cosine"] < duplicate_gate
            for row in measurement["pairs"]
        )
        assert any(
            row["cosine"] is not None and row["cosine"] < profile.default_search_threshold
            for row in measurement["query_scores"]
        )
        report = full_report(project, measurement)
        assert report["selection"] is None
        assert report["duplicate"]["complete_published"]["selection_eligible"] is True

    queue = review_queue(project, list(measurements.values()))
    assert queue["seed"] == 20
    assert queue["sample_size"] == 20
    assert queue["pending"] == []
    for model in models:
        comparison = compare_devices(
            measurements[(model, "cpu")], measurements[(model, "mps")], project
        )
        assert comparison["effective_devices"] == {"cpu": ["cpu"], "mps": ["mps"]}
        assert comparison["duplicate_decision_changes"] == []
        assert comparison["search_threshold_decision_changes"] == []
        assert comparison["search_top_k_decision_changes"] == []
