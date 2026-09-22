"""Measurement capture, fingerprinting, and provenance rejection tests."""

from __future__ import annotations

import sys
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
    calibration_measurements,
)
from scripts.calibration_contract import (
    ProjectAnalyzer,
    analyzer_config,
    eligibility_reason,
    extract_project,
    load_projects,
    resolve_annotations,
    unit_ids,
    write_json,
)
from scripts.calibration_evaluation import (
    compare_devices,
    measurement_digest,
    measurement_digests,
    validate_measurement_digests,
    validate_measurement_provenance,
)
from scripts.calibration_measurements import (
    ARTIFACT_VERSION,
    capture,
    load_measurement,
    measurement_fingerprint,
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
            "mps_operator_fallback": False,
            "runtime_versions": semantic.get_semantic_runtime_versions(),
            "captured_profile": {},
            "timing_seconds": {"duplicate": 0.0, "search": 0.0},
            "execution": {
                "duplicate": {"execution_device": "cpu", "cache_hit_rows": 0},
                "search": {"execution_device": "cpu", "cache_hit_rows": 0},
            },
            "query_execution": [
                {"probe": probe, "execution_device": "cpu", "cache_hit": False} for probe in probes
            ],
            "live_default": [],
            "input_fingerprint": measurement_fingerprint(project, model, "cpu"),
            "measurement_pipeline_version": calibration_measurements.MEASUREMENT_PIPELINE_VERSION,
            "codedupes_version": calibration_measurements.__version__,
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


def test_measurements_bind_capture_inputs_but_load_on_other_runtimes(tmp_path: Path, monkeypatch):
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

    def no_live_runtime(*_args, **_kwargs):
        pytest.fail("reading recorded scores must not inspect the live inference runtime")

    monkeypatch.setattr(semantic, "get_semantic_runtime_versions", no_live_runtime)
    monkeypatch.setattr(semantic, "_resolve_model_dtype", no_live_runtime)
    monkeypatch.setattr(semantic, "_mps_fast_math_variant", no_live_runtime)
    assert load_measurement(path, project)["metadata"]["input_fingerprint"] == fingerprint


@pytest.mark.parametrize("changed_package", ["numpy", "tokenizers"])
def test_measurement_fingerprint_includes_numeric_runtime(monkeypatch, changed_package):
    """Numeric runtime releases can change vectors and raw score evidence."""
    project = load_projects(project_ids=["ledger"])[0]
    versions = {
        package: semantic._safe_package_version(package) or "missing"
        for package in ("numpy", "torch", "transformers", "tokenizers", "sentence-transformers")
    }
    monkeypatch.setattr(semantic, "_safe_package_version", versions.get)

    assert set(semantic.get_semantic_runtime_versions()) == (
        calibration_measurements.RUNTIME_VERSION_KEYS
    )
    before = measurement_fingerprint(project, "gte-modernbert-base", "cpu")
    versions[changed_package] = f"{versions[changed_package]}+different"
    after = measurement_fingerprint(project, "gte-modernbert-base", "cpu")

    assert after != before


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


@pytest.mark.parametrize("mutation", ["batch", "dtype", "fallback", "pipeline_version"])
def test_measurements_reject_behavior_identity_mismatches(
    tmp_path: Path, monkeypatch, mutation: str
):
    project = load_projects(project_ids=["ledger"])[0]
    measurement = _empty_measurement(project)
    path = tmp_path / "measurement.json"

    if mutation == "batch":
        assert (
            measurement_fingerprint(project, "gte-modernbert-base", "cpu", batch_size=5)
            != measurement["metadata"]["input_fingerprint"]
        )
        measurement["metadata"]["batch_size"] = 5
    elif mutation == "dtype":
        measurement["metadata"]["inference_dtype"] = "float16"
    elif mutation == "fallback":
        measurement["metadata"]["mps_operator_fallback"] = True
    else:
        monkeypatch.setattr(
            calibration_measurements,
            "MEASUREMENT_PIPELINE_VERSION",
            calibration_measurements.MEASUREMENT_PIPELINE_VERSION + 1,
        )

    write_json(path, measurement)
    with pytest.raises(ValueError, match="stale measurement"):
        load_measurement(path, project)


def test_capture_disables_mps_operator_fallback_before_dtype_resolution(
    tmp_path: Path, monkeypatch
):
    project = load_projects(project_ids=["ledger"])[0]
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    order = []

    def configure(device, *, mps_fallback):
        order.append(("configure", device, mps_fallback))

    def resolve_dtype(_family, _device):
        assert order == [("configure", "mps", False)]
        raise RuntimeError("dtype probe reached")

    monkeypatch.setattr(semantic, "_configure_semantic_runtime_env", configure)
    monkeypatch.setattr(semantic, "_resolve_model_dtype", resolve_dtype)

    with pytest.raises(RuntimeError, match="dtype probe reached"):
        capture(project, "gte-modernbert-base", "mps", tmp_path)


def test_capture_rejects_an_already_imported_mps_runtime(tmp_path: Path, monkeypatch):
    project = load_projects(project_ids=["ledger"])[0]
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    monkeypatch.setitem(sys.modules, "torch", object())

    with pytest.raises(ValueError, match="fresh process"):
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


def test_derived_selections_ignore_diagnostic_package_version(tmp_path):
    project = load_projects(project_ids=["ledger"])[0]
    measurement = _empty_measurement(project)
    original = measurement_digest(measurement)

    measurement["metadata"]["codedupes_version"] = "99.0.0"

    assert measurement_digest(measurement) == original
    path = tmp_path / "measurement.json"
    write_json(path, measurement)
    assert load_measurement(path, project) == measurement


@pytest.mark.parametrize(
    "field", ["execution", "query_execution", "live_default", "timing_seconds"]
)
def test_derived_selections_bind_claimed_execution_provenance(field: str):
    project = load_projects(project_ids=["ledger"])[0]
    measurement = _empty_measurement(project)
    original = measurement_digest(measurement)
    if field == "execution":
        measurement["metadata"][field]["duplicate"]["cache_hit_rows"] = 17
    elif field == "query_execution":
        measurement["metadata"][field][0]["execution_device"] = "mps"
    elif field == "live_default":
        measurement["metadata"][field].append({"a": "forged", "b": "pair", "tier": "exact"})
    else:
        measurement["metadata"][field]["duplicate"] = 99.0

    assert measurement_digest(measurement) != original


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("fallback", "did not execute on cpu"),
        ("cache", "did not execute on cpu"),
        ("missing", "incomplete query execution provenance"),
    ],
)
def test_measurements_reject_invalid_query_execution_provenance(
    tmp_path: Path, mutation: str, message: str
):
    project = load_projects(project_ids=["ledger"])[0]
    measurement = _empty_measurement(project)
    if mutation == "fallback":
        measurement["metadata"]["query_execution"][0]["execution_device"] = "mps"
    elif mutation == "cache":
        measurement["metadata"]["query_execution"][0]["cache_hit"] = True
    else:
        measurement["metadata"]["query_execution"].pop()
    path = tmp_path / "measurement.json"
    write_json(path, measurement)
    with pytest.raises(ValueError, match=message):
        if mutation == "missing":
            load_measurement(path, project)
        else:
            load_measurement(path, project, expected_device="cpu")


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

    measurement["metadata"]["query_execution"][0]["execution_device"] = "mps"
    with pytest.raises(ValueError, match="invalid query execution provenance"):
        validate_measurement_provenance(project, measurement)
    measurement["metadata"]["query_execution"][0]["execution_device"] = "cpu"

    # Captured gates audit the analyzer run but do not affect the raw vectors or
    # scores, so a threshold-only policy edit must not force model re-inference.
    measurement["metadata"]["captured_profile"]["semantic_threshold"] = 0.86
    validate_measurement_provenance(project, measurement)
    measurement["metadata"]["captured_profile"]["semantic_threshold"] = 0.87

    measurement["metadata"]["captured_profile"]["high_gate"] = 0.5
    with pytest.raises(ValueError, match="invalid captured threshold profile"):
        validate_measurement_provenance(project, measurement)
    measurement["metadata"]["captured_profile"]["high_gate"] = 0.88

    recorded_runtime = measurement["metadata"]["runtime_versions"].copy()
    measurement["metadata"]["runtime_versions"]["torch"] = "forged"
    with pytest.raises(ValueError, match="stale measurement"):
        validate_measurement_provenance(project, measurement)

    measurement["metadata"]["runtime_versions"] = recorded_runtime
    measurement["metadata"]["requested_device"] = "mps"
    measurement["metadata"]["input_fingerprint"] = measurement_fingerprint(
        project, "gte-modernbert-base", "mps"
    )
    for stats in measurement["metadata"]["execution"].values():
        stats["execution_device"] = "mps"
    for row in measurement["metadata"]["query_execution"]:
        row["execution_device"] = "mps"
    # Saved faithful-MPS evidence is readable even on a host configured for
    # fast math; only capture executes Metal and must reject that setting.
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")
    monkeypatch.setattr(
        semantic,
        "get_semantic_runtime_versions",
        lambda: pytest.fail("provenance validation must not inspect the reader's runtime"),
    )
    validate_measurement_provenance(project, measurement)
    measurement["metadata"]["math_policy"] = "mps_fast_math"
    with pytest.raises(ValueError, match="stale measurement"):
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


def test_measurement_identity_ignores_threshold_only_profile_edits(monkeypatch):
    project = load_projects(project_ids=["ledger"])[0]
    original = measurement_fingerprint(project, "gte-modernbert-base", "cpu")
    profile = resolve_model_profile("gte-modernbert-base")
    edited_profile = replace(
        profile,
        language_semantic_thresholds=profile.language_semantic_thresholds | {"python": 0.86},
    )
    monkeypatch.setattr(
        "scripts.calibration_measurements.resolve_model_profile", lambda _model: edited_profile
    )

    assert measurement_fingerprint(project, "gte-modernbert-base", "cpu") == original


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
