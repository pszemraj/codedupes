"""Measure calibration fixtures without applying production thresholds."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

import numpy as np

from codedupes import __version__, semantic
from codedupes.analyzer import _statement_count_ratio
from codedupes.constants import (
    DEFAULT_CHECK_SEMANTIC_TASK,
    DEFAULT_SEARCH_SEMANTIC_TASK,
    DEFAULT_TRADITIONAL_THRESHOLD,
)
from codedupes.pairs import ordered_pair_key
from codedupes.semantic_profiles import resolve_model_profile
from codedupes.traditional import find_exact_pair_keys, jaccard_similarity

try:
    from .calibration_contract import (
        REPO,
        Project,
        ProjectAnalyzer,
        analyzer_config,
        eligibility_reason,
        extract_project,
        read_json,
        relative_file,
        resolve_annotations,
        unit_ids,
        validate_project,
        write_json,
    )
except ImportError:
    from calibration_contract import (
        REPO,
        Project,
        ProjectAnalyzer,
        analyzer_config,
        eligibility_reason,
        extract_project,
        read_json,
        relative_file,
        resolve_annotations,
        unit_ids,
        validate_project,
        write_json,
    )

ARTIFACT_VERSION = 10
CALIBRATION_BATCH_SIZE = 4
CALIBRATION_MPS_OPERATOR_FALLBACK = False
RUNTIME_VERSION_KEYS = {
    "python",
    "numpy",
    "torch",
    "transformers",
    "tokenizers",
    "sentence-transformers",
}
DEFAULT_MEASUREMENTS = REPO / "scratch/calibration"
# Bump whenever capture, extraction, scoring, replay, or their provenance
# changes in a way that could change a measured artifact. Package versions are
# diagnostic provenance; documentation-only releases do not invalidate scores.
MEASUREMENT_PIPELINE_VERSION = 5


def _identity_digest(identity: dict[str, Any]) -> str:
    """Return a deterministic SHA-256 digest for one identity mapping."""
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def measurement_behavior_identity(project: Project, model: str) -> dict[str, Any]:
    """Describe project and model inputs shared by raw and checked measurement evidence.

    :param project: Calibration project whose source and queries are measured.
    :param model: Built-in semantic model key or alias.
    :return: Structured behavior identity independent of a measurement host.
    """
    profile = resolve_model_profile(model)
    source_units, _ = extract_project(project)
    source_paths = {unit.file_path.resolve() for unit in source_units} | {
        relative_file(project.root, unit["selector"]["path"]).resolve()
        for unit in project.annotations["units"]
    }
    source_files = {
        path.relative_to(project.root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_paths)
    }
    return {
        "project": {
            "pipeline_version": MEASUREMENT_PIPELINE_VERSION,
            "embedding_pipeline_schema": semantic.EMBEDDING_PIPELINE_SCHEMA,
            "search_document": "source",
            "traditional_threshold": DEFAULT_TRADITIONAL_THRESHOLD,
            "source": source_files,
            "units": {unit["id"]: unit["selector"] for unit in project.annotations["units"]},
            "queries": {probe["id"]: probe["query"] for probe in project.annotations["probes"]},
            "policy": project.policy,
        },
        "model": profile.canonical_name,
        "revision": profile.default_revision,
        "family": profile.family,
        "trust_remote_code": profile.default_trust_remote_code,
        "tasks": [DEFAULT_CHECK_SEMANTIC_TASK, DEFAULT_SEARCH_SEMANTIC_TASK],
    }


def measurement_identity(
    project: Project,
    model: str,
    device: str,
    *,
    batch_size: int = CALIBRATION_BATCH_SIZE,
) -> dict[str, Any]:
    """Describe every behavior-affecting input to one raw measurement.

    :param project: Calibration project whose source and queries are measured.
    :param model: Built-in semantic model key or alias.
    :param device: Calibration reference device, either ``cpu`` or ``mps``.
    :param batch_size: Inference batch size used for the uncached measurement.
    :return: Structured measurement identity suitable for stable hashing.
    :raises ValueError: If the device or batch size is invalid.
    """
    if device not in {"cpu", "mps"}:
        raise ValueError("calibration fingerprint device must be cpu or mps")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("calibration fingerprint batch size must be a positive integer")
    behavior = measurement_behavior_identity(project, model)
    math_policy = semantic._mps_fast_math_variant(device) or "standard"
    return behavior["project"] | {
        "device": device,
        "batch_size": batch_size,
        "inference_dtype": "float32",
        "math_policy": math_policy,
        "mps_operator_fallback": CALIBRATION_MPS_OPERATOR_FALLBACK,
        "runtime_versions": semantic.get_semantic_runtime_versions(),
        **{
            key: behavior[key]
            for key in ("model", "revision", "family", "trust_remote_code", "tasks")
        },
    }


def measurement_fingerprint(
    project: Project,
    model: str,
    device: str,
    *,
    batch_size: int = CALIBRATION_BATCH_SIZE,
) -> str:
    """Fingerprint behavior-affecting inputs to one raw measurement.

    :param project: Calibration project whose source and queries are measured.
    :param model: Built-in semantic model key or alias.
    :param device: Calibration reference device, either ``cpu`` or ``mps``.
    :param batch_size: Inference batch size used for the uncached measurement.
    :return: Stable SHA-256 identity for one raw measurement input.
    :raises ValueError: If the device or batch size is invalid.
    """
    return _identity_digest(measurement_identity(project, model, device, batch_size=batch_size))


def artifact_path(root: Path, project: Project, model: str, device: str) -> Path:
    """Return the raw measurement path for one project/model/device run."""
    key = resolve_model_profile(model).key
    return root / project.id / key / f"{device}.json"


def validate_measurement_identity(project: Project, metadata: dict[str, Any]) -> None:
    """Validate saved inputs using the capture's execution metadata, without live inference.

    Replaying saved scores does not execute the model. The reader's Python,
    accelerator availability, and inference libraries cannot change those scores.
    The fingerprint still binds the recorded runtime and execution policy to the
    current corpus and model inputs, so metadata edits cannot silently pass.

    :param project: Calibration project whose inputs must still match.
    :param metadata: Execution and input metadata recorded by capture.
    :raises ValueError: If the recorded identity is invalid or stale.
    """
    runtime = metadata.get("runtime_versions")
    if (
        not isinstance(runtime, dict)
        or set(runtime) != RUNTIME_VERSION_KEYS
        or any(not isinstance(value, str) or not value for value in runtime.values())
        or metadata.get("requested_device") not in {"cpu", "mps"}
        or type(metadata.get("batch_size")) is not int
        or metadata["batch_size"] <= 0
        or metadata.get("inference_dtype") != "float32"
        or metadata.get("math_policy") != "standard"
        or metadata.get("mps_operator_fallback") is not CALIBRATION_MPS_OPERATOR_FALLBACK
    ):
        raise ValueError(f"{project.id}: stale measurement or invalid execution metadata")
    behavior = measurement_behavior_identity(project, metadata["model"])
    identity = behavior["project"] | {
        "device": metadata["requested_device"],
        **{
            key: metadata[key]
            for key in (
                "batch_size",
                "inference_dtype",
                "math_policy",
                "mps_operator_fallback",
                "runtime_versions",
            )
        },
        **{
            key: behavior[key]
            for key in ("model", "revision", "family", "trust_remote_code", "tasks")
        },
    }
    if metadata.get("measurement_pipeline_version") != MEASUREMENT_PIPELINE_VERSION or metadata.get(
        "input_fingerprint"
    ) != _identity_digest(identity):
        raise ValueError(f"{project.id}: stale measurement")


def _pair_scores(embeddings: np.ndarray) -> dict[tuple[int, int], float]:
    """Calculate every upper-triangle cosine score from normalized embeddings."""
    scores: dict[tuple[int, int], float] = {}
    for start in range(0, len(embeddings), semantic._PAIRWISE_SCAN_BLOCK_SIZE):
        matrix = embeddings[start : start + semantic._PAIRWISE_SCAN_BLOCK_SIZE] @ embeddings.T
        for offset in range(len(matrix)):
            for column in range(start + offset + 1, len(embeddings)):
                value = float(matrix[offset, column])
                if not np.isfinite(value):
                    raise ValueError("nonfinite duplicate score")
                scores[(start + offset, column)] = max(-1.0, min(value, 1.0))
    return scores


def _execution(analyzer: ProjectAnalyzer, requested: str) -> dict[str, Any]:
    """Require a fresh run on the requested device."""
    stats = analyzer.embedding_stats
    if stats is None or stats.execution_device != requested or stats.cache_hit_rows:
        raise ValueError(f"measurement did not execute independently on {requested}: {stats}")
    return asdict(stats)


def _query_execution(
    analyzer: ProjectAnalyzer,
    requested: str,
    probe: str,
    expected_count: int,
) -> dict[str, Any]:
    """Require one newly recorded, uncached query encode on the requested device."""
    records = analyzer.query_execution
    if len(records) != expected_count:
        raise ValueError(
            f"measurement did not record exactly one execution for probe {probe!r}: {records}"
        )
    record = records[-1]
    if record.execution_device != requested or record.cache_hit:
        raise ValueError(f"probe {probe!r} did not execute independently on {requested}: {record}")
    return {
        "probe": probe,
        "execution_device": record.execution_device,
        "cache_hit": record.cache_hit,
    }


def capture(
    project: Project,
    model: str,
    device: str,
    output: Path,
    batch_size: int = CALIBRATION_BATCH_SIZE,
) -> Path:
    """Capture duplicate-pair and search scores in two uncached model passes."""
    if device not in {"cpu", "mps"}:
        raise ValueError("device must be cpu or mps")
    if os.getenv("CODEDUPES_CPU_BF16") == "1":
        raise ValueError("disable experimental CPU bf16 while calibrating")
    math_policy = semantic._mps_fast_math_variant(device) or "standard"
    if math_policy != "standard":
        raise ValueError("disable PYTORCH_MPS_FAST_MATH while calibrating")
    if device == "mps" and "torch" in sys.modules:
        raise ValueError(
            "calibration requires a fresh process so MPS operator fallback can be disabled "
            "before importing PyTorch"
        )
    semantic._configure_semantic_runtime_env(device, mps_fallback=CALIBRATION_MPS_OPERATOR_FALLBACK)

    # A fresh deterministic finding changes the reviewed corpus contract even
    # when it is excluded from semantic scoring. Refuse to spend model time or
    # emit raw evidence until every such pair has an explicit judgment.
    validate_project(project, require_adjudicated=True)

    profile = resolve_model_profile(model)
    if profile.default_revision is None or len(profile.default_revision) != 40:
        raise ValueError("calibration requires a pinned built-in model revision")
    inference_dtype = str(semantic._resolve_model_dtype(profile.family, device)).removeprefix(
        "torch."
    )
    if inference_dtype != "float32":
        raise ValueError("calibration requires float32 inference")

    inventory, _ = extract_project(project, inventory=True)
    resolved = resolve_annotations(project, inventory)
    source_units, _ = extract_project(project)
    measured_units = list({unit.uid: unit for unit in [*source_units, *resolved.values()]}.values())
    ids = unit_ids(project, measured_units, resolved)

    config = analyzer_config(
        project,
        model_name=profile.canonical_name,
        model_revision=profile.default_revision,
        device=device,
        batch_size=batch_size,
        embedding_cache=False,
        progress="never",
        mps_fallback=CALIBRATION_MPS_OPERATOR_FALLBACK,
    )
    analyzer = ProjectAnalyzer(project, config)
    started = time.perf_counter()
    result = analyzer.analyze(project.root)
    duplicate_seconds = time.perf_counter() - started
    duplicate_execution = _execution(analyzer, device)
    candidates = analyzer._semantic_units or []
    embeddings = analyzer._embeddings
    if embeddings is None or not candidates:
        raise ValueError(f"{project.id}: no semantic candidates")

    rows_by_uid = {unit.uid: row for row, unit in enumerate(candidates)}
    candidate_uids = set(rows_by_uid)
    scores = _pair_scores(embeddings)
    exact = find_exact_pair_keys(candidates)
    traditional: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for duplicate in result.traditional_duplicates:
        traditional.setdefault(ordered_pair_key(duplicate.unit_a, duplicate.unit_b), []).append(
            {"method": duplicate.method, "similarity": duplicate.similarity}
        )

    units = []
    for unit in sorted(measured_units, key=lambda item: ids[item.uid]):
        units.append(
            {
                "id": ids[unit.uid],
                "language": unit.language,
                "kind": unit.unit_type.name.lower(),
                "statement_count": semantic.get_code_unit_statement_count(unit),
                "embedded": unit.uid in candidate_uids,
            }
        )

    pairs = []
    ordered_units = sorted(measured_units, key=lambda item: ids[item.uid])
    for a, b in combinations(ordered_units, 2):
        embedded = a.uid in candidate_uids and b.uid in candidate_uids
        cosine = None
        if embedded:
            row_key = tuple(sorted((rows_by_uid[a.uid], rows_by_uid[b.uid])))
            cosine = scores[row_key]
        reason = eligibility_reason(
            a,
            b,
            candidate_uids,
            exact,
            suppress_tests=config.suppress_test_semantic_matches,
        )
        pairs.append(
            {
                "a": ids[a.uid],
                "b": ids[b.uid],
                "cosine": cosine,
                "comparable": reason is None,
                "exclusion_reason": reason,
                "identifier_jaccard": jaccard_similarity(a.identifiers, b.identifiers),
                "statement_ratio": _statement_count_ratio(a, b),
                "traditional": traditional.get(ordered_pair_key(a, b), []),
            }
        )

    search = ProjectAnalyzer(
        project,
        analyzer_config(
            project,
            model_name=profile.canonical_name,
            model_revision=profile.default_revision,
            device=device,
            batch_size=batch_size,
            embedding_cache=False,
            progress="never",
            mps_fallback=CALIBRATION_MPS_OPERATOR_FALLBACK,
            mode="search",
            run_traditional=False,
        ),
    )
    started = time.perf_counter()
    count = search.index(project.root)
    query_scores = []
    query_execution = []
    for probe in project.annotations["probes"]:
        hits = search.search(probe["query"], top_k=count, threshold=-1.0)
        query_execution.append(
            _query_execution(search, device, probe["id"], len(query_execution) + 1)
        )
        by_uid = {
            unit.uid: (max(-1.0, min(float(score), 1.0)), rank)
            for rank, (unit, score) in enumerate(hits, 1)
        }
        for unit in ordered_units:
            score, rank = by_uid.get(unit.uid, (None, None))
            query_scores.append(
                {"probe": probe["id"], "unit": ids[unit.uid], "cosine": score, "rank": rank}
            )
    search_seconds = time.perf_counter() - started
    search_execution = _execution(search, device)

    payload = {
        "schema_version": ARTIFACT_VERSION,
        "metadata": {
            "project": project.id,
            "model": profile.key,
            "canonical_model": profile.canonical_name,
            "revision": profile.default_revision,
            "requested_device": device,
            "batch_size": batch_size,
            "inference_dtype": inference_dtype,
            "math_policy": math_policy,
            "mps_operator_fallback": CALIBRATION_MPS_OPERATOR_FALLBACK,
            "runtime_versions": semantic.get_semantic_runtime_versions(),
            "measurement_pipeline_version": MEASUREMENT_PIPELINE_VERSION,
            "input_fingerprint": measurement_fingerprint(
                project, profile.key, device, batch_size=batch_size
            ),
            "codedupes_version": __version__,
            "captured_profile": {
                "semantic_threshold": profile.semantic_threshold_for_language(
                    project.spec["languages"][0]
                ),
                "weak_identifier_jaccard_min": profile.hybrid_weak_identifier_jaccard_min,
                "statement_ratio_min": profile.hybrid_statement_ratio_min,
                "high_gate": profile.high_confidence_threshold_for_language(
                    project.spec["languages"][0]
                ),
            },
            "timing_seconds": {"duplicate": duplicate_seconds, "search": search_seconds},
            "execution": {"duplicate": duplicate_execution, "search": search_execution},
            "query_execution": query_execution,
            "live_default": [
                {"a": ids[item.unit_a.uid], "b": ids[item.unit_b.uid], "tier": item.tier}
                for item in result.hybrid_duplicates
            ],
        },
        "units": units,
        "pairs": pairs,
        "query_scores": query_scores,
    }
    path = artifact_path(output, project, model, device)
    write_json(path, payload)
    semantic.clear_model_cache()
    return path


def _validate_measurement_payload(measurement: dict[str, Any], project: Project) -> None:
    """Reject score payloads that differ from the current non-model corpus evidence."""
    inventory, _ = extract_project(project, inventory=True)
    resolved = resolve_annotations(project, inventory)
    source_units, _ = extract_project(project)
    measured_units = list({unit.uid: unit for unit in [*source_units, *resolved.values()]}.values())
    ids = unit_ids(project, measured_units, resolved)
    expected_ids = {ids[unit.uid] for unit in measured_units}
    units_by_id = {ids[unit.uid]: unit for unit in measured_units}

    # These fields route scores into the admission and hybrid sweeps, so they
    # cannot be trusted merely because a raw artifact has the right matrix shape.
    # Recompute them without loading an embedding model.
    selection_config = analyzer_config(project)
    selector = ProjectAnalyzer(project, selection_config)
    candidates = selector._select_semantic_candidates(source_units)
    candidate_uids = {unit.uid for unit in candidates}
    exact = find_exact_pair_keys(candidates)
    traditional_result = ProjectAnalyzer(project, analyzer_config(project, semantic=False)).analyze(
        project.root
    )
    traditional: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for duplicate in traditional_result.traditional_duplicates:
        key = tuple(sorted((ids[duplicate.unit_a.uid], ids[duplicate.unit_b.uid])))
        traditional.setdefault(key, []).append((duplicate.method, duplicate.similarity))
    for evidence in traditional.values():
        evidence.sort()

    def traditional_signature(value: Any, key: tuple[str, str]) -> list[tuple[str, float]]:
        if not isinstance(value, list):
            raise ValueError(  # noqa: TRY004 -- malformed artifact is one validation failure type
                f"{project.id}: invalid traditional evidence for pair {key}"
            )
        signature = []
        for item in value:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("method"), str)
                or not isinstance(item.get("similarity"), int | float)
                or isinstance(item["similarity"], bool)
                or not math.isfinite(item["similarity"])
            ):
                raise ValueError(f"{project.id}: invalid traditional evidence for pair {key}")
            signature.append((item["method"], item["similarity"]))
        return sorted(signature)

    units = measurement.get("units")
    if not isinstance(units, list) or any(not isinstance(unit, dict) for unit in units):
        raise ValueError("measurement units must be a list of objects")
    actual_ids = [unit.get("id") for unit in units]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise ValueError(f"{project.id}: incomplete or duplicate measurement units")
    embedded = {}
    for unit in units:
        if type(unit.get("embedded")) is not bool:
            raise ValueError(f"{project.id}: measurement unit has invalid embedded state")
        canonical = units_by_id[unit["id"]]
        if (
            unit.get("language") != canonical.language
            or unit.get("kind") != canonical.unit_type.name.lower()
            or unit.get("statement_count") != semantic.get_code_unit_statement_count(canonical)
            or unit["embedded"] != (canonical.uid in candidate_uids)
        ):
            raise ValueError(f"{project.id}: measurement unit differs from corpus evidence")
        embedded[unit["id"]] = unit["embedded"]

    pairs = measurement.get("pairs")
    if not isinstance(pairs, list) or any(not isinstance(row, dict) for row in pairs):
        raise ValueError("measurement pairs must be a list of objects")
    pair_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in pairs:
        a, b = row.get("a"), row.get("b")
        if a not in expected_ids or b not in expected_ids or a == b:
            raise ValueError(f"{project.id}: invalid measurement pair endpoints")
        key = tuple(sorted((a, b)))
        if key in pair_rows:
            raise ValueError(f"{project.id}: duplicate measurement pair {key}")
        pair_rows[key] = row
        score = row.get("cosine")
        should_be_scored = embedded[a] and embedded[b]
        if should_be_scored != (
            isinstance(score, int | float)
            and not isinstance(score, bool)
            and math.isfinite(score)
            and -1.0 <= score <= 1.0
        ):
            raise ValueError(f"{project.id}: inconsistent score state for pair {key}")
        unit_a, unit_b = units_by_id[a], units_by_id[b]
        reason = eligibility_reason(
            unit_a,
            unit_b,
            candidate_uids,
            exact,
            suppress_tests=selection_config.suppress_test_semantic_matches,
        )
        if row.get("comparable") is not (reason is None) or row.get("exclusion_reason") != reason:
            raise ValueError(
                f"{project.id}: pair eligibility differs from corpus evidence for {key}"
            )
        expected_identifier_jaccard = jaccard_similarity(unit_a.identifiers, unit_b.identifiers)
        expected_statement_ratio = _statement_count_ratio(unit_a, unit_b)
        if (
            not isinstance(row.get("identifier_jaccard"), int | float)
            or isinstance(row["identifier_jaccard"], bool)
            or not math.isfinite(row["identifier_jaccard"])
            or row["identifier_jaccard"] != expected_identifier_jaccard
            or not isinstance(row.get("statement_ratio"), int | float)
            or isinstance(row["statement_ratio"], bool)
            or not math.isfinite(row["statement_ratio"])
            or row["statement_ratio"] != expected_statement_ratio
            or traditional_signature(row.get("traditional"), key) != traditional.get(key, [])
        ):
            raise ValueError(f"{project.id}: pair evidence differs from corpus evidence for {key}")
    expected_pairs = {tuple(sorted(pair)) for pair in combinations(expected_ids, 2)}
    if pair_rows.keys() != expected_pairs:
        raise ValueError(f"{project.id}: incomplete measurement pair matrix")

    probes = {probe["id"] for probe in project.annotations["probes"]}
    query_execution = measurement.get("metadata", {}).get("query_execution")
    if (
        not isinstance(query_execution, list)
        or any(
            not isinstance(row, dict) or set(row) != {"probe", "execution_device", "cache_hit"}
            for row in query_execution
        )
        or len(query_execution) != len(probes)
        or {row["probe"] for row in query_execution} != probes
    ):
        raise ValueError(f"{project.id}: incomplete query execution provenance")
    query_scores = measurement.get("query_scores")
    if not isinstance(query_scores, list) or any(not isinstance(row, dict) for row in query_scores):
        raise ValueError("measurement query_scores must be a list of objects")
    query_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in query_scores:
        probe, unit = row.get("probe"), row.get("unit")
        key = (probe, unit)
        if probe not in probes or unit not in expected_ids:
            raise ValueError(f"{project.id}: invalid measurement query row {key}")
        if key in query_rows:
            raise ValueError(f"{project.id}: duplicate measurement query row {key}")
        query_rows[key] = row
        score, rank = row.get("cosine"), row.get("rank")
        scored = (
            isinstance(score, int | float)
            and not isinstance(score, bool)
            and math.isfinite(score)
            and -1.0 <= score <= 1.0
        )
        ranked = type(rank) is int and rank > 0
        if embedded[unit] != scored or scored != ranked:
            raise ValueError(f"{project.id}: inconsistent score state for query row {key}")
    expected_queries = {(probe, unit) for probe in probes for unit in expected_ids}
    if query_rows.keys() != expected_queries:
        raise ValueError(f"{project.id}: incomplete measurement query matrix")
    expected_ranks = set(range(1, sum(embedded.values()) + 1))
    for probe in probes:
        ranked_rows = [row for (row_probe, _), row in query_rows.items() if row_probe == probe]
        ranks = {row["rank"] for row in ranked_rows if row["rank"] is not None}
        if ranks != expected_ranks:
            raise ValueError(f"{project.id}: incomplete query ranking for {probe}")
        ordered = sorted(
            (row["rank"], row["cosine"]) for row in ranked_rows if row["rank"] is not None
        )
        if any(left[1] < right[1] for left, right in pairwise(ordered)):
            raise ValueError(f"{project.id}: query scores disagree with ranks for {probe}")


def load_measurement(
    path: Path,
    project: Project | None = None,
    *,
    expected_model: str | None = None,
    expected_device: str | None = None,
) -> dict[str, Any]:
    """Load one raw measurement and verify its basic identity."""
    measurement = read_json(path)
    if measurement.get("schema_version") != ARTIFACT_VERSION:
        raise ValueError(f"unsupported measurement schema: {path}")
    if project is not None and measurement["metadata"]["project"] != project.id:
        raise ValueError(f"measurement belongs to another project: {path}")
    metadata = measurement["metadata"]
    if type(metadata.get("batch_size")) is not int or metadata["batch_size"] <= 0:
        raise ValueError(f"measurement has invalid batch size: {path}")
    if metadata.get("math_policy") != "standard":
        raise ValueError(f"measurement has invalid math policy: {path}")
    if metadata.get("mps_operator_fallback") is not CALIBRATION_MPS_OPERATOR_FALLBACK:
        raise ValueError(f"stale measurement or invalid execution metadata: {path}")
    if (
        expected_model is not None
        and metadata["model"] != resolve_model_profile(expected_model).key
    ):
        raise ValueError(f"measurement belongs to another model: {path}")
    if expected_device is not None:
        query_execution = metadata.get("query_execution")
        if (
            metadata["requested_device"] != expected_device
            or any(
                stats["execution_device"] != expected_device or stats["cache_hit_rows"]
                for stats in metadata["execution"].values()
            )
            or not isinstance(query_execution, list)
            or not query_execution
            or any(
                not isinstance(row, dict)
                or row.get("execution_device") != expected_device
                or row.get("cache_hit") is not False
                for row in query_execution
            )
        ):
            raise ValueError(f"measurement did not execute on {expected_device}: {path}")
    if project is not None:
        validate_measurement_identity(project, metadata)
        _validate_measurement_payload(measurement, project)
    return measurement
