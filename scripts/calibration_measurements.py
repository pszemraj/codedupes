"""Measure calibration fixtures without applying production thresholds."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from codedupes import semantic
from codedupes.analyzer import _statement_count_ratio
from codedupes.constants import DEFAULT_CHECK_SEMANTIC_TASK, DEFAULT_SEARCH_SEMANTIC_TASK
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
        resolve_annotations,
        unit_ids,
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
        resolve_annotations,
        unit_ids,
        write_json,
    )

ARTIFACT_VERSION = 3
DEFAULT_MEASUREMENTS = REPO / "scratch/calibration"


def measurement_fingerprint(project: Project, model: str) -> str:
    """Fingerprint source, queries, model, and the embedding/extraction pipeline."""
    profile = resolve_model_profile(model)
    source_files = {}
    roots = list(project.spec["analysis_roots"])
    if project.policy.get("include_tests", False):
        roots.extend(project.spec["test_roots"])
    for relative in roots:
        root = project.root / relative
        for path in sorted(root.rglob("*") if root.is_dir() else [root]):
            if path.is_file():
                source_files[path.relative_to(project.root).as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    pipeline_paths = [
        REPO / "src/codedupes/analyzer.py",
        REPO / "src/codedupes/devices.py",
        REPO / "src/codedupes/extractor.py",
        REPO / "src/codedupes/models.py",
        REPO / "src/codedupes/semantic.py",
        REPO / "src/codedupes/traditional.py",
        *sorted((REPO / "src/codedupes/languages").glob("*.py")),
    ]
    identity = {
        "source": source_files,
        "queries": {probe["id"]: probe["query"] for probe in project.annotations["probes"]},
        "policy": project.policy,
        "model": profile.canonical_name,
        "revision": profile.default_revision,
        "family": profile.family,
        "trust_remote_code": profile.default_trust_remote_code,
        "tasks": [DEFAULT_CHECK_SEMANTIC_TASK, DEFAULT_SEARCH_SEMANTIC_TASK],
        "runtime_versions": semantic.get_semantic_runtime_versions(),
        "pipeline": {
            path.relative_to(REPO).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in pipeline_paths
        },
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def artifact_path(root: Path, project: Project, model: str, device: str) -> Path:
    """Return the raw measurement path for one project/model/device run."""
    key = resolve_model_profile(model).key
    return root / project.id / key / f"{device}.json"


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
                scores[(start + offset, column)] = min(value, 1.0)
    return scores


def _execution(analyzer: ProjectAnalyzer, requested: str) -> dict[str, Any]:
    """Require a fresh run on the requested device."""
    stats = analyzer.embedding_stats
    if stats is None or stats.execution_device != requested or stats.cache_hit_rows:
        raise ValueError(f"measurement did not execute independently on {requested}: {stats}")
    return asdict(stats)


def capture(project: Project, model: str, device: str, output: Path, batch_size: int = 4) -> Path:
    """Capture duplicate-pair and search scores in two uncached model passes."""
    if device not in {"cpu", "mps"}:
        raise ValueError("device must be cpu or mps")
    if os.getenv("CODEDUPES_CPU_BF16") == "1":
        raise ValueError("disable experimental CPU bf16 while calibrating")

    profile = resolve_model_profile(model)
    if profile.default_revision is None or len(profile.default_revision) != 40:
        raise ValueError("calibration requires a pinned built-in model revision")
    inference_dtype = str(semantic._resolve_model_dtype(profile.family, device)).removeprefix(
        "torch."
    )

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
            mode="search",
            run_traditional=False,
        ),
    )
    started = time.perf_counter()
    count = search.index(project.root)
    query_scores = []
    for probe in project.annotations["probes"]:
        hits = search.search(probe["query"], top_k=count, threshold=-1.0)
        by_uid = {unit.uid: (float(score), rank) for rank, (unit, score) in enumerate(hits, 1)}
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
            "inference_dtype": inference_dtype,
            "runtime_versions": semantic.get_semantic_runtime_versions(),
            "input_fingerprint": measurement_fingerprint(project, profile.key),
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
    if (
        expected_model is not None
        and metadata["model"] != resolve_model_profile(expected_model).key
    ):
        raise ValueError(f"measurement belongs to another model: {path}")
    if expected_device is not None and (
        metadata["requested_device"] != expected_device
        or any(
            stats["execution_device"] != expected_device or stats["cache_hit_rows"]
            for stats in metadata["execution"].values()
        )
    ):
        raise ValueError(f"measurement did not execute on {expected_device}: {path}")
    if project is not None and measurement["metadata"][
        "input_fingerprint"
    ] != measurement_fingerprint(project, measurement["metadata"]["model"]):
        raise ValueError(f"stale measurement: {path}")
    return measurement
