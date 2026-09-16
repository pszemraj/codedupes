"""Capture threshold-independent measurements and reject stale replay artifacts."""

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
from codedupes.semantic_profiles import list_supported_models, resolve_model_profile
from codedupes.traditional import find_exact_pair_keys, jaccard_similarity

try:
    from .calibration_contract import (
        REPO,
        Project,
        ProjectAnalyzer,
        analyzer_config,
        digest,
        eligibility_reason,
        evidence_identity,
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
        digest,
        eligibility_reason,
        evidence_identity,
        extract_project,
        read_json,
        resolve_annotations,
        unit_ids,
        write_json,
    )

ARTIFACT_VERSION = 1
DEFAULT_MEASUREMENTS = REPO / "test_fixtures/calibration/measurements"


def frozen_defaults() -> dict[str, Any]:
    """Snapshot production settings without claiming statistical optimality."""
    fields = (
        "canonical_name",
        "default_revision",
        "default_semantic_threshold",
        "default_search_threshold",
        "hybrid_weak_identifier_jaccard_min",
        "hybrid_statement_ratio_min",
        "language_semantic_thresholds",
        "language_high_confidence_thresholds",
    )
    return {
        profile.key: {
            key: dict(value) if hasattr(value, "items") else value
            for key in fields
            if (value := getattr(profile, key)) is not ...
        }
        for profile in list_supported_models()
    }


def file_sha256(path: Path) -> str:
    """Hash a file as bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(project: Project) -> dict[str, Any]:
    """Describe source and candidate policy independently of labels and test evidence."""
    sources = {}
    roots = list(project.spec["analysis_roots"])
    if project.policy.get("include_tests", False):
        roots += project.spec["test_roots"]
    for relative in roots:
        root = project.root / relative
        for path in sorted(root.rglob("*") if root.is_dir() else [root]):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            sources[path.relative_to(project.root).as_posix()] = file_sha256(path)
    pipeline_files = [
        REPO / "src/codedupes/analyzer.py",
        REPO / "src/codedupes/constants.py",
        REPO / "src/codedupes/devices.py",
        REPO / "src/codedupes/embedding_cache.py",
        REPO / "src/codedupes/extractor.py",
        REPO / "src/codedupes/models.py",
        REPO / "src/codedupes/pairs.py",
        REPO / "src/codedupes/semantic.py",
        REPO / "src/codedupes/traditional.py",
        *list((REPO / "src/codedupes/languages").rglob("*.py")),
        Path(__file__),
        Path(__file__).with_name("calibration_contract.py"),
    ]
    pipeline = {
        p.relative_to(REPO).as_posix(): file_sha256(p)
        for p in sorted(pipeline_files)
        if p.name != "_version.py"
    }
    return {
        "project": project.id,
        "source_files": sources,
        "analysis_roots": roots,
        "languages": project.spec["languages"],
        "policy": project.policy,
        "pipeline_sha256": digest(pipeline),
        "runtime_versions": semantic.get_semantic_runtime_versions(),
    }


def query_identity(project: Project) -> dict[str, str]:
    """Query wording affects scores; relevance judgments do not."""
    return {p["id"]: p["query"] for p in project.annotations["probes"]}


def artifact_path(root: Path, project: Project, model: str, device: str) -> Path:
    """Return the portable directory for one measurement configuration."""
    return root / project.id / project.policy_name / resolve_model_profile(model).key / device


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Persist one reviewable JSON record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        )
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read an artifact table."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pair_scores(embeddings: np.ndarray) -> dict[tuple[int, int], float]:
    """Use production row blocking and matrix width before applying any threshold."""
    scores = {}
    for start in range(0, len(embeddings), semantic._PAIRWISE_SCAN_BLOCK_SIZE):
        matrix = embeddings[start : start + semantic._PAIRWISE_SCAN_BLOCK_SIZE] @ embeddings.T
        for row in range(len(matrix)):
            for column in range(start + row + 1, len(embeddings)):
                value = float(matrix[row, column])
                if not np.isfinite(value):
                    raise ValueError("nonfinite duplicate score")
                scores[(start + row, column)] = min(value, 1.0)
    return scores


def _assert_device(analyzer: ProjectAnalyzer, requested: str) -> dict[str, Any]:
    """Require actual uncached execution on the device named by the artifact."""
    stats = analyzer.embedding_stats
    if stats is None or stats.execution_device != requested or stats.cache_hit_rows:
        raise ValueError(f"measurement did not independently execute on {requested}: {stats}")
    return asdict(stats)


def capture(project: Project, model: str, device: str, output: Path, batch_size: int = 4) -> Path:
    """Measure one project/model/device with separate duplicate and search task spaces."""
    if device not in {"cpu", "mps"}:
        raise ValueError("pilot measurements require explicit cpu or mps")
    if os.getenv("CODEDUPES_CPU_BF16") == "1":
        raise ValueError("disable experimental CPU bf16 for fp32 reference calibration")
    profile = resolve_model_profile(model)
    if profile.default_revision is None or len(profile.default_revision) != 40:
        raise ValueError("measurement requires an immutable built-in model revision")
    inventory, _ = extract_project(project, inventory=True)
    resolved = resolve_annotations(project, inventory)
    # Include source inventory and annotated policy exclusions, not every behavior-test unit.
    source_units, _ = extract_project(project)
    inventory = list({u.uid: u for u in [*source_units, *resolved.values()]}.values())
    ids = unit_ids(project, inventory, resolved)
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
    duplicate_stats = _assert_device(analyzer, device)
    candidates = analyzer._semantic_units or []
    embeddings = analyzer._embeddings
    if embeddings is None or not candidates:
        raise ValueError("measurement has no embedded candidates")
    candidate_uids = {u.uid for u in candidates}
    rows_by_uid = {u.uid: row for row, u in enumerate(candidates)}
    matrix_scores = pair_scores(embeddings)
    exact = find_exact_pair_keys(candidates)
    traditional = {}
    for duplicate in result.traditional_duplicates:
        key = ordered_pair_key(duplicate.unit_a, duplicate.unit_b)
        traditional.setdefault(key, []).append(
            {"method": duplicate.method, "similarity": duplicate.similarity}
        )
    units = []
    for unit in sorted(inventory, key=lambda u: ids[u.uid]):
        embedded = unit.uid in candidate_uids
        reason = None
        if not embedded:
            if unit.uid not in {u.uid for u in source_units}:
                reason = "excluded_by_source_policy"
            elif unit.unit_type.name.lower() not in config.semantic_unit_types:
                reason = "unit_kind"
            else:
                reason = "minimum_statements"
        units.append(
            {
                "id": ids[unit.uid],
                "path": unit.file_path.relative_to(project.root).as_posix(),
                "qualified_name": unit.qualified_name,
                "kind": unit.unit_type.name.lower(),
                "language": unit.language,
                "start_line": unit.lineno,
                "end_line": unit.end_lineno,
                "statement_count": semantic.get_code_unit_statement_count(unit),
                "identifiers": sorted(unit.identifiers),
                "is_public": unit.is_public,
                "source_sha256": hashlib.sha256(unit.source.encode()).hexdigest(),
                "embedded": embedded,
                "missing_score_reason": reason,
            }
        )
    pairs = []
    for a, b in combinations(sorted(inventory, key=lambda u: ids[u.uid]), 2):
        embedded = a.uid in candidate_uids and b.uid in candidate_uids
        score = None
        if embedded:
            key = tuple(sorted((rows_by_uid[a.uid], rows_by_uid[b.uid])))
            score = matrix_scores[key]
        reason = eligibility_reason(
            a, b, candidate_uids, exact, suppress_tests=config.suppress_test_semantic_matches
        )
        pairs.append(
            {
                "a": ids[a.uid],
                "b": ids[b.uid],
                "cosine": score,
                "embedded": embedded,
                "comparable": reason is None,
                "missing_score_reason": None if embedded else "endpoint_not_embedded",
                "exclusion_reason": reason,
                "identifier_jaccard": jaccard_similarity(a.identifiers, b.identifiers),
                "statement_ratio": _statement_count_ratio(a, b),
                "token_equal": bool(a.token_hash and a.token_hash == b.token_hash),
                "structural_equal": bool(
                    a.structural_hash and a.structural_hash == b.structural_hash
                ),
                "traditional": traditional.get(ordered_pair_key(a, b), []),
            }
        )
    duplicate_identity = asdict(analyzer._embedding_space_identity)
    live_default = [
        {"a": ids[p.unit_a.uid], "b": ids[p.unit_b.uid], "tier": p.tier}
        for p in result.hybrid_duplicates
    ]
    explicit_analyzer = ProjectAnalyzer(
        project,
        analyzer_config(
            project,
            model_name=profile.canonical_name,
            model_revision=profile.default_revision,
            device=device,
            batch_size=batch_size,
            embedding_cache=False,
            progress="never",
            semantic_threshold=profile.default_semantic_threshold,
        ),
    )
    started = time.perf_counter()
    explicit_result = explicit_analyzer.analyze(project.root)
    explicit_seconds = time.perf_counter() - started
    explicit_stats = _assert_device(explicit_analyzer, device)
    live_explicit_override = [
        {"a": ids[p.unit_a.uid], "b": ids[p.unit_b.uid], "tier": p.tier}
        for p in explicit_result.hybrid_duplicates
    ]
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
    search_stats = _assert_device(search, device)
    query_rows = []
    for probe in project.annotations["probes"]:
        hits = search.search(probe["query"], top_k=count, threshold=-1.0)
        scores = {u.uid: (score, rank) for rank, (u, score) in enumerate(hits, 1)}
        for unit in sorted(inventory, key=lambda u: ids[u.uid]):
            score, rank = scores.get(unit.uid, (None, None))
            query_rows.append(
                {
                    "probe": probe["id"],
                    "unit": ids[unit.uid],
                    "cosine": score,
                    "rank": rank,
                    "missing_score_reason": None if score is not None else "endpoint_not_embedded",
                }
            )
    search_seconds = time.perf_counter() - started
    task_identity = {
        "duplicate": {
            "task": DEFAULT_CHECK_SEMANTIC_TASK,
            "embedding_space": duplicate_identity,
            "encode_plan": asdict(
                semantic.resolve_encode_plan(
                    profile.canonical_name, "code", None, DEFAULT_CHECK_SEMANTIC_TASK
                )
            ),
        },
        "search": {
            "task": DEFAULT_SEARCH_SEMANTIC_TASK,
            "embedding_space": asdict(search._embedding_space_identity),
            "code_plan": asdict(
                semantic.resolve_encode_plan(
                    profile.canonical_name, "code", None, DEFAULT_SEARCH_SEMANTIC_TASK
                )
            ),
            "query_plan": asdict(
                semantic.resolve_encode_plan(
                    profile.canonical_name, "query", None, DEFAULT_SEARCH_SEMANTIC_TASK
                )
            ),
            "document": "source",
        },
    }
    directory = artifact_path(output, project, model, device)
    for filename, table in [
        ("units.jsonl", units),
        ("pairs.jsonl", pairs),
        ("query_scores.jsonl", query_rows),
    ]:
        write_rows(directory / filename, table)
    identity = {
        "source": source_identity(project),
        "queries": query_identity(project),
        "model": profile.canonical_name,
        "revision": profile.default_revision,
        "requested_device": device,
        "tasks": task_identity,
        "batch_size": batch_size,
    }
    metadata = {
        "schema_version": ARTIFACT_VERSION,
        "identity": identity,
        "measurement_id": digest(identity),
        "annotation_sha256_at_capture": digest(project.annotations),
        "evidence_sha256_at_capture": evidence_identity(project),
        "timing_seconds": {
            "duplicate": duplicate_seconds,
            "explicit_override": explicit_seconds,
            "search": search_seconds,
        },
        "execution": {
            "duplicate": duplicate_stats,
            "explicit_override": explicit_stats,
            "search": search_stats,
        },
        "live_default": live_default,
        "live_explicit_override": {
            "threshold": profile.default_semantic_threshold,
            "findings": live_explicit_override,
        },
        "tables": {
            name: file_sha256(directory / name)
            for name in ("units.jsonl", "pairs.jsonl", "query_scores.jsonl")
        },
    }
    write_json(directory / "metadata.json", metadata)
    semantic.clear_model_cache()
    return directory


def load_measurement(directory: Path, project: Project) -> dict[str, Any]:
    """Fail closed on corrupt or stale source/model/query/pipeline measurements."""
    metadata = read_json(directory / "metadata.json")
    if metadata.get("schema_version") != ARTIFACT_VERSION:
        raise ValueError("unsupported measurement schema")
    identity = metadata["identity"]
    if metadata["measurement_id"] != digest(identity):
        raise ValueError("corrupt measurement identity")
    if identity["source"] != source_identity(project):
        raise ValueError(f"stale source/pipeline/policy measurement: {directory}")
    if identity["queries"] != query_identity(project):
        raise ValueError("stale query measurements; relevance-only edits can reuse scores")
    if identity["revision"] != resolve_model_profile(identity["model"]).default_revision:
        raise ValueError("stale pinned model revision")
    result = {"metadata": metadata}
    for filename, checksum in metadata["tables"].items():
        if file_sha256(directory / filename) != checksum:
            raise ValueError(f"corrupt measurement table: {filename}")
        result[filename.split(".")[0]] = read_rows(directory / filename)
    return result
