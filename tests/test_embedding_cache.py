from __future__ import annotations

import contextlib
import copy
import hashlib
import itertools
import json
import os
import stat
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from codedupes import embedding_cache, semantic
from codedupes.embedding_cache import CorpusSelectionManifest, EmbeddingCache, diff_manifest
from codedupes.models import CodeUnit
from codedupes.semantic import (
    EmbeddingRunStats,
    compute_embeddings,
    compute_embeddings_with_identity,
    find_similar_to_query,
)
from tests.conftest import extract_units

REVISION_1 = "1" * 40
REVISION_2 = "2" * 40

FIVE_FUNCTION_SOURCE = """
def alpha(x):
    return x + 1

def beta(x):
    return x + 2

def gamma(x):
    return x + 3

def delta(x):
    return x + 4

def epsilon(x):
    return x + 5
"""


def _vector_for_text(text: str, dim: int = 4) -> np.ndarray:
    """Derive a deterministic unit-normalized float32 vector from a text's MD5 digest."""
    digest = hashlib.md5(text.encode()).digest()
    raw = np.array([float(b) + 1.0 for b in digest[:dim]], dtype=np.float32)
    return (raw / np.linalg.norm(raw)).astype(np.float32)


class CountingModel:
    """Deterministic fake embedding model that records every encode call."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.encode_calls: list[list[str]] = []
        self.prompts_seen: list[str | None] = []

    def encode(self, texts, **kwargs):
        self.encode_calls.append(list(texts))
        self.prompts_seen.append(kwargs.get("prompt"))
        return np.stack([_vector_for_text(text, self.dim) for text in texts], axis=0)


def _corpus_manifest(
    units: dict[str, str],
    *,
    orphans: dict[str, int] | None = None,
) -> CorpusSelectionManifest:
    """Build a minimal complete manifest for transition-diff tests."""
    return CorpusSelectionManifest(
        last_seen_generation=1,
        complete_scan=True,
        units=units,
        orphans=dict(orphans or {}),
    )


@pytest.mark.parametrize(
    ("previous", "current", "moved", "deleted", "orphaned"),
    [
        ({"old": "key"}, {"new": "key"}, ["new"], [], set()),
        ({"old": "key"}, {}, [], ["old"], {"key"}),
        ({"old": "old-key"}, {"new": "new-key"}, [], ["old"], {"old-key"}),
        ({"first": "key"}, {"first": "key", "second": "key"}, [], [], set()),
        ({"first": "key", "second": "key"}, {"first": "key"}, [], ["second"], set()),
        ({"first": "key", "second": "key"}, {}, [], ["first", "second"], {"key"}),
        ({"first": "key", "second": "key"}, {"new": "key"}, ["new"], ["first"], set()),
        ({"old": "key"}, {"first": "key", "second": "key"}, ["first"], [], set()),
    ],
    ids=[
        "move",
        "delete",
        "edit",
        "identical-body-added",
        "identical-body-deleted",
        "all-identical-bodies-deleted",
        "identical-body-moved-and-deleted",
        "identical-body-moved-and-added",
    ],
)
def test_diff_manifest_classifies_corpus_transitions(
    previous,
    current,
    moved,
    deleted,
    orphaned,
) -> None:
    diff = diff_manifest(_corpus_manifest(previous), current)

    assert diff.moved == moved
    assert diff.deleted == deleted
    assert diff.orphaned == orphaned


def test_manifest_revert_clears_orphan_age(tmp_path) -> None:
    cache = EmbeddingCache()
    scope = tmp_path / "repo"
    scope.mkdir()
    key = "content-key"
    cache.put_many(scope, "model", "revision", [(key, np.ones(2, dtype=np.float32))])
    cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={"unit": key},
        complete_scan=True,
    )
    orphaned = cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={},
        complete_scan=True,
    )
    restored = cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={"unit": key},
        complete_scan=True,
    )

    assert orphaned is not None
    assert orphaned.orphan_rows_retained == 1
    assert restored is not None
    assert restored.orphan_rows_retained == 0
    assert restored.orphan_rows_collected == 0


def test_manifest_gc_never_collects_query_rows(tmp_path) -> None:
    cache = EmbeddingCache()
    scope = tmp_path / "repo"
    scope.mkdir()
    code_key = "code-key"
    query_key = "query-key"
    vector = np.ones(2, dtype=np.float32)
    cache.put_many(scope, "model", "revision", [(code_key, vector)], namespace="code")
    cache.put_many(scope, "model", "revision", [(query_key, vector)], namespace="query")
    cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={"unit": code_key},
        complete_scan=True,
    )

    published = None
    for _ in range(4):
        published = cache.publish_corpus_manifest(
            scope,
            "model",
            "revision",
            selection="selection",
            units={},
            complete_scan=True,
        )

    assert published is not None
    assert published.orphan_rows_collected == 1
    assert cache.get_many(scope, "model", "revision", [code_key]) == {}
    assert query_key in cache.get_many(scope, "model", "revision", [query_key])


def test_manifest_gc_expires_unrefreshed_selection_pin(tmp_path) -> None:
    cache = EmbeddingCache()
    scope = tmp_path / "repo"
    scope.mkdir()
    key = "shared-key"
    cache.put_many(scope, "model", "revision", [(key, np.ones(2, dtype=np.float32))])
    for selection in ("first", "second"):
        cache.publish_corpus_manifest(
            scope,
            "model",
            "revision",
            selection=selection,
            units={"unit": key},
            complete_scan=True,
        )

    orphaned = cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="first",
        units={},
        complete_scan=True,
    )
    assert orphaned is not None
    assert orphaned.orphan_rows_retained == 1
    assert orphaned.orphan_rows_collected == 0

    for expected_generation in (4, 5):
        retained = cache.publish_corpus_manifest(
            scope,
            "model",
            "revision",
            selection="first",
            units={},
            complete_scan=True,
        )
        assert retained is not None
        assert retained.generation == expected_generation
        assert retained.orphan_rows_retained == 1
        assert retained.orphan_rows_collected == 0
    assert key in cache.get_many(scope, "model", "revision", [key])

    published = cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="first",
        units={},
        complete_scan=True,
    )

    assert published is not None
    assert published.generation == 6
    assert published.orphan_rows_retained == 0
    assert published.orphan_rows_collected == 1
    assert cache.get_many(scope, "model", "revision", [key]) == {}

    rediscovered = cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="second",
        units={},
        complete_scan=True,
    )
    assert rediscovered is not None
    assert len(rediscovered.diff.deleted) == 1
    assert rediscovered.orphan_rows_retained == 0
    assert rediscovered.orphan_rows_collected == 0


def test_manifest_gc_revalidates_references_before_collection(tmp_path, monkeypatch) -> None:
    cache = EmbeddingCache()
    concurrent_cache = EmbeddingCache(cache.cache_root)
    scope = tmp_path / "repo"
    scope.mkdir()
    key = "reintroduced-key"
    cache.put_many(scope, "model", "revision", [(key, np.ones(2, dtype=np.float32))])
    cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={"unit": key},
        complete_scan=True,
    )
    cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={},
        complete_scan=True,
    )
    cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={},
        complete_scan=True,
    )
    cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={},
        complete_scan=True,
    )
    collect_orphans = cache.collect_orphans

    def reintroduce_before_collection(*args, **kwargs):
        concurrent_cache.publish_corpus_manifest(
            scope,
            "model",
            "revision",
            selection="selection",
            units={"unit": key},
            complete_scan=True,
        )
        return collect_orphans(*args, **kwargs)

    monkeypatch.setattr(cache, "collect_orphans", reintroduce_before_collection)

    raced = cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={},
        complete_scan=True,
    )

    assert raced is not None
    assert raced.orphan_rows_collected == 0
    manifest = cache.load_manifest(scope, "model", "revision")
    assert manifest is not None
    assert manifest.selections["selection"].units == {"unit": key}
    assert key in cache.get_many(scope, "model", "revision", [key])


def test_cache_stats_accepts_manifest_without_selections(tmp_path) -> None:
    cache = EmbeddingCache()
    scope = tmp_path / "repo"
    scope.mkdir()
    cache.put_many(
        scope,
        "model",
        "revision",
        [("key", np.ones(2, dtype=np.float32))],
    )
    shard_dir = cache.shard_dir(scope, "model", "revision")
    embedding_cache.atomic_write_json(
        shard_dir / embedding_cache.MANIFEST_FILENAME,
        {
            "schema": embedding_cache.MANIFEST_SCHEMA,
            "generation": 7,
            "selections": {},
        },
    )

    stats = cache.stats()

    assert stats["repos"][0]["last_complete_generation"] == 7


def test_manifest_limits_inactive_selection_baselines(tmp_path) -> None:
    cache = EmbeddingCache()
    scope = tmp_path / "repo"
    scope.mkdir()
    selection_count = embedding_cache.MANIFEST_SELECTION_LIMIT + 3

    for index in range(selection_count):
        cache.publish_corpus_manifest(
            scope,
            "model",
            "revision",
            selection=f"selection-{index:02d}",
            units={"unit": f"key-{index}"},
            complete_scan=True,
        )

    manifest = cache.load_manifest(scope, "model", "revision")
    assert manifest is not None
    assert len(manifest.selections) == embedding_cache.MANIFEST_SELECTION_LIMIT
    assert "selection-00" not in manifest.selections
    assert f"selection-{selection_count - 1:02d}" in manifest.selections


def test_manifest_selection_limit_uses_incomplete_publish_recency(tmp_path) -> None:
    cache = EmbeddingCache()
    scope = tmp_path / "repo"
    scope.mkdir()
    for index in range(embedding_cache.MANIFEST_SELECTION_LIMIT):
        cache.publish_corpus_manifest(
            scope,
            "model",
            "revision",
            selection=f"z-old-{index:02d}",
            units={"unit": f"old-key-{index}"},
            complete_scan=False,
        )

    cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="a-recent",
        units={"unit": "recent-key"},
        complete_scan=False,
    )
    cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="b-newest",
        units={"unit": "newest-key"},
        complete_scan=False,
    )

    manifest = cache.load_manifest(scope, "model", "revision")
    assert manifest is not None
    assert len(manifest.selections) == embedding_cache.MANIFEST_SELECTION_LIMIT
    assert "a-recent" in manifest.selections
    assert "b-newest" in manifest.selections
    assert "z-old-00" not in manifest.selections


def test_incomplete_manifest_matches_observed_files_by_exact_path(tmp_path) -> None:
    cache = EmbeddingCache()
    scope = tmp_path / "repo"
    scope.mkdir()
    first_path = "/repo/a.py"
    sibling_path = "/repo/a.py::sibling.py"
    first_uid = f"{first_path}::python::a.alpha::0"
    sibling_uid = f"{sibling_path}::python::sibling.beta::0"
    cache.put_many(
        scope,
        "model",
        "revision",
        [
            ("first-key", np.ones(2, dtype=np.float32)),
            ("sibling-key", np.zeros(2, dtype=np.float32)),
        ],
    )
    cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={first_uid: "first-key", sibling_uid: "sibling-key"},
        unit_paths={first_uid: first_path, sibling_uid: sibling_path},
        complete_scan=True,
    )

    narrow = cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={first_uid: "first-key"},
        unit_paths={first_uid: first_path},
        observed_files=(first_path,),
        complete_scan=False,
    )

    assert narrow is not None
    assert narrow.diff.deleted == []
    assert narrow.orphan_rows_retained == 0
    manifest = cache.load_manifest(scope, "model", "revision")
    assert manifest is not None
    assert manifest.selections["selection"].units == {
        first_uid: "first-key",
        sibling_uid: "sibling-key",
    }


def _five_units(tmp_path: Path) -> list[CodeUnit]:
    return extract_units(tmp_path, FIVE_FUNCTION_SOURCE, filename="mod.py")


def _patch_get_model(monkeypatch, model: CountingModel) -> dict[str, int]:
    counts = {"count": 0}

    def fake_get_model(*_args, **_kwargs):
        counts["count"] += 1
        return model

    monkeypatch.setattr(semantic, "get_model", fake_get_model)
    return counts


def _active_vectors_path(shard_dir: Path) -> Path:
    payload = json.loads((shard_dir / embedding_cache.INDEX_FILENAME).read_text(encoding="utf-8"))
    return shard_dir / embedding_cache._vectors_filename(payload["generation"])


def test_full_cache_hit_skips_model_load_and_encode(tmp_path, monkeypatch):
    units = _five_units(tmp_path)
    model = CountingModel()
    get_model_counts = _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    first = compute_embeddings(
        units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path
    )
    assert get_model_counts["count"] == 1
    assert len(model.encode_calls) == 1
    assert len(model.encode_calls[0]) == 5

    second = compute_embeddings(
        units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path
    )
    assert get_model_counts["count"] == 1
    assert len(model.encode_calls) == 1
    np.testing.assert_array_equal(first, second)


def test_strict_symbolic_revision_revalidates_before_cache_hit(tmp_path, monkeypatch):
    units = _five_units(tmp_path)
    model = CountingModel()
    get_model_counts = _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", lambda *_args: None)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    first = compute_embeddings(
        units,
        model_name="test-model",
        revision="main",
        cache_scope=tmp_path,
        strict_revision_cache=True,
    )
    second = compute_embeddings(
        units,
        model_name="test-model",
        revision="main",
        cache_scope=tmp_path,
        strict_revision_cache=True,
    )

    assert get_model_counts["count"] == 2
    assert len(model.encode_calls) == 2
    np.testing.assert_array_equal(first, second)


def test_strict_query_load_uses_resolved_commit(tmp_path, monkeypatch):
    units = _five_units(tmp_path)
    model = CountingModel()
    commit = "a" * 40
    loaded_revisions: list[str | None] = []

    def fake_get_model(*_args, **kwargs):
        loaded_revisions.append(kwargs.get("revision"))
        return model

    monkeypatch.setattr(semantic, "get_model", fake_get_model)
    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", lambda *_args: commit)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: commit)

    embeddings, identity = compute_embeddings_with_identity(
        units,
        model_name="drift-model",
        revision="main",
        cache_scope=tmp_path,
        strict_revision_cache=True,
    )
    find_similar_to_query(
        "find addition",
        units,
        embeddings,
        model_name="drift-model",
        revision="main",
        threshold=0.0,
        cache_scope=tmp_path,
        strict_revision_cache=True,
        corpus_identity=identity,
    )

    assert loaded_revisions == [commit, commit]


def test_embeddinggemma_cache_variant_scopes_only_nondefault_dtype(monkeypatch):
    profile = semantic.resolve_model_profile("embeddinggemma-300m")
    monkeypatch.setattr(
        semantic,
        "_resolve_semantic_device_request",
        lambda *_args, **_kwargs: "cuda",
    )
    selected_dtype = {"value": "torch.bfloat16"}
    monkeypatch.setattr(
        semantic,
        "_resolve_model_dtype",
        lambda _family, _device, **_kwargs: selected_dtype["value"],
    )

    assert semantic._dtype_variant_for(profile, "cuda", mps_fallback=None) == "dtype=torch.bfloat16"

    selected_dtype["value"] = "torch.float32"
    assert semantic._dtype_variant_for(profile, "cuda", mps_fallback=None) == ""


def test_cuda_dtype_variant_applies_to_every_family(monkeypatch):
    profile = semantic.resolve_model_profile("test-model")
    monkeypatch.setattr(
        semantic,
        "_resolve_semantic_device_request",
        lambda *_args, **_kwargs: "cuda",
    )
    monkeypatch.setattr(
        semantic,
        "_resolve_model_dtype",
        lambda _family, _device, **_kwargs: "torch.bfloat16",
    )

    assert semantic._dtype_variant_for(profile, "cuda", mps_fallback=None) == "dtype=torch.bfloat16"
    # CPU/MPS requests never resolve a device and stay in the shared float32 space.
    assert semantic._dtype_variant_for(profile, "cpu", mps_fallback=None) == ""
    assert semantic._dtype_variant_for(profile, "mps", mps_fallback=None) == ""


def test_runtime_upgrade_invalidates_whole_corpus_not_row_subset(tmp_path, monkeypatch):
    units = _five_units(tmp_path)
    model = CountingModel()
    _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 1

    # Same model, revision, and texts — only the installed inference stack differs.
    monkeypatch.setattr(semantic, "_safe_package_version", lambda _name: "99.0.0-upgraded")

    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 2
    # Every unit re-embedded: no partial reuse of vectors from the old runtime.
    assert len(model.encode_calls[-1]) == len(units)


def test_cache_variant_includes_encode_plan_identity():
    profile = semantic.resolve_model_profile("test-model")
    plain = semantic._cache_variant_for(
        profile, "cpu", semantic.EncodePlan(route="symmetric"), mps_fallback=None
    )
    prompted = semantic._cache_variant_for(
        profile, "cpu", semantic.EncodePlan(route="symmetric", prompt="custom: "), mps_fallback=None
    )
    routed = semantic._cache_variant_for(
        profile, "cpu", semantic.EncodePlan(route="document"), mps_fallback=None
    )
    assert len({plain, prompted, routed}) == 3


def test_cache_variant_keys_mps_fast_math_policy(monkeypatch):
    profile = semantic.resolve_model_profile("test-model")
    plan = semantic.EncodePlan(route="symmetric")

    monkeypatch.delenv("PYTORCH_MPS_FAST_MATH", raising=False)
    baseline = semantic._cache_variant_for(profile, "mps", plan, mps_fallback=None)
    cpu_baseline = semantic._cache_variant_for(profile, "cpu", plan, mps_fallback=None)

    # Disabled fast math is the same policy as an unset variable.
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "0")
    assert semantic._cache_variant_for(profile, "mps", plan, mps_fallback=None) == baseline

    # An enabled policy must split the key space wherever MPS can execute.
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")
    assert semantic._cache_variant_for(profile, "mps", plan, mps_fallback=None) != baseline

    # torch enables fast math for any set value except the literal "0": the
    # empty string and whitespace-wrapped zeros must key as fast-math variants.
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "")
    assert semantic._cache_variant_for(profile, "mps", plan, mps_fallback=None) != baseline
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", " 0")
    assert semantic._cache_variant_for(profile, "mps", plan, mps_fallback=None) != baseline
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")

    # Devices that can never execute Metal kernels ignore the policy.
    assert semantic._cache_variant_for(profile, "cpu", plan, mps_fallback=None) == cpu_baseline

    # On macOS, ``auto`` can resolve to MPS, so it splits with the policy too.
    monkeypatch.setattr(sys, "platform", "darwin")
    auto_fast = semantic._cache_variant_for(profile, "auto", plan, mps_fallback=None)
    monkeypatch.delenv("PYTORCH_MPS_FAST_MATH")
    assert semantic._cache_variant_for(profile, "auto", plan, mps_fallback=None) != auto_fast


def test_mps_fast_math_policy_change_invalidates_warm_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("PYTORCH_MPS_FAST_MATH", raising=False)
    units = _five_units(tmp_path)
    model = CountingModel()
    _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 1

    # Same request under an altered Metal math policy: faithful-float32 rows
    # must not satisfy hits, so the whole corpus re-embeds. The fake reports
    # MPS execution so the fast-math key space accepts its writes.
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")
    model.device = "mps"
    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 2
    assert len(model.encode_calls[-1]) == len(units)

    # The fast-math key space warms independently.
    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 2


def test_fast_math_variant_restarts_under_faithful_cpu_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")
    units = _five_units(tmp_path)
    model = CountingModel()
    _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    # The request is keyed for fast math, but execution lands on CPU. Corpus
    # assembly restarts under the faithful policy and publishes that complete
    # matrix to the CPU/MPS-float32 key space.
    model.device = "cpu"
    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 1

    # Another fallback-shaped request reuses the complete faithful matrix.
    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 1


class _MidEncodeCpuFallbackModel(CountingModel):
    """Fake that lands on CPU during encode, like the OOM/invalid-output retry ladder."""

    def encode(self, texts, **kwargs):
        result = super().encode(texts, **kwargs)
        self.device = "cpu"
        return result


def test_fast_math_variant_rebuilds_corpus_after_mid_encode_cpu_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")
    units = _five_units(tmp_path)
    model = _MidEncodeCpuFallbackModel()
    model.device = "mps"
    _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    # Execution starts on MPS but lands on CPU mid-encode. The first vectors are
    # discarded and every row is encoded again under the faithful CPU policy.
    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 2
    assert all(len(call) == len(units) for call in model.encode_calls)

    # A later request cannot consume the faithful rows as fast-math hits. This
    # model lands on CPU again, after which the faithful restart is already warm.
    model.device = "mps"
    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 3


def test_fast_math_query_aborts_before_mixed_policy_dot_product(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")
    units = _five_units(tmp_path)
    model = _MidEncodeCpuFallbackModel()
    model.device = "mps"
    _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    embeddings = compute_embeddings(
        units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path
    )

    model.device = "mps"
    with pytest.raises(RuntimeError, match="Fast-math query execution left MPS"):
        find_similar_to_query(
            "find addition",
            units,
            embeddings,
            model_name="test-model",
            revision=REVISION_1,
            cache_scope=tmp_path,
            top_k=3,
        )
    query_encodes = len(model.encode_calls)

    # The rejected CPU query was not cached in the fast-math key space.
    model.device = "mps"
    with pytest.raises(RuntimeError, match="Fast-math query execution left MPS"):
        find_similar_to_query(
            "find addition",
            units,
            embeddings,
            model_name="test-model",
            revision=REVISION_1,
            cache_scope=tmp_path,
            top_k=3,
        )
    assert len(model.encode_calls) == query_encodes + 1


def test_compute_embeddings_passes_raw_text_with_prompt_config(tmp_path, monkeypatch):
    units = _five_units(tmp_path)
    model = CountingModel()
    _patch_get_model(monkeypatch, model)

    compute_embeddings(
        units,
        model_name="test-model",
        instruction_prefix="custom: ",
        cache_scope=None,
    )

    # The instruction travels as the backend prompt; input texts stay raw.
    assert model.encode_calls == [[unit.source.strip() for unit in units]]
    assert model.prompts_seen == ["custom: "]


def test_partial_update_only_reencodes_changed_unit(tmp_path, monkeypatch):
    units = _five_units(tmp_path)
    model = CountingModel()
    get_model_counts = _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 1

    changed = copy.copy(units[2])
    changed.source = "def gamma(x):\n    return x + 999\n"
    updated_units = list(units)
    updated_units[2] = changed

    stats = EmbeddingRunStats()
    compute_embeddings(
        updated_units,
        model_name="test-model",
        revision=REVISION_1,
        cache_scope=tmp_path,
        stats=stats,
    )
    assert get_model_counts["count"] == 2
    assert len(model.encode_calls) == 2
    assert len(model.encode_calls[-1]) == 1
    assert stats.encoded_inputs == 1
    assert stats.cache_hit_rows == 4
    assert stats.model_loaded is True

    warm_stats = EmbeddingRunStats()
    compute_embeddings(
        updated_units,
        model_name="test-model",
        revision=REVISION_1,
        cache_scope=tmp_path,
        stats=warm_stats,
    )
    assert warm_stats.cache_hit_rows == 5
    assert warm_stats.encoded_inputs == 0
    assert warm_stats.model_loaded is False


def test_cache_key_sensitive_to_model_revision_prefix_and_task(tmp_path, monkeypatch):
    units = _five_units(tmp_path)
    model = CountingModel()
    _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    base: dict[str, object] = {
        "model_name": "embeddinggemma-300m",
        "revision": REVISION_1,
        "cache_scope": tmp_path,
    }
    compute_embeddings(units, **base)
    assert len(model.encode_calls) == 1

    compute_embeddings(units, **{**base, "revision": REVISION_2})
    assert len(model.encode_calls) == 2

    compute_embeddings(units, **{**base, "model_name": "other-model"})
    assert len(model.encode_calls) == 3

    compute_embeddings(units, **{**base, "instruction_prefix": "CUSTOM: "})
    assert len(model.encode_calls) == 4

    compute_embeddings(units, **{**base, "semantic_task": "classification"})
    assert len(model.encode_calls) == 5


def test_shuffled_partial_hit_matches_fully_uncached_compute(tmp_path, monkeypatch):
    units = _five_units(tmp_path)
    model = CountingModel()
    _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 1

    shuffled = [units[3], units[0], units[4], units[1], units[2]]
    mutated = copy.copy(shuffled[1])
    mutated.source = "def other(x):\n    return x + 12345\n"
    shuffled[1] = mutated

    stats = EmbeddingRunStats()
    cached_result = compute_embeddings(
        shuffled,
        model_name="test-model",
        revision=REVISION_1,
        cache_scope=tmp_path,
        stats=stats,
    )
    assert len(model.encode_calls) == 2
    assert len(model.encode_calls[-1]) == 1
    assert stats.encoded_inputs == 1
    assert stats.cache_hit_rows == 4

    uncached_result = compute_embeddings(
        shuffled, model_name="test-model", revision=REVISION_1, cache_scope=None
    )
    assert len(model.encode_calls) == 3
    assert len(model.encode_calls[-1]) == 5

    np.testing.assert_allclose(cached_result, uncached_result)


def test_strict_revision_drift_after_model_load_discards_stale_prefetched_hits(
    tmp_path, monkeypatch
):
    units = _five_units(tmp_path)
    model = CountingModel()
    _patch_get_model(monkeypatch, model)

    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", lambda _model: "rev-a")
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "rev-a")
    compute_embeddings(
        units,
        model_name="drift-model",
        revision=None,
        cache_scope=tmp_path,
        strict_revision_cache=True,
    )
    assert len(model.encode_calls) == 1
    assert len(model.encode_calls[0]) == 5

    changed = copy.copy(units[1])
    changed.source = "def other(x):\n    return x + 777\n"
    mixed_units = [units[0], changed, units[2]]
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "rev-b")

    result = compute_embeddings(
        mixed_units,
        model_name="drift-model",
        revision=None,
        cache_scope=tmp_path,
        strict_revision_cache=True,
    )

    assert len(model.encode_calls) == 2
    assert len(model.encode_calls[-1]) == 3
    assert result.shape == (3, model.dim)


def test_confirm_source_commit_matches_and_purges_on_drift(tmp_path):
    cache = EmbeddingCache(tmp_path)
    scope = tmp_path / "repo"
    scope.mkdir()
    key = embedding_cache.compute_cache_key("some/model", "main", "text-a")
    cache.put_many(
        scope,
        "some/model",
        "main",
        [(key, np.array([1.0, 0.0], dtype=np.float32))],
        expected_source_commit="a" * 40,
    )

    # The write stamped provenance; a matching commit confirms and keeps hits.
    assert cache.confirm_source_commit(scope, "some/model", "main", "a" * 40) is True
    assert set(cache.get_many(scope, "some/model", "main", [key])) == {key}

    # A different loaded commit is drift: the whole label shard is purged.
    assert cache.confirm_source_commit(scope, "some/model", "main", "b" * 40) is False
    assert cache.get_many(scope, "some/model", "main", [key]) == {}
    assert cache.confirm_source_commit(scope, "some/model", "main", "b" * 40) is True


def test_provenance_lives_inside_the_atomic_index(tmp_path):
    cache = EmbeddingCache(tmp_path)
    scope = tmp_path / "repo"
    scope.mkdir()
    key_a = embedding_cache.compute_cache_key("some/model", "main", "text-a")
    key_b = embedding_cache.compute_cache_key("some/model", "main", "text-b")
    vector = np.array([1.0, 0.0], dtype=np.float32)

    cache.put_many(scope, "some/model", "main", [(key_a, vector)], expected_source_commit="a" * 40)
    cache.put_many(scope, "some/model", "main", [(key_b, vector)], expected_source_commit="a" * 40)

    shard_dir = cache.shard_dir(scope, "some/model", "main")
    meta = embedding_cache._read_shard_meta(shard_dir)
    assert meta is not None
    assert meta["source_commit"] == "a" * 40
    assert set(meta["keys"]) == {key_a, key_b}

    # Immutable-revision shards carry no provenance: the revision is the truth.
    cache.put_many(scope, "some/model", REVISION_1, [(key_a, vector)])
    pinned_meta = embedding_cache._read_shard_meta(cache.shard_dir(scope, "some/model", REVISION_1))
    assert pinned_meta is not None
    assert pinned_meta["source_commit"] is None


def test_rows_without_provenance_are_purged_not_adopted(tmp_path):
    """Reviewer repro (round 2): lost provenance must degrade to recompute, never
    to trusting unverifiable rows under a newly observed commit."""
    cache = EmbeddingCache(tmp_path)
    scope = tmp_path / "repo"
    scope.mkdir()
    key = embedding_cache.compute_cache_key("some/model", "main", "text-a")
    vector = np.array([1.0, 0.0], dtype=np.float32)
    cache.put_many(scope, "some/model", "main", [(key, vector)], expected_source_commit="a" * 40)

    # Strip the provenance in place, simulating corruption or a legacy writer.
    shard_dir = cache.shard_dir(scope, "some/model", "main")
    index_path = shard_dir / embedding_cache.INDEX_FILENAME
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["source_commit"] = None
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    assert cache.confirm_source_commit(scope, "some/model", "main", "a" * 40) is False
    assert cache.get_many(scope, "some/model", "main", [key]) == {}

    # Rows written with no provenance at all behave identically on first confirm.
    cache.put_many(scope, "some/model", "main", [(key, vector)])
    assert cache.confirm_source_commit(scope, "some/model", "main", "a" * 40) is False
    assert cache.get_many(scope, "some/model", "main", [key]) == {}


def test_put_many_rejects_batch_after_provenance_moved(tmp_path):
    """Reviewer repro (round 2): a writer that confirmed commit a must not publish
    its batch after another process re-confirmed the shard under commit b."""
    cache = EmbeddingCache(tmp_path)
    scope = tmp_path / "repo"
    scope.mkdir()
    key_a = embedding_cache.compute_cache_key("some/model", "main", "text-a")
    key_b = embedding_cache.compute_cache_key("some/model", "main", "text-b")
    vector = np.array([1.0, 0.0], dtype=np.float32)

    # Writer A confirms on the empty shard, then stalls during inference.
    assert cache.confirm_source_commit(scope, "some/model", "main", "a" * 40) is True

    # Writer B confirms under commit b and publishes its rows.
    assert cache.confirm_source_commit(scope, "some/model", "main", "b" * 40) is True
    cache.put_many(scope, "some/model", "main", [(key_b, vector)], expected_source_commit="b" * 40)

    # Writer A wakes up and publishes late: the stale batch must be dropped.
    cache.put_many(scope, "some/model", "main", [(key_a, vector)], expected_source_commit="a" * 40)

    hits = cache.get_many(scope, "some/model", "main", [key_a, key_b])
    assert set(hits) == {key_b}
    meta = embedding_cache._read_shard_meta(cache.shard_dir(scope, "some/model", "main"))
    assert meta is not None
    assert meta["source_commit"] == "b" * 40


def test_get_many_with_provenance_returns_the_snapshot_commit(tmp_path):
    cache = EmbeddingCache(tmp_path)
    scope = tmp_path / "repo"
    scope.mkdir()
    key = embedding_cache.compute_cache_key("some/model", "main", "text-a")
    vector = np.array([1.0, 0.0], dtype=np.float32)

    empty = cache.get_many_with_provenance(scope, "some/model", "main", [key])
    assert empty.vectors == {}
    assert empty.source_commit is None

    cache.put_many(scope, "some/model", "main", [(key, vector)], expected_source_commit="a" * 40)
    lookup = cache.get_many_with_provenance(scope, "some/model", "main", [key])
    assert set(lookup.vectors) == {key}
    assert lookup.source_commit == "a" * 40

    # Immutable-revision shards report no provenance: the revision is the truth.
    cache.put_many(scope, "some/model", REVISION_1, [(key, vector)])
    pinned = cache.get_many_with_provenance(scope, "some/model", REVISION_1, [key])
    assert set(pinned.vectors) == {key}
    assert pinned.source_commit is None


def test_get_many_touches_shard_after_interval_elapses(tmp_path, monkeypatch):
    # _touch_shard feeds LRU eviction ordering: a read hit only refreshes the
    # shard's recorded last_used_at once it is more than _TOUCH_INTERVAL_SECONDS
    # stale, throttling index rewrites on a hot read path.
    fake_time = {"value": 1_000_000.0}
    monkeypatch.setattr(embedding_cache.time, "time", lambda: fake_time["value"])
    touch_calls: list[Path] = []
    original_touch = embedding_cache._touch_shard

    def spying_touch(shard_dir: Path) -> None:
        touch_calls.append(shard_dir)
        original_touch(shard_dir)

    monkeypatch.setattr(embedding_cache, "_touch_shard", spying_touch)

    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    vector = np.array([1.0, 2.0], dtype=np.float32)
    cache.put_many(scope, "model-a", "rev1", [("key", vector)])
    shard_dir = cache.shard_dir(scope, "model-a", "rev1")
    written_meta = embedding_cache._read_shard_meta(shard_dir)
    assert written_meta is not None
    assert written_meta["last_used_at"] == fake_time["value"]

    # Advance past the touch interval; the read hit must refresh the stamp.
    fake_time["value"] += embedding_cache._TOUCH_INTERVAL_SECONDS + 1.0
    assert "key" in cache.get_many(scope, "model-a", "rev1", ["key"])

    assert touch_calls == [shard_dir]
    refreshed_meta = embedding_cache._read_shard_meta(shard_dir)
    assert refreshed_meta is not None
    assert refreshed_meta["last_used_at"] == fake_time["value"]
    assert refreshed_meta["last_used_at"] > written_meta["last_used_at"]


def test_get_many_skips_touch_within_interval(tmp_path, monkeypatch):
    fake_time = {"value": 1_000_000.0}
    monkeypatch.setattr(embedding_cache.time, "time", lambda: fake_time["value"])
    touch_calls: list[Path] = []
    monkeypatch.setattr(
        embedding_cache, "_touch_shard", lambda shard_dir: touch_calls.append(shard_dir)
    )

    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    vector = np.array([1.0, 2.0], dtype=np.float32)
    cache.put_many(scope, "model-a", "rev1", [("key", vector)])
    shard_dir = cache.shard_dir(scope, "model-a", "rev1")

    # Advance, but stay strictly under the touch interval: no refresh happens.
    fake_time["value"] += embedding_cache._TOUCH_INTERVAL_SECONDS - 1.0
    assert "key" in cache.get_many(scope, "model-a", "rev1", ["key"])

    assert touch_calls == []
    meta = embedding_cache._read_shard_meta(shard_dir)
    assert meta is not None
    assert meta["last_used_at"] == 1_000_000.0


def test_concurrent_republish_between_lookup_and_load_discards_stale_hits(tmp_path, monkeypatch):
    """Reviewer repro (round 3): partial checkpoint-a hits copied before the model
    load must not survive a concurrent purge-and-republish under checkpoint b, even
    though the shard's current provenance matches the loaded commit."""
    units = _five_units(tmp_path)
    model = CountingModel()
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "a" * 40)

    cache = embedding_cache.get_embedding_cache()
    assert cache is not None
    canonical = semantic.resolve_model_profile("drift-model").canonical_name
    loads = {"count": 0}

    def fake_get_model(*_args, **_kwargs):
        loads["count"] += 1
        if loads["count"] == 2:
            # Concurrent run: sees main move to b upstream, purges the
            # checkpoint-a shard, and publishes its own b generation - all
            # after this run copied its warm hits out of the a snapshot.
            assert cache.confirm_source_commit(tmp_path, canonical, "main", "b" * 40) is False
            cache.put_many(
                tmp_path,
                canonical,
                "main",
                [
                    (
                        embedding_cache.compute_cache_key(canonical, "main", "concurrent-unit"),
                        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    )
                ],
                expected_source_commit="b" * 40,
            )
            # This run's own load then observes the moved branch.
            monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "b" * 40)
        return model

    monkeypatch.setattr(semantic, "get_model", fake_get_model)

    compute_embeddings(units, model_name="drift-model", revision="main", cache_scope=tmp_path)
    assert loads["count"] == 1

    # One local source change gives the next run a genuine miss beside warm hits.
    changed = copy.copy(units[1])
    changed.source = "def other(x):\n    return x + 777\n"
    mixed_units = [units[0], changed, units[2]]

    result = compute_embeddings(
        mixed_units, model_name="drift-model", revision="main", cache_scope=tmp_path
    )

    # confirm_source_commit passes here (the current shard already says b), so
    # only the snapshot provenance carried by the lookup can catch the stale
    # hits: every row must re-encode under b, never two a hits beside one b row.
    assert loads["count"] == 2
    assert len(model.encode_calls[-1]) == 3
    assert result.shape == (3, model.dim)

    meta = embedding_cache._read_shard_meta(cache.shard_dir(tmp_path, canonical, "main"))
    assert meta is not None
    assert meta["source_commit"] == "b" * 40

    # The rebuilt shard is coherent: an identical rerun is fully warm, no load.
    compute_embeddings(mixed_units, model_name="drift-model", revision="main", cache_scope=tmp_path)
    assert loads["count"] == 2


def test_loose_branch_move_never_mixes_two_checkpoints(tmp_path, monkeypatch):
    """Reviewer repro (Issue 4): a branch move plus a partial warm hit must re-embed
    the whole corpus rather than assembling old-commit hits beside new-commit rows."""
    units = _five_units(tmp_path)
    model = CountingModel()
    _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "a" * 40)

    compute_embeddings(units, model_name="drift-model", revision="main", cache_scope=tmp_path)
    assert len(model.encode_calls) == 1

    # Upstream, "main" moves to checkpoint b; locally one source unit changes,
    # so the next run has a genuine miss and must load the model.
    changed = copy.copy(units[1])
    changed.source = "def other(x):\n    return x + 777\n"
    mixed_units = [units[0], changed, units[2]]
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "b" * 40)

    result = compute_embeddings(
        mixed_units, model_name="drift-model", revision="main", cache_scope=tmp_path
    )

    # All three rows were re-embedded under checkpoint b - never one changed
    # row computed by b assembled beside two checkpoint-a cache hits.
    assert len(model.encode_calls) == 2
    assert len(model.encode_calls[-1]) == 3
    assert result.shape == (3, model.dim)

    # The purged shard was rebuilt coherently: a repeat run is fully warm.
    compute_embeddings(mixed_units, model_name="drift-model", revision="main", cache_scope=tmp_path)
    assert len(model.encode_calls) == 2


def test_unreportable_mutable_revision_never_mixes_cached_and_fresh_rows(tmp_path, monkeypatch):
    """A label-keyed run with unknown loaded provenance must bypass its whole shard."""

    class EpochModel(CountingModel):
        def __init__(self) -> None:
            super().__init__(dim=2)
            self.epoch = 0

        def encode(self, texts, **kwargs):
            self.encode_calls.append(list(texts))
            vector = np.array([1.0, 0.0] if self.epoch == 0 else [0.0, 1.0])
            return np.repeat(vector[None, :], len(texts), axis=0)

    units = _five_units(tmp_path)[:3]
    model = EpochModel()
    _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    first = compute_embeddings(
        units, model_name="drift-model", revision="main", cache_scope=tmp_path
    )
    np.testing.assert_array_equal(first, np.tile([1.0, 0.0], (3, 1)))

    model.epoch = 1
    changed = copy.copy(units[1])
    changed.source = "def beta(x):\n    return x + 777\n"
    second = compute_embeddings(
        [units[0], changed, units[2]],
        model_name="drift-model",
        revision="main",
        cache_scope=tmp_path,
    )

    np.testing.assert_array_equal(second, np.tile([0.0, 1.0], (3, 1)))
    assert [len(call) for call in model.encode_calls] == [3, 3]
    assert EmbeddingCache().stats()["entries"] == 0


def test_loose_branch_move_purges_shard_and_aborts_search(tmp_path, monkeypatch):
    units = _five_units(tmp_path)
    model = CountingModel()
    _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "a" * 40)

    embeddings = compute_embeddings(
        units, model_name="drift-model", revision="main", cache_scope=tmp_path
    )

    # The branch moves before the first uncached query: the query load
    # discovers the drift, purges the label shard, and refuses to compare a
    # checkpoint-b query vector against the checkpoint-a corpus matrix.
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "b" * 40)
    with pytest.raises(RuntimeError, match="moved to a different commit"):
        find_similar_to_query(
            "find addition",
            units,
            embeddings,
            model_name="drift-model",
            revision="main",
            cache_scope=tmp_path,
        )

    # The purge emptied the shard, so reindexing re-embeds everything under b.
    compute_embeddings(units, model_name="drift-model", revision="main", cache_scope=tmp_path)
    assert len(model.encode_calls[-1]) == len(units)


def test_republished_shard_query_hit_never_reaches_stale_corpus(tmp_path, monkeypatch):
    """A warm query hit must carry the corpus's checkpoint, not just its dimension.

    A concurrent process can purge and republish the label shard under a new
    commit while a long-lived analyzer still holds the old-commit matrix in
    memory; the republished shard is internally coherent, so only the corpus
    identity's recorded source commit can refuse the comparison.
    """
    units = _five_units(tmp_path)
    model = CountingModel()
    _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "a" * 40)

    embeddings, identity = compute_embeddings_with_identity(
        units, model_name="drift-model", revision="main", cache_scope=tmp_path
    )
    assert identity.source_commit == "a" * 40
    find_similar_to_query(
        "find addition",
        units,
        embeddings,
        model_name="drift-model",
        revision="main",
        cache_scope=tmp_path,
        corpus_identity=identity,
    )

    # A concurrent process loads commit b, purges the drifted shard, and
    # republishes everything - code rows and the same query vector - under b.
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "b" * 40)
    cache = embedding_cache.get_embedding_cache()
    assert cache is not None
    canonical = semantic.resolve_model_profile("drift-model").canonical_name
    assert cache.confirm_source_commit(tmp_path, canonical, "main", "b" * 40) is False
    fresh_embeddings, fresh_identity = compute_embeddings_with_identity(
        units, model_name="drift-model", revision="main", cache_scope=tmp_path
    )
    assert fresh_identity.source_commit == "b" * 40
    find_similar_to_query(
        "find addition",
        units,
        fresh_embeddings,
        model_name="drift-model",
        revision="main",
        cache_scope=tmp_path,
        corpus_identity=fresh_identity,
    )

    # Provenance is metadata, not identity: the label-keyed coordinate-system
    # policy is unchanged, so only the source-commit comparison stands between
    # the commit-b query vector and the commit-a matrix.
    assert fresh_identity == identity
    with pytest.raises(RuntimeError, match="moved to a different commit"):
        find_similar_to_query(
            "find addition",
            units,
            embeddings,
            model_name="drift-model",
            revision="main",
            cache_scope=tmp_path,
            corpus_identity=identity,
        )


def test_loose_corpus_writes_stamp_the_loaded_commit(tmp_path, monkeypatch):
    """The semantic write path must hand its loaded commit to the cache, so the
    TOCTOU rejection in put_many has a truth to compare against."""
    units = _five_units(tmp_path)
    model = CountingModel()
    _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "a" * 40)

    compute_embeddings(units, model_name="drift-model", revision="main", cache_scope=tmp_path)

    cache = embedding_cache.get_embedding_cache()
    assert cache is not None
    canonical = semantic.resolve_model_profile("drift-model").canonical_name
    meta = embedding_cache._read_shard_meta(cache.shard_dir(tmp_path, canonical, "main"))
    assert meta is not None
    assert meta["source_commit"] == "a" * 40


def test_pinned_revision_shards_skip_the_source_commit_guard(tmp_path, monkeypatch):
    units = _five_units(tmp_path)
    model = CountingModel()
    _patch_get_model(monkeypatch, model)
    # Immutable commit-hash keys can never drift, whatever the backend reports.
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: REVISION_2)

    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)

    changed = copy.copy(units[1])
    changed.source = "def other(x):\n    return x + 777\n"
    result = compute_embeddings(
        [units[0], changed, units[2]],
        model_name="test-model",
        revision=REVISION_1,
        cache_scope=tmp_path,
    )

    assert len(model.encode_calls[-1]) == 1
    assert result.shape == (3, model.dim)


def test_revision_is_mutable_label_classification(tmp_path, monkeypatch):
    assert semantic._revision_is_mutable_label("test-model", "main") is True
    assert semantic._revision_is_mutable_label("test-model", REVISION_1) is False
    assert semantic._revision_is_mutable_label("test-model", None) is False

    monkeypatch.setattr(semantic, "resolve_local_model_path", lambda _name: tmp_path)
    assert semantic._revision_is_mutable_label("test-model", "content-fingerprint") is False


def test_repeated_identical_search_skips_model_load_when_corpus_cached(tmp_path, monkeypatch):
    units = _five_units(tmp_path)
    model = CountingModel()
    get_model_counts = _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    embeddings = compute_embeddings(
        units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path
    )
    assert get_model_counts["count"] == 1

    first_hits = find_similar_to_query(
        "find addition",
        units,
        embeddings,
        model_name="test-model",
        revision=REVISION_1,
        cache_scope=tmp_path,
        top_k=3,
    )
    assert get_model_counts["count"] == 2
    assert len(model.encode_calls) == 2

    second_hits = find_similar_to_query(
        "find addition",
        units,
        embeddings,
        model_name="test-model",
        revision=REVISION_1,
        cache_scope=tmp_path,
        top_k=3,
    )
    assert get_model_counts["count"] == 2
    assert len(model.encode_calls) == 2
    assert [unit.qualified_name for unit, _score in first_hits] == [
        unit.qualified_name for unit, _score in second_hits
    ]


def test_corrupt_vectors_file_recomputes_without_crash(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(embedding_cache, "_warned_cache_error", False)
    units = _five_units(tmp_path)
    model = CountingModel()
    _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 1

    cache = EmbeddingCache()
    shard_dir = cache.shard_dir(tmp_path, "test-model", REVISION_1)
    original_vectors_path = _active_vectors_path(shard_dir)
    original_vectors_path.write_bytes(b"garbage, not a valid npy file")

    with caplog.at_level("WARNING"):
        result = compute_embeddings(
            units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path
        )

    assert result.shape == (5, model.dim)
    assert len(model.encode_calls) == 2
    assert "Embedding cache" in caplog.text
    rebuilt_vectors_path = _active_vectors_path(shard_dir)
    assert rebuilt_vectors_path != original_vectors_path
    assert rebuilt_vectors_path.exists()
    assert not original_vectors_path.exists()


def test_reader_discards_shard_replaced_during_vector_load(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_cache, "_warned_cache_error", False)
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    first = np.array([1.0, 2.0], dtype=np.float32)
    second = np.array([3.0, 4.0], dtype=np.float32)
    cache.put_many(scope, "model-a", "rev1", [("first", first)])

    original_load = embedding_cache.np.load
    raced = False

    def racing_load(*args, **kwargs):
        nonlocal raced
        vectors = original_load(*args, **kwargs)
        if not raced:
            raced = True
            cache.put_many(scope, "model-a", "rev1", [("second", second)])
        return vectors

    monkeypatch.setattr(embedding_cache.np, "load", racing_load)
    assert cache.get_many(scope, "model-a", "rev1", ["first"]) == {}

    monkeypatch.setattr(embedding_cache.np, "load", original_load)
    hits = cache.get_many(scope, "model-a", "rev1", ["first", "second"])
    np.testing.assert_array_equal(hits["first"], first)
    np.testing.assert_array_equal(hits["second"], second)


def test_reader_treats_whole_shard_deletion_during_vector_load_as_miss(tmp_path, monkeypatch):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    vector = np.array([1.0, 2.0], dtype=np.float32)
    cache.put_many(scope, "model-a", "rev1", [("key", vector)])
    shard_dir = cache.shard_dir(scope, "model-a", "rev1")

    original_load = embedding_cache.np.load

    def deleting_load(*args, **kwargs):
        vectors = original_load(*args, **kwargs)
        assert embedding_cache._delete_cache_tree(shard_dir, action="test eviction").removed is True
        return vectors

    monkeypatch.setattr(embedding_cache.np, "load", deleting_load)

    assert cache.get_many(scope, "model-a", "rev1", ["key"]) == {}


def test_stats_and_eviction_continue_after_one_shard_vanishes(tmp_path, monkeypatch):
    cache = EmbeddingCache()
    first_scope = tmp_path / "first"
    second_scope = tmp_path / "second"
    first_scope.mkdir()
    second_scope.mkdir()
    cache.put_many(
        first_scope,
        "model-a",
        "rev1",
        [("first", np.zeros(256, dtype=np.float32))],
    )
    cache.put_many(
        second_scope,
        "model-b",
        "rev1",
        [("second", np.ones(256, dtype=np.float32))],
    )
    vanished = cache.shard_dir(first_scope, "model-a", "rev1")
    surviving = cache.shard_dir(second_scope, "model-b", "rev1")
    original_size = embedding_cache._shard_size_bytes

    def racing_size(shard_dir: Path) -> int:
        if shard_dir == vanished:
            raise FileNotFoundError(shard_dir)
        return original_size(shard_dir)

    monkeypatch.setattr(embedding_cache, "_shard_size_bytes", racing_size)

    stats = cache.stats()
    assert stats["entries"] == 1
    assert stats["models"] == {"model-b": 1}

    monkeypatch.setattr(embedding_cache, "_resolve_max_bytes", lambda: 1)
    embedding_cache._maybe_evict(cache.repos_dir)
    assert not surviving.exists()


def test_stale_index_row_out_of_range_recomputes_without_crash(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(embedding_cache, "_warned_cache_error", False)
    units = _five_units(tmp_path)
    model = CountingModel()
    _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 1

    cache = EmbeddingCache()
    shard_dir = cache.shard_dir(tmp_path, "test-model", REVISION_1)
    index_path = shard_dir / embedding_cache.INDEX_FILENAME
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["keys"] = dict.fromkeys(payload["keys"], 999)
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    with caplog.at_level("WARNING"):
        result = compute_embeddings(
            units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path
        )

    assert result.shape == (5, model.dim)
    assert len(model.encode_calls) == 2
    assert "Embedding cache" in caplog.text


def test_invalid_last_used_metadata_is_a_cache_miss(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(embedding_cache, "_warned_cache_error", False)
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    vector = np.array([1.0, 2.0], dtype=np.float32)
    cache.put_many(scope, "model-a", "rev1", [("key", vector)])

    index_path = cache.shard_dir(scope, "model-a", "rev1") / embedding_cache.INDEX_FILENAME
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["last_used_at"] = "corrupt"
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    with caplog.at_level("WARNING"):
        assert cache.get_many(scope, "model-a", "rev1", ["key"]) == {}

    assert "Embedding cache read shard failed" in caplog.text


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("keys", 3),
        ("last_used_at", True),
        ("last_used_at", float("nan")),
        ("dim", False),
    ],
)
def test_malformed_metadata_is_safe_for_stats_and_clear(tmp_path, field, invalid_value):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    cache.put_many(
        scope,
        "model-a",
        "rev1",
        [("key", np.array([1.0, 2.0], dtype=np.float32))],
    )
    shard_dir = cache.shard_dir(scope, "model-a", "rev1")
    index_path = shard_dir / embedding_cache.INDEX_FILENAME
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload[field] = invalid_value
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    stats = cache.stats()
    assert stats["entries"] == 0
    assert stats["models"] == {}
    assert cache.clear().removed_entries == 0
    assert not shard_dir.exists()


def test_use_cache_false_creates_no_cache_files(tmp_path, monkeypatch):
    units = _five_units(tmp_path)
    model = CountingModel()
    _patch_get_model(monkeypatch, model)

    compute_embeddings(
        units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path, use_cache=False
    )
    assert not embedding_cache.resolve_cache_dir().exists()


def test_codedupes_no_cache_env_creates_no_cache_files(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEDUPES_NO_CACHE", "1")
    units = _five_units(tmp_path)
    model = CountingModel()
    _patch_get_model(monkeypatch, model)

    compute_embeddings(
        units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path, use_cache=True
    )
    assert not embedding_cache.resolve_cache_dir().exists()


def test_size_cap_prunes_least_recently_used_shards(tmp_path, monkeypatch):
    counter = itertools.count()
    monkeypatch.setattr(embedding_cache.time, "time", lambda: next(counter))
    monkeypatch.setattr(embedding_cache, "_resolve_max_bytes", lambda: 4000)

    cache = EmbeddingCache()
    dim = 256
    for i in range(6):
        scope = tmp_path / f"proj{i}"
        scope.mkdir()
        vector = np.full(dim, float(i), dtype=np.float32)
        cache.put_many(scope, "model-x", "rev1", [(f"key{i}", vector)])

    stats = cache.stats()
    assert stats["size_bytes"] <= 4000
    assert len(stats["repos"]) < 6


def test_size_cap_preserves_fresh_shard_larger_than_cap(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(embedding_cache, "_resolve_max_bytes", lambda: 1000)
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    vector = np.zeros(256, dtype=np.float32)

    with caplog.at_level("WARNING"):
        cache.put_many(scope, "model-x", "rev1", [("key", vector)])

    np.testing.assert_array_equal(
        cache.get_many(scope, "model-x", "rev1", ["key"])["key"],
        vector,
    )
    assert cache.stats()["size_bytes"] > 1000
    assert "still exceeds its size target after eviction" in caplog.text


def test_size_cap_keeps_failed_deletion_in_total(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(embedding_cache, "_warned_cache_error", False)
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    shard_dir = cache.shard_dir(scope, "model-x", "rev1")
    cache.put_many(
        scope,
        "model-x",
        "rev1",
        [("key", np.zeros(256, dtype=np.float32))],
    )
    monkeypatch.setattr(embedding_cache, "_resolve_max_bytes", lambda: 1)

    def fail_delete(path, *_args, **_kwargs):
        if Path(path) == shard_dir:
            raise PermissionError("read-only cache shard")

    monkeypatch.setattr(embedding_cache.shutil, "rmtree", fail_delete)

    with caplog.at_level("WARNING"):
        embedding_cache._maybe_evict(cache.repos_dir)

    assert shard_dir.exists()
    assert "Embedding cache evict shard failed" in caplog.text
    assert "still exceeds its size target after eviction" in caplog.text


# "invalid" fails float() itself; "nan" passes float() and hits the isfinite
# rejection — "inf"/"-inf" would exercise that identical branch again. "0",
# "-5", and "0.5" all pass float() and isfinite but must fall back to the
# default like any other unparsable value, rather than being clamped up to a
# thrashing 1 MB cache: "0.5" is the fractional-below-1-MB case (reviewer
# regression — it used to silently clamp to exactly 1 MB instead of rejecting).
@pytest.mark.parametrize("value", ["invalid", "nan", "0", "-5", "0.5"])
def test_invalid_size_cap_uses_default(monkeypatch, value: str):
    monkeypatch.setattr(embedding_cache, "_warned_invalid_cache_max_mb", False)
    monkeypatch.setenv("CODEDUPES_CACHE_MAX_MB", value)

    assert (
        embedding_cache._resolve_max_bytes() == embedding_cache.DEFAULT_CACHE_MAX_MB * 1024 * 1024
    )


def test_non_positive_size_cap_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(embedding_cache, "_warned_invalid_cache_max_mb", False)
    monkeypatch.setenv("CODEDUPES_CACHE_MAX_MB", "0")

    with caplog.at_level("WARNING"):
        embedding_cache._resolve_max_bytes()
        embedding_cache._resolve_max_bytes()

    assert caplog.text.count("CODEDUPES_CACHE_MAX_MB") == 1


def test_fractional_size_cap_below_one_mb_falls_back_with_warning(monkeypatch, caplog):
    # Reviewer fix: "0.5" MB used to silently clamp to a thrashing 1 MB cache,
    # contradicting the docstring's claim that degenerate values are rejected.
    # It must now be rejected through the same warn-once/fallback path as "0".
    monkeypatch.setattr(embedding_cache, "_warned_invalid_cache_max_mb", False)
    monkeypatch.setenv("CODEDUPES_CACHE_MAX_MB", "0.5")

    with caplog.at_level("WARNING"):
        max_bytes = embedding_cache._resolve_max_bytes()

    assert max_bytes == embedding_cache.DEFAULT_CACHE_MAX_MB * 1024 * 1024
    assert "CODEDUPES_CACHE_MAX_MB" in caplog.text


def test_one_mb_size_cap_is_the_accepted_boundary(monkeypatch, caplog):
    # "1" is the smallest value the cap accepts outright: resolves to exactly
    # 1 MB with no fallback and no warning, unlike "0.5" just below it.
    monkeypatch.setattr(embedding_cache, "_warned_invalid_cache_max_mb", False)
    monkeypatch.setenv("CODEDUPES_CACHE_MAX_MB", "1")

    with caplog.at_level("WARNING"):
        max_bytes = embedding_cache._resolve_max_bytes()

    assert max_bytes == 1024 * 1024
    assert "CODEDUPES_CACHE_MAX_MB" not in caplog.text


def test_put_many_retains_keys_absent_from_current_write(tmp_path):
    # A write never treats its own key set as the whole live corpus: keys from
    # other invocations (and other namespaces) survive until eviction.
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()

    def vector(value: float) -> np.ndarray:
        return np.array([value, value + 1.0], dtype=np.float32)

    cache.put_many(
        scope,
        "model-a",
        "rev1",
        [("code-a", vector(1.0)), ("code-b", vector(2.0))],
        namespace="check",
    )
    cache.put_many(scope, "model-a", "rev1", [("query", vector(3.0))], namespace="query")
    for index in range(3):
        cache.put_many(
            scope,
            "model-a",
            "rev1",
            [(f"edited-{index}", vector(10.0 + index))],
            namespace="check",
        )

    all_keys = ["code-a", "code-b", "edited-0", "edited-1", "edited-2", "query"]
    hits = cache.get_many(scope, "model-a", "rev1", all_keys)
    assert set(hits) == set(all_keys)
    assert cache.stats()["entries"] == 6


def test_dimension_change_warns_before_replacing_incompatible_shard(tmp_path, caplog):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    cache.put_many(
        scope,
        "model-a",
        "rev1",
        [
            ("first", np.array([1.0, 2.0], dtype=np.float32)),
            ("second", np.array([3.0, 4.0], dtype=np.float32)),
        ],
    )
    replacement = np.array([5.0, 6.0, 7.0], dtype=np.float32)

    with caplog.at_level("WARNING", logger="codedupes.embedding_cache"):
        cache.put_many(scope, "model-a", "rev1", [("replacement", replacement)])

    hits = cache.get_many(scope, "model-a", "rev1", ["first", "second", "replacement"])
    assert set(hits) == {"replacement"}
    np.testing.assert_array_equal(hits["replacement"], replacement)
    assert "vector dimension changed from 2 to 3" in caplog.text
    assert "replacing all 2 entries" in caplog.text


def test_narrow_invocation_keeps_full_directory_run_warm(tmp_path, monkeypatch):
    # Full directory -> single file -> full directory: the narrow middle run
    # must not evict its siblings' vectors, so the final run encodes nothing.
    units = _five_units(tmp_path)
    model = CountingModel()
    get_model_counts = _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert get_model_counts["count"] == 1
    assert len(model.encode_calls) == 1

    compute_embeddings(
        units[:1],
        model_name="test-model",
        revision=REVISION_1,
        cache_scope=tmp_path,
    )
    full_run = compute_embeddings(
        units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path
    )

    assert full_run.shape == (5, 4)
    assert get_model_counts["count"] == 1
    assert len(model.encode_calls) == 1
    assert EmbeddingCache().stats()["entries"] == 5


def test_get_embedding_cache_degrades_when_construction_raises(monkeypatch, caplog):
    monkeypatch.setattr(embedding_cache, "_warned_cache_error", False)
    monkeypatch.delenv("CODEDUPES_NO_CACHE", raising=False)
    monkeypatch.delenv("CODEDUPES_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    def raising_home():
        raise RuntimeError("Could not determine home directory")

    monkeypatch.setattr(Path, "home", raising_home)

    with caplog.at_level("WARNING"):
        cache = embedding_cache.get_embedding_cache()

    # Construction failure degrades to the same disabled shape as
    # CODEDUPES_NO_CACHE, never raises into the caller (analysis path).
    assert cache is None
    assert "Embedding cache initialize failed" in caplog.text


def test_resolve_cache_dir_env_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv("CODEDUPES_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    assert embedding_cache.resolve_cache_dir() == tmp_path / "home" / ".cache" / "codedupes"

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert embedding_cache.resolve_cache_dir() == tmp_path / "xdg" / "codedupes"

    monkeypatch.setenv("CODEDUPES_CACHE_DIR", str(tmp_path / "explicit"))
    assert embedding_cache.resolve_cache_dir() == tmp_path / "explicit"


def test_clear_scopes_to_one_model(tmp_path):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    cache.put_many(scope, "model-a", "rev1", [("k1", np.array([1.0, 2.0], dtype=np.float32))])
    cache.put_many(scope, "model-b", "rev1", [("k2", np.array([3.0, 4.0], dtype=np.float32))])

    result = cache.clear(model="model-a")
    assert result.removed_entries == 1
    assert result.failed_deletions == 0

    remaining = cache.stats()
    assert remaining["entries"] == 1
    assert remaining["models"] == {"model-b": 1}


def test_clear_does_not_count_failed_shard_deletion(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(embedding_cache, "_warned_cache_error", False)
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    shard_dir = cache.shard_dir(scope, "model-a", "rev1")
    cache.put_many(
        scope,
        "model-a",
        "rev1",
        [("key", np.array([1.0, 2.0], dtype=np.float32))],
    )
    original_rmtree = embedding_cache.shutil.rmtree

    def fail_shard_delete(path, *args, **kwargs):
        if Path(path) == shard_dir:
            raise PermissionError("read-only cache shard")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(embedding_cache.shutil, "rmtree", fail_shard_delete)

    with caplog.at_level("WARNING"):
        result = cache.clear()

    assert result.removed_entries == 0
    assert result.failed_deletions == 1
    assert shard_dir.exists()
    assert "Embedding cache clear shard failed" in caplog.text


def test_clear_counts_entries_added_before_lock_acquisition(tmp_path, monkeypatch):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    cache.put_many(
        scope,
        "model-a",
        "rev1",
        [("first", np.array([1.0, 2.0], dtype=np.float32))],
    )
    original_lock = embedding_cache._shard_write_lock
    injected_write = False

    @contextlib.contextmanager
    def lock_after_concurrent_write(shard_dir, *, blocking=False):
        nonlocal injected_write
        if blocking and not injected_write:
            injected_write = True
            cache.put_many(
                scope,
                "model-a",
                "rev1",
                [("second", np.array([3.0, 4.0], dtype=np.float32))],
            )
        with original_lock(shard_dir, blocking=blocking) as acquired:
            yield acquired

    monkeypatch.setattr(embedding_cache, "_shard_write_lock", lock_after_concurrent_write)

    result = cache.clear()
    assert result.removed_entries == 2
    assert result.failed_deletions == 0


def test_strict_unconfirmable_loaded_revision_disables_cache(tmp_path, monkeypatch):
    units = _five_units(tmp_path)
    model = CountingModel()
    get_model_counts = _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", lambda _model: "rev-a")
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    first = compute_embeddings(
        units, model_name="test-model", cache_scope=tmp_path, strict_revision_cache=True
    )
    assert len(model.encode_calls) == 1

    second = compute_embeddings(
        units, model_name="test-model", cache_scope=tmp_path, strict_revision_cache=True
    )
    assert get_model_counts["count"] == 2
    assert len(model.encode_calls) == 2
    assert EmbeddingCache().stats()["entries"] == 0
    np.testing.assert_array_equal(first, second)


def test_finite_on_disk_mutation_degrades_to_per_key_miss(tmp_path):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    entries = [
        ("k1", np.array([1.0, 0.0], dtype=np.float32)),
        ("k2", np.array([0.0, 1.0], dtype=np.float32)),
    ]
    cache.put_many(scope, "model-a", "rev1", entries)
    shard_dir = cache.shard_dir(scope, "model-a", "rev1")

    # Corrupt k1's stored row with different *finite* values, bypassing put_many.
    payload = json.loads((shard_dir / embedding_cache.INDEX_FILENAME).read_text(encoding="utf-8"))
    vectors_path = shard_dir / embedding_cache._vectors_filename(payload["generation"])
    vectors = np.load(vectors_path, allow_pickle=False)
    vectors[payload["keys"]["k1"]] = np.array([0.5, 0.5], dtype=np.float32)
    with open(vectors_path, "wb") as handle:
        np.save(handle, vectors)

    hits = cache.get_many(scope, "model-a", "rev1", ["k1", "k2"])
    assert "k1" not in hits
    np.testing.assert_array_equal(hits["k2"], entries[1][1])

    # A recompute for the corrupted key heals it in place.
    cache.put_many(scope, "model-a", "rev1", [("k1", entries[0][1])])
    healed = cache.get_many(scope, "model-a", "rev1", ["k1", "k2"])
    np.testing.assert_array_equal(healed["k1"], entries[0][1])


def test_nonfinite_cached_vector_treated_as_miss(tmp_path):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    poisoned = np.array([1.0, float("nan")], dtype=np.float32)
    healthy = np.array([3.0, 4.0], dtype=np.float32)
    cache.put_many(scope, "model-a", "rev1", [("bad", poisoned), ("good", healthy)])

    hits = cache.get_many(scope, "model-a", "rev1", ["bad", "good"])
    assert "bad" not in hits
    np.testing.assert_array_equal(hits["good"], healthy)


def test_dim_mismatched_hits_after_revision_correction_recover(tmp_path, monkeypatch):
    units = _five_units(tmp_path)
    model = CountingModel(dim=4)
    _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", lambda _model: "rev-a")
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "rev-b")

    stale_text = units[0].source.strip()
    stale_variant = semantic._cache_variant_for(
        semantic.resolve_model_profile("test-model"),
        "auto",
        semantic.resolve_encode_plan("test-model", mode="code"),
        mps_fallback=None,
    )
    stale_key = embedding_cache.compute_cache_key(
        "test-model", "rev-b", stale_text, variant=stale_variant
    )
    EmbeddingCache().put_many(
        tmp_path, "test-model", "rev-b", [(stale_key, np.array([9.0, 9.0], dtype=np.float32))]
    )

    result = compute_embeddings(units, model_name="test-model", cache_scope=tmp_path)
    assert result.shape == (5, 4)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[0], _vector_for_text(stale_text))


def test_put_many_skips_write_when_shard_lock_held(tmp_path):
    fcntl = pytest.importorskip("fcntl")
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    shard_dir = cache.shard_dir(scope, "model-a", "rev1")
    shard_dir.mkdir(parents=True)
    lock_path = embedding_cache._shard_lock_path(shard_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        cache.put_many(scope, "model-a", "rev1", [("k1", np.array([1.0], dtype=np.float32))])
        assert cache.get_many(scope, "model-a", "rev1", ["k1"]) == {}
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    cache.put_many(scope, "model-a", "rev1", [("k1", np.array([1.0], dtype=np.float32))])
    assert "k1" in cache.get_many(scope, "model-a", "rev1", ["k1"])


def _fake_local_model_dir(tmp_path: Path, name: str = "gemma-work-copy") -> Path:
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}')
    (model_dir / "model.safetensors").write_text("weights-v1")
    return model_dir


def test_local_model_dir_cache_uses_fingerprint_not_revision(tmp_path, monkeypatch):
    units = _five_units(tmp_path)
    model = CountingModel()
    get_model_counts = _patch_get_model(monkeypatch, model)
    model_dir = _fake_local_model_dir(tmp_path)

    compute_embeddings(
        units,
        model_name=str(model_dir),
        revision="requested-revision-a",
        cache_scope=tmp_path,
    )
    assert get_model_counts["count"] == 1
    assert len(model.encode_calls) == 1

    second = compute_embeddings(
        units,
        model_name=str(model_dir),
        revision="requested-revision-b",
        cache_scope=tmp_path,
    )
    assert get_model_counts["count"] == 1
    assert len(model.encode_calls) == 1
    assert second.shape == (5, 4)

    # Replacing the weights in place must change the fingerprint revision and
    # invalidate every cached vector for this model directory.
    (model_dir / "model.safetensors").write_text("weights-v2-longer")
    compute_embeddings(
        units,
        model_name=str(model_dir),
        revision="requested-revision-c",
        cache_scope=tmp_path,
    )
    assert get_model_counts["count"] == 2
    assert len(model.encode_calls) == 2
    assert len(model.encode_calls[-1]) == 5


def test_local_model_dir_relative_and_absolute_share_cache(tmp_path, monkeypatch):
    units = _five_units(tmp_path)
    model = CountingModel()
    get_model_counts = _patch_get_model(monkeypatch, model)
    model_dir = _fake_local_model_dir(tmp_path)

    compute_embeddings(units, model_name=str(model_dir), cache_scope=tmp_path)
    assert get_model_counts["count"] == 1

    monkeypatch.chdir(tmp_path)
    compute_embeddings(units, model_name="./gemma-work-copy", cache_scope=tmp_path)
    assert get_model_counts["count"] == 1
    assert len(model.encode_calls) == 1


def test_fingerprint_local_model_dir_stability_and_edge_cases(tmp_path):
    model_dir = _fake_local_model_dir(tmp_path)
    first = semantic._fingerprint_local_model_dir(model_dir)
    assert first is not None and first.startswith("dir-")
    assert semantic._fingerprint_local_model_dir(model_dir) == first

    hf_metadata = model_dir / ".cache" / "huggingface" / "download"
    hf_metadata.mkdir(parents=True)
    (hf_metadata / "config.json.metadata").write_text("updated-download-metadata")
    assert semantic._fingerprint_local_model_dir(model_dir) == first

    empty = tmp_path / "empty-model"
    empty.mkdir()
    assert semantic._fingerprint_local_model_dir(empty) is None


def test_local_model_content_change_with_preserved_size_and_mtime_invalidates(
    tmp_path, monkeypatch
):
    # A byte-for-byte-length rewrite with a restored mtime defeats a
    # metadata-only fingerprint; the content-backed fingerprint must still miss.
    units = _five_units(tmp_path)
    model = CountingModel()
    get_model_counts = _patch_get_model(monkeypatch, model)
    model_dir = _fake_local_model_dir(tmp_path)
    weights_path = model_dir / "model.safetensors"
    original_stat = weights_path.stat()

    compute_embeddings(units, model_name=str(model_dir), cache_scope=tmp_path)
    assert get_model_counts["count"] == 1
    assert len(model.encode_calls) == 1

    weights_path.write_text("weights-v2")
    os.utime(weights_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    rewritten_stat = weights_path.stat()
    assert rewritten_stat.st_size == original_stat.st_size
    assert rewritten_stat.st_mtime_ns == original_stat.st_mtime_ns

    compute_embeddings(units, model_name=str(model_dir), cache_scope=tmp_path)
    assert get_model_counts["count"] == 2
    assert len(model.encode_calls) == 2
    assert len(model.encode_calls[-1]) == 5


def test_local_model_swap_during_load_discards_preload_hits(tmp_path, monkeypatch):
    # Weights swapped between key derivation and model load must not let
    # vectors cached for the old weights survive into the new model's matrix.
    units = _five_units(tmp_path)
    model = CountingModel()
    model_dir = _fake_local_model_dir(tmp_path)
    weights_path = model_dir / "model.safetensors"
    get_model_counts = {"count": 0}

    def swapping_get_model(*_args, **_kwargs):
        get_model_counts["count"] += 1
        if get_model_counts["count"] == 2:
            weights_path.write_text("weights-v2-swapped-mid-load")
        return model

    monkeypatch.setattr(semantic, "get_model", swapping_get_model)

    compute_embeddings(units, model_name=str(model_dir), cache_scope=tmp_path)
    assert len(model.encode_calls) == 1

    extra_source = "def brand_new():\n    return 99\n"
    all_units = units + extract_units(tmp_path, extra_source, filename="extra.py")
    second = compute_embeddings(all_units, model_name=str(model_dir), cache_scope=tmp_path)
    assert second.shape[0] == 6
    assert get_model_counts["count"] == 2
    # All six units re-embed with the freshly loaded weights; zero stale reuse.
    assert len(model.encode_calls[-1]) == 6

    third = compute_embeddings(all_units, model_name=str(model_dir), cache_scope=tmp_path)
    assert third.shape[0] == 6
    assert get_model_counts["count"] == 2
    assert len(model.encode_calls) == 2


def test_fingerprint_manifest_reuses_digests_and_ignores_touch(tmp_path, monkeypatch):
    model_dir = _fake_local_model_dir(tmp_path)
    hash_calls = {"count": 0}
    real_hash = semantic._hash_file_content

    def counting_hash(path):
        hash_calls["count"] += 1
        return real_hash(path)

    monkeypatch.setattr(semantic, "_hash_file_content", counting_hash)

    first = semantic._fingerprint_local_model_dir(model_dir)
    assert first is not None
    assert hash_calls["count"] == 2

    assert semantic._fingerprint_local_model_dir(model_dir) == first
    assert hash_calls["count"] == 2

    # A metadata-only touch rehashes that file, but the fingerprint is
    # content-based so cached vectors survive.
    os.utime(model_dir / "model.safetensors")
    assert semantic._fingerprint_local_model_dir(model_dir) == first
    assert hash_calls["count"] == 3

    # The on-disk manifest keeps a fresh process cheap: drop the in-process
    # memo and confirm nothing is rehashed.
    with semantic._local_model_manifest_lock:
        semantic._local_model_manifest_memo.pop(str(model_dir), None)
    assert semantic._fingerprint_local_model_dir(model_dir) == first
    assert hash_calls["count"] == 3

    (model_dir / "model.safetensors").write_text("weights-v2-longer")
    assert semantic._fingerprint_local_model_dir(model_dir) != first


def test_model_slug_for_local_paths_is_bounded_and_collision_safe():
    hub = embedding_cache._model_slug("Alibaba-NLP/gte-modernbert-base")
    assert hub == "Alibaba-NLP--gte-modernbert-base"

    deep = "/very/deep/nested/path/to/models/gte-modernbert-base"
    other = "/other/location/gte-modernbert-base"
    slug_a = embedding_cache._model_slug(deep)
    slug_b = embedding_cache._model_slug(other)
    assert slug_a.startswith("local--gte-modernbert-base-")
    assert "/" not in slug_a
    assert slug_a != slug_b
    assert len(slug_a) < 60


def test_duplicate_source_units_share_keys_and_warm_run_full_hits(tmp_path, monkeypatch):
    # Two copies of the same functions in different files collapse to one cache
    # key each; the warm-path coverage check must not confuse unique hits with
    # covered units (regression: IndexError on the second cached run).
    units = _five_units(tmp_path)
    duplicate_units = extract_units(tmp_path, FIVE_FUNCTION_SOURCE, filename="copy.py")
    all_units = units + duplicate_units
    model = CountingModel()
    get_model_counts = _patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    first = compute_embeddings(
        all_units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path
    )
    assert first.shape[0] == 10
    assert get_model_counts["count"] == 1
    assert len(model.encode_calls[0]) == 5

    cache = EmbeddingCache()
    shard_dir = cache.shard_dir(tmp_path, "test-model", REVISION_1)
    vectors = np.load(_active_vectors_path(shard_dir), allow_pickle=False)
    payload = json.loads((shard_dir / embedding_cache.INDEX_FILENAME).read_text(encoding="utf-8"))
    assert vectors.shape[0] == 5
    assert len(payload["keys"]) == 5

    second = compute_embeddings(
        all_units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path
    )
    assert get_model_counts["count"] == 1
    assert len(model.encode_calls) == 1
    np.testing.assert_array_equal(first, second)

    # A partial warm run (one changed unit) alongside duplicate keys must
    # re-encode exactly the changed unit.
    changed = copy.copy(all_units[1])
    changed.source = "def beta(x):\n    return x + 222\n"
    mutated = list(all_units)
    mutated[1] = changed
    compute_embeddings(mutated, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 2
    assert len(model.encode_calls[-1]) == 1


def test_put_many_coalesces_duplicate_keys(tmp_path):
    scope = tmp_path / "project"
    scope.mkdir()
    cache = EmbeddingCache()
    first = np.array([1.0, 2.0], dtype=np.float32)
    replacement = np.array([3.0, 4.0], dtype=np.float32)

    cache.put_many(
        scope,
        "model-a",
        "rev1",
        [("shared", first), ("shared", replacement)],
    )

    shard_dir = cache.shard_dir(scope, "model-a", "rev1")
    vectors = np.load(_active_vectors_path(shard_dir), allow_pickle=False)
    assert vectors.shape == (1, 2)
    np.testing.assert_array_equal(
        cache.get_many(scope, "model-a", "rev1", ["shared"])["shared"],
        replacement,
    )


def test_poisoned_cached_row_is_healed_by_next_put(tmp_path):
    # Reuses the corruption setup from test_nonfinite_cached_vector_treated_as_miss:
    # a NaN-poisoned row is a permanent miss until put_many heals it in place.
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    poisoned = np.array([1.0, float("nan")], dtype=np.float32)
    corrected = np.array([5.0, 6.0], dtype=np.float32)

    cache.put_many(scope, "model-a", "rev1", [("key", poisoned)])
    assert cache.get_many(scope, "model-a", "rev1", ["key"]) == {}

    cache.put_many(scope, "model-a", "rev1", [("key", corrected)])

    hits = cache.get_many(scope, "model-a", "rev1", ["key"])
    np.testing.assert_array_equal(hits["key"], corrected)


def test_put_many_never_overwrites_a_valid_existing_row(tmp_path):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    original = np.array([1.0, 2.0], dtype=np.float32)
    bogus = np.array([float("nan"), float("nan")], dtype=np.float32)

    cache.put_many(scope, "model-a", "rev1", [("key", original)])
    # A second write for the same already-valid key must be a no-op for that
    # row, even if the candidate vector is poisoned: only a stored row that
    # fails the finiteness predicate is eligible for the healing overwrite.
    cache.put_many(scope, "model-a", "rev1", [("key", bogus)])

    hits = cache.get_many(scope, "model-a", "rev1", ["key"])
    np.testing.assert_array_equal(hits["key"], original)


def test_eviction_skips_shard_whose_lock_is_held(tmp_path, monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    monkeypatch.setattr(embedding_cache, "_resolve_max_bytes", lambda: 1000)
    cache = EmbeddingCache()
    dim = 256

    locked_scope = tmp_path / "locked-proj"
    locked_scope.mkdir()
    locked_shard_dir = cache.shard_dir(locked_scope, "model-x", "rev1")
    cache.put_many(locked_scope, "model-x", "rev1", [("k0", np.zeros(dim, dtype=np.float32))])
    assert locked_shard_dir.exists()

    lock_path = embedding_cache._shard_lock_path(locked_shard_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        # Write enough other shards to push the cache well past its tiny cap and
        # force eviction; the locked shard must survive every sweep.
        for i in range(6):
            scope = tmp_path / f"proj{i}"
            scope.mkdir()
            vector = np.full(dim, float(i), dtype=np.float32)
            cache.put_many(scope, "model-x", "rev1", [(f"key{i}", vector)])

        assert locked_shard_dir.exists()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def test_clear_waits_for_held_lock_then_removes_shard(tmp_path):
    fcntl = pytest.importorskip("fcntl")
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    cache.put_many(scope, "model-a", "rev1", [("k1", np.array([1.0], dtype=np.float32))])
    shard_dir = cache.shard_dir(scope, "model-a", "rev1")

    lock_path = embedding_cache._shard_lock_path(shard_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)

    def release_after_delay() -> None:
        time.sleep(0.2)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    releaser = threading.Thread(target=release_after_delay)
    releaser.start()

    result: dict[str, embedding_cache.CacheClearResult] = {}

    def run_clear() -> None:
        result["removed"] = cache.clear()

    clearer = threading.Thread(target=run_clear)
    clearer.start()
    clearer.join(timeout=5)
    releaser.join(timeout=5)

    # Bounds the test: clear() must block-and-wait, not hang forever or skip.
    assert not clearer.is_alive()
    assert result["removed"].removed_entries == 1
    assert result["removed"].failed_deletions == 0
    assert not shard_dir.exists()


def test_shard_deletion_cannot_split_the_advisory_lock_domain(tmp_path):
    pytest.importorskip("fcntl")
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    cache.put_many(scope, "model-a", "rev1", [("k1", np.array([1.0], dtype=np.float32))])
    shard_dir = cache.shard_dir(scope, "model-a", "rev1")

    with embedding_cache._shard_write_lock(shard_dir, blocking=True) as outer_acquired:
        assert outer_acquired is True
        assert embedding_cache._delete_cache_tree(shard_dir, action="test delete").removed is True
        shard_dir.mkdir(parents=True)

        # Recreating a shard directory must not create a fresh lock inode that
        # bypasses the still-held lock for the same logical shard.
        with embedding_cache._shard_write_lock(shard_dir) as inner_acquired:
            assert inner_acquired is False

    assert embedding_cache._shard_lock_path(shard_dir).is_file()


@pytest.mark.parametrize("managed_component", ["repos", "repo", "shard", "locks"])
def test_cache_write_refuses_symlinked_managed_directory(tmp_path, managed_component):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    shard_dir = cache.shard_dir(scope, "model-a", "rev1")
    redirected = tmp_path / "redirected"
    redirected.mkdir()

    if managed_component == "repos":
        cache.cache_root.mkdir(parents=True)
        link_path = cache.repos_dir
    elif managed_component == "repo":
        cache.repos_dir.mkdir(parents=True)
        link_path = shard_dir.parent
    elif managed_component == "shard":
        shard_dir.parent.mkdir(parents=True)
        link_path = shard_dir
    else:
        cache.cache_root.mkdir(parents=True)
        link_path = cache.cache_root / embedding_cache.LOCKS_SUBDIR
    link_path.symlink_to(redirected, target_is_directory=True)

    cache.put_many(
        scope,
        "model-a",
        "rev1",
        [("key", np.array([1.0, 2.0], dtype=np.float32))],
    )

    assert list(redirected.iterdir()) == []


def test_cache_writes_private_directories_and_files(tmp_path):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    cache.put_many(
        scope,
        "model-a",
        "rev1",
        [("key", np.array([1.0, 2.0], dtype=np.float32))],
    )
    shard_dir = cache.shard_dir(scope, "model-a", "rev1")

    directories = (
        cache.cache_root,
        cache.repos_dir,
        shard_dir.parent,
        shard_dir,
        cache.cache_root / embedding_cache.LOCKS_SUBDIR,
    )
    files = (
        shard_dir / embedding_cache.INDEX_FILENAME,
        _active_vectors_path(shard_dir),
        embedding_cache._shard_lock_path(shard_dir),
    )

    assert all(stat.S_IMODE(path.stat().st_mode) == 0o700 for path in directories)
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in files)


def test_orphaned_tmp_file_reclaimed_by_next_write(tmp_path):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    cache.put_many(scope, "model-a", "rev1", [("k1", np.array([1.0, 2.0], dtype=np.float32))])
    shard_dir = cache.shard_dir(scope, "model-a", "rev1")

    orphan = shard_dir / f"vectors-deadbeef.npy{embedding_cache._tmp_suffix()}"
    orphan.write_bytes(b"leftover from a writer that never reached its own cleanup")
    assert orphan.exists()

    cache.put_many(scope, "model-a", "rev1", [("k2", np.array([3.0, 4.0], dtype=np.float32))])

    assert not orphan.exists()


def test_unpublished_vector_generation_reclaimed_by_next_write_attempt(tmp_path):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    vector = np.array([1.0, 2.0], dtype=np.float32)
    cache.put_many(scope, "model-a", "rev1", [("k1", vector)])
    shard_dir = cache.shard_dir(scope, "model-a", "rev1")
    active_vectors = _active_vectors_path(shard_dir)

    # Reproduce a crash after the generation rename but before index publication.
    orphan = shard_dir / "vectors-deadbeefdeadbeefdeadbeefdeadbeef.npy"
    np.save(orphan, np.array([[9.0, 9.0]], dtype=np.float32))
    assert orphan.exists()

    # The incoming row is already valid, so this exercises cleanup even when the
    # write attempt does not need to publish a replacement generation.
    cache.put_many(scope, "model-a", "rev1", [("k1", vector)])

    assert active_vectors.exists()
    assert not orphan.exists()
    np.testing.assert_array_equal(cache.get_many(scope, "model-a", "rev1", ["k1"])["k1"], vector)


def test_max_namespace_keys_drops_oldest_and_spares_other_namespaces(tmp_path):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()

    for i in range(5):
        cache.put_many(
            scope,
            "model-a",
            "rev1",
            [(f"query-{i}", np.array([float(i), float(i) + 1.0], dtype=np.float32))],
            namespace="query",
            max_namespace_keys=3,
        )

    all_query_keys = [f"query-{i}" for i in range(5)]
    hits = cache.get_many(scope, "model-a", "rev1", all_query_keys)
    assert set(hits) == {"query-2", "query-3", "query-4"}

    # A key in a different namespace is unaffected by another namespace's cap.
    cache.put_many(
        scope,
        "model-a",
        "rev1",
        [("code-a", np.array([9.0, 9.0], dtype=np.float32))],
        namespace="check",
    )
    hits_with_code = cache.get_many(scope, "model-a", "rev1", [*all_query_keys, "code-a"])
    assert set(hits_with_code) == {"code-a", "query-2", "query-3", "query-4"}


def test_namespace_cap_amortizes_matrix_compaction(tmp_path, monkeypatch):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    cache.put_many(
        scope,
        "model-a",
        "rev1",
        [("code", np.array([9.0, 9.0], dtype=np.float32))],
        namespace="check",
    )
    rebuild_count = 0
    original_rebuild = embedding_cache._rebuild_matrix_retaining

    def recording_rebuild(*args, **kwargs):
        nonlocal rebuild_count
        rebuild_count += 1
        return original_rebuild(*args, **kwargs)

    monkeypatch.setattr(embedding_cache, "_rebuild_matrix_retaining", recording_rebuild)

    for index in range(7):
        cache.put_many(
            scope,
            "model-a",
            "rev1",
            [(f"query-{index}", np.array([float(index), 1.0], dtype=np.float32))],
            namespace="query",
            max_namespace_keys=5,
        )

    hits = cache.get_many(
        scope,
        "model-a",
        "rev1",
        ["code", *(f"query-{index}" for index in range(7))],
    )
    assert set(hits) == {"code", "query-2", "query-3", "query-4", "query-5", "query-6"}
    assert rebuild_count == 1


@pytest.mark.skipif(os.geteuid() == 0, reason="permission bits do not bind as root")
def test_get_many_degrades_to_miss_on_unreadable_shard(tmp_path):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    vector = np.array([1.0, 2.0], dtype=np.float32)
    cache.put_many(scope, "model-a", "rev1", [("key", vector)])
    shard_dir = cache.shard_dir(scope, "model-a", "rev1")

    shard_dir.chmod(0o000)
    try:
        assert cache.get_many(scope, "model-a", "rev1", ["key"]) == {}
    finally:
        shard_dir.chmod(0o700)

    hits = cache.get_many(scope, "model-a", "rev1", ["key"])
    np.testing.assert_array_equal(hits["key"], vector)


@pytest.mark.skipif(os.geteuid() == 0, reason="permission bits do not bind as root")
def test_clear_continues_past_unreadable_shard(tmp_path):
    cache = EmbeddingCache()
    scopes = []
    for name in ("aaa", "bbb", "ccc"):
        scope = tmp_path / name
        scope.mkdir()
        scopes.append(scope)
        cache.put_many(scope, "model-a", "rev1", [(name, np.array([1.0, 2.0], dtype=np.float32))])
    shard_dirs = [cache.shard_dir(scope, "model-a", "rev1") for scope in scopes]

    # Repo directories sweep in sorted order, so blocking the middle shard
    # proves the sweep continued past a failure rather than never reaching it.
    shard_dirs[1].chmod(0o000)
    try:
        result = cache.clear()
    finally:
        shard_dirs[1].chmod(0o700)

    assert result.removed_entries == 2
    assert result.failed_deletions == 1
    assert not shard_dirs[0].exists()
    assert shard_dirs[1].exists()
    assert not shard_dirs[2].exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="permission bits do not bind as root")
def test_eviction_survives_unreadable_shard(tmp_path, monkeypatch):
    cache = EmbeddingCache()
    readable_scope = tmp_path / "readable"
    blocked_scope = tmp_path / "blocked"
    readable_scope.mkdir()
    blocked_scope.mkdir()
    cache.put_many(
        readable_scope,
        "model-a",
        "rev1",
        [("r", np.zeros(256, dtype=np.float32))],
    )
    cache.put_many(
        blocked_scope,
        "model-b",
        "rev1",
        [("b", np.ones(256, dtype=np.float32))],
    )
    readable_shard = cache.shard_dir(readable_scope, "model-a", "rev1")
    blocked_shard = cache.shard_dir(blocked_scope, "model-b", "rev1")
    monkeypatch.setattr(embedding_cache, "_resolve_max_bytes", lambda: 1)

    blocked_shard.chmod(0o000)
    try:
        embedding_cache._maybe_evict(cache.repos_dir)
    finally:
        blocked_shard.chmod(0o700)

    assert not readable_shard.exists()
    assert blocked_shard.exists()


def test_hostile_deeply_nested_index_degrades_for_stats_and_clear(tmp_path):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    cache.put_many(scope, "model-a", "rev1", [("key", np.array([1.0, 2.0], dtype=np.float32))])
    shard_dir = cache.shard_dir(scope, "model-a", "rev1")
    depth = 100_000
    (shard_dir / embedding_cache.INDEX_FILENAME).write_text(
        "[" * depth + "]" * depth, encoding="utf-8"
    )

    stats = cache.stats()
    assert stats["entries"] == 0

    result = cache.clear()
    assert result.removed_entries == 0
    assert result.failed_deletions == 0
    assert not shard_dir.exists()


def test_eviction_scan_throttled_across_put_many_calls_but_still_enforces_cap(
    tmp_path, monkeypatch
):
    # Reviewer fix: put_many used to pay for a full cache-tree scan on every
    # call, making every single-entry query-cache miss O(total cache size).
    fake_time = {"value": 1_000_000.0}
    monkeypatch.setattr(embedding_cache.time, "time", lambda: fake_time["value"])
    monkeypatch.setattr(embedding_cache, "_resolve_max_bytes", lambda: 100_000)
    cache = EmbeddingCache()
    scan_calls: list[Path] = []
    original_evict = embedding_cache._maybe_evict

    def spying_evict(repos_dir, protect=None):
        scan_calls.append(repos_dir)
        original_evict(repos_dir, protect=protect)

    monkeypatch.setattr(embedding_cache, "_maybe_evict", spying_evict)

    dim = 64  # 256 bytes/vector, well under the 2% (2000 byte) scan threshold.
    for i in range(5):
        scope = tmp_path / f"proj{i}"
        scope.mkdir()
        cache.put_many(scope, "model-x", "rev1", [(f"key{i}", np.zeros(dim, dtype=np.float32))])

    # The first call always scans (nothing scanned yet for this cache root);
    # every later write stays under both the time and byte gates, so none of
    # them should pay for a rescan of the whole cache tree.
    assert len(scan_calls) == 1

    # Once the scan interval elapses the next write must scan again, so the
    # cap is still enforced eventually rather than postponed forever.
    fake_time["value"] += embedding_cache._EVICT_SCAN_INTERVAL_SECONDS + 1.0
    last_scope = tmp_path / "proj-last"
    last_scope.mkdir()
    cache.put_many(last_scope, "model-x", "rev1", [("key-last", np.zeros(dim, dtype=np.float32))])
    assert len(scan_calls) == 2


def test_eviction_scan_throttle_also_trips_on_accumulated_write_bytes(tmp_path, monkeypatch):
    # Even with no time elapsed, a burst of writes whose total crosses the
    # byte-ratio gate must force a scan rather than waiting out the interval.
    monkeypatch.setattr(embedding_cache, "_resolve_max_bytes", lambda: 10_000)
    cache = EmbeddingCache()
    scan_calls: list[Path] = []
    original_evict = embedding_cache._maybe_evict

    def spying_evict(repos_dir, protect=None):
        scan_calls.append(repos_dir)
        original_evict(repos_dir, protect=protect)

    monkeypatch.setattr(embedding_cache, "_maybe_evict", spying_evict)

    # Threshold is 2% of 10_000 = 200 bytes; each 64-float32 vector is 256
    # bytes, so every call after the bootstrap scan should also cross it.
    dim = 64
    for i in range(3):
        scope = tmp_path / f"proj{i}"
        scope.mkdir()
        cache.put_many(scope, "model-x", "rev1", [(f"key{i}", np.zeros(dim, dtype=np.float32))])

    assert len(scan_calls) == 3


def test_read_shard_reuses_cached_snapshot_until_index_replaced(tmp_path, monkeypatch):
    # Reviewer fix: _read_shard used to re-parse index.json and re-mmap the
    # vectors file on every call, even for repeated lookups against an
    # unchanged shard.
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    vector = np.array([1.0, 2.0], dtype=np.float32)
    cache.put_many(scope, "model-a", "rev1", [("key", vector)])
    shard_dir = cache.shard_dir(scope, "model-a", "rev1")

    load_calls = {"count": 0}
    original_load = embedding_cache.np.load

    def counting_load(*args, **kwargs):
        load_calls["count"] += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(embedding_cache.np, "load", counting_load)

    first = embedding_cache._read_shard(shard_dir)
    assert first is not None
    assert load_calls["count"] == 1

    # A second read of the same, unchanged shard must reuse the cached
    # snapshot rather than re-parsing the index and re-mmapping the vectors.
    second = embedding_cache._read_shard(shard_dir)
    assert second is first
    assert load_calls["count"] == 1

    # A write that replaces index.json (a new generation) must invalidate the
    # cached snapshot: the next read observes the new content, never stale data.
    cache.put_many(scope, "model-a", "rev1", [("key2", np.array([3.0, 4.0], dtype=np.float32))])
    third = embedding_cache._read_shard(shard_dir)
    assert third is not None
    assert third is not first
    assert load_calls["count"] == 2
    assert set(third.keys) == {"key", "key2"}


def test_read_shard_cache_invalidated_immediately_on_delete(tmp_path):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    cache.put_many(scope, "model-a", "rev1", [("key", np.array([1.0, 2.0], dtype=np.float32))])
    shard_dir = cache.shard_dir(scope, "model-a", "rev1")

    assert embedding_cache._read_shard(shard_dir) is not None
    assert str(shard_dir) in embedding_cache._shard_read_cache

    result = cache.clear()
    assert result.removed_entries == 1
    assert result.failed_deletions == 0
    assert str(shard_dir) not in embedding_cache._shard_read_cache
    assert embedding_cache._read_shard(shard_dir) is None


def test_atomic_write_shard_rejects_inconsistent_shard_state(tmp_path):
    # Reviewer fix: keys/namespaces/digests used to be mutated as three
    # separate dicts with no atomic setter; a mutation bug could publish a
    # shard where they had drifted apart, only surfacing as a silent
    # invalidate-on-read on some later run. The write path must now fail
    # loudly instead.
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    shard_dir = cache.shard_dir(scope, "model-a", "rev1")
    vectors = np.zeros((1, 2), dtype=np.float32)
    keys_map = {"key": 0}
    namespaces: dict[str, str] = {}  # missing "key": diverged from keys_map.
    digests = {"key": "deadbeef"}

    with pytest.raises(AssertionError, match="diverged"):
        embedding_cache._atomic_write_shard(
            shard_dir, "model-a", "rev1", vectors, keys_map, namespaces, digests, 2, None
        )


def test_shard_data_mutation_methods_keep_keys_namespaces_digests_atomic():
    shard = embedding_cache._ShardData(
        vectors=np.zeros((1, 2), dtype=np.float32),
        keys={"a": 0},
        namespaces={"a": "check"},
        digests={"a": embedding_cache._row_digest(np.zeros(2, dtype=np.float32))},
        last_used_at=0.0,
        generation="0" * 32,
        source_commit=None,
    )
    shard.assert_consistent()

    shard.append_rows([("b", np.array([1.0, 1.0], dtype=np.float32))], "check")
    shard.assert_consistent()
    assert set(shard.keys) == {"a", "b"}
    assert shard.vectors.shape == (2, 2)

    shard.overwrite_rows([("a", np.array([9.0, 9.0], dtype=np.float32))], "check")
    shard.assert_consistent()
    np.testing.assert_array_equal(shard.vectors[shard.keys["a"]], np.array([9.0, 9.0]))

    shard.retain(["b"])
    shard.assert_consistent()
    assert set(shard.keys) == {"b"}
    assert shard.vectors.shape == (1, 2)


def test_public_cache_seams_are_directly_usable(tmp_path):
    # embedding_cache exposes these names for reuse by other modules
    # (semantic.py in particular imports ensure_cache_subdirectory,
    # atomic_write_json, and log_warning_once directly); this covers each one
    # standalone, independent of any caller module.
    managed = embedding_cache.ensure_cache_subdirectory(tmp_path / "root-a", "child")
    assert managed.name == "child"
    assert managed.is_dir()

    payload = {"hello": "world"}
    target = tmp_path / "manifest.json"
    embedding_cache.atomic_write_json(target, payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload

    class _StubExc(Exception):
        pass

    embedding_cache.warn_once("stub action", _StubExc("boom"))


def test_log_warning_once_gates_on_the_given_namespace_and_flag(caplog):
    # semantic.py reuses this helper for its own ad hoc warn-once booleans
    # (_warned_mlx_mps_contention, _warned_cpu_fallback_reuse); the flag must
    # live in and mutate the caller's own namespace (so tests can monkeypatch
    # it directly, e.g. `semantic._warned_mlx_mps_contention = False`) and the
    # message must log through the caller's own logger (so caplog scoped to
    # that logger name still captures it).
    namespace = {"_warned_stub": False}

    with caplog.at_level("WARNING"):
        embedding_cache.log_warning_once(namespace, "_warned_stub", "stub warning one")
        embedding_cache.log_warning_once(namespace, "_warned_stub", "stub warning two")

    assert namespace["_warned_stub"] is True
    assert caplog.text.count("stub warning") == 1
    assert "stub warning one" in caplog.text
