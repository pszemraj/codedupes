"""Query-vector caching and search-path checkpoint guards."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from codedupes import embedding_cache, semantic
from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from codedupes.embedding_cache import EmbeddingCache
from codedupes.semantic import (
    compute_embeddings,
    compute_embeddings_with_identity,
    find_similar_to_query,
)
from tests.embedding_cache_helpers import (
    REVISION_1,
    CountingModel,
    MidEncodeCpuFallbackModel,
    five_units,
    patch_get_model,
    vector_for_text,
)


def test_strict_query_load_uses_resolved_commit(tmp_path, monkeypatch):
    units = five_units(tmp_path)
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


def test_fast_math_query_aborts_before_mixed_policy_dot_product(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")
    units = five_units(tmp_path)
    model = MidEncodeCpuFallbackModel()
    model.device = "mps"
    patch_get_model(monkeypatch, model)
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


@pytest.mark.parametrize("corpus_cache", ["enabled", "disabled", "environment-disabled"])
@pytest.mark.parametrize("query_cache", [True, False], ids=["cached", "uncached"])
@pytest.mark.parametrize(
    "query_commit", ["a" * 40, "b" * 40, None], ids=["match", "mismatch", "missing"]
)
def test_index_requires_query_checkpoint_before_encoding(
    tmp_path, monkeypatch, corpus_cache, query_cache, query_commit
):
    class CheckpointModel(CountingModel):
        def __getitem__(self, index):
            return SimpleNamespace(
                auto_model=SimpleNamespace(config=SimpleNamespace(_commit_hash=self.commit))
            )

        def encode(self, texts, **kwargs):
            text_list = self._record_encode_call(texts, **kwargs)
            vectors = np.array(
                [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in text_list],
                dtype=np.float32,
            )
            return vectors if self.commit == "a" * 40 else vectors[:, ::-1]

    five_units(tmp_path)
    original = CheckpointModel(dim=2)
    original.commit = "a" * 40
    loads = patch_get_model(monkeypatch, original)
    config = AnalyzerConfig(
        model_name="drift-model",
        model_revision="main",
        device="cpu",
        min_semantic_statements=0,
        embedding_cache=corpus_cache != "disabled",
        strict_revision_cache=False,
    )
    analyzer = CodeAnalyzer(config)
    with monkeypatch.context() as corpus_patch:
        if corpus_cache == "environment-disabled":
            corpus_patch.setenv("CODEDUPES_NO_CACHE", "1")
        assert CodeAnalyzer(config).index(tmp_path) == 5
        assert analyzer.index(tmp_path) == 5
    assert loads["count"] == (1 if corpus_cache == "enabled" else 2)
    assert analyzer._embedding_space_identity.source_commit == original.commit
    semantic.clear_model_cache()
    analyzer.config.embedding_cache = query_cache

    cache = EmbeddingCache()
    shard_dir = cache.shard_dir(tmp_path, "drift-model", "main")
    before = embedding_cache._read_shard_meta(shard_dir)
    write_calls = []
    put_many = EmbeddingCache.put_many

    def record_write(self, *args, **kwargs):
        write_calls.append(kwargs)
        return put_many(self, *args, **kwargs)

    monkeypatch.setattr(EmbeddingCache, "put_many", record_write)
    query_model = CheckpointModel(dim=2)
    query_model.commit = query_commit
    patch_get_model(monkeypatch, query_model)
    if query_commit != original.commit:
        message = "cannot be verified" if query_commit is None else "moved to a different commit"
        with pytest.raises(RuntimeError, match=message):
            analyzer.search("find alpha", top_k=1, threshold=0.0)
        assert query_model.encode_calls == []
        assert write_calls == []
        assert embedding_cache._read_shard_meta(shard_dir) == before
        query_model = original

    recovery_loads = patch_get_model(monkeypatch, query_model)
    assert analyzer.search("find alpha", top_k=1, threshold=0.0)[0][0].name == "alpha"
    assert analyzer.search("find alpha", top_k=1, threshold=0.0)[0][0].name == "alpha"
    assert recovery_loads["count"] == (1 if query_cache else 2)
    assert len(write_calls) == int(query_cache)
    if query_cache:
        assert write_calls[0]["require_source_commit"] is True
        assert write_calls[0]["expected_source_commit"] == original.commit


@pytest.mark.parametrize("revision", ["main", REVISION_1], ids=["mutable", "pinned"])
def test_query_provenance_namespace_only_invalidates_mutable_query_rows(
    tmp_path, monkeypatch, revision
):
    units = five_units(tmp_path)
    model = CountingModel()
    loads = patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "a" * 40)
    embeddings, identity = compute_embeddings_with_identity(
        units,
        model_name="drift-model",
        revision=revision,
        device="cpu",
        cache_scope=tmp_path,
        strict_revision_cache=False,
    )
    profile = semantic.resolve_model_profile("drift-model")
    query = "find addition"
    plan = semantic.resolve_encode_plan("drift-model", mode="query")
    old_variant = semantic._cache_variant_for(profile, "cpu", plan, mps_fallback=None)
    old_key = semantic.compute_cache_key(
        profile.canonical_name,
        revision,
        query,
        mode="query",
        variant=old_variant,
    )
    cache = EmbeddingCache()
    cache.put_many(
        tmp_path,
        profile.canonical_name,
        revision,
        [(old_key, -vector_for_text(query))],
        namespace=semantic._embedding_cache_namespace("query", old_variant),
        expected_source_commit="a" * 40 if revision == "main" else None,
    )

    warm_embeddings, warm_identity = compute_embeddings_with_identity(
        units,
        model_name="drift-model",
        revision=revision,
        device="cpu",
        cache_scope=tmp_path,
        strict_revision_cache=False,
    )
    np.testing.assert_array_equal(warm_embeddings, embeddings)
    assert warm_identity == identity
    assert loads["count"] == 1
    results = find_similar_to_query(
        query,
        units,
        warm_embeddings,
        model_name="drift-model",
        revision=revision,
        device="cpu",
        cache_scope=tmp_path,
        strict_revision_cache=False,
        corpus_identity=warm_identity,
        threshold=-1.0,
    )
    assert (model.encode_calls[-1] == [query]) == (revision == "main")
    assert loads["count"] == (2 if revision == "main" else 1)
    assert all((score > 0.0) == (revision == "main") for _unit, score in results)


def test_republished_shard_query_hit_never_reaches_stale_corpus(tmp_path, monkeypatch):
    """A warm query hit must carry the corpus's checkpoint, not just its dimension.

    A concurrent process can purge and republish the label shard under a new
    commit while a long-lived analyzer still holds the old-commit matrix in
    memory; the republished shard is internally coherent, so only the corpus
    identity's recorded source commit can refuse the comparison.
    """
    units = five_units(tmp_path)
    model = CountingModel()
    patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "a" * 40)

    embeddings, identity = compute_embeddings_with_identity(
        units,
        model_name="drift-model",
        revision="main",
        cache_scope=tmp_path,
        strict_revision_cache=False,
    )
    assert identity.source_commit == "a" * 40
    find_similar_to_query(
        "find addition",
        units,
        embeddings,
        model_name="drift-model",
        revision="main",
        cache_scope=tmp_path,
        strict_revision_cache=False,
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
        units,
        model_name="drift-model",
        revision="main",
        cache_scope=tmp_path,
        strict_revision_cache=False,
    )
    assert fresh_identity.source_commit == "b" * 40
    find_similar_to_query(
        "find addition",
        units,
        fresh_embeddings,
        model_name="drift-model",
        revision="main",
        cache_scope=tmp_path,
        strict_revision_cache=False,
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
            strict_revision_cache=False,
            corpus_identity=identity,
        )


def test_repeated_identical_search_skips_model_load_when_corpus_cached(tmp_path, monkeypatch):
    units = five_units(tmp_path)
    model = CountingModel()
    get_model_counts = patch_get_model(monkeypatch, model)
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
