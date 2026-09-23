"""Corruption and concurrency safeguards for cache reads, writes, and locks."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from codedupes import embedding_cache, semantic
from codedupes.embedding_cache import EmbeddingCache
from codedupes.semantic import (
    compute_embeddings,
    find_similar_to_query,
)
from tests.embedding_cache_helpers import (
    REVISION_1,
    CountingModel,
    active_vectors_path,
    five_units,
    patch_get_model,
)


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


def test_concurrent_republish_between_lookup_and_load_discards_stale_hits(tmp_path, monkeypatch):
    """Reviewer repro (round 3): partial checkpoint-a hits copied before the model
    load must not survive a concurrent purge-and-republish under checkpoint b, even
    though the shard's current provenance matches the loaded commit."""
    units = five_units(tmp_path)
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

    compute_embeddings(
        units,
        model_name="drift-model",
        revision="main",
        cache_scope=tmp_path,
        strict_revision_cache=False,
    )
    assert loads["count"] == 1

    # One local source change gives the next run a genuine miss beside warm hits.
    changed = copy.copy(units[1])
    changed.source = "def other(x):\n    return x + 777\n"
    mixed_units = [units[0], changed, units[2]]

    result = compute_embeddings(
        mixed_units,
        model_name="drift-model",
        revision="main",
        cache_scope=tmp_path,
        strict_revision_cache=False,
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
    compute_embeddings(
        mixed_units,
        model_name="drift-model",
        revision="main",
        cache_scope=tmp_path,
        strict_revision_cache=False,
    )
    assert loads["count"] == 2


def test_loose_branch_move_never_mixes_two_checkpoints(tmp_path, monkeypatch):
    """Reviewer repro (Issue 4): a branch move plus a partial warm hit must re-embed
    the whole corpus rather than assembling old-commit hits beside new-commit rows."""
    units = five_units(tmp_path)
    model = CountingModel()
    patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "a" * 40)

    compute_embeddings(
        units,
        model_name="drift-model",
        revision="main",
        cache_scope=tmp_path,
        strict_revision_cache=False,
    )
    assert len(model.encode_calls) == 1

    # Upstream, "main" moves to checkpoint b; locally one source unit changes,
    # so the next run has a genuine miss and must load the model.
    changed = copy.copy(units[1])
    changed.source = "def other(x):\n    return x + 777\n"
    mixed_units = [units[0], changed, units[2]]
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "b" * 40)

    result = compute_embeddings(
        mixed_units,
        model_name="drift-model",
        revision="main",
        cache_scope=tmp_path,
        strict_revision_cache=False,
    )

    # All three rows were re-embedded under checkpoint b - never one changed
    # row computed by b assembled beside two checkpoint-a cache hits.
    assert len(model.encode_calls) == 2
    assert len(model.encode_calls[-1]) == 3
    assert result.shape == (3, model.dim)

    # The purged shard was rebuilt coherently: a repeat run is fully warm.
    compute_embeddings(
        mixed_units,
        model_name="drift-model",
        revision="main",
        cache_scope=tmp_path,
        strict_revision_cache=False,
    )
    assert len(model.encode_calls) == 2


def test_unreportable_mutable_revision_never_mixes_cached_and_fresh_rows(tmp_path, monkeypatch):
    """A label-keyed run with unknown loaded provenance must bypass its whole shard."""

    class EpochModel(CountingModel):
        def __init__(self) -> None:
            super().__init__(dim=2)
            self.epoch = 0

        def encode(self, texts, **kwargs):
            text_list = self._record_encode_call(texts, **kwargs)
            vector = np.array([1.0, 0.0] if self.epoch == 0 else [0.0, 1.0])
            return np.repeat(vector[None, :], len(text_list), axis=0)

    units = five_units(tmp_path)[:3]
    model = EpochModel()
    patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    first = compute_embeddings(
        units,
        model_name="drift-model",
        revision="main",
        cache_scope=tmp_path,
        strict_revision_cache=False,
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
        strict_revision_cache=False,
    )

    np.testing.assert_array_equal(second, np.tile([0.0, 1.0], (3, 1)))
    assert [len(call) for call in model.encode_calls] == [3, 3]
    assert EmbeddingCache().stats()["entries"] == 0


def test_loose_branch_move_purges_shard_and_aborts_search(tmp_path, monkeypatch):
    units = five_units(tmp_path)
    model = CountingModel()
    patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "a" * 40)

    embeddings = compute_embeddings(
        units,
        model_name="drift-model",
        revision="main",
        cache_scope=tmp_path,
        strict_revision_cache=False,
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
            strict_revision_cache=False,
        )

    # The purge emptied the shard, so reindexing re-embeds everything under b.
    compute_embeddings(
        units,
        model_name="drift-model",
        revision="main",
        cache_scope=tmp_path,
        strict_revision_cache=False,
    )
    assert len(model.encode_calls[-1]) == len(units)


def test_corrupt_vectors_file_recomputes_without_crash(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(embedding_cache, "_warned_cache_error", False)
    units = five_units(tmp_path)
    model = CountingModel()
    patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    compute_embeddings(units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path)
    assert len(model.encode_calls) == 1

    cache = EmbeddingCache()
    shard_dir = cache.shard_dir(tmp_path, "test-model", REVISION_1)
    original_vectors_path = active_vectors_path(shard_dir)
    original_vectors_path.write_bytes(b"garbage, not a valid npy file")

    with caplog.at_level("WARNING"):
        result = compute_embeddings(
            units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path
        )

    assert result.shape == (5, model.dim)
    assert len(model.encode_calls) == 2
    assert "Embedding cache" in caplog.text
    rebuilt_vectors_path = active_vectors_path(shard_dir)
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
    units = five_units(tmp_path)
    model = CountingModel()
    patch_get_model(monkeypatch, model)
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
