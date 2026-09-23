"""On-disk cache layout and IO: shard files, permissions, snapshot caching, and cache-directory resolution."""

from __future__ import annotations

import copy
import json
import stat
from pathlib import Path

import numpy as np
import pytest

from codedupes import embedding_cache, semantic
from codedupes.embedding_cache import EmbeddingCache
from codedupes.semantic import (
    EmbeddingRunStats,
    compute_embeddings,
)
from tests.conftest import extract_units
from tests.embedding_cache_helpers import (
    FIVE_FUNCTION_SOURCE,
    REVISION_1,
    CountingModel,
    active_vectors_path,
    five_units,
    patch_get_model,
)


def test_partial_update_only_reencodes_changed_unit(tmp_path, monkeypatch):
    units = five_units(tmp_path)
    model = CountingModel()
    get_model_counts = patch_get_model(monkeypatch, model)
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


def test_shuffled_partial_hit_matches_fully_uncached_compute(tmp_path, monkeypatch):
    units = five_units(tmp_path)
    model = CountingModel()
    patch_get_model(monkeypatch, model)
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


def test_use_cache_false_creates_no_cache_files(tmp_path, monkeypatch):
    units = five_units(tmp_path)
    model = CountingModel()
    patch_get_model(monkeypatch, model)

    compute_embeddings(
        units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path, use_cache=False
    )
    assert not embedding_cache.resolve_cache_dir().exists()


def test_codedupes_no_cache_env_creates_no_cache_files(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEDUPES_NO_CACHE", "1")
    units = five_units(tmp_path)
    model = CountingModel()
    patch_get_model(monkeypatch, model)

    compute_embeddings(
        units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path, use_cache=True
    )
    assert not embedding_cache.resolve_cache_dir().exists()


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


def test_narrow_invocation_keeps_full_directory_run_warm(tmp_path, monkeypatch):
    # Full directory -> single file -> full directory: the narrow middle run
    # must not evict its siblings' vectors, so the final run encodes nothing.
    units = five_units(tmp_path)
    model = CountingModel()
    get_model_counts = patch_get_model(monkeypatch, model)
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


def test_duplicate_source_units_share_keys_and_warm_run_full_hits(tmp_path, monkeypatch):
    # Two copies of the same functions in different files collapse to one cache
    # key each; the warm-path coverage check must not confuse unique hits with
    # covered units (regression: IndexError on the second cached run).
    units = five_units(tmp_path)
    duplicate_units = extract_units(tmp_path, FIVE_FUNCTION_SOURCE, filename="copy.py")
    all_units = units + duplicate_units
    model = CountingModel()
    get_model_counts = patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    first = compute_embeddings(
        all_units, model_name="test-model", revision=REVISION_1, cache_scope=tmp_path
    )
    assert first.shape[0] == 10
    assert get_model_counts["count"] == 1
    assert len(model.encode_calls[0]) == 5

    cache = EmbeddingCache()
    shard_dir = cache.shard_dir(tmp_path, "test-model", REVISION_1)
    vectors = np.load(active_vectors_path(shard_dir), allow_pickle=False)
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
    vectors = np.load(active_vectors_path(shard_dir), allow_pickle=False)
    assert vectors.shape == (1, 2)
    np.testing.assert_array_equal(
        cache.get_many(scope, "model-a", "rev1", ["shared"])["shared"],
        replacement,
    )


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
        active_vectors_path(shard_dir),
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
    active_vectors = active_vectors_path(shard_dir)

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
