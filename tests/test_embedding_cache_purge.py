"""Cache size caps, namespace caps, and clear/eviction behavior."""

from __future__ import annotations

import contextlib
import itertools
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from codedupes import embedding_cache
from codedupes.embedding_cache import EmbeddingCache


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
