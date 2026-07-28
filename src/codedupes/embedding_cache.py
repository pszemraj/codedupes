"""Persistent, content-addressed on-disk cache for semantic embedding vectors.

Cached vectors live under ``<cache_root>/repos/<repo-shard>/<model>@<revision>/`` as a
pair of files: ``vectors.npy`` (a float32 matrix) and ``index.json`` (a key-to-row
map plus metadata). The primary key hashes the model, resolved revision, and the
prepared (pre-truncation) embedding text, so unchanged code units keep hitting the
cache across runs and partial edits only miss for the units that actually changed.
Every public operation is wrapped so on-disk corruption or filesystem errors never
raise into the caller; a shard that cannot be trusted is simply treated as empty and
rebuilt on the next write.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import shutil
import time
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CACHE_SUBDIR = "repos"
VECTORS_FILENAME = "vectors.npy"
INDEX_FILENAME = "index.json"
DEFAULT_CACHE_MAX_MB = 2048
_SCHEMA_VERSION = 1
_PRUNE_TARGET_RATIO = 0.8
_TOUCH_INTERVAL_SECONDS = 3600.0
_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")

_warned_cache_error = False


def _warn_once(action: str, exc: Exception) -> None:
    """Log one process-wide warning for a cache failure, then stay quiet.

    :param action: Short label identifying the failing cache operation.
    :param exc: Captured exception.
    :return: ``None``.
    """
    global _warned_cache_error
    if _warned_cache_error:
        return
    _warned_cache_error = True
    logger.warning(
        "Embedding cache %s failed (%s: %s); continuing without cache benefits for this run.",
        action,
        type(exc).__name__,
        exc,
    )


def is_cache_disabled() -> bool:
    """Return whether the global embedding-cache kill switch is set.

    :return: ``True`` when ``CODEDUPES_NO_CACHE`` is set to a truthy value.
    """
    return os.environ.get("CODEDUPES_NO_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_cache_dir() -> Path:
    """Resolve the embedding cache root directory from environment overrides.

    :return: ``CODEDUPES_CACHE_DIR`` if set, else ``$XDG_CACHE_HOME/codedupes`` if
        ``XDG_CACHE_HOME`` is set, else ``~/.cache/codedupes``.
    """
    override = os.environ.get("CODEDUPES_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "codedupes"
    return Path.home() / ".cache" / "codedupes"


def _resolve_max_bytes() -> int:
    """Resolve the opportunistic cache size cap in bytes.

    :return: Size cap in bytes from ``CODEDUPES_CACHE_MAX_MB``, defaulting to
        ``DEFAULT_CACHE_MAX_MB`` megabytes when unset or unparsable.
    """
    raw = os.environ.get("CODEDUPES_CACHE_MAX_MB")
    if raw:
        try:
            return max(1, int(float(raw))) * 1024 * 1024
        except ValueError:
            pass
    return DEFAULT_CACHE_MAX_MB * 1024 * 1024


def compute_cache_key(
    canonical_model: str,
    revision: str,
    text: str,
    mode: str = "code",
    variant: str = "",
) -> str:
    """Derive a content-addressed cache key for one prepared embedding input.

    The embedding mode participates in the key because some model families route
    modes through different encode entry points with their own internal prompts
    (for example EmbeddingGemma's ``encode_document`` vs ``encode_query``), so an
    identical prepared text does not guarantee an identical vector across modes.
    The ``variant`` component carries any additional vector-affecting fingerprint,
    such as the requested device for families whose torch dtype is device-dependent.

    :param canonical_model: Canonical model identifier.
    :param revision: Resolved model revision/commit hash.
    :param text: Prepared (pre-truncation) embedding input text.
    :param mode: Embedding input mode, ``"code"`` or ``"query"``.
    :param variant: Extra vector-affecting fingerprint, empty when not applicable.
    :return: Stable 32-character hex digest used as the shard row key.
    """
    payload = f"{canonical_model}\x00{revision}\x00{mode}\x00{variant}\x00{text}".encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _sanitize_component(value: str) -> str:
    """Sanitize a string for safe use as a single path component.

    :param value: Raw value to sanitize.
    :return: Value with unsafe characters collapsed to ``_``, never empty.
    """
    cleaned = _SANITIZE_PATTERN.sub("_", value).strip("._")
    return cleaned or "root"


def _scope_hash(resolved_scope: Path) -> str:
    """Hash a resolved repository root path for collision-safe shard naming.

    :param resolved_scope: Absolute, resolved repository root path.
    :return: 12-character hex digest of the path.
    """
    return hashlib.blake2b(str(resolved_scope).encode(), digest_size=6).hexdigest()


def _model_slug(canonical_model: str) -> str:
    """Build a filesystem-safe slug for a canonical model name.

    Local model directories (absolute paths) slug to their basename plus a
    short path hash so shard names stay readable and bounded regardless of
    how deep the directory lives.

    :param canonical_model: Canonical model identifier, for example
        ``Alibaba-NLP/gte-modernbert-base`` or ``/models/gte-modernbert-base``.
    :return: Sanitized slug with ``/`` replaced by ``--``.
    """
    path = Path(canonical_model)
    if path.is_absolute():
        digest = hashlib.blake2b(canonical_model.encode(), digest_size=6).hexdigest()
        return f"local--{_sanitize_component(path.name)}-{digest}"
    return _sanitize_component(canonical_model.replace("/", "--"))


def _revision_slug(revision: str | None) -> str:
    """Build a filesystem-safe slug for a resolved model revision.

    :param revision: Resolved revision/commit hash, or ``None`` when unpinned.
    :return: Sanitized revision slug, or ``"unpinned"`` when ``revision`` is ``None``.
    """
    return _sanitize_component(revision) if revision else "unpinned"


def _repo_dir_name(cache_scope: Path) -> str:
    """Build the per-repository shard directory name for a corpus root path.

    :param cache_scope: Analyzed corpus root path.
    :return: ``"<sanitized-basename>-<pathhash>"`` directory name.
    """
    resolved = Path(cache_scope).resolve()
    return f"{_sanitize_component(resolved.name)}-{_scope_hash(resolved)}"


def _shard_dir_for(
    repos_dir: Path,
    cache_scope: Path,
    canonical_model: str,
    revision: str | None,
) -> Path:
    """Resolve the shard directory for one (repo, model, revision) combination.

    :param repos_dir: Root directory holding all per-repo shard directories.
    :param cache_scope: Analyzed corpus root path.
    :param canonical_model: Canonical model identifier.
    :param revision: Resolved model revision, or ``None`` when unpinned.
    :return: Shard directory path (not guaranteed to exist).
    """
    shard_name = f"{_model_slug(canonical_model)}@{_revision_slug(revision)}"
    return repos_dir / _repo_dir_name(cache_scope) / shard_name


def _tmp_suffix() -> str:
    """Build a collision-resistant temp-file suffix for atomic replacement writes.

    :return: Suffix unique across processes and threads.
    """
    return f".tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"


@contextlib.contextmanager
def _shard_write_lock(shard_dir: Path) -> Iterator[bool]:
    """Hold an exclusive advisory lock serializing writers of one shard.

    Two unserialized read-modify-write writers can interleave their vectors/index
    replacements so a reader pairs one writer's vectors with the other's index,
    silently serving a wrong vector under a valid key. Writers therefore must hold
    this lock across the whole read-append-replace sequence. Lock contention or an
    unavailable lock API yields ``False`` and the caller skips its write (lost
    cache entries are acceptable; wrong ones are not). Readers never need the lock.

    :param shard_dir: Shard directory the caller intends to rewrite.
    :return: Context manager yielding ``True`` when the exclusive lock was acquired.
    """
    try:
        import fcntl
    except ImportError:
        # No advisory-lock API (non-POSIX): allow the write rather than disabling
        # caching entirely; single-writer machines remain correct.
        yield True
        return

    lock_fd: int | None = None
    try:
        lock_fd = os.open(shard_dir / ".lock", os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        if lock_fd is not None:
            os.close(lock_fd)
        yield False
        return
    try:
        yield True
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _shard_is_consistent(index: Any, vectors: Any) -> bool:
    """Validate that a loaded index and vector matrix are mutually consistent.

    :param index: Parsed ``index.json`` payload.
    :param vectors: Loaded ``vectors.npy`` array.
    :return: ``True`` when the schema, dimensions, and row references all line up.
    """
    if not isinstance(index, dict) or index.get("schema") != _SCHEMA_VERSION:
        return False
    dim = index.get("dim")
    keys_map = index.get("keys")
    if not isinstance(dim, int) or not isinstance(keys_map, dict):
        return False
    last_used_at = index.get("last_used_at", 0.0)
    if isinstance(last_used_at, bool) or not isinstance(last_used_at, (int, float)):
        return False
    try:
        if not math.isfinite(float(last_used_at)):
            return False
    except OverflowError:
        return False
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        return False
    if vectors.dtype != np.float32 or vectors.shape[1] != dim:
        return False
    n_rows = vectors.shape[0]
    return all(isinstance(row, int) and 0 <= row < n_rows for row in keys_map.values())


def _read_shard(shard_dir: Path) -> tuple[np.ndarray, dict[str, int], float] | None:
    """Load and validate one shard's vectors and key index.

    Any structural inconsistency (corrupt ``vectors.npy``, a stale ``index.json``
    pointing past the end of the vector matrix, schema drift, and so on) is treated
    as an empty shard rather than raised, matching the never-fatal cache contract.

    :param shard_dir: Shard directory to load.
    :return: ``(vectors, key_to_row, last_used_at)``, or ``None`` when the shard is
        missing, unreadable, or internally inconsistent.
    """
    index_path = shard_dir / INDEX_FILENAME
    vectors_path = shard_dir / VECTORS_FILENAME
    if not index_path.exists() or not vectors_path.exists():
        return None

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        # Memory-mapped so sparse lookups (for example a single query key) only
        # fault in the rows they touch; hit rows are copied before being returned.
        vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - corrupt on-disk data can fail in many ways
        _warn_once("read shard", exc)
        return None

    if not _shard_is_consistent(index, vectors):
        _warn_once("read shard", ValueError(f"inconsistent shard at {shard_dir}"))
        return None

    keys_map = {str(key): int(row) for key, row in index["keys"].items()}
    last_used_at = float(index.get("last_used_at", 0.0))
    return vectors, keys_map, last_used_at


def _atomic_write_shard(
    shard_dir: Path,
    canonical_model: str,
    revision: str | None,
    vectors: np.ndarray,
    keys_map: dict[str, int],
    dim: int,
) -> None:
    """Write a shard's vectors and index atomically via temp-file-then-replace.

    The vectors file is replaced first and the index second: an older index only
    ever references a subset of an already-superset vectors file, so a crash
    between the two replacements still leaves a consistent shard on disk.

    :param shard_dir: Shard directory to write into (created if missing).
    :param canonical_model: Canonical model identifier.
    :param revision: Resolved model revision, or ``None`` when unpinned.
    :param vectors: Full float32 vector matrix to persist.
    :param keys_map: Key-to-row mapping to persist.
    :param dim: Embedding dimensionality.
    :return: ``None``.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    suffix = _tmp_suffix()
    vectors_path = shard_dir / VECTORS_FILENAME
    index_path = shard_dir / INDEX_FILENAME
    vectors_tmp = shard_dir / f"{VECTORS_FILENAME}{suffix}"
    index_tmp = shard_dir / f"{INDEX_FILENAME}{suffix}"
    try:
        with open(vectors_tmp, "wb") as handle:
            np.save(handle, np.ascontiguousarray(vectors, dtype=np.float32))
        os.replace(vectors_tmp, vectors_path)

        payload = {
            "schema": _SCHEMA_VERSION,
            "model": canonical_model,
            "revision": revision if revision is not None else "unpinned",
            "dim": dim,
            "keys": keys_map,
            "last_used_at": time.time(),
        }
        index_tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(index_tmp, index_path)
    finally:
        for tmp_path in (vectors_tmp, index_tmp):
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass


def _touch_shard(shard_dir: Path) -> None:
    """Refresh a shard's recency stamp after a read hit, without touching vectors.

    Re-reads the current index under the writer lock so the touch can never clobber
    entries a concurrent writer appended after this reader loaded the shard; when
    the lock is contended the touch is simply skipped (recency only feeds LRU
    eviction, so a missed refresh is harmless).

    :param shard_dir: Shard directory to update.
    :return: ``None``.
    """
    try:
        with _shard_write_lock(shard_dir) as acquired:
            if not acquired:
                return
            payload = _read_shard_meta(shard_dir)
            if payload is None:
                return
            payload["last_used_at"] = time.time()
            index_tmp = shard_dir / f"{INDEX_FILENAME}{_tmp_suffix()}"
            index_tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(index_tmp, shard_dir / INDEX_FILENAME)
    except OSError as exc:
        _warn_once("touch shard", exc)


def _write_shard_entries(
    shard_dir: Path,
    canonical_model: str,
    revision: str | None,
    entries: Sequence[tuple[str, np.ndarray]],
) -> None:
    """Append new embedding rows to a shard, rebuilding it fresh if inconsistent.

    :param shard_dir: Shard directory to update.
    :param canonical_model: Canonical model identifier.
    :param revision: Resolved model revision, or ``None`` when unpinned.
    :param entries: Sequence of ``(key, vector)`` pairs to append.
    :return: ``None``.
    """
    if not entries:
        return
    try:
        unique_entries = list(dict(entries).items())
        dim = int(np.asarray(unique_entries[0][1]).reshape(-1).shape[0])

        shard_dir.mkdir(parents=True, exist_ok=True)
        with _shard_write_lock(shard_dir) as acquired:
            if not acquired:
                return
            existing = _read_shard(shard_dir)
            if existing is not None and existing[0].shape[1] == dim:
                vectors, keys_map, _ = existing
            else:
                vectors = np.empty((0, dim), dtype=np.float32)
                keys_map = {}
            existing = None

            start_row = vectors.shape[0]
            new_rows = np.stack(
                [
                    np.ascontiguousarray(vector, dtype=np.float32).reshape(dim)
                    for _key, vector in unique_entries
                ],
                axis=0,
            )
            # Rebinding releases the memory-mapped source before the replace below,
            # so the vectors file is never replaced while still mapped.
            vectors = np.concatenate([vectors, new_rows], axis=0)
            for offset, (key, _vector) in enumerate(unique_entries):
                keys_map[key] = start_row + offset

            _atomic_write_shard(shard_dir, canonical_model, revision, vectors, keys_map, dim)
    except Exception as exc:  # noqa: BLE001 - cache writes must never break analysis
        _warn_once("write shard", exc)


def _iter_shard_dirs(repos_dir: Path) -> list[Path]:
    """List every shard directory under the cache repos root.

    :param repos_dir: Root directory holding all per-repo shard directories.
    :return: Sorted list of shard directory paths, empty when ``repos_dir`` is absent.
    """
    if not repos_dir.exists():
        return []
    return [
        shard_dir
        for repo_dir in sorted(p for p in repos_dir.iterdir() if p.is_dir())
        for shard_dir in sorted(p for p in repo_dir.iterdir() if p.is_dir())
    ]


def _read_shard_meta(shard_dir: Path) -> dict[str, Any] | None:
    """Read a shard's ``index.json`` without loading its vector matrix.

    :param shard_dir: Shard directory to inspect.
    :return: Parsed index payload, or ``None`` when missing/unreadable.
    """
    index_path = shard_dir / INDEX_FILENAME
    if not index_path.exists():
        return None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError):
        return None


def _shard_size_bytes(shard_dir: Path) -> int:
    """Sum on-disk file sizes for one shard directory.

    :param shard_dir: Shard directory to measure.
    :return: Total bytes used by files directly inside ``shard_dir``.
    """
    return sum(f.stat().st_size for f in shard_dir.glob("*") if f.is_file())


def _prune_empty_repo_dirs(repos_dir: Path) -> None:
    """Remove per-repo directories left empty after shard deletion.

    :param repos_dir: Root directory holding all per-repo shard directories.
    :return: ``None``.
    """
    if not repos_dir.exists():
        return
    for repo_dir in list(repos_dir.iterdir()):
        if repo_dir.is_dir() and not any(repo_dir.iterdir()):
            repo_dir.rmdir()


def _maybe_evict(repos_dir: Path, protect: Path | None = None) -> None:
    """Evict least-recently-used shards once the cache exceeds its size cap.

    :param repos_dir: Root directory holding all per-repo shard directories.
    :param protect: Shard directory exempt from eviction (the one just written),
        so a cap smaller than one shard cannot instantly delete fresh work.
    :return: ``None``.
    """
    try:
        max_bytes = _resolve_max_bytes()
        sized = [
            (shard_dir, _shard_size_bytes(shard_dir)) for shard_dir in _iter_shard_dirs(repos_dir)
        ]
        total = sum(size for _, size in sized)
        if total <= max_bytes:
            return

        def _last_used(shard_dir: Path) -> float:
            """Resolve a shard's recency stamp, falling back to mtime.

            :param shard_dir: Shard directory to inspect.
            :return: ``last_used_at`` from the index, or directory mtime as a fallback.
            """
            meta = _read_shard_meta(shard_dir)
            if meta is not None and isinstance(meta.get("last_used_at"), (int, float)):
                return float(meta["last_used_at"])
            try:
                return shard_dir.stat().st_mtime
            except OSError:
                return 0.0

        target = int(max_bytes * _PRUNE_TARGET_RATIO)
        for shard_dir, size in sorted(sized, key=lambda item: _last_used(item[0])):
            if total <= target:
                break
            if protect is not None and shard_dir == protect:
                continue
            shutil.rmtree(shard_dir, ignore_errors=True)
            total -= size
        _prune_empty_repo_dirs(repos_dir)
        if total > target:
            logger.warning(
                "Embedding cache still exceeds its size target after eviction "
                "(%d bytes > %d); consider raising CODEDUPES_CACHE_MAX_MB.",
                total,
                target,
            )
    except OSError as exc:
        _warn_once("evict", exc)


class EmbeddingCache:
    """File-based embedding cache addressed by repo-root, model, and revision shards."""

    def __init__(self, cache_root: Path | None = None) -> None:
        """Bind a cache handle to a cache root directory.

        :param cache_root: Explicit cache root, defaults to :func:`resolve_cache_dir`.
        """
        self.cache_root = cache_root or resolve_cache_dir()

    @property
    def repos_dir(self) -> Path:
        """Return the directory holding all per-repository shard directories.

        :return: ``<cache_root>/repos``.
        """
        return self.cache_root / CACHE_SUBDIR

    def shard_dir(self, cache_scope: Path, canonical_model: str, revision: str | None) -> Path:
        """Resolve the shard directory for one (repo, model, revision) combination.

        :param cache_scope: Analyzed corpus root path.
        :param canonical_model: Canonical model identifier.
        :param revision: Resolved model revision, or ``None`` when unpinned.
        :return: Shard directory path (not guaranteed to exist).
        """
        return _shard_dir_for(self.repos_dir, cache_scope, canonical_model, revision)

    def get_many(
        self,
        cache_scope: Path,
        canonical_model: str,
        revision: str | None,
        keys: list[str],
    ) -> dict[str, np.ndarray]:
        """Look up cached embedding vectors, refreshing the recency stamp at most hourly.

        The recency stamp only feeds shard-granularity LRU eviction, so refreshing it
        on every read would waste an index rewrite per lookup and widen the window in
        which a reader's stale key map can clobber a concurrent writer's index.

        :param cache_scope: Analyzed corpus root path.
        :param canonical_model: Canonical model identifier.
        :param revision: Resolved model revision, or ``None`` when unpinned.
        :param keys: Cache keys to look up.
        :return: Mapping of hit keys to owned float32 embedding vectors.
        """
        if not keys:
            return {}
        shard_dir = self.shard_dir(cache_scope, canonical_model, revision)
        loaded = _read_shard(shard_dir)
        if loaded is None:
            return {}
        vectors, keys_map, last_used_at = loaded
        hits: dict[str, np.ndarray] = {}
        for key in keys:
            row = keys_map.get(key)
            if row is None:
                continue
            vector = np.array(vectors[row], dtype=np.float32, copy=True)
            # A NaN/Inf row would silently poison every similarity it touches on
            # every future run; treat it as a miss so it gets recomputed.
            if np.isfinite(vector).all():
                hits[key] = vector
        if hits and (time.time() - last_used_at) > _TOUCH_INTERVAL_SECONDS:
            _touch_shard(shard_dir)
        return hits

    def put_many(
        self,
        cache_scope: Path,
        canonical_model: str,
        revision: str | None,
        entries: Sequence[tuple[str, np.ndarray]],
    ) -> None:
        """Insert new embedding vectors into a shard and enforce the size cap.

        :param cache_scope: Analyzed corpus root path.
        :param canonical_model: Canonical model identifier.
        :param revision: Resolved model revision, or ``None`` when unpinned.
        :param entries: Sequence of ``(key, vector)`` pairs to store.
        :return: ``None``.
        """
        if not entries:
            return
        shard_dir = self.shard_dir(cache_scope, canonical_model, revision)
        _write_shard_entries(shard_dir, canonical_model, revision, entries)
        _maybe_evict(self.repos_dir, protect=shard_dir)

    def stats(self) -> dict[str, Any]:
        """Summarize cache location, size, and per-model/per-repo entry counts.

        :return: Dict with ``path``, ``disabled``, ``entries``, ``size_bytes``,
            ``models``, and ``repos`` keys.
        """
        info: dict[str, Any] = {
            "path": str(self.cache_root),
            "disabled": is_cache_disabled(),
            "entries": 0,
            "size_bytes": 0,
            "models": {},
            "repos": [],
        }
        repo_totals: dict[str, dict[str, int]] = {}
        try:
            for shard_dir in _iter_shard_dirs(self.repos_dir):
                meta = _read_shard_meta(shard_dir)
                count = len(meta.get("keys", {})) if meta else 0
                model_name = (meta.get("model") if meta else None) or shard_dir.name
                size = _shard_size_bytes(shard_dir)
                info["entries"] += count
                info["size_bytes"] += size
                info["models"][model_name] = info["models"].get(model_name, 0) + count
                totals = repo_totals.setdefault(
                    shard_dir.parent.name, {"shards": 0, "entries": 0, "size_bytes": 0}
                )
                totals["shards"] += 1
                totals["entries"] += count
                totals["size_bytes"] += size
        except OSError as exc:
            _warn_once("stats", exc)
        info["repos"] = [{"repo": name, **totals} for name, totals in sorted(repo_totals.items())]
        return info

    def clear(self, model: str | None = None) -> int:
        """Delete cached embeddings, optionally scoped to one canonical model.

        :param model: Canonical model name to scope deletion to, or ``None`` to
            clear every shard across every repo.
        :return: Number of cached entries removed, ``0`` on failure.
        """
        removed = 0
        try:
            for shard_dir in _iter_shard_dirs(self.repos_dir):
                meta = _read_shard_meta(shard_dir)
                if model is not None and (meta is None or meta.get("model") != model):
                    continue
                removed += len(meta.get("keys", {})) if meta else 0
                shutil.rmtree(shard_dir, ignore_errors=True)
            _prune_empty_repo_dirs(self.repos_dir)
        except OSError as exc:
            _warn_once("clear", exc)
        return removed


def get_embedding_cache() -> EmbeddingCache | None:
    """Build a fresh embedding cache handle unless caching is globally disabled.

    :return: New :class:`EmbeddingCache` instance, or ``None`` when
        ``CODEDUPES_NO_CACHE`` is set.
    """
    if is_cache_disabled():
        return None
    return EmbeddingCache()
