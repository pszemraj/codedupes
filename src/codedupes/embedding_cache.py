"""Persistent, content-addressed on-disk cache for semantic embedding vectors.

Cached vectors live under ``<cache_root>/repos/<repo-shard>/<model>@<revision>/`` as
an immutable generation-named float32 matrix and an ``index.json`` key-to-row map
that atomically selects the active generation. The primary key hashes the model,
resolved revision, and prepared (pre-truncation) embedding text, so unchanged code
units keep hitting the cache across runs and partial edits only miss for units that
actually changed. Every public operation is wrapped so on-disk corruption or
filesystem errors never raise into the caller; an untrusted shard is treated as
empty and rebuilt on the next write.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CACHE_SUBDIR = "repos"
LOCAL_MODELS_SUBDIR = "local-models"
INDEX_FILENAME = "index.json"
DEFAULT_CACHE_MAX_MB = 2048
_SCHEMA_VERSION = 3
_PRUNE_TARGET_RATIO = 0.8
_TOUCH_INTERVAL_SECONDS = 3600.0
_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_GENERATION_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_TMP_FILE_GLOB = "*.tmp-*"

_warned_cache_error = False


@dataclass
class _ShardData:
    """Validated vector shard loaded from one immutable generation."""

    vectors: np.ndarray
    keys: dict[str, int]
    namespaces: dict[str, str]
    digests: dict[str, str]
    last_used_at: float
    generation: str


def _row_digest(vector: np.ndarray) -> str:
    """Digest one embedding row for read-time integrity verification.

    :param vector: Contiguous float32 embedding row.
    :return: Hex digest of the row's exact bytes.
    """
    return hashlib.blake2b(
        np.ascontiguousarray(vector, dtype=np.float32).tobytes(),
        digest_size=16,
    ).hexdigest()


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
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError
            return max(1, int(value)) * 1024 * 1024
        except (OverflowError, ValueError):
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
    such as a non-default inference dtype.

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


def _hex_digest(value: str) -> str:
    """Build the short hex digest used for collision-safe cache path components.

    :param value: String to digest.
    :return: 12-character blake2b hex digest.
    """
    return hashlib.blake2b(value.encode(), digest_size=6).hexdigest()


def _scope_hash(resolved_scope: Path) -> str:
    """Hash a resolved repository root path for collision-safe shard naming.

    :param resolved_scope: Absolute, resolved repository root path.
    :return: 12-character hex digest of the path.
    """
    return _hex_digest(str(resolved_scope))


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
        digest = _hex_digest(canonical_model)
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


def _vectors_filename(generation: str) -> str:
    """Return the immutable vector filename for one shard generation.

    :param generation: Validated hexadecimal generation identifier.
    :return: Filename local to a shard directory.
    """
    return f"vectors-{generation}.npy"


def _is_finite_row(vector: np.ndarray) -> bool:
    """Check whether a stored or candidate embedding row is usable.

    Shared by reads and writes so the two can never disagree on what counts as a
    poisoned (NaN/Inf) row.

    :param vector: Embedding row to validate.
    :return: ``True`` when every element is finite.
    """
    return bool(np.isfinite(vector).all())


@contextlib.contextmanager
def _shard_write_lock(shard_dir: Path, *, blocking: bool = False) -> Iterator[bool]:
    """Hold an exclusive advisory lock serializing writers of one shard.

    Writers must hold this lock across the whole read-update-publish sequence so
    concurrent updates cannot discard one another. Lock contention or an unavailable
    lock API yields ``False`` and the caller skips its write (lost cache entries are
    acceptable; wrong ones are not). Readers need no lock because published vector
    generations are immutable.

    :param shard_dir: Shard directory the caller intends to rewrite.
    :param blocking: When ``True``, block until the lock is available instead of
        yielding ``False`` on contention; used by callers (like directory deletion)
        that must not race a concurrent writer under any circumstance.
    :return: Context manager yielding ``True`` when the exclusive lock was acquired.
    """
    try:
        import fcntl
    except ImportError:
        # No advisory-lock API (non-POSIX): allow the write rather than disabling
        # caching entirely; single-writer machines remain correct.
        yield True
        return

    lock_flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
    lock_fd: int | None = None
    try:
        lock_fd = os.open(shard_dir / ".lock", os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(lock_fd, lock_flags)
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


def _validate_shard_metadata(payload: Any) -> dict[str, Any] | None:
    """Validate and normalize one parsed shard index.

    :param payload: Parsed ``index.json`` payload.
    :return: Normalized metadata dictionary, or ``None`` when invalid.
    """
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA_VERSION:
        return None

    model = payload.get("model")
    revision = payload.get("revision")
    dim = payload.get("dim")
    keys_map = payload.get("keys")
    namespaces = payload.get("namespaces")
    digests = payload.get("digests")
    generation = payload.get("generation")
    if not isinstance(model, str) or not isinstance(revision, str):
        return None
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        return None
    if not isinstance(keys_map, dict) or not all(
        isinstance(key, str) and not isinstance(row, bool) and isinstance(row, int) and row >= 0
        for key, row in keys_map.items()
    ):
        return None
    if (
        not isinstance(namespaces, dict)
        or set(namespaces) != set(keys_map)
        or not all(
            isinstance(key, str) and isinstance(namespace, str)
            for key, namespace in namespaces.items()
        )
    ):
        return None
    if (
        not isinstance(digests, dict)
        or set(digests) != set(keys_map)
        or not all(
            isinstance(key, str) and isinstance(digest, str) for key, digest in digests.items()
        )
    ):
        return None
    if not isinstance(generation, str) or _GENERATION_PATTERN.fullmatch(generation) is None:
        return None

    last_used_at = payload.get("last_used_at", 0.0)
    if isinstance(last_used_at, bool) or not isinstance(last_used_at, (int, float)):
        return None
    try:
        normalized_last_used_at = float(last_used_at)
        if not math.isfinite(normalized_last_used_at):
            return None
    except OverflowError:
        return None

    return {
        "schema": _SCHEMA_VERSION,
        "model": model,
        "revision": revision,
        "dim": dim,
        "keys": dict(keys_map),
        "namespaces": dict(namespaces),
        "digests": dict(digests),
        "last_used_at": normalized_last_used_at,
        "generation": generation,
    }


def _validate_shard(index: Any, vectors: Any) -> dict[str, Any] | None:
    """Validate one shard's metadata and vector matrix together.

    :param index: Parsed ``index.json`` payload.
    :param vectors: Loaded ``vectors.npy`` array.
    :return: Normalized metadata dictionary, or ``None`` when inconsistent.
    """
    metadata = _validate_shard_metadata(index)
    if metadata is None:
        return None
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        return None
    if vectors.dtype != np.float32 or vectors.shape[1] != metadata["dim"]:
        return None
    n_rows = vectors.shape[0]
    if any(row >= n_rows for row in metadata["keys"].values()):
        return None
    return metadata


def _peek_generation(payload: Any) -> str | None:
    """Extract a well-formed generation identifier from a parsed index payload.

    Cheap first-pass check for :func:`_read_shard`: only the generation is needed
    to pick the vectors file, so full metadata validation waits for the confirmed
    re-read.

    :param payload: Parsed ``index.json`` payload.
    :return: Generation hex string, or ``None`` when absent or malformed.
    """
    if not isinstance(payload, dict):
        return None
    generation = payload.get("generation")
    if not isinstance(generation, str) or _GENERATION_PATTERN.fullmatch(generation) is None:
        return None
    return generation


def _read_shard(shard_dir: Path) -> _ShardData | None:
    """Load and validate one shard's vectors and key index.

    Any structural inconsistency (a corrupt vector generation, a stale ``index.json``
    pointing past the end of its matrix, schema drift, and so on) is treated as an
    empty shard rather than raised, matching the never-fatal cache contract.

    :param shard_dir: Shard directory to load.
    :return: Validated shard data, or ``None`` when the shard is missing, unreadable,
        internally inconsistent, or replaced by a concurrent writer.
    """
    index_path = shard_dir / INDEX_FILENAME
    if not index_path.exists():
        return None

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        initial_generation = _peek_generation(index)
        if initial_generation is None:
            raise ValueError(f"invalid shard metadata at {shard_dir}")
        vectors_path = shard_dir / _vectors_filename(initial_generation)
        # Memory-mapped so sparse lookups (for example a single query key) only
        # fault in the rows they touch; hit rows are copied before being returned.
        vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
        confirmed_index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - corrupt on-disk data can fail in many ways
        _warn_once("read shard", exc)
        return None

    metadata = _validate_shard(confirmed_index, vectors)
    if metadata is None or metadata["generation"] != initial_generation:
        _warn_once("read shard", ValueError(f"inconsistent shard at {shard_dir}"))
        return None

    return _ShardData(
        vectors=vectors,
        keys=metadata["keys"],
        namespaces=metadata["namespaces"],
        digests=metadata["digests"],
        last_used_at=metadata["last_used_at"],
        generation=metadata["generation"],
    )


def _reclaim_stale_tmp_files(shard_dir: Path, keep: frozenset[Path] = frozenset()) -> None:
    """Delete leftover tmp files abandoned by a writer that never reached cleanup.

    Tmp files are only ever created by a writer holding this shard's exclusive
    lock, so any tmp file found here while that lock is held (as it is by every
    caller of this helper) was orphaned by a process that died mid-write (SIGKILL,
    power loss) before its own ``finally`` block could remove it. Left alone, they
    linger forever and inflate the shard's size-cap accounting.

    :param shard_dir: Shard directory to sweep; caller must hold its write lock.
    :param keep: Tmp paths belonging to the current write, left untouched.
    :return: ``None``.
    """
    for stale_tmp in shard_dir.glob(_TMP_FILE_GLOB):
        if stale_tmp not in keep:
            with contextlib.suppress(OSError):
                stale_tmp.unlink()


def _publish_index(shard_dir: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a shard's ``index.json``, cleaning its tmp file on failure.

    :param shard_dir: Shard directory holding the index; caller must hold its write lock.
    :param payload: JSON-serializable index payload.
    :return: ``None``.
    """
    index_tmp = shard_dir / f"{INDEX_FILENAME}{_tmp_suffix()}"
    try:
        index_tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(index_tmp, shard_dir / INDEX_FILENAME)
    finally:
        if index_tmp.exists():
            with contextlib.suppress(OSError):
                index_tmp.unlink()


def _atomic_write_shard(
    shard_dir: Path,
    canonical_model: str,
    revision: str | None,
    vectors: np.ndarray,
    keys_map: dict[str, int],
    namespaces: dict[str, str],
    digests: dict[str, str],
    dim: int,
) -> None:
    """Publish a complete shard generation through one atomic index replacement.

    Each vector matrix has an immutable generation-specific filename. The matrix
    is fully written before ``index.json`` atomically switches to that generation,
    so readers can never pair an older key map with a rebuilt matrix. A crash
    before the index replacement leaves only an unreferenced file; a crash after
    it leaves the new generation complete. Always runs under the shard write lock,
    so it also reclaims any tmp files orphaned by a prior writer that crashed.

    :param shard_dir: Shard directory to write into (created if missing).
    :param canonical_model: Canonical model identifier.
    :param revision: Resolved model revision, or ``None`` when unpinned.
    :param vectors: Full float32 vector matrix to persist.
    :param keys_map: Key-to-row mapping to persist.
    :param namespaces: Key-to-input-namespace mapping used for namespace capping.
    :param digests: Key-to-row-digest mapping used for read-time integrity checks.
    :param dim: Embedding dimensionality.
    :return: ``None``.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    vectors_filename = _vectors_filename(generation)
    vectors_path = shard_dir / vectors_filename
    vectors_tmp = shard_dir / f"{vectors_filename}{_tmp_suffix()}"
    _reclaim_stale_tmp_files(shard_dir, keep=frozenset({vectors_tmp}))
    try:
        with open(vectors_tmp, "wb") as handle:
            np.save(handle, np.ascontiguousarray(vectors, dtype=np.float32))
        os.replace(vectors_tmp, vectors_path)

        _publish_index(
            shard_dir,
            {
                "schema": _SCHEMA_VERSION,
                "model": canonical_model,
                "revision": revision if revision is not None else "unpinned",
                "dim": dim,
                "keys": keys_map,
                "namespaces": namespaces,
                "digests": digests,
                "last_used_at": time.time(),
                "generation": generation,
            },
        )

        for stale_vectors in shard_dir.glob("vectors-*.npy"):
            if stale_vectors != vectors_path:
                with contextlib.suppress(OSError):
                    stale_vectors.unlink()
    finally:
        if vectors_tmp.exists():
            with contextlib.suppress(OSError):
                vectors_tmp.unlink()


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
            _reclaim_stale_tmp_files(shard_dir)
            payload = _read_shard_meta(shard_dir)
            if payload is None:
                return
            payload["last_used_at"] = time.time()
            _publish_index(shard_dir, payload)
    except OSError as exc:
        _warn_once("touch shard", exc)


def _rebuild_matrix_retaining(
    vectors: np.ndarray,
    keys_map: dict[str, int],
    namespaces: dict[str, str],
    digests: dict[str, str],
    dim: int,
    retained_keys: list[str],
) -> tuple[np.ndarray, dict[str, int], dict[str, str], dict[str, str]]:
    """Rebuild a shard's vector matrix and key maps keeping only the given keys.

    :param vectors: Current vector matrix indexed by ``keys_map``.
    :param keys_map: Current key-to-row mapping.
    :param namespaces: Current key-to-input-namespace mapping.
    :param digests: Current key-to-row-digest mapping.
    :param dim: Embedding dimensionality.
    :param retained_keys: Keys to keep, in their new row order.
    :return: Rebuilt ``(vectors, keys_map, namespaces, digests)`` with densely
        renumbered rows.
    """
    new_vectors = (
        np.stack(
            [
                np.ascontiguousarray(vectors[keys_map[key]], dtype=np.float32)
                for key in retained_keys
            ],
            axis=0,
        )
        if retained_keys
        else np.empty((0, dim), dtype=np.float32)
    )
    new_keys_map = {key: row for row, key in enumerate(retained_keys)}
    new_namespaces = {key: namespaces[key] for key in retained_keys}
    new_digests = {key: digests[key] for key in retained_keys}
    return new_vectors, new_keys_map, new_namespaces, new_digests


def _write_shard_entries(
    shard_dir: Path,
    canonical_model: str,
    revision: str | None,
    entries: Sequence[tuple[str, np.ndarray]],
    *,
    namespace: str,
    max_namespace_keys: int | None = None,
) -> None:
    """Append/heal embedding rows and cap overflowing namespace keys.

    :param shard_dir: Shard directory to update.
    :param canonical_model: Canonical model identifier.
    :param revision: Resolved model revision, or ``None`` when unpinned.
    :param entries: Sequence of ``(key, vector)`` pairs to append or, for keys that
        already exist with a poisoned (NaN/Inf) stored row, heal in place.
    :param namespace: Stable identifier for one mode/instruction/dtype combination.
    :param max_namespace_keys: Maximum keys to retain in ``namespace`` after this
        write, oldest (lowest row index) dropped first, or ``None`` for no cap.
    :return: ``None``.
    """
    if not entries:
        return
    try:
        unique_entries = list(dict(entries).items())
        entry_dim = (
            int(np.asarray(unique_entries[0][1]).reshape(-1).shape[0]) if unique_entries else None
        )

        shard_dir.mkdir(parents=True, exist_ok=True)
        with _shard_write_lock(shard_dir) as acquired:
            if not acquired:
                return
            existing = _read_shard(shard_dir)
            if existing is not None and (
                entry_dim is None or existing.vectors.shape[1] == entry_dim
            ):
                vectors = existing.vectors
                keys_map = existing.keys
                namespaces = existing.namespaces
                digests = existing.digests
                dim = int(vectors.shape[1])
            else:
                if entry_dim is None:
                    return
                dim = entry_dim
                vectors = np.empty((0, dim), dtype=np.float32)
                keys_map = {}
                namespaces = {}
                digests = {}
            existing = None

            missing_entries: list[tuple[str, np.ndarray]] = []
            overwrite_entries: list[tuple[str, np.ndarray]] = []
            for key, vector in unique_entries:
                row = keys_map.get(key)
                if row is None:
                    missing_entries.append((key, vector))
                elif not _is_finite_row(vectors[row]) or digests.get(key) != _row_digest(
                    vectors[row]
                ):
                    # A poisoned or digest-mismatched stored row is a permanent
                    # miss for get_many (see its matching checks), so a
                    # recomputed value for the same key must overwrite it here
                    # or the unit would re-embed on every future run forever.
                    overwrite_entries.append((key, vector))

            if overwrite_entries:
                # existing.vectors is a read-only mmap; copy before mutating rows.
                vectors = np.array(vectors, dtype=np.float32, copy=True)
                for key, vector in overwrite_entries:
                    healed = np.ascontiguousarray(vector, dtype=np.float32).reshape(dim)
                    vectors[keys_map[key]] = healed
                    namespaces[key] = namespace
                    digests[key] = _row_digest(healed)

            if missing_entries:
                start_row = vectors.shape[0]
                new_rows = np.stack(
                    [
                        np.ascontiguousarray(vector, dtype=np.float32).reshape(dim)
                        for _key, vector in missing_entries
                    ],
                    axis=0,
                )
                vectors = np.concatenate([vectors, new_rows], axis=0)
                for offset, (key, _vector) in enumerate(missing_entries):
                    keys_map[key] = start_row + offset
                    namespaces[key] = namespace
                    digests[key] = _row_digest(new_rows[offset])

            capped_namespace = False
            if max_namespace_keys is not None:
                namespace_keys_by_row = sorted(
                    (
                        key
                        for key, key_namespace in namespaces.items()
                        if key_namespace == namespace
                    ),
                    key=keys_map.__getitem__,
                )
                overflow = len(namespace_keys_by_row) - max_namespace_keys
                if overflow > 0:
                    drop_keys = set(namespace_keys_by_row[:overflow])
                    retained_keys = sorted(
                        (key for key in keys_map if key not in drop_keys),
                        key=keys_map.__getitem__,
                    )
                    vectors, keys_map, namespaces, digests = _rebuild_matrix_retaining(
                        vectors, keys_map, namespaces, digests, dim, retained_keys
                    )
                    capped_namespace = True

            if missing_entries or overwrite_entries or capped_namespace:
                _atomic_write_shard(
                    shard_dir,
                    canonical_model,
                    revision,
                    vectors,
                    keys_map,
                    namespaces,
                    digests,
                    dim,
                )
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
        return _validate_shard_metadata(payload)
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
            if meta is not None:
                return meta["last_used_at"]
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
            # A concurrent writer may be mid-publish under this shard's lock; skip
            # rather than rmtree out from under it (lost eviction opportunity is
            # fine, destroying an in-flight write is not).
            with _shard_write_lock(shard_dir) as acquired:
                if not acquired:
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
        hits: dict[str, np.ndarray] = {}
        for key in keys:
            row = loaded.keys.get(key)
            if row is None:
                continue
            vector = np.array(loaded.vectors[row], dtype=np.float32, copy=True)
            # A NaN/Inf row would silently poison every similarity it touches,
            # and a finite row whose bytes no longer match the digest recorded
            # at write time is corruption that would otherwise look valid
            # forever. Both degrade to a per-key miss so the unit is recomputed
            # (and, in _write_shard_entries, healed in place).
            if _is_finite_row(vector) and loaded.digests.get(key) == _row_digest(vector):
                hits[key] = vector
        if hits and (time.time() - loaded.last_used_at) > _TOUCH_INTERVAL_SECONDS:
            _touch_shard(shard_dir)
        return hits

    def put_many(
        self,
        cache_scope: Path,
        canonical_model: str,
        revision: str | None,
        entries: Sequence[tuple[str, np.ndarray]],
        *,
        namespace: str = "default",
        max_namespace_keys: int | None = None,
    ) -> None:
        """Insert vectors, cap overflowing namespaces, enforce the global size cap.

        :param cache_scope: Analyzed corpus root path.
        :param canonical_model: Canonical model identifier.
        :param revision: Resolved model revision, or ``None`` when unpinned.
        :param entries: Sequence of ``(key, vector)`` pairs to store.
        :param namespace: Stable identifier for one mode/instruction/dtype combination.
        :param max_namespace_keys: Maximum keys to retain in ``namespace`` after this
            write, oldest dropped first, or ``None`` for no cap.
        :return: ``None``.
        """
        if not entries:
            return
        shard_dir = self.shard_dir(cache_scope, canonical_model, revision)
        _write_shard_entries(
            shard_dir,
            canonical_model,
            revision,
            entries,
            namespace=namespace,
            max_namespace_keys=max_namespace_keys,
        )
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
                size = _shard_size_bytes(shard_dir)
                info["entries"] += count
                info["size_bytes"] += size
                if meta is not None:
                    model_name = meta["model"]
                    info["models"][model_name] = info["models"].get(model_name, 0) + count
                totals = repo_totals.setdefault(
                    shard_dir.parent.name, {"shards": 0, "entries": 0, "size_bytes": 0}
                )
                totals["shards"] += 1
                totals["entries"] += count
                totals["size_bytes"] += size
            local_models_dir = self.cache_root / LOCAL_MODELS_SUBDIR
            if local_models_dir.is_dir():
                info["size_bytes"] += sum(
                    f.stat().st_size for f in local_models_dir.glob("*") if f.is_file()
                )
        except OSError as exc:
            _warn_once("stats", exc)
        info["repos"] = [{"repo": name, **totals} for name, totals in sorted(repo_totals.items())]
        return info

    def clear(self, model: str | None = None) -> int:
        """Delete cached embeddings, optionally scoped to one canonical model.

        :param model: Canonical model name to scope deletion to, or ``None`` to
            clear every shard across every repo plus local-model digest manifests.
        :return: Number of cached entries removed, ``0`` on failure.
        """
        removed = 0
        try:
            for shard_dir in _iter_shard_dirs(self.repos_dir):
                meta = _read_shard_meta(shard_dir)
                if model is not None and (meta is None or meta.get("model") != model):
                    continue
                removed += len(meta.get("keys", {})) if meta else 0
                # Wait for any concurrent writer rather than deleting under it: writers
                # hold this lock only briefly, and a dead holder's flock self-releases.
                with _shard_write_lock(shard_dir, blocking=True) as acquired:
                    if not acquired:
                        continue
                    shutil.rmtree(shard_dir, ignore_errors=True)
            _prune_empty_repo_dirs(self.repos_dir)
            if model is None:
                # Manifests are keyed by local model directory, not canonical model
                # name, so they are only removed on a full clear.
                shutil.rmtree(self.cache_root / LOCAL_MODELS_SUBDIR, ignore_errors=True)
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
