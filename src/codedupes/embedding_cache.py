"""Persistent, content-addressed on-disk cache for semantic embedding vectors.

Cached vectors live under ``<cache_root>/repos/<repo-shard>/<model>@<revision>/`` as
an immutable generation-named float32 matrix and an ``index.json`` key-to-row map
that atomically selects the active generation. The primary key hashes the model,
resolved revision, and complete prepared embedding text, so unchanged code
units keep hitting the cache across runs and partial edits only miss for units that
actually changed. Every public operation is wrapped so on-disk corruption or
filesystem errors never raise into the caller; an untrusted shard is treated as
empty and rebuilt on the next write.

The cache tracks three different things:

* A unit UID identifies one occurrence of code. Two UIDs can share a content key.
* A content key identifies reusable embedding input. ``index.json`` maps it to
  a vector row; compaction can change that row number without changing the key.
* ``manifest.json`` records which units each analysis selection last observed.
  It counts unit moves/deletions separately from keys that become unreferenced.

There are also two independent generations: vector generations are immutable
snapshot IDs, while the manifest generation counts complete corpus scans. Only
the scan counter ages orphaned keys toward collection.
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
from collections.abc import Iterator, MutableMapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CACHE_SUBDIR = "repos"
LOCAL_MODELS_SUBDIR = "local-models"
LOCKS_SUBDIR = "locks"
INDEX_FILENAME = "index.json"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA = 4
ORPHAN_GC_GENERATIONS = 3
MANIFEST_SELECTION_LIMIT = 16
DEFAULT_CACHE_MAX_MB = 2048
_SCHEMA_VERSION = 3
_PRUNE_TARGET_RATIO = 0.8
_NAMESPACE_PRUNE_TARGET_RATIO = 0.8
_TOUCH_INTERVAL_SECONDS = 3600.0
_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_GENERATION_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_TMP_FILE_GLOB = "*.tmp-*"
_CACHE_DIRECTORY_MODE = 0o700
_CACHE_FILE_MODE = 0o600

_warned_cache_error = False
_warned_invalid_cache_max_mb = False
_cache_warning_collector: ContextVar[list[str] | None] = ContextVar(
    "codedupes_cache_warning_collector",
    default=None,
)


@dataclass
class CorpusSelectionManifest:
    """Describe one selection's embedded corpus and aged unreferenced keys."""

    last_seen_generation: int  # Last complete scan, or the selection's first observation.
    complete_scan: bool
    units: dict[str, str]  # Unit UID -> shared content key.
    orphans: dict[str, int]  # Content key -> generation when it became unreferenced.
    unit_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class CorpusManifest:
    """Describe independently comparable corpus selections in one cache shard."""

    schema: int
    generation: int
    selections: dict[str, CorpusSelectionManifest]


@dataclass(frozen=True)
class ManifestDiff:
    """Classify unit-identity changes between two corpus manifests."""

    moved: list[str]  # New UIDs matched to departed UIDs by content key.
    deleted: list[str]  # Departed UIDs left after matching moves.
    orphaned: set[str]  # Previous content keys with no current reference.


@dataclass(frozen=True)
class ManifestPublishResult:
    """Report one successful manifest publication and optional orphan collection."""

    diff: ManifestDiff
    generation: int
    orphan_rows_retained: int
    orphan_rows_collected: int


def diff_manifest(
    previous: CorpusSelectionManifest | None,
    current: dict[str, str],
) -> ManifestDiff:
    """Classify moves, deletions, and newly unreferenced keys.

    :param previous: Previous comparable selection manifest, if one exists.
    :param current: Current unit-to-cache-key mapping.
    :return: Classified manifest transition.
    """
    previous_units = previous.units if previous is not None else {}

    # First find missing occurrences, grouped by their embedding content. A
    # shared key can have several departed UIDs, each needing its own match.
    departed_uids_by_key: dict[str, list[str]] = {}
    for uid, key in previous_units.items():
        if uid in current:
            continue
        departed_uids_by_key.setdefault(key, []).append(uid)

    # Infer a move only when a new UID can consume one departed UID with the
    # same content. An extra copy without a departure is an addition, not a move.
    moved_uids: list[str] = []
    moved_from_uids: set[str] = set()
    for uid, key in current.items():
        if uid in previous_units:
            continue
        matching_departures = departed_uids_by_key.get(key)
        if not matching_departures:
            continue
        moved_from_uids.add(matching_departures.pop())
        moved_uids.append(uid)

    # Unmatched departures are deletions even if another UID still shares their
    # vector. Keep the previous manifest's order for deterministic reporting.
    deleted_uids: list[str] = []
    for uid in previous_units:
        if uid in current or uid in moved_from_uids:
            continue
        deleted_uids.append(uid)

    # Row reclamation depends on content references, independently of UID counts.
    previous_keys = set(previous_units.values())
    current_keys = set(current.values())
    return ManifestDiff(
        moved=moved_uids,
        deleted=deleted_uids,
        orphaned=previous_keys - current_keys,
    )


@dataclass
class _ShardData:
    """Validated vector shard loaded from one immutable generation.

    ``keys``, ``namespaces``, and ``digests`` must always agree on membership
    (same key set) and stay in row-bounds with ``vectors``; the mutation
    methods below are the only sanctioned way to change any of the three
    together, so a caller can never advance one without the others.
    """

    vectors: np.ndarray
    keys: dict[str, int]
    namespaces: dict[str, str]
    digests: dict[str, str]
    last_used_at: float
    generation: str
    source_commit: str | None

    def overwrite_rows(self, entries: Sequence[tuple[str, np.ndarray]], namespace: str) -> None:
        """Heal existing rows in place, keeping keys/namespaces/digests atomic.

        :param entries: ``(key, vector)`` pairs; every ``key`` must already
            have a row in :attr:`keys`.
        :param namespace: Namespace label to stamp on every healed key.
        :return: ``None``.
        """
        if not entries:
            return
        if not self.vectors.flags.writeable:
            # ``self.vectors`` may still be the read-only mmap a loaded shard
            # was returned with; healing rows in place requires an owned copy.
            self.vectors = np.array(self.vectors, dtype=np.float32, copy=True)
        dim = self.vectors.shape[1]
        for key, vector in entries:
            row = self.keys[key]
            healed = np.ascontiguousarray(vector, dtype=np.float32).reshape(dim)
            self.vectors[row] = healed
            self.namespaces[key] = namespace
            self.digests[key] = _row_digest(healed)

    def append_rows(self, entries: Sequence[tuple[str, np.ndarray]], namespace: str) -> None:
        """Append new rows, keeping keys/namespaces/digests atomic and densely numbered.

        :param entries: ``(key, vector)`` pairs; every ``key`` must be absent
            from :attr:`keys`.
        :param namespace: Namespace label to stamp on every appended key.
        :return: ``None``.
        """
        if not entries:
            return
        dim = self.vectors.shape[1]
        start_row = self.vectors.shape[0]
        new_rows = np.empty((len(entries), dim), dtype=np.float32)
        for offset, (_key, vector) in enumerate(entries):
            new_rows[offset] = np.asarray(vector, dtype=np.float32).reshape(dim)
        self.vectors = np.concatenate([self.vectors, new_rows], axis=0)
        for offset, (key, _vector) in enumerate(entries):
            self.keys[key] = start_row + offset
            self.namespaces[key] = namespace
            self.digests[key] = _row_digest(new_rows[offset])

    def retain(self, retained_keys: list[str]) -> None:
        """Rebuild vectors/keys/namespaces/digests keeping only the given keys.

        Delegates the matrix compaction itself to
        :func:`_rebuild_matrix_retaining` and publishes the four results back
        onto this shard together, so no caller can observe a partially
        rebuilt shard.

        :param retained_keys: Keys to keep, in their new row order.
        :return: ``None``.
        """
        dim = self.vectors.shape[1]
        self.vectors, self.keys, self.namespaces, self.digests = _rebuild_matrix_retaining(
            self.vectors, self.keys, self.namespaces, self.digests, dim, retained_keys
        )

    def assert_consistent(self) -> None:
        """Fail loudly when keys/namespaces/digests/vectors have drifted apart.

        Cheap invariant check meant to run at write time (see
        :func:`_atomic_write_shard`), catching a mutation-path bug before it
        reaches disk instead of letting it silently degrade to an
        invalid-shard miss on the next read.

        :raises AssertionError: If the structures disagree on membership, a
            key references a row outside the matrix, or two keys collide on
            one row.
        :return: ``None``.
        """
        assert set(self.keys) == set(self.namespaces) == set(self.digests), (
            "embedding cache shard keys/namespaces/digests diverged"
        )
        n_rows = self.vectors.shape[0]
        rows = list(self.keys.values())
        assert all(0 <= row < n_rows for row in rows), (
            "embedding cache shard key references a row outside the vector matrix"
        )
        assert len(set(rows)) == len(rows), "embedding cache shard rows are not uniquely assigned"


@dataclass(frozen=True)
class CacheClearResult:
    """Outcome of a best-effort embedding-cache clear operation.

    :param int removed_entries: Number of cached embedding entries removed.
    :param int failed_deletions: Number of cache trees or shards that could not be cleared.
    """

    removed_entries: int
    failed_deletions: int


@dataclass(frozen=True)
class _CacheTreeDeletion:
    """Internal outcome that distinguishes an absent tree from a failed deletion."""

    removed: bool
    failed: bool


@dataclass(frozen=True)
class CacheLookup:
    """Cached vectors plus the provenance of the one shard snapshot they were read from."""

    vectors: dict[str, np.ndarray]
    source_commit: str | None


def _row_digest(vector: np.ndarray) -> str:
    """Digest one embedding row for read-time integrity verification.

    :param vector: Contiguous float32 embedding row.
    :return: Hex digest of the row's exact bytes.
    """
    return hashlib.blake2b(
        np.ascontiguousarray(vector, dtype=np.float32).tobytes(),
        digest_size=16,
    ).hexdigest()


def log_warning_once(
    namespace: MutableMapping[str, Any],
    flag_name: str,
    message: str,
    *,
    warning_logger: logging.Logger = logger,
) -> None:
    """Log one warning gated by a named boolean flag stored in ``namespace``.

    Generalizes this module's own one-shot warning gate for reuse by other
    modules. ``namespace`` is typically the caller's own ``globals()``, so the
    flag stays a real module-level attribute that existing tests can
    monkeypatch directly (for example ``semantic._warned_mlx_mps_contention =
    False``), and ``warning_logger`` lets the message keep the calling
    module's logger name so a caller scoping ``caplog`` to its own logger
    (``caplog.at_level(..., logger="codedupes.semantic")``) still captures it.
    Callers passing the same ``(namespace, flag_name)`` pair share one warning
    budget, so distinct warning categories must use distinct flag names or one
    would silence the other.

    :param namespace: Mutable mapping holding the named boolean flag, typically
        a module's ``globals()``.
    :param flag_name: Name of the boolean flag gating this warning.
    :param message: Fully formatted warning text to log the first time.
    :param warning_logger: Logger to emit through; defaults to this module's logger.
    :return: ``None``.
    """
    if namespace[flag_name]:
        return
    namespace[flag_name] = True
    warning_logger.warning(message)


def _log_warning_once(flag_name: str, message: str) -> None:
    """Log one process-wide warning gated by a named module-level flag.

    Thin wrapper around :func:`log_warning_once` bound to this module's own
    globals and logger, kept for this module's internal call sites.

    :param flag_name: Name of the module-level boolean flag gating this warning.
    :param message: Fully formatted warning text to log the first time.
    :return: ``None``.
    """
    log_warning_once(globals(), flag_name, message)


def warn_once(action: str, exc: Exception) -> None:
    """Log one process-wide warning for a cache failure, then stay quiet.

    :param action: Short label identifying the failing cache operation.
    :param exc: Captured exception.
    :return: ``None``.
    """
    message = (
        f"Embedding cache {action} failed ({type(exc).__name__}: {exc}); "
        "continuing without cache benefits for this run."
    )
    collector = _cache_warning_collector.get()
    if collector is not None and message not in collector:
        collector.append(message)
    _log_warning_once("_warned_cache_error", message)


@contextlib.contextmanager
def capture_cache_warnings(collector: list[str] | None) -> Iterator[None]:
    """Route cache failure messages into one caller-owned run collector.

    :param collector: Mutable warning list, or ``None`` to collect nothing.
    :return: Context manager restoring the previous collector on exit.
    """
    token = _cache_warning_collector.set(collector)
    try:
        yield
    finally:
        _cache_warning_collector.reset(token)


def is_cache_disabled() -> bool:
    """Return whether the global embedding-cache kill switch is set.

    :return: ``True`` when ``CODEDUPES_NO_CACHE`` is set to a truthy value.
    """
    return os.environ.get("CODEDUPES_NO_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_cache_dir() -> Path:
    """Resolve the embedding cache root directory from environment overrides.

    :return: ``CODEDUPES_CACHE_DIR`` if set, else ``$XDG_CACHE_HOME/codedupes`` if
        ``XDG_CACHE_HOME`` is set, else ``~/.cache/codedupes``; always fully
        resolved (symlinks dereferenced, relative paths absolutized).
    """
    override = os.environ.get("CODEDUPES_CACHE_DIR")
    if override:
        root = Path(override).expanduser()
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        root = (
            Path(xdg_cache).expanduser() / "codedupes"
            if xdg_cache
            else Path.home() / ".cache" / "codedupes"
        )
    # Resolve every branch so each spelling of one physical root (a relative
    # override, a symlinked ~/.cache) yields one identity: shard lock names key
    # on the absolute path, so unresolved spellings would split the lock domain.
    return root.resolve()


def _resolve_max_bytes() -> int:
    """Resolve the opportunistic cache size cap in bytes.

    ``0``, negative, and sub-1-MB fractional values (for example ``"0.5"``) are
    all rejected the same way an unparsable value is, rather than clamped up
    to a minimum 1 MB: silently substituting a thrashing 1 MB cache for a
    value that looks like an attempt to disable the cap, or a typo, would
    surprise the caller, and there is no supported way to disable the cap via
    this variable (use ``CODEDUPES_NO_CACHE`` instead).

    :return: Size cap in bytes from ``CODEDUPES_CACHE_MAX_MB``, defaulting to
        ``DEFAULT_CACHE_MAX_MB`` megabytes when unset, unparsable, or less than
        1 MB; values at or above 1 MB are floored to a whole number of
        megabytes.
    """
    raw = os.environ.get("CODEDUPES_CACHE_MAX_MB")
    if raw:
        try:
            value = float(raw)
            if not math.isfinite(value) or value < 1:
                raise ValueError
            return int(value) * 1024 * 1024
        except (OverflowError, ValueError):
            _log_warning_once(
                "_warned_invalid_cache_max_mb",
                f"Ignoring CODEDUPES_CACHE_MAX_MB={raw!r} (must be a positive number "
                f"of megabytes); using the default {DEFAULT_CACHE_MAX_MB} MB cap.",
            )
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
    :param text: Complete prepared embedding input text.
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


def _path_exists(path: Path) -> bool:
    """Check existence without letting a stat failure escape.

    ``Path.exists()`` re-raises stat errors other than the ENOENT family, so an
    unreadable cache tree (foreign-owned ``0700`` directories, restrictive ACLs)
    would crash code paths that promise never-fatal degradation. A failed stat
    reads as absent; the write path still warns once when it also fails.

    :param path: Path to check.
    :return: ``True`` when the path exists and is stat-able, else ``False``.
    """
    try:
        return path.exists()
    except OSError:
        return False


def _shard_lock_path(shard_dir: Path) -> Path:
    """Resolve the stable advisory-lock path for one cache shard.

    Lock files must live outside ``shard_dir`` because cache clearing and LRU
    eviction delete that directory recursively. Unlinking a held lock file lets
    another process recreate the pathname on a new inode and acquire an
    independent lock while the original inode is still locked. A digest of the
    absolute shard path keeps the lock identity stable across shard deletion and
    recreation without exposing long model/revision names in lock filenames.

    :param shard_dir: Cache shard path under ``<cache-root>/repos/<repo>/<model>``.
    :return: Lock path under the cache root's persistent ``locks`` directory.
    """
    cache_root = shard_dir.parents[2]
    # Hash the logical path without following a pre-planted symlink. Following
    # one here would split lock identity from the cache-managed shard name before
    # the writer has a chance to reject that symlink. ``absolute()`` rather than
    # ``resolve()`` keeps symlinks unfollowed; inputs are already normalized
    # because shards derive from the resolved cache root.
    identity = str(shard_dir.absolute())
    lock_name = f"{hashlib.blake2b(identity.encode(), digest_size=16).hexdigest()}.lock"
    return cache_root / LOCKS_SUBDIR / lock_name


def _remove_shard_lock_file(shard_dir: Path) -> None:
    """Unlink a deleted shard's lock file so ``locks/`` cannot grow forever.

    Must be called while still holding the shard's write lock, and only when
    this process itself deleted the shard directory - an already-absent shard
    means a concurrent deleter owns reclamation, and unlinking here could take
    out a lock file a recreating writer currently holds. A waiter already
    blocked on the old inode can still acquire it after release while a
    newcomer locks the recreated pathname, leaving two writers serialized on
    different inodes; that needs two overlapping deleters (concurrent ``clear``
    calls, or ``clear`` racing eviction) plus a writer recreating the shard,
    and the generation re-confirmation in ``_read_shard`` keeps the fallout
    availability-only, so reclaiming the directory entry is worth the narrow
    race.

    :param shard_dir: Shard directory whose lock file should be reclaimed.
    :return: ``None``.
    """
    with contextlib.suppress(OSError):
        _shard_lock_path(shard_dir).unlink()


def _ensure_managed_directory(path: Path) -> None:
    """Create one cache-managed directory without accepting a symlink in its place.

    The user-selected cache root may intentionally be a symlink, but deterministic
    descendants owned by codedupes (``repos``, repo/model shards, and ``locks``)
    must be real directories so a pre-planted link cannot redirect cache writes.

    :param path: One cache-managed directory whose parent already exists.
    :raises OSError: If ``path`` is or becomes a symlink/non-directory.
    :return: ``None``.
    """
    if path.is_symlink():
        raise OSError(f"Refusing symlinked cache directory: {path}")
    path.mkdir(mode=_CACHE_DIRECTORY_MODE, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise OSError(f"Cache path is not a real directory: {path}")
    path.chmod(_CACHE_DIRECTORY_MODE)


def ensure_cache_subdirectory(cache_root: Path, name: str) -> Path:
    """Create a private cache root and one real managed child directory.

    The configured root may intentionally be a symlink, so only its target is
    validated and permission-hardened. Deterministic child names remain subject
    to the stricter no-symlink rule in :func:`_ensure_managed_directory`.

    :param cache_root: User-selected codedupes cache root.
    :param name: Direct child directory managed by codedupes.
    :raises OSError: If the root/child is not a directory or permissions fail.
    :return: Created managed child path.
    """
    root_existed = cache_root.is_dir()
    cache_root.mkdir(mode=_CACHE_DIRECTORY_MODE, parents=True, exist_ok=True)
    if not cache_root.is_dir():
        raise OSError(f"Cache root is not a directory: {cache_root}")
    if not root_existed:
        # Harden only roots this process created; a pre-existing user-chosen
        # root may be deliberately shared with other tools or accounts.
        cache_root.chmod(_CACHE_DIRECTORY_MODE)
    managed_dir = cache_root / name
    _ensure_managed_directory(managed_dir)
    return managed_dir


def _ensure_shard_directory(shard_dir: Path) -> None:
    """Create the deterministic cache hierarchy for one shard without symlinks.

    :param shard_dir: Target shard under ``<cache-root>/repos/<repo>/<model>``.
    :raises OSError: If a cache-managed component is a symlink/non-directory.
    :return: ``None``.
    """
    cache_root = shard_dir.parents[2]
    repos_dir = ensure_cache_subdirectory(cache_root, CACHE_SUBDIR)
    if shard_dir.parents[1] != repos_dir:
        raise OSError(f"Cache shard is outside the managed repository directory: {shard_dir}")
    for managed_dir in (shard_dir.parent, shard_dir):
        _ensure_managed_directory(managed_dir)


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
        lock_path = _shard_lock_path(shard_dir)
        ensure_cache_subdirectory(lock_path.parent.parent, LOCKS_SUBDIR)
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, _CACHE_FILE_MODE)
        os.fchmod(lock_fd, _CACHE_FILE_MODE)
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
    if not isinstance(keys_map, dict):
        return None
    for key, row in keys_map.items():
        if not isinstance(key, str):
            return None
        if isinstance(row, bool) or not isinstance(row, int) or row < 0:
            return None

    # Every indexed row needs a namespace and digest for that same content key.
    # Row bounds are checked later, once the corresponding matrix is loaded.
    cache_keys = set(keys_map)
    if not isinstance(namespaces, dict) or set(namespaces) != cache_keys:
        return None
    for namespace in namespaces.values():
        if not isinstance(namespace, str):
            return None
    if not isinstance(digests, dict) or set(digests) != cache_keys:
        return None
    for digest in digests.values():
        if not isinstance(digest, str):
            return None
    if not isinstance(generation, str) or _GENERATION_PATTERN.fullmatch(generation) is None:
        return None

    source_commit = payload.get("source_commit")
    if source_commit is not None and (not isinstance(source_commit, str) or not source_commit):
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
        "source_commit": source_commit,
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


def _index_stat_signature(index_path: Path) -> tuple[int, int, int] | None:
    """Cheaply fingerprint an index file's identity for the shard-read cache.

    :param index_path: Path to one shard's ``index.json``.
    :return: ``(inode, mtime_ns, size)``, or ``None`` when the file cannot be stat-ed.
    """
    try:
        stat_result = index_path.stat()
    except OSError:
        return None
    return (stat_result.st_ino, stat_result.st_mtime_ns, stat_result.st_size)


# Keep one validated snapshot per shard in this process. Writers atomically
# replace index.json, so its stat signature lets repeated lookups reuse a
# snapshot without parsing JSON and opening the vector mmap again.
_shard_read_cache: dict[str, tuple[tuple[int, int, int], _ShardData]] = {}


def _read_shard(shard_dir: Path) -> _ShardData | None:
    """Load and validate one shard's vectors and key index.

    Any structural inconsistency (a corrupt vector generation, a stale ``index.json``
    pointing past the end of its matrix, schema drift, and so on) is treated as an
    empty shard rather than raised, matching the never-fatal cache contract.

    Reuses the last validated snapshot when the index's stat signature is
    unchanged. A writer can advance the shard after this check; the returned
    snapshot still pairs its own row map, matrix, and source commit together.

    :param shard_dir: Shard directory to load.
    :return: Validated shard data, or ``None`` when the shard is missing, unreadable,
        internally inconsistent, or replaced by a concurrent writer.
    """
    index_path = shard_dir / INDEX_FILENAME
    cache_key = str(shard_dir)
    pre_signature = _index_stat_signature(index_path)
    if pre_signature is None:
        _shard_read_cache.pop(cache_key, None)
        return None

    cached = _shard_read_cache.get(cache_key)
    if cached is not None:
        cached_signature, cached_shard = cached
        if cached_signature == pre_signature:
            return cached_shard

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        initial_generation = _peek_generation(index)
        if initial_generation is None:
            raise ValueError(f"invalid shard metadata at {shard_dir}")
        vectors_path = shard_dir / _vectors_filename(initial_generation)
        # Memory-mapped so sparse lookups (for example a single query key) only
        # fault in the rows they touch; hit rows are copied before being returned.
        vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
        # A writer may have switched the index while the matrix was opening.
        # Confirm its generation before trusting any of the row mappings.
        confirmed_index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - corrupt on-disk data can fail in many ways
        warn_once("read shard", exc)
        _shard_read_cache.pop(cache_key, None)
        return None

    metadata = _validate_shard(confirmed_index, vectors)
    if metadata is None or metadata["generation"] != initial_generation:
        warn_once("read shard", ValueError(f"inconsistent shard at {shard_dir}"))
        _shard_read_cache.pop(cache_key, None)
        return None

    shard_data = _ShardData(
        vectors=vectors,
        keys=metadata["keys"],
        namespaces=metadata["namespaces"],
        digests=metadata["digests"],
        last_used_at=metadata["last_used_at"],
        generation=metadata["generation"],
        source_commit=metadata["source_commit"],
    )

    # This snapshot is internally consistent. Remember it for reuse only when
    # the index stayed unchanged throughout the read; otherwise re-read next time.
    post_signature = _index_stat_signature(index_path)
    if post_signature == pre_signature:
        _shard_read_cache[cache_key] = (post_signature, shard_data)
    else:
        _shard_read_cache.pop(cache_key, None)
    return shard_data


def _reclaim_stale_shard_files(shard_dir: Path, keep: frozenset[Path] = frozenset()) -> None:
    """Delete unpublished files abandoned by a writer that never reached cleanup.

    Tmp files are only ever created by a writer holding this shard's exclusive
    lock, so any tmp file found here while that lock is held (as it is by every
    caller of this helper) was orphaned by a process that died mid-write (SIGKILL,
    power loss) before its own ``finally`` block could remove it. A process that
    dies after renaming its vector matrix but before publishing the index leaves a
    properly named generation orphan instead. The index is authoritative, so every
    generation other than the one it names is equally safe to reclaim under lock.
    When an index exists but no well-formed generation can be peeked from it
    (foreign schema, torn write, transient read error), the vectors sweep is
    skipped entirely: deleting what might be another codedupes version's active
    generation would silently destroy its shard, and orphans cost only disk.

    :param shard_dir: Shard directory to sweep; caller must hold its write lock.
    :param keep: Tmp paths belonging to the current write, left untouched.
    :return: ``None``.
    """
    for stale_tmp in shard_dir.glob(_TMP_FILE_GLOB):
        if stale_tmp not in keep:
            with contextlib.suppress(OSError):
                stale_tmp.unlink()

    active_vectors: Path | None = None
    index_path = shard_dir / INDEX_FILENAME
    if _path_exists(index_path):
        try:
            payload: Any = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            payload = None
        generation = _peek_generation(payload)
        if generation is None:
            return
        active_vectors = shard_dir / _vectors_filename(generation)
    for stale_vectors in shard_dir.glob("vectors-*.npy"):
        if stale_vectors != active_vectors and stale_vectors not in keep:
            with contextlib.suppress(OSError):
                stale_vectors.unlink()


def atomic_write_json(path: Path, obj: Any) -> None:
    """Atomically publish one JSON file via a temp-write-then-replace.

    Writes ``obj`` to a collision-resistant temp path beside ``path`` (created
    with ``O_EXCL`` so two writers can never share one temp file), then swaps
    it into place with :func:`os.replace` so readers always see either the
    prior complete file or the new one, never a partial write. The temp file
    is best-effort cleaned up if anything raises before the swap.

    :param path: Destination file path; its parent directory must already exist.
    :param obj: JSON-serializable payload to write.
    :raises OSError: If the temp file cannot be created, or the payload cannot
        be serialized or written.
    :return: ``None``.
    """
    tmp_path = path.parent / f"{path.name}{_tmp_suffix()}"
    try:
        tmp_fd = os.open(
            tmp_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            _CACHE_FILE_MODE,
        )
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            json.dump(obj, handle)
        os.replace(tmp_path, path)
    finally:
        if _path_exists(tmp_path):
            with contextlib.suppress(OSError):
                tmp_path.unlink()


def _read_corpus_manifest(shard_dir: Path) -> CorpusManifest | None:
    """Read one valid corpus manifest, returning ``None`` when unavailable.

    :param shard_dir: Cache shard containing the manifest.
    :return: Parsed manifest, or ``None`` when absent or invalid.
    """
    path = shard_dir / MANIFEST_FILENAME
    if not _path_exists(path):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA:
        return None
    manifest_generation = payload.get("generation")
    selections = payload.get("selections")
    if (
        not isinstance(manifest_generation, int)
        or manifest_generation < 0
        or not isinstance(selections, dict)
    ):
        return None
    parsed: dict[str, CorpusSelectionManifest] = {}
    for selection, entry in selections.items():
        if not isinstance(selection, str) or not isinstance(entry, dict):
            return None
        last_seen_generation = entry.get("last_seen_generation")
        complete_scan = entry.get("complete_scan")
        units = entry.get("units")
        orphans = entry.get("orphans")
        unit_paths = entry.get("unit_paths")
        if not isinstance(last_seen_generation, int):
            return None
        if not 0 <= last_seen_generation <= manifest_generation:
            return None
        if not isinstance(complete_scan, bool):
            return None

        # Validate each map separately so its meaning and membership rules are
        # visible. In particular, an orphan stores a scan generation, not an age.
        if not isinstance(units, dict):
            return None
        for uid, key in units.items():
            if not isinstance(uid, str) or not isinstance(key, str):
                return None
        if not isinstance(orphans, dict):
            return None
        for key, orphaned_generation in orphans.items():
            if not isinstance(key, str) or not isinstance(orphaned_generation, int):
                return None
            if orphaned_generation < 0:
                return None
        if not isinstance(unit_paths, dict):
            return None
        for uid, path in unit_paths.items():
            if not isinstance(uid, str) or not isinstance(path, str):
                return None
            if uid not in units:
                return None
        parsed[selection] = CorpusSelectionManifest(
            last_seen_generation=last_seen_generation,
            complete_scan=complete_scan,
            units=dict(units),
            orphans=dict(orphans),
            unit_paths=dict(unit_paths),
        )
    return CorpusManifest(
        schema=MANIFEST_SCHEMA,
        generation=manifest_generation,
        selections=parsed,
    )


def _write_corpus_manifest(shard_dir: Path, manifest: CorpusManifest) -> None:
    """Atomically publish one corpus manifest under the shard lock.

    :param shard_dir: Cache shard receiving the manifest.
    :param manifest: Complete manifest to publish.
    :return: ``None``.
    """
    selections = {}
    for name, entry in manifest.selections.items():
        selections[name] = {
            "last_seen_generation": entry.last_seen_generation,
            "complete_scan": entry.complete_scan,
            "units": entry.units,
            "orphans": entry.orphans,
            "unit_paths": entry.unit_paths,
        }
    payload = {
        "schema": manifest.schema,
        "generation": manifest.generation,
        "selections": selections,
    }
    atomic_write_json(shard_dir / MANIFEST_FILENAME, payload)


def _manifest_referenced_keys(manifest: CorpusManifest) -> set[str]:
    """Return cache keys referenced by a recently refreshed selection.

    :param manifest: Multi-selection corpus manifest.
    :return: Cache keys pinned by at least one active selection.
    """
    referenced_keys: set[str] = set()
    for selection in manifest.selections.values():
        scans_since_refresh = manifest.generation - selection.last_seen_generation
        if scans_since_refresh >= ORPHAN_GC_GENERATIONS:
            continue
        referenced_keys.update(selection.units.values())
    return referenced_keys


def _manifest_orphan_keys(manifest: CorpusManifest) -> set[str]:
    """Return tracked orphan keys, including keys pinned by another selection.

    :param manifest: Multi-selection corpus manifest.
    :return: Cache keys classified as deleted by at least one selection.
    """
    orphan_keys: set[str] = set()
    for selection in manifest.selections.values():
        orphan_keys.update(selection.orphans)
    return orphan_keys


def _manifest_collectable_keys(manifest: CorpusManifest) -> set[str]:
    """Find unpinned keys whose orphan records have all aged through the grace period.

    :param manifest: Manifest after a complete scan has advanced its generation.
    :return: Orphaned content keys eligible for row compaction.
    """
    # Start with every recorded orphan, then let any active reference or recent
    # orphan record protect the shared row. The newest orphan record wins.
    collectable = _manifest_orphan_keys(manifest)
    collectable.difference_update(_manifest_referenced_keys(manifest))
    for selection in manifest.selections.values():
        for key, orphaned_generation in selection.orphans.items():
            orphan_age = manifest.generation - orphaned_generation
            if orphan_age < ORPHAN_GC_GENERATIONS:
                collectable.discard(key)
    return collectable


def _merge_partial_manifest_units(
    previous: CorpusSelectionManifest,
    units: dict[str, str],
    unit_paths: dict[str, str],
    observed_files: tuple[str, ...],
) -> tuple[dict[str, str], dict[str, str]]:
    """Replace observed file slices while retaining units from unseen files.

    :param previous: Selection baseline before the partial scan.
    :param units: Unit UID-to-content-key observations from this run.
    :param unit_paths: File paths for the observed units.
    :param observed_files: Exact file paths fully covered by this scan.
    :return: Merged unit and path maps for the selection.
    """
    observed_file_set = set(observed_files)
    merged_units = dict(previous.units)
    for uid in previous.units:
        if previous.unit_paths.get(uid) in observed_file_set:
            del merged_units[uid]
    merged_units.update(units)

    merged_paths = {}
    for uid, path in previous.unit_paths.items():
        if uid in merged_units:
            merged_paths[uid] = path
    merged_paths.update(unit_paths)
    return merged_units, merged_paths


def _limit_manifest_selections(
    selections: dict[str, CorpusSelectionManifest],
    *,
    current_selection: str,
) -> dict[str, CorpusSelectionManifest]:
    """Keep pending orphan state plus the most recently used selection baselines.

    :param selections: Candidate selection entries for one shard manifest.
    :param current_selection: Selection published by the current run.
    :return: Selection entries retained in the bounded manifest.
    """
    if len(selections) <= MANIFEST_SELECTION_LIMIT:
        return selections

    # Pending orphan records must survive until collection, even if keeping
    # them exceeds the baseline cap. Always retain the selection just published.
    retained_names = {current_selection}
    for name, entry in selections.items():
        if entry.orphans:
            retained_names.add(name)

    # Insertion order records publication recency. Fill any remaining slots
    # from newest to oldest, then return entries in their original order.
    for name in reversed(selections):
        if len(retained_names) >= MANIFEST_SELECTION_LIMIT:
            break
        retained_names.add(name)
    retained = {}
    for name, entry in selections.items():
        if name in retained_names:
            retained[name] = entry
    return retained


def _publish_index(shard_dir: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a shard's ``index.json``, cleaning its tmp file on failure.

    :param shard_dir: Shard directory holding the index; caller must hold its write lock.
    :param payload: JSON-serializable index payload.
    :return: ``None``.
    """
    atomic_write_json(shard_dir / INDEX_FILENAME, payload)


def _atomic_write_shard(
    shard_dir: Path,
    canonical_model: str,
    revision: str | None,
    vectors: np.ndarray,
    keys_map: dict[str, int],
    namespaces: dict[str, str],
    digests: dict[str, str],
    dim: int,
    source_commit: str | None,
) -> None:
    """Publish a complete shard generation through one atomic index replacement.

    Each vector matrix has an immutable generation-specific filename. The matrix
    is fully written before ``index.json`` atomically switches to that generation,
    so readers can never pair an older key map with a rebuilt matrix. A crash
    before the index replacement leaves only an unreferenced file; a crash after
    it leaves the new generation complete. Always runs under the shard write lock,
    so it also reclaims files orphaned by a prior writer that crashed.

    :param shard_dir: Shard directory to write into (created if missing).
    :param canonical_model: Canonical model identifier.
    :param revision: Resolved model revision, or ``None`` when unpinned.
    :param vectors: Full float32 vector matrix to persist.
    :param keys_map: Key-to-row mapping to persist.
    :param namespaces: Key-to-input-namespace mapping used for namespace capping.
    :param digests: Key-to-row-digest mapping used for read-time integrity checks.
    :param dim: Embedding dimensionality.
    :param source_commit: Concrete checkpoint commit the vectors derive from, or
        ``None`` when the revision itself is immutable. Lives inside the atomically
        switched index so provenance can never desynchronize from the generation
        it describes.
    :raises AssertionError: If ``keys_map``/``namespaces``/``digests`` have
        drifted apart (see :meth:`_ShardData.assert_consistent`); this is the
        single choke point every shard publish passes through, so it is
        where a mutation-path bug fails loudly instead of quietly writing an
        inconsistent shard that only degrades to a miss on the next read.
    :return: ``None``.
    """
    _ShardData(
        vectors=vectors,
        keys=keys_map,
        namespaces=namespaces,
        digests=digests,
        last_used_at=0.0,
        generation="",
        source_commit=source_commit,
    ).assert_consistent()

    _ensure_shard_directory(shard_dir)
    generation = uuid.uuid4().hex
    vectors_filename = _vectors_filename(generation)
    vectors_path = shard_dir / vectors_filename
    vectors_tmp = shard_dir / f"{vectors_filename}{_tmp_suffix()}"
    _reclaim_stale_shard_files(shard_dir, keep=frozenset({vectors_tmp}))
    try:
        vectors_fd = os.open(
            vectors_tmp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            _CACHE_FILE_MODE,
        )
        with os.fdopen(vectors_fd, "wb") as handle:
            np.save(handle, np.ascontiguousarray(vectors, dtype=np.float32))
        os.replace(vectors_tmp, vectors_path)

        # Readers discover generations through the index. Publish it only after
        # the complete matrix is in place, with row maps and provenance together.
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
                "source_commit": source_commit,
            },
        )

        _reclaim_stale_shard_files(shard_dir, keep=frozenset({vectors_path}))
    finally:
        if _path_exists(vectors_tmp):
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
            _reclaim_stale_shard_files(shard_dir)
            payload = _read_shard_meta(shard_dir)
            if payload is None:
                return
            payload["last_used_at"] = time.time()
            _publish_index(shard_dir, payload)
    except OSError as exc:
        warn_once("touch shard", exc)


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
    new_vectors = np.empty((len(retained_keys), dim), dtype=np.float32)
    new_keys_map = {}
    new_namespaces = {}
    new_digests = {}
    # Compaction changes row numbers, not content keys or row contents. Copy each
    # retained vector and its metadata together in the requested row order.
    for new_row, key in enumerate(retained_keys):
        old_row = keys_map[key]
        new_vectors[new_row] = vectors[old_row]
        new_keys_map[key] = new_row
        new_namespaces[key] = namespaces[key]
        new_digests[key] = digests[key]
    return new_vectors, new_keys_map, new_namespaces, new_digests


def _write_shard_entries(
    shard_dir: Path,
    canonical_model: str,
    revision: str | None,
    entries: Sequence[tuple[str, np.ndarray]],
    *,
    namespace: str,
    max_namespace_keys: int | None = None,
    expected_source_commit: str | None = None,
    require_source_commit: bool = False,
) -> None:
    """Append/heal embedding rows and cap overflowing namespace keys.

    :param shard_dir: Shard directory to update.
    :param canonical_model: Canonical model identifier.
    :param revision: Resolved model revision, or ``None`` when unpinned.
    :param entries: Sequence of ``(key, vector)`` pairs to append or, for keys that
        already exist with a poisoned (NaN/Inf) stored row, heal in place.
    :param namespace: Stable identifier for one mode/instruction/dtype combination.
    :param max_namespace_keys: Maximum keys allowed in ``namespace`` before an
        amortized prune drops the oldest rows to 80% of the cap, or ``None`` for
        no cap.
    :param expected_source_commit: Checkpoint commit the entries were computed
        under, for mutable-label shards. Revalidated here, under the same lock
        that publishes the generation: a shard whose recorded provenance no
        longer matches rejects the whole batch rather than assembling rows from
        two checkpoints into one generation.
    :param require_source_commit: Whether the caller requires verified incoming
        checkpoint provenance; reject an unstamped batch instead of inheriting
        any existing shard's source commit.
    :return: ``None``.
    """
    if not entries:
        return
    if require_source_commit and expected_source_commit is None:
        logger.warning(
            f"Discarding {len(entries)} computed embeddings for {shard_dir.name}: "
            "the incoming batch has no verified source commit"
        )
        return
    try:
        # One row per content key: the last supplied value wins, while insertion
        # order keeps the first occurrence's position in this batch.
        unique_entries = list(dict(entries).items())
        entry_dim = int(np.asarray(unique_entries[0][1]).size)

        _ensure_shard_directory(shard_dir)
        with _shard_write_lock(shard_dir) as acquired:
            if not acquired:
                return
            _reclaim_stale_shard_files(shard_dir)
            existing = _read_shard(shard_dir)
            if (
                expected_source_commit is not None
                and existing is not None
                and existing.keys
                and existing.source_commit != expected_source_commit
            ):
                # Another process may have rebuilt the shard while this batch
                # was encoding. Its checkpoint must still match before we merge.
                logger.warning(
                    f"Discarding {len(unique_entries)} computed embeddings for {shard_dir.name}: "
                    "the shard's recorded source commit changed while they were being computed"
                )
                return
            if existing is not None and existing.vectors.shape[1] == entry_dim:
                dim = int(existing.vectors.shape[1])
                # Readers may share this snapshot. Copy its maps now; mutation
                # methods replace the matrix or copy its read-only mmap first.
                working = _ShardData(
                    vectors=existing.vectors,
                    keys=dict(existing.keys),
                    namespaces=dict(existing.namespaces),
                    digests=dict(existing.digests),
                    last_used_at=existing.last_used_at,
                    generation=existing.generation,
                    source_commit=existing.source_commit,
                )
                publish_source_commit = (
                    expected_source_commit
                    if expected_source_commit is not None
                    else existing.source_commit
                )
            else:
                if existing is not None:
                    logger.warning(
                        "Embedding cache vector dimension changed from "
                        f"{existing.vectors.shape[1]} to {entry_dim} for {shard_dir}; "
                        f"replacing all {len(existing.keys)} entries in the incompatible shard."
                    )
                dim = entry_dim
                working = _ShardData(
                    vectors=np.empty((0, dim), dtype=np.float32),
                    keys={},
                    namespaces={},
                    digests={},
                    last_used_at=0.0,
                    generation="",
                    source_commit=None,
                )
                # Any prior rows are discarded with the incompatible matrix, so
                # only the incoming batch's provenance describes this generation.
                publish_source_commit = expected_source_commit
            existing = None

            missing_entries: list[tuple[str, np.ndarray]] = []
            overwrite_entries: list[tuple[str, np.ndarray]] = []
            for key, vector in unique_entries:
                row = working.keys.get(key)
                if row is None:
                    missing_entries.append((key, vector))
                    continue

                # A valid existing row needs no write. A row rejected by reads
                # must be repaired or the same key will miss on every future run.
                cached_vector = working.vectors[row]
                if not _is_finite_row(cached_vector):
                    overwrite_entries.append((key, vector))
                    continue
                if working.digests[key] != _row_digest(cached_vector):
                    overwrite_entries.append((key, vector))

            # overwrite_rows/append_rows update vectors/keys/namespaces/digests
            # together, so the four structures can never observe a partial update.
            working.overwrite_rows(overwrite_entries, namespace)
            working.append_rows(missing_entries, namespace)

            capped_namespace = False
            if max_namespace_keys is not None:
                # Row order preserves insertion order through compaction. This
                # namespace uses FIFO eviction; other namespaces keep their rows.
                namespace_keys_by_row = []
                for key, key_namespace in working.namespaces.items():
                    if key_namespace == namespace:
                        namespace_keys_by_row.append(key)
                namespace_keys_by_row.sort(key=working.keys.__getitem__)
                if len(namespace_keys_by_row) > max_namespace_keys:
                    # Leave headroom so the next new query does not immediately
                    # force another full-matrix compaction.
                    prune_target = 0
                    if max_namespace_keys > 0:
                        prune_target = max(
                            1, int(max_namespace_keys * _NAMESPACE_PRUNE_TARGET_RATIO)
                        )
                    drop_count = len(namespace_keys_by_row) - prune_target
                    drop_keys = set(namespace_keys_by_row[:drop_count])
                    retained_keys = []
                    for key in working.keys:
                        if key not in drop_keys:
                            retained_keys.append(key)
                    retained_keys.sort(key=working.keys.__getitem__)
                    working.retain(retained_keys)
                    capped_namespace = True

            if missing_entries or overwrite_entries or capped_namespace:
                _atomic_write_shard(
                    shard_dir,
                    canonical_model,
                    revision,
                    working.vectors,
                    working.keys,
                    working.namespaces,
                    working.digests,
                    dim,
                    publish_source_commit,
                )
    except Exception as exc:  # noqa: BLE001 - cache writes must never break analysis
        warn_once("write shard", exc)


def _iter_shard_dirs(repos_dir: Path) -> list[Path]:
    """List every shard directory under the cache repos root.

    :param repos_dir: Root directory holding all per-repo shard directories.
    :return: Sorted list of shard directory paths, empty when ``repos_dir`` is absent.
    """
    if not _path_exists(repos_dir):
        return []
    try:
        repo_dirs = sorted(path for path in repos_dir.iterdir() if path.is_dir())
    except OSError:
        return []

    shard_dirs: list[Path] = []
    for repo_dir in repo_dirs:
        try:
            shard_dirs.extend(sorted(path for path in repo_dir.iterdir() if path.is_dir()))
        except OSError:
            # Cache clear or another process's eviction may remove this repo
            # directory after the root snapshot. The remaining repos are still
            # useful and safe to inventory.
            continue
    return shard_dirs


def _read_shard_meta(shard_dir: Path) -> dict[str, Any] | None:
    """Read a shard's ``index.json`` without loading its vector matrix.

    :param shard_dir: Shard directory to inspect.
    :return: Parsed index payload, or ``None`` when missing/unreadable.
    """
    index_path = shard_dir / INDEX_FILENAME
    if not _path_exists(index_path):
        return None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        return _validate_shard_metadata(payload)
    except (OSError, ValueError, RecursionError):
        return None


def _shard_size_bytes(shard_dir: Path) -> int:
    """Sum on-disk file sizes for one shard directory.

    :param shard_dir: Shard directory to measure.
    :return: Total bytes used by files directly inside ``shard_dir``.
    """
    total = 0
    try:
        files = list(shard_dir.glob("*"))
    except OSError:
        return 0
    for file_path in files:
        try:
            if file_path.is_file():
                total += file_path.stat().st_size
        except OSError:
            # A whole-shard deletion can race the gap between is_file() and
            # stat(); omit that vanished file and continue the inventory.
            continue
    return total


def _prune_empty_repo_dirs(repos_dir: Path) -> None:
    """Remove per-repo directories left empty after shard deletion.

    :param repos_dir: Root directory holding all per-repo shard directories.
    :return: ``None``.
    """
    if not _path_exists(repos_dir):
        return
    try:
        repo_dirs = list(repos_dir.iterdir())
    except OSError:
        return
    for repo_dir in repo_dirs:
        try:
            if repo_dir.is_dir() and not any(repo_dir.iterdir()):
                repo_dir.rmdir()
        except OSError:
            # Another process may populate or delete the directory after the
            # emptiness check; either outcome needs no corrective action here.
            continue


def _delete_cache_tree(path: Path, *, action: str) -> _CacheTreeDeletion:
    """Delete one cache directory without hiding whether removal failed.

    :param path: Cache directory to remove recursively.
    :param action: Short label used in the best-effort cache warning.
    :return: Outcome distinguishing successful removal, an already-absent tree,
        and a failed deletion.
    """
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return _CacheTreeDeletion(removed=False, failed=False)
    except OSError as exc:
        warn_once(action, exc)
        return _CacheTreeDeletion(removed=False, failed=True)
    # Drop any cached read snapshot promptly (see _shard_read_cache) instead of
    # waiting for the next _read_shard call's stat mismatch to evict it, so a
    # deleted shard's mmap file handle is released as soon as it is deleted.
    _shard_read_cache.pop(str(path), None)
    return _CacheTreeDeletion(removed=True, failed=False)


_EVICT_SCAN_INTERVAL_SECONDS = 300.0
_EVICT_SCAN_BYTES_RATIO = 0.02

# Per-repos_dir throttle state for _should_scan_for_eviction: (last scan time,
# bytes written since that scan). Keyed by the resolved string path so
# multiple cache roots in one process (as in the test suite) never share a
# throttle budget.
_evict_scan_state: dict[str, tuple[float, int]] = {}


def _should_scan_for_eviction(repos_dir: Path, written_bytes: int, max_bytes: int) -> bool:
    """Decide whether one ``put_many`` call should pay for a full eviction scan.

    ``_maybe_evict`` walks and ``stat()``s every file in every shard, so paying
    that cost on every ``put_many`` call - as semantic.py does once per
    single-entry query-cache miss - makes every miss O(total cache size). This
    gate scans only when it is due: on the first call ever seen for
    ``repos_dir`` (nothing scanned yet), once ``_EVICT_SCAN_INTERVAL_SECONDS``
    has elapsed since the last scan, or once accumulated bytes written since
    the last scan reach ``_EVICT_SCAN_BYTES_RATIO`` of the size cap.

    Trade-off: between scans the cache can grow past its cap by up to
    whichever bound fires first - roughly ``_EVICT_SCAN_BYTES_RATIO`` of the
    cap in writes, or ``_EVICT_SCAN_INTERVAL_SECONDS`` of write traffic. The
    cap is still enforced eventually (every call updates this gate's state, so
    a burst of small writes cannot postpone a scan forever), just not on every
    single call. A missed scan only delays eviction, never skips it outright.

    :param repos_dir: Root directory holding all per-repo shard directories;
        the throttle key.
    :param written_bytes: Approximate bytes this call is about to write.
    :param max_bytes: Resolved cache size cap in bytes.
    :return: ``True`` when the caller should run :func:`_maybe_evict` now.
    """
    key = str(repos_dir)
    now = time.time()
    state = _evict_scan_state.get(key)
    if state is None:
        _evict_scan_state[key] = (now, 0)
        return True

    last_scan_at, pending_bytes = state
    pending_bytes += max(written_bytes, 0)
    threshold = max(1, int(max_bytes * _EVICT_SCAN_BYTES_RATIO))
    due = (now - last_scan_at) >= _EVICT_SCAN_INTERVAL_SECONDS or pending_bytes >= threshold
    _evict_scan_state[key] = (now, 0) if due else (last_scan_at, pending_bytes)
    return due


def _maybe_evict(repos_dir: Path, protect: Path | None = None) -> None:
    """Evict least-recently-used shards once the cache exceeds its size cap.

    Callers on a hot path should gate this behind :func:`_should_scan_for_eviction`
    (as ``put_many`` does) rather than calling it unconditionally, since every
    call here walks and stats the whole cache tree; this function itself always
    scans when called; it does not throttle itself.

    :param repos_dir: Root directory holding all per-repo shard directories.
    :param protect: Shard directory exempt from eviction (the one just written),
        so a cap smaller than one shard cannot instantly delete fresh work.
    :return: ``None``.
    """
    try:
        max_bytes = _resolve_max_bytes()
        sized: list[tuple[Path, int]] = []
        for shard_dir in _iter_shard_dirs(repos_dir):
            try:
                sized.append((shard_dir, _shard_size_bytes(shard_dir)))
            except OSError:
                # A concurrently deleted shard must not prevent other shards
                # from being considered for the global cap.
                continue
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
                if _delete_cache_tree(shard_dir, action="evict shard").removed:
                    total -= size
                    _remove_shard_lock_file(shard_dir)
        _prune_empty_repo_dirs(repos_dir)
        if total > target:
            logger.warning(
                "Embedding cache still exceeds its size target after eviction "
                f"({total} bytes > {target}); consider raising CODEDUPES_CACHE_MAX_MB."
            )
    except Exception as exc:  # noqa: BLE001 - eviction must never break analysis
        warn_once("evict", exc)


class EmbeddingCache:
    """File-based embedding cache addressed by repo-root, model, and revision shards."""

    def __init__(self, cache_root: Path | None = None) -> None:
        """Bind a cache handle to a cache root directory.

        :param cache_root: Explicit cache root, defaults to :func:`resolve_cache_dir`.
            Always fully resolved (symlinks dereferenced, relative paths
            absolutized) so an unresolved spelling of one physical root can
            never split the advisory-lock domain :func:`_shard_lock_path` keys
            on resolved paths.
        """
        self.cache_root = (cache_root or resolve_cache_dir()).resolve()

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

    def get_many_with_provenance(
        self,
        cache_scope: Path,
        canonical_model: str,
        revision: str | None,
        keys: list[str],
    ) -> CacheLookup:
        """Look up cached vectors plus the source commit of the snapshot they came from.

        The vectors and the source commit are taken from one ``_read_shard``
        snapshot, so the caller can later prove which checkpoint produced every
        returned row even after a concurrent writer purges and republishes the
        shard - the shard's *current* provenance says nothing about vectors
        copied out of an earlier generation.

        The recency stamp only feeds shard-granularity LRU eviction, so refreshing it
        on every read would waste an index rewrite per lookup and widen the window in
        which a reader's stale key map can clobber a concurrent writer's index.

        :param cache_scope: Analyzed corpus root path.
        :param canonical_model: Canonical model identifier.
        :param revision: Resolved model revision, or ``None`` when unpinned.
        :param keys: Cache keys to look up.
        :return: Hit vectors and the snapshot's recorded source commit (``None``
            for a miss, an immutable-revision shard, or provenance-less rows).
        """
        if not keys:
            return CacheLookup(vectors={}, source_commit=None)
        shard_dir = self.shard_dir(cache_scope, canonical_model, revision)
        loaded = _read_shard(shard_dir)
        if loaded is None:
            return CacheLookup(vectors={}, source_commit=None)
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
        return CacheLookup(vectors=hits, source_commit=loaded.source_commit)

    def get_many(
        self,
        cache_scope: Path,
        canonical_model: str,
        revision: str | None,
        keys: list[str],
    ) -> dict[str, np.ndarray]:
        """Look up cached embedding vectors, refreshing the recency stamp at most hourly.

        :param cache_scope: Analyzed corpus root path.
        :param canonical_model: Canonical model identifier.
        :param revision: Resolved model revision, or ``None`` when unpinned.
        :param keys: Cache keys to look up.
        :return: Mapping of hit keys to owned float32 embedding vectors.
        """
        return self.get_many_with_provenance(cache_scope, canonical_model, revision, keys).vectors

    def put_many(
        self,
        cache_scope: Path,
        canonical_model: str,
        revision: str | None,
        entries: Sequence[tuple[str, np.ndarray]],
        *,
        namespace: str = "default",
        max_namespace_keys: int | None = None,
        expected_source_commit: str | None = None,
        require_source_commit: bool = False,
    ) -> None:
        """Insert vectors, cap overflowing namespaces, enforce the global size cap.

        :param cache_scope: Analyzed corpus root path.
        :param canonical_model: Canonical model identifier.
        :param revision: Resolved model revision, or ``None`` when unpinned.
        :param entries: Sequence of ``(key, vector)`` pairs to store.
        :param namespace: Stable identifier for one mode/instruction/dtype combination.
        :param max_namespace_keys: Maximum keys allowed in ``namespace`` before an
            amortized prune drops the oldest rows to 80% of the cap, or ``None``
            for no cap.
        :param expected_source_commit: Checkpoint commit the entries were computed
            under, required for mutable-label shards; the write is rejected under
            the shard lock when the shard's recorded provenance no longer matches.
        :param require_source_commit: Whether this write requires a non-null
            ``expected_source_commit``, as required for mutable-label embeddings.
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
            expected_source_commit=expected_source_commit,
            require_source_commit=require_source_commit,
        )
        # Eviction scans the whole cache tree, so it is throttled rather than
        # run on every call (see _should_scan_for_eviction) - callers that need
        # an unconditional scan (tests, admin flows) call _maybe_evict directly.
        max_bytes = _resolve_max_bytes()
        written_bytes = sum(int(np.asarray(vector).nbytes) for _key, vector in entries)
        if _should_scan_for_eviction(self.repos_dir, written_bytes, max_bytes):
            _maybe_evict(self.repos_dir, protect=shard_dir)

    def load_manifest(
        self,
        cache_scope: Path,
        canonical_model: str,
        revision: str | None,
    ) -> CorpusManifest | None:
        """Load the corpus manifest for one cache shard.

        :param cache_scope: Analyzed corpus root.
        :param canonical_model: Canonical model identifier.
        :param revision: Cache revision addressing the shard.
        :return: Parsed manifest, or ``None`` when unavailable.
        """
        return _read_corpus_manifest(self.shard_dir(cache_scope, canonical_model, revision))

    def collect_orphans(
        self,
        cache_scope: Path,
        canonical_model: str,
        revision: str | None,
        drop_keys: set[str],
        *,
        expected_manifest: CorpusManifest | None = None,
        expected_shard_generation: str | None = None,
    ) -> int:
        """Compact one shard by removing the requested orphaned code keys.

        :param cache_scope: Analyzed corpus root.
        :param canonical_model: Canonical model identifier.
        :param revision: Cache revision addressing the shard.
        :param drop_keys: Manifest-owned code keys eligible for collection.
        :param expected_manifest: Manifest snapshot that made the keys eligible;
            a newer publication cancels collection.
        :param expected_shard_generation: Vector-index generation observed with
            ``expected_manifest``; a newer vector write cancels collection.
        :return: Number of rows removed.
        """
        if not drop_keys:
            return 0
        shard_dir = self.shard_dir(cache_scope, canonical_model, revision)
        try:
            with _shard_write_lock(shard_dir, blocking=True) as acquired:
                if not acquired:
                    return 0
                if (
                    expected_manifest is not None
                    and _read_corpus_manifest(shard_dir) != expected_manifest
                ):
                    return 0
                if expected_shard_generation is not None:
                    current_meta = _read_shard_meta(shard_dir)
                    if (
                        current_meta is None
                        or current_meta["generation"] != expected_shard_generation
                    ):
                        return 0
                _reclaim_stale_shard_files(shard_dir)
                existing = _read_shard(shard_dir)
                if existing is None:
                    return 0
                actual_drop = set(existing.keys) & drop_keys
                if not actual_drop:
                    return 0
                working = _ShardData(
                    vectors=existing.vectors,
                    keys=dict(existing.keys),
                    namespaces=dict(existing.namespaces),
                    digests=dict(existing.digests),
                    last_used_at=existing.last_used_at,
                    generation=existing.generation,
                    source_commit=existing.source_commit,
                )
                # Preserve row order so GC cannot change FIFO eviction order for
                # query rows sharing this shard with the corpus vectors.
                retained_keys = []
                for key in working.keys:
                    if key not in actual_drop:
                        retained_keys.append(key)
                retained_keys.sort(key=working.keys.__getitem__)
                working.retain(retained_keys)
                _atomic_write_shard(
                    shard_dir,
                    canonical_model,
                    revision,
                    working.vectors,
                    working.keys,
                    working.namespaces,
                    working.digests,
                    int(working.vectors.shape[1]),
                    working.source_commit,
                )
                return len(actual_drop)
        except Exception as exc:  # noqa: BLE001 - cache maintenance must never break analysis
            warn_once("collect orphan rows", exc)
            return 0

    def publish_corpus_manifest(
        self,
        cache_scope: Path,
        canonical_model: str,
        revision: str | None,
        *,
        selection: str,
        units: dict[str, str],
        complete_scan: bool,
        unit_paths: dict[str, str] | None = None,
        observed_files: tuple[str, ...] = (),
    ) -> ManifestPublishResult | None:
        """Publish a completed corpus run and age or collect unreferenced rows.

        :param cache_scope: Analyzed corpus root.
        :param canonical_model: Canonical model identifier.
        :param revision: Cache revision addressing the shard.
        :param selection: Digest of semantic candidate-selection settings.
        :param units: Current unit-to-cache-key mapping.
        :param complete_scan: Whether missing units represent the complete selection.
        :param unit_paths: Unit UID-to-file-path mapping for ``units``.
        :param observed_files: Exact file paths fully observed by an incomplete
            scan, allowing those slices of the baseline to be replaced.
        :return: Publication and GC telemetry, or ``None`` on cache failure.
        """
        shard_dir = self.shard_dir(cache_scope, canonical_model, revision)
        try:
            _ensure_shard_directory(shard_dir)
            with _shard_write_lock(shard_dir, blocking=True) as acquired:
                if not acquired:
                    return None
                manifest = _read_corpus_manifest(shard_dir) or CorpusManifest(
                    schema=MANIFEST_SCHEMA,
                    generation=0,
                    selections={},
                )
                shard_meta = _read_shard_meta(shard_dir)
                cached_keys = set(shard_meta["keys"]) if shard_meta is not None else None
                expected_shard_generation = (
                    shard_meta["generation"] if shard_meta is not None else None
                )
                # Only complete scans advance the shared orphan-aging clock.
                generation = manifest.generation
                if complete_scan:
                    generation += 1

                # Copy the previous baselines, dropping orphan records for rows
                # already gone or content observed again in this run. Unknown
                # shard metadata gives no evidence that a row is gone.
                current_keys = set(units.values())
                selections = {}
                for name, entry in manifest.selections.items():
                    pending_orphans = {}
                    for key, orphaned_generation in entry.orphans.items():
                        if key in current_keys:
                            continue
                        if cached_keys is not None and key not in cached_keys:
                            continue
                        pending_orphans[key] = orphaned_generation
                    selections[name] = CorpusSelectionManifest(
                        last_seen_generation=entry.last_seen_generation,
                        complete_scan=entry.complete_scan,
                        units=dict(entry.units),
                        orphans=pending_orphans,
                        unit_paths=dict(entry.unit_paths),
                    )

                # Reinsertion below moves this entry to the end, which records
                # publication recency for deterministic selection-cap eviction.
                previous = selections.pop(selection, None)

                # A complete scan replaces its baseline. A partial scan can
                # replace only observed files; missing siblings remain unknown.
                current_units = dict(units)
                current_unit_paths = dict(unit_paths or {})
                if previous is not None and not complete_scan:
                    current_units, current_unit_paths = _merge_partial_manifest_units(
                        previous, current_units, current_unit_paths, observed_files
                    )

                if previous is None:
                    # The first observation establishes a baseline, so there is
                    # no earlier population from which to infer moves/deletions.
                    diff = ManifestDiff(moved=[], deleted=[], orphaned=set())
                    orphans = {}
                else:
                    diff = diff_manifest(previous, current_units)
                    orphans = dict(previous.orphans)
                    for key in diff.orphaned:
                        if cached_keys is not None and key not in cached_keys:
                            continue
                        # Keep the first orphan generation across repeated scans.
                        orphans.setdefault(key, generation)

                if complete_scan or previous is None:
                    last_seen_generation = generation
                else:
                    # A partial run cannot refresh pins for the unseen units
                    # carried forward in current_units.
                    last_seen_generation = previous.last_seen_generation
                selections[selection] = CorpusSelectionManifest(
                    last_seen_generation=last_seen_generation,
                    complete_scan=complete_scan,
                    units=current_units,
                    orphans=orphans,
                    unit_paths=current_unit_paths,
                )
                selections = _limit_manifest_selections(
                    selections,
                    current_selection=selection,
                )
                manifest = CorpusManifest(
                    schema=MANIFEST_SCHEMA,
                    generation=generation,
                    selections=selections,
                )
                collectable: set[str] = set()
                if complete_scan:
                    collectable = _manifest_collectable_keys(manifest)
                _write_corpus_manifest(shard_dir, manifest)

            # Collection takes its own lock and rechecks both snapshots. A
            # concurrent manifest update or vector write cancels this attempt.
            collected = self.collect_orphans(
                cache_scope,
                canonical_model,
                revision,
                collectable,
                expected_manifest=manifest,
                expected_shard_generation=expected_shard_generation,
            )
            if collected:
                # Remove completed orphan records only if this is still our
                # manifest. A newer publication must retain its own bookkeeping.
                with _shard_write_lock(shard_dir, blocking=True) as acquired:
                    if acquired:
                        current = _read_corpus_manifest(shard_dir)
                        if current == manifest:
                            for entry in current.selections.values():
                                for key in collectable:
                                    entry.orphans.pop(key, None)
                            current.selections = _limit_manifest_selections(
                                current.selections,
                                current_selection=selection,
                            )
                            _write_corpus_manifest(shard_dir, current)
                            manifest = current
            return ManifestPublishResult(
                diff=diff,
                generation=generation,
                orphan_rows_retained=len(_manifest_orphan_keys(manifest)),
                orphan_rows_collected=collected,
            )
        except Exception as exc:  # noqa: BLE001 - cache metadata must never break analysis
            warn_once("publish corpus manifest", exc)
            return None

    def confirm_source_commit(
        self,
        cache_scope: Path,
        canonical_model: str,
        revision: str | None,
        loaded_commit: str,
    ) -> bool:
        """Confirm a mutable-label shard's vectors derive from one loaded commit.

        Loose revision keying addresses a shard by the requested branch/tag
        label, so an upstream branch move would otherwise let cache hits
        computed under the old commit assemble into one matrix with fresh
        vectors from the new commit. Mixing requires a model load, and a load
        knows its commit: provenance lives inside the atomically switched
        shard index, so vectors and the commit that produced them can never
        desynchronize on disk. Rows whose recorded commit differs from the
        loaded one - or whose index records no commit at all, the
        corruption/legacy case - cannot be tied to this checkpoint and the
        whole shard is purged (old-commit code and query vectors alike). The
        write path stamps the loaded commit when the fresh rows publish. Never
        raises; a cache-layer error keeps degrading to plain loose-keying
        behavior.

        Confirmation vouches only for the shard's *current* generation. A
        concurrent run may have purged and republished the shard under the
        loaded commit after this caller copied hits out of an earlier
        snapshot, so callers holding pre-load hits must additionally compare
        the snapshot's own provenance from ``get_many_with_provenance``.

        :param cache_scope: Analyzed corpus root path.
        :param canonical_model: Canonical model identifier.
        :param revision: Mutable revision label addressing the shard.
        :param loaded_commit: Commit hash reported by the loaded model.
        :return: ``False`` when the shard was purged, so callers must discard
            any pre-load hits; ``True`` when the current shard is coherent
            with ``loaded_commit``.
        """
        try:
            shard_dir = self.shard_dir(cache_scope, canonical_model, revision)
            meta = _read_shard_meta(shard_dir)
            if meta is None or not meta["keys"] or meta["source_commit"] == loaded_commit:
                # Nothing readable to protect (misses recompute anyway), or
                # provenance already matches: lock-free fast path.
                return True
            with _shard_write_lock(shard_dir, blocking=True) as acquired:
                if not acquired:
                    return True
                meta = _read_shard_meta(shard_dir)
                if meta is None or not meta["keys"] or meta["source_commit"] == loaded_commit:
                    return True
                # Everything under this label predates the branch move or has
                # no provable checkpoint. The shard lock file stays: the shard
                # is recreated immediately and generation re-confirmation
                # covers racing readers.
                _delete_cache_tree(shard_dir, action="purge drifted shard")
                return False
        except Exception as exc:  # noqa: BLE001 - cache must never break analysis
            warn_once("confirm source commit", exc)
            return True

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
                try:
                    meta = _read_shard_meta(shard_dir)
                    count = len(meta.get("keys", {})) if meta else 0
                    size = _shard_size_bytes(shard_dir)
                except OSError:
                    # Report the rest of the cache when clear/eviction removes
                    # this shard between directory discovery and aggregation.
                    continue
                info["entries"] += count
                info["size_bytes"] += size
                if meta is not None:
                    model_name = meta["model"]
                    info["models"][model_name] = info["models"].get(model_name, 0) + count
                totals = repo_totals.setdefault(
                    shard_dir.parent.name,
                    {
                        "shards": 0,
                        "entries": 0,
                        "size_bytes": 0,
                        "orphan_rows": 0,
                        "last_complete_generation": 0,
                    },
                )
                totals["shards"] += 1
                totals["entries"] += count
                totals["size_bytes"] += size
                manifest = _read_corpus_manifest(shard_dir)
                if manifest is not None:
                    totals["orphan_rows"] += len(_manifest_orphan_keys(manifest))
                    totals["last_complete_generation"] = max(
                        totals["last_complete_generation"],
                        manifest.generation,
                    )
            local_models_dir = self.cache_root / LOCAL_MODELS_SUBDIR
            if local_models_dir.is_dir():
                info["size_bytes"] += sum(
                    f.stat().st_size for f in local_models_dir.glob("*") if f.is_file()
                )
        except Exception as exc:  # noqa: BLE001 - stats must never break analysis
            warn_once("stats", exc)
        info["repos"] = [{"repo": name, **totals} for name, totals in sorted(repo_totals.items())]
        return info

    def clear(self, model: str | None = None) -> CacheClearResult:
        """Delete cached embeddings, optionally scoped to one canonical model.

        When ``model`` is given, a shard whose ``index.json`` cannot be read or
        parsed is skipped rather than deleted, since its recorded model name is
        unknown and cannot be matched against the filter; a corrupt shard for
        the targeted model can therefore survive a scoped clear. A full
        unscoped ``clear()`` (``model=None``) does not consult per-shard
        metadata to decide whether to delete, so it still removes such shards.

        :param model: Canonical model name to scope deletion to, or ``None`` to
            clear every shard across every repo plus local-model digest manifests.
        :return: Entry removal count and number of failed deletion operations.
        """
        removed = 0
        failures = 0
        try:
            for shard_dir in _iter_shard_dirs(self.repos_dir):
                try:
                    # Wait for any concurrent writer rather than deleting under it:
                    # writers hold this lock only briefly, and a dead holder's flock
                    # self-releases.
                    with _shard_write_lock(shard_dir, blocking=True) as acquired:
                        if not acquired:
                            failures += 1
                            continue
                        meta = _read_shard_meta(shard_dir)
                        if model is not None and (meta is None or meta.get("model") != model):
                            continue
                        deletion = _delete_cache_tree(shard_dir, action="clear shard")
                        failures += int(deletion.failed)
                        if deletion.removed:
                            removed += len(meta.get("keys", {})) if meta else 0
                            _remove_shard_lock_file(shard_dir)
                except OSError as exc:
                    # One unreadable or undeletable shard must not abort the sweep;
                    # every other shard still gets cleared and counted.
                    warn_once("clear", exc)
                    failures += 1
            _prune_empty_repo_dirs(self.repos_dir)
            if model is None:
                # Manifests are keyed by local model directory, not canonical model
                # name, so they are only removed on a full clear. Machine capability
                # records are per-environment, not per-model, and follow the same rule.
                deletion = _delete_cache_tree(
                    self.cache_root / LOCAL_MODELS_SUBDIR,
                    action="clear local-model manifests",
                )
                failures += int(deletion.failed)
        except Exception as exc:  # noqa: BLE001 - clear must never break analysis
            warn_once("clear", exc)
            failures += 1
        return CacheClearResult(removed_entries=removed, failed_deletions=failures)


def get_embedding_cache() -> EmbeddingCache | None:
    """Build a fresh embedding cache handle unless caching is globally disabled.

    Construction itself (resolving the cache root, which may call ``Path.home()``)
    is wrapped like every other public cache operation: a failure here must
    degrade to the same cache-disabled shape ``CODEDUPES_NO_CACHE`` produces,
    never propagate into the analysis path.

    :return: New :class:`EmbeddingCache` instance, or ``None`` when
        ``CODEDUPES_NO_CACHE`` is set or cache construction itself fails.
    """
    if is_cache_disabled():
        return None
    try:
        return EmbeddingCache()
    except Exception as exc:  # noqa: BLE001 - cache construction must never break analysis
        warn_once("initialize", exc)
        return None
