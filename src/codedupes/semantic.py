"""Semantic duplicate detection using embedding similarity."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import json
import logging
import os
import sys
import textwrap
import threading
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import numpy as np
from packaging.version import InvalidVersion, Version

from codedupes.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECK_SEMANTIC_TASK,
    DEFAULT_MODEL,
    DEFAULT_SEARCH_SEMANTIC_TASK,
    DEFAULT_SEMANTIC_DEVICE,
    DEFAULT_TOP_K,
    SemanticTask,
    normalize_semantic_task,
)
from codedupes.devices import (
    DeviceConfigurationError,
    clear_device_cache,
    configure_mps_environment,
    configure_mps_memory_fraction,
    format_mps_memory_snapshot,
    is_mlx_loaded,
    resolve_cpu_bf16_inference,
    resolve_semantic_device,
    restore_mps_memory_fraction_if_managed,
)
from codedupes.embedding_cache import (
    LOCAL_MODELS_SUBDIR,
    EmbeddingCache,
    _ensure_cache_subdirectory,
    compute_cache_key,
    get_embedding_cache,
    is_cache_disabled,
    resolve_cache_dir,
)
from codedupes.models import CodeUnit, DuplicatePair
from codedupes.pairs import ordered_pair_key
from codedupes.semantic_profiles import (
    SemanticModelProfile,
    get_default_search_threshold,
    get_default_semantic_threshold,
    is_explicit_local_model_path,
    resolve_local_model_path,
    resolve_model_profile,
)

logger = logging.getLogger(__name__)

# Lazy-loaded model
_model = None
_model_name: str | None = None
_model_revision: str | None = None
_model_trust_remote_code: bool | None = None
_model_local_fingerprint: str | None = None
_model_device_key: str | None = None
_model_execution_device: str | None = None
_model_lock = threading.RLock()
_warned_mlx_mps_contention = False
_warned_cpu_fallback_reuse = False

# Per-file content digests for local model directories, memoized so warm-path
# fingerprinting stays a stat walk instead of rehashing gigabytes of weights.
_LOCAL_MODEL_MANIFEST_VERSION = 1
_local_model_manifest_lock = threading.Lock()


@dataclass(frozen=True)
class _LocalModelManifestState:
    """In-memory digest manifest and whether the same content is on disk."""

    files: dict[str, dict[str, object]]
    persisted: bool


_local_model_manifest_memo: dict[str, _LocalModelManifestState] = {}

_TORCH_MIN_RELEASE = (2, 13)
_TORCH_MAX_EXCLUSIVE_RELEASE = (3,)

EMBEDDINGGEMMA_QUERY_PREFIXES: dict[SemanticTask, str] = {
    "semantic-similarity": "task: sentence similarity | query: ",
    "code-retrieval": "task: code retrieval | query: ",
    "retrieval": "task: search result | query: ",
    "question-answering": "task: question answering | query: ",
    "fact-verification": "task: fact checking | query: ",
    "classification": "task: classification | query: ",
    "clustering": "task: clustering | query: ",
}
EMBEDDINGGEMMA_DOCUMENT_PREFIX = "title: none | text: "

EncodeRoute = Literal["symmetric", "query", "document"]


@dataclass(frozen=True)
class EncodePlan:
    """Structured encode configuration: backend route plus explicit prompt.

    The prompt is passed to the backend's encode call so it is applied exactly
    once. SentenceTransformers ``encode_query``/``encode_document`` prepend the
    model's saved query/document prompt whenever no explicit prompt is given,
    so prompts must never also be prepended to the input text.
    """

    route: EncodeRoute
    prompt: str | None = None

    def cache_identity(self) -> str:
        """Serialize the vector-affecting encode configuration for cache keys.

        ``None`` (backend applies its saved prompt) and ``""`` (explicitly
        empty prompt) reach the backend differently, so they must not share
        an identity.

        :return: Stable string covering route and effective prompt.
        """
        return (
            f"route={self.route}\x00prompt_set={int(self.prompt is not None)}"
            f"\x00prompt={self.prompt or ''}"
        )


@dataclass(frozen=True)
class EmbeddingSpaceIdentity:
    """Identity of the coordinate system used for one corpus matrix."""

    model_name: str
    resolved_revision: str | None
    runtime_variant: str


def _resolve_encode_plan(
    profile: SemanticModelProfile,
    mode: Literal["code", "query"],
    semantic_task: SemanticTask,
    instruction_prefix: str | None,
) -> EncodePlan:
    """Resolve the encode route and prompt for one model/mode/task combination.

    Duplicate detection is a symmetric similarity task, so EmbeddingGemma code
    inputs use the symmetric route with the task's query prompt. Retrieval-task
    code inputs are search corpus documents and use the document route; query
    inputs always use the query route. A custom instruction prefix replaces the
    model prompt while preserving the route.

    :param profile: Resolved model profile.
    :param mode: Embedding input mode.
    :param semantic_task: Normalized semantic task.
    :param instruction_prefix: Optional explicit prompt override.
    :return: Encode plan applied exactly once at the backend call.
    """
    if profile.family == "embeddinggemma":
        if mode == "query":
            route: EncodeRoute = "query"
            prompt: str | None = EMBEDDINGGEMMA_QUERY_PREFIXES[semantic_task]
        elif semantic_task in {"retrieval", "code-retrieval"}:
            route = "document"
            prompt = EMBEDDINGGEMMA_DOCUMENT_PREFIX
        else:
            route = "symmetric"
            prompt = EMBEDDINGGEMMA_QUERY_PREFIXES[semantic_task]
        if instruction_prefix is not None:
            prompt = instruction_prefix
        return EncodePlan(route=route, prompt=prompt)

    return EncodePlan(route="symmetric", prompt=instruction_prefix)


def _select_encode_fn(model: Any, route: EncodeRoute) -> Callable[..., np.ndarray]:
    """Select the backend encode callable for one route.

    :param model: Loaded embedding model.
    :param route: Encode route from the resolved plan.
    :return: Bound encode callable, falling back to ``model.encode``.
    """
    if route == "query" and hasattr(model, "encode_query"):
        return cast(Callable[..., np.ndarray], model.encode_query)
    if route == "document" and hasattr(model, "encode_document"):
        return cast(Callable[..., np.ndarray], model.encode_document)
    return cast(Callable[..., np.ndarray], model.encode)


class SemanticBackendError(RuntimeError):
    """Raised when semantic model loading or inference backend is incompatible."""


class InvalidEmbeddingError(RuntimeError):
    """Raised when an embedding matrix violates the similarity-math invariants.

    ``retryable`` marks value-level corruption (NaN/Inf/zero rows) that a CPU
    re-encode may fix; structural mismatches (wrong shape or row count) indicate
    a code or backend bug and must fail immediately.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        """Record the failure message and whether a CPU re-encode may fix it.

        :param message: Human-readable description of the violated invariant.
        :param retryable: ``True`` when the corruption is value-level and worth
            one CPU retry.
        """
        super().__init__(message)
        self.retryable = retryable


def canonicalize_embeddings(
    values: object,
    *,
    expected_rows: int,
    expected_dim: int | None = None,
) -> np.ndarray:
    """Validate and renormalize one embedding matrix.

    Dot product equals cosine similarity only for finite, nonzero,
    unit-normalized rows, so every freshly encoded matrix passes through here
    before it is cached, compared, or returned.

    :param values: Backend encode output.
    :param expected_rows: Required row count (one per input text).
    :param expected_dim: Required embedding dimensionality, or ``None`` to accept any.
    :return: Contiguous float32 matrix with unit-normalized rows.
    :raises InvalidEmbeddingError: If shape, row count, dimensionality, finiteness,
        or norm invariants are violated.
    """
    matrix = np.asarray(values, dtype=np.float32)

    if matrix.ndim != 2:
        raise InvalidEmbeddingError(f"Expected a 2D embedding matrix, got shape {matrix.shape!r}")
    if matrix.shape[0] != expected_rows:
        raise InvalidEmbeddingError(f"Expected {expected_rows} rows, got {matrix.shape[0]}")
    if expected_dim is not None and matrix.shape[1] != expected_dim:
        raise InvalidEmbeddingError(f"Expected dimension {expected_dim}, got {matrix.shape[1]}")

    if not np.isfinite(matrix).all():
        raise InvalidEmbeddingError(
            "Embedding matrix contains NaN or infinity",
            retryable=True,
        )

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise InvalidEmbeddingError(
            "Embedding matrix contains a zero or invalid vector",
            retryable=True,
        )

    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _configure_semantic_runtime_env(
    device: str | None = DEFAULT_SEMANTIC_DEVICE,
    *,
    mps_fallback: bool | None = None,
) -> None:
    """Set semantic runtime guards before importing model frameworks.

    :param device: Requested semantic execution device.
    :param mps_fallback: Explicit MPS unsupported-op fallback setting, or ``None``
        for automatic behavior that respects an existing environment override.
    :return: ``None``.
    """
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    configure_mps_environment(device, fallback=mps_fallback)


T = TypeVar("T")


def _resolve_profile_default(
    model_name: str,
    override: T | None,
    *,
    accessor: Callable[[Any], T],
) -> T:
    """Resolve a profile-derived default with optional explicit override.

    :param model_name: Requested model identifier.
    :param override: Explicit override value.
    :param accessor: Profile accessor for default lookup.
    :return: Explicit override when provided, otherwise profile default.
    """
    if override is not None:
        return override
    profile = resolve_model_profile(model_name)
    return accessor(profile)


def _resolve_model_revision(model_name: str, revision: str | None) -> str | None:
    """Resolve model revision for a model, honoring explicit overrides.

    :param model_name: Requested model identifier.
    :param revision: Optional explicit revision.
    :return: Profile default revision when no explicit revision is provided.
    """
    return _resolve_profile_default(
        model_name,
        revision,
        accessor=lambda profile: cast(str | None, profile.default_revision),
    )


def _resolve_trust_remote_code(model_name: str, trust_remote_code: bool | None) -> bool:
    """Resolve trust-remote-code mode for a model, honoring explicit overrides.

    :param model_name: Requested model identifier.
    :param trust_remote_code: Optional explicit trust setting.
    :return: Profile default trust setting when no override is provided.
    """
    return _resolve_profile_default(
        model_name,
        trust_remote_code,
        accessor=lambda profile: cast(bool, profile.default_trust_remote_code),
    )


def _resolve_hf_cached_revision(
    canonical_model: str,
    revision: str = "main",
) -> str | None:
    """Resolve a locally cached Hub revision to its commit hash without downloading.

    :param canonical_model: Canonical HuggingFace model identifier.
    :param revision: Branch, tag, or commit revision to resolve, defaults to ``"main"``.
    :return: Resolved commit hash, or ``None`` when it cannot be determined offline.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        cached = try_to_load_from_cache(canonical_model, "config.json", revision=revision)
    except Exception:  # noqa: BLE001 - offline revision lookup must never block analysis
        return None
    if not isinstance(cached, str):
        return None
    parts = Path(cached).parts
    try:
        snapshots_index = parts.index("snapshots")
    except ValueError:
        return None
    if snapshots_index + 1 >= len(parts):
        return None
    return parts[snapshots_index + 1]


def _is_hf_commit_hash(revision: str) -> bool:
    """Return whether a revision is an immutable 40-character Git commit hash.

    :param revision: Hub revision string to classify.
    :return: ``True`` for a full hexadecimal SHA-1 commit hash.
    """
    return len(revision) == 40 and all(
        character in "0123456789abcdefABCDEF" for character in revision
    )


def _hash_file_content(file_path: Path) -> str:
    """Stream a file's bytes into a stable content digest.

    :param file_path: File to hash.
    :return: Hexadecimal blake2b digest of the file contents.
    """
    digest = hashlib.blake2b(digest_size=16)
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_model_manifest_path(model_dir: Path) -> Path:
    """Resolve the on-disk digest-manifest path for a local model directory.

    :param model_dir: Resolved local model directory.
    :return: Manifest path under the codedupes cache root.
    """
    key = hashlib.blake2b(str(model_dir).encode(), digest_size=8).hexdigest()
    return resolve_cache_dir() / LOCAL_MODELS_SUBDIR / f"{key}.json"


def _manifest_digest_for(entry: object, identity: tuple[int, int, int, int]) -> str | None:
    """Return a manifest entry's content digest when its stat identity still matches.

    :param entry: Manifest entry payload for one file, of unverified shape.
    :param identity: Current ``(size, mtime_ns, ctime_ns, ino)`` of the file.
    :return: Reusable content digest, or ``None`` when the file must be rehashed.
    """
    if not isinstance(entry, dict):
        return None
    recorded = (
        entry.get("size"),
        entry.get("mtime_ns"),
        entry.get("ctime_ns"),
        entry.get("ino"),
    )
    digest = entry.get("digest")
    if recorded != identity or not isinstance(digest, str) or not digest:
        return None
    return digest


def _load_local_model_manifest(
    model_dir: Path,
    *,
    persist_manifest: bool,
) -> dict[str, dict[str, object]]:
    """Load the digest manifest for a local model directory, tolerating any failure.

    :param model_dir: Resolved local model directory.
    :param persist_manifest: Whether the on-disk digest manifest may be read.
    :return: Mapping of relative path to digest entry; empty when unavailable.
    """
    memo_key = str(model_dir)
    with _local_model_manifest_lock:
        state = _local_model_manifest_memo.get(memo_key)
        if state is not None:
            return state.files
    if not persist_manifest or is_cache_disabled():
        return {}
    try:
        payload = json.loads(_local_model_manifest_path(model_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict) or payload.get("version") != _LOCAL_MODEL_MANIFEST_VERSION:
        return {}
    files = payload.get("files")
    if not isinstance(files, dict):
        return {}
    with _local_model_manifest_lock:
        _local_model_manifest_memo[memo_key] = _LocalModelManifestState(
            files=files,
            persisted=True,
        )
    return files


def _store_local_model_manifest(
    model_dir: Path,
    files: dict[str, dict[str, object]],
    *,
    persist_manifest: bool,
) -> None:
    """Persist the digest manifest in-process and, best effort, on disk.

    :param model_dir: Resolved local model directory.
    :param files: Fresh manifest entries keyed by relative path.
    :param persist_manifest: Whether the digest manifest may be written to disk.
    :return: ``None``.
    """
    memo_key = str(model_dir)
    with _local_model_manifest_lock:
        previous_state = _local_model_manifest_memo.get(memo_key)
        already_persisted = (
            previous_state is not None
            and previous_state.persisted
            and previous_state.files == files
        )
        _local_model_manifest_memo[memo_key] = _LocalModelManifestState(
            files=files,
            persisted=already_persisted,
        )
    if not persist_manifest or is_cache_disabled():
        return
    manifest_path = _local_model_manifest_path(model_dir)
    if already_persisted and manifest_path.is_file():
        return
    tmp_path = manifest_path.with_name(f"{manifest_path.name}.{os.getpid()}.tmp")
    try:
        _ensure_cache_subdirectory(resolve_cache_dir(), LOCAL_MODELS_SUBDIR)
        payload = json.dumps(
            {
                "version": _LOCAL_MODEL_MANIFEST_VERSION,
                "model_dir": str(model_dir),
                "files": files,
            }
        )
        manifest_fd = os.open(
            tmp_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        with os.fdopen(manifest_fd, "w", encoding="utf-8") as handle:
            # File mode bits are a POSIX concern; Windows has no ``fchmod``.
            with contextlib.suppress(AttributeError):
                os.fchmod(handle.fileno(), 0o600)
            handle.write(payload)
        os.replace(tmp_path, manifest_path)
        with _local_model_manifest_lock:
            current_state = _local_model_manifest_memo.get(memo_key)
            if current_state is not None and current_state.files == files:
                _local_model_manifest_memo[memo_key] = _LocalModelManifestState(
                    files=files,
                    persisted=True,
                )
    except OSError:
        logger.debug(
            f"Could not persist local-model digest manifest for {model_dir}", exc_info=True
        )
    finally:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()


def _fingerprint_local_model_dir(
    model_dir: Path,
    *,
    persist_manifest: bool = True,
) -> str | None:
    """Fingerprint a local model directory's contents for use as a cache revision.

    The fingerprint hashes each model file's relative path and content digest,
    excluding Hugging Face's ``--local-dir`` download metadata, so replacing or
    retraining weights in place changes the cache revision while metadata-only
    touches keep it stable. The walk follows symlinks
    (``os.walk(..., followlinks=True)``) so weight shards stored behind a
    symlinked subdirectory move the fingerprint too; a visited-realpath set
    guards against symlink cycles. Digests are reused from a manifest keyed on
    the full stat identity ``(size, mtime_ns, ctime_ns, inode)``; unchanged
    files are never rehashed, keeping warm-path key derivation cheap enough to
    run before any model import. The in-place guarantee is POSIX-specific:
    there ``st_ctime`` is the inode-change time, which every content write
    moves even when size and mtime are restored. On Windows ``st_ctime`` is
    the creation time, so a same-size in-place overwrite with a preserved
    mtime could reuse a stale digest.

    :param model_dir: Resolved local model directory.
    :param persist_manifest: Whether per-file digests may be loaded from and saved
        to the persistent manifest. In-memory digest reuse remains enabled.
    :return: ``"dir-<hex>"`` fingerprint, or ``None`` when the walk fails.
    """
    manifest = _load_local_model_manifest(
        model_dir,
        persist_manifest=persist_manifest,
    )
    fresh_manifest: dict[str, dict[str, object]] = {}
    entries: list[tuple[str, str]] = []
    hf_download_metadata = model_dir / ".cache" / "huggingface"
    visited_dirs: set[str] = set()
    try:
        for dirpath, dirnames, filenames in os.walk(model_dir, followlinks=True):
            real_dir = os.path.realpath(dirpath)
            if real_dir in visited_dirs:
                # A symlink cycle looped back to an already-processed real
                # directory: stop descending and skip re-processing its files.
                dirnames[:] = []
                continue
            visited_dirs.add(real_dir)
            for filename in filenames:
                file_path = Path(dirpath) / filename
                if not file_path.is_file():
                    continue
                if file_path.is_relative_to(hf_download_metadata):
                    continue
                stat = file_path.stat()
                relative = file_path.relative_to(model_dir).as_posix()
                identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
                content_digest = _manifest_digest_for(manifest.get(relative), identity)
                if content_digest is None:
                    content_digest = _hash_file_content(file_path)
                fresh_manifest[relative] = {
                    "size": identity[0],
                    "mtime_ns": identity[1],
                    "ctime_ns": identity[2],
                    "ino": identity[3],
                    "digest": content_digest,
                }
                entries.append((relative, content_digest))
    except OSError:
        return None
    if not entries:
        return None
    entries.sort()
    _store_local_model_manifest(
        model_dir,
        fresh_manifest,
        persist_manifest=persist_manifest,
    )
    digest = hashlib.blake2b(repr(entries).encode(), digest_size=12).hexdigest()
    return f"dir-{digest}"


def _validate_local_model_directory(model_dir: Path) -> None:
    """Validate the minimum files needed to load a local embedding model.

    :param model_dir: Resolved local model directory.
    :raises SemanticBackendError: If model configuration or weights are missing.
    """
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise SemanticBackendError(
            f"Local model directory is missing config.json: {model_dir}. "
            "Download the complete model repository, not selected files: "
            "`hf download <repo-id> --local-dir <directory>`."
        )

    weight_patterns = ("*.safetensors", "pytorch_model*.bin")
    has_weights = any(
        candidate.is_file() for pattern in weight_patterns for candidate in model_dir.rglob(pattern)
    )
    if not has_weights:
        raise SemanticBackendError(
            f"Local model directory contains no safetensors or PyTorch model weights: {model_dir}. "
            "Download the complete model repository, not selected files: "
            "`hf download <repo-id> --local-dir <directory>`."
        )


def _resolve_load_revision(model_name: str, explicit_revision: str | None) -> str | None:
    """Resolve the revision used for model loading and post-load verification.

    Local model directories always resolve to ``None``: on-disk weights have no
    hub revision, so an explicit override would only poison model-cache keys and
    post-load cache-revision confirmation with a meaningless string.

    :param model_name: Requested model identifier.
    :param explicit_revision: Optional explicit revision override.
    :return: Revision for hub models, ``None`` for local model directories.
    """
    canonical_model = resolve_model_profile(model_name).canonical_name
    if resolve_local_model_path(canonical_model) is not None:
        return None
    return _resolve_model_revision(model_name, explicit_revision)


def _resolve_revision_for_cache(
    model_name: str,
    explicit_revision: str | None,
    *,
    strict: bool = False,
) -> str | None:
    """Resolve a revision usable as a cache key component, without loading the model.

    Pinned profiles (or an explicit full commit hash override) resolve to that
    hash immediately in both modes. Local model directories resolve to a
    content fingerprint of the directory (an explicit revision is ignored for
    them because nothing pins on-disk weights) regardless of ``strict``.

    The default (loose, ``strict=False``) policy keys an unpinned hub revision
    by the requested LABEL itself - the explicit ``--model-revision`` value,
    or ``"main"`` when none was given - without ever resolving it to a
    concrete commit or disabling persistent caching. An upstream branch move
    (even a metadata-only commit) never invalidates the cache under this
    policy; the cost is that a real weight change behind a moving branch is
    not tracked, so run ``codedupes cache clear --model`` or pass
    ``strict=True`` when that matters.

    ``strict=True`` restores the pre-loose policy: an unpinned hub model
    falls back to reading the locally cached HuggingFace commit hash so cache
    keys stay stable across runs even before the model is loaded, and returns
    ``None`` (disabling persistent caching for the run) when a branch or tag
    cannot be mapped offline - loading the model would be required before
    cached vectors could be trusted.

    :param model_name: Requested model identifier.
    :param explicit_revision: Optional explicit revision override.
    :param strict: Whether to resolve an unpinned hub revision to a concrete
        commit hash (and disable caching when that mapping fails) instead of
        keying by the requested revision label, defaults to ``False``.
    :return: Concrete revision string, revision label, or (strict mode only)
        ``None`` when it cannot be resolved offline.
    """
    canonical_model = resolve_model_profile(model_name).canonical_name
    local_dir = resolve_local_model_path(canonical_model)
    if local_dir is not None:
        return _fingerprint_local_model_dir(local_dir)
    load_revision = _resolve_model_revision(model_name, explicit_revision)

    if not strict:
        if load_revision is not None and _is_hf_commit_hash(load_revision):
            return load_revision
        return load_revision or "main"

    if load_revision is None:
        return _resolve_hf_cached_revision(canonical_model)

    cached_revision = _resolve_hf_cached_revision(canonical_model, load_revision)
    if cached_revision is not None:
        return cached_revision
    if _is_hf_commit_hash(load_revision):
        return load_revision

    # A branch or tag may move. Without an offline mapping to a concrete
    # snapshot, loading the model is required before cached vectors are trusted.
    return None


def _get_loaded_model_commit_hash(model: object) -> str | None:
    """Best-effort read of the actual loaded commit hash for cache-revision verification.

    :param model: Loaded ``SentenceTransformer`` model instance.
    :return: Commit hash reported by the underlying transformers config, or ``None``.
    """
    try:
        first_module = model[0]  # type: ignore[index]
        auto_model = getattr(first_module, "auto_model", None)
        config = getattr(auto_model, "config", None)
        commit_hash = getattr(config, "_commit_hash", None)
        return commit_hash if isinstance(commit_hash, str) else None
    except Exception:  # noqa: BLE001 - defensive introspection of an arbitrary model object
        return None


def _confirm_cache_revision_after_load(
    model: object,
    model_name: str,
    resolved_revision: str | None,
    *,
    strict: bool = False,
) -> str | None:
    """Resolve a vector-safe cache revision after loading an embedding model.

    Local directories resolve to the content fingerprint verified while this
    model was loaded, never the caller's pre-load assumption, regardless of
    ``strict``: a directory swapped mid-load would otherwise key fresh vectors
    — and retain stale hits — under a fingerprint the loaded weights no
    longer match.

    For hub models, the default (loose, ``strict=False``) policy never
    reconciles against what the backend actually loaded: an explicit or
    profile-pinned full commit hash keys as-is (unchanged either way);
    otherwise the pre-load revision label (or ``"main"``) is trusted as-is,
    mirroring :func:`_resolve_revision_for_cache` exactly so the two can never
    disagree and force a spurious rekey.

    ``strict=True`` restores the pre-loose policy: it requires either the
    loaded config's concrete commit hash or an explicitly pinned full commit
    hash, returning ``None`` (disabling persistent reuse for this run) when a
    symbolic branch/tag is unsafe because the backend cannot report what it
    loaded.

    :param model: Loaded embedding model.
    :param model_name: Requested model alias, Hub ID, or local directory.
    :param resolved_revision: Revision passed to the model loader.
    :param strict: Whether to require post-load commit-hash confirmation for
        hub models instead of trusting the pre-load revision label, defaults
        to ``False``.
    :return: Safe cache revision, or (strict mode only) ``None`` when
        persistent reuse must be disabled.
    """
    canonical_model = resolve_model_profile(model_name).canonical_name
    local_dir = resolve_local_model_path(canonical_model)
    if local_dir is not None:
        with _model_lock:
            if model is _model:
                return _model_local_fingerprint
        # The model did not come from the process-wide cache (injected by tests
        # or displaced by a concurrent load), so no load-time verification
        # bracket exists for it; the directory's current state is the best
        # available identity.
        return _fingerprint_local_model_dir(local_dir)

    if not strict:
        if resolved_revision is not None and _is_hf_commit_hash(resolved_revision):
            return resolved_revision
        return resolved_revision or "main"

    loaded_commit = _get_loaded_model_commit_hash(model)
    if loaded_commit is not None:
        return loaded_commit
    if resolved_revision is not None and _is_hf_commit_hash(resolved_revision):
        return resolved_revision
    return None


def _revision_is_mutable_label(model_name: str, cache_revision: str | None) -> bool:
    """Check whether a cache revision is a mutable hub label rather than immutable identity.

    Pinned or explicit full commit hashes and local-directory content
    fingerprints identify exact weights, so their shards can never drift.
    Everything else under loose keying is a branch/tag label whose upstream
    target can move, which is what the per-shard source-commit guard exists
    to detect.

    :param model_name: Requested model alias, Hub ID, or local directory.
    :param cache_revision: Revision component the cache shard is keyed by.
    :return: ``True`` when the shard is keyed by a movable label.
    """
    if cache_revision is None or _is_hf_commit_hash(cache_revision):
        return False
    return resolve_local_model_path(resolve_model_profile(model_name).canonical_name) is None


def _assemble_cached_matrix(keys: list[str], hits: dict[str, np.ndarray]) -> np.ndarray:
    """Assemble a row-aligned embedding matrix entirely from cache hits.

    :param keys: Cache keys row-aligned with the target unit order.
    :param hits: Mapping of cache key to cached embedding vector.
    :return: Float32 matrix with one row per key, in ``keys`` order.
    """
    dim = next(iter(hits.values())).shape[-1]
    matrix = np.empty((len(keys), dim), dtype=np.float32)
    for i, key in enumerate(keys):
        matrix[i] = hits[key]
    return matrix


def _select_cache_miss_indices(
    cache_keys: list[str] | None,
    hits: dict[str, np.ndarray],
    unit_count: int,
) -> list[int]:
    """Select one representative row for each missing cache key.

    :param cache_keys: Row-aligned cache keys, or ``None`` when caching is unavailable.
    :param hits: Cached vectors keyed by cache key.
    :param unit_count: Number of input units.
    :return: Representative row indices that require encoding.
    """
    if cache_keys is None:
        return list(range(unit_count))

    covered_keys = set(hits)
    miss_indices: list[int] = []
    for index, key in enumerate(cache_keys):
        if key in covered_keys:
            continue
        covered_keys.add(key)
        miss_indices.append(index)
    return miss_indices


# Bump whenever codedupes' own embedding pipeline changes in a vector-affecting
# way (prompt handling, routing, normalization, truncation policy, load dtype),
# so cached vectors from an older pipeline can never mix into a new matrix.
EMBEDDING_PIPELINE_SCHEMA = 3


def _embedding_runtime_fingerprint() -> str:
    """Digest the embedding runtime that determines vector values.

    Two runs whose installed inference stack differs may place the same text at
    different coordinates even for the same model revision, so the runtime is
    part of cache identity. ``importlib.metadata`` never imports the packages,
    keeping the model-free warm path intact.

    :return: Stable hex digest of pipeline schema and runtime package versions.
    """
    payload = "\x00".join(
        (
            f"pipeline={EMBEDDING_PIPELINE_SCHEMA}",
            f"sentence-transformers={_safe_package_version('sentence-transformers') or 'missing'}",
            f"transformers={_safe_package_version('transformers') or 'missing'}",
            f"tokenizers={_safe_package_version('tokenizers') or 'missing'}",
            f"torch={_safe_package_version('torch') or 'missing'}",
        )
    ).encode()
    return hashlib.blake2b(payload, digest_size=10).hexdigest()


# Query texts are unbounded user input with no live-corpus set to compact
# against, so cap them FIFO per namespace instead of letting unique searches
# accumulate until whole-shard LRU eviction.
_MAX_CACHED_QUERY_KEYS = 512


def _dtype_variant_for(
    profile: SemanticModelProfile,
    device: str | None,
    *,
    mps_fallback: bool | None,
    resolved_device: str | None = None,
) -> str:
    """Build the dtype component of the cache variant for one model family.

    Every load pins an explicit dtype, so the empty fingerprint truthfully
    means float32. ``mps`` always resolves float32 (measured: bf16 gains only
    ~13% runtime while drifting pair similarities ~1e-2, not worth splitting
    the shared key space) and returns without importing PyTorch. ``cpu``
    resolves float32 or bfloat16 from the CPU inference policy
    (:func:`codedupes.devices.resolve_cpu_bf16_inference` - the experimental
    ``CODEDUPES_CPU_BF16=1`` opt-in plus this machine's live capability gate);
    a non-opted-in run resolves float32 without ever importing torch. On
    darwin, ``auto`` can only select MPS or CPU: when the CPU policy is
    float32 both possible targets agree (every run without the opt-in, and
    every Mac without an mkldnn backend), so the variant resolves without
    picking a concrete device; when the policy enables bf16, resolution falls
    through to inspect the concrete target because only a CPU pick would
    differ from MPS. Requests that can reach CUDA, and non-darwin ``auto``,
    always resolve the device and record a selected bfloat16 as a non-default
    dtype.

    :param profile: Resolved model profile.
    :param device: Requested device string (``auto``, ``cpu``, ``cuda``, ``mps``),
        ``None`` meaning the default request.
    :param mps_fallback: MPS unsupported-op CPU fallback behavior.
    :param resolved_device: Already resolved execution target, when available.
    :return: Dtype fingerprint, empty for float32.
    """
    normalized_device = (device or DEFAULT_SEMANTIC_DEVICE).strip().lower()
    if normalized_device == "mps":
        return ""
    if normalized_device == "cpu":
        gate = resolve_cpu_bf16_inference()
        return "dtype=torch.bfloat16" if gate else ""
    if (
        normalized_device == "auto"
        and sys.platform == "darwin"
        and not resolve_cpu_bf16_inference()
    ):
        return ""

    concrete_device = resolved_device or _resolve_semantic_device_request(
        device,
        mps_fallback=mps_fallback,
    )
    selected_dtype = _resolve_model_dtype(profile.family, concrete_device)
    dtype_name = str(selected_dtype) if selected_dtype is not None else "default"
    if dtype_name in {"float32", "fp32", "torch.float32"}:
        return ""
    return f"dtype={dtype_name}"


def _mps_fast_math_variant(device: str | None) -> str:
    """Build the Metal math-policy component of the cache variant.

    ``PYTORCH_MPS_FAST_MATH`` switches MPS kernels to approximate math, which can
    move similarity scores across tuned thresholds, so vectors computed under a
    fast-math policy must never satisfy hits for the shared faithful-float32 key
    space (or vice versa). The raw setting is recorded so distinct policies never
    mix even if PyTorch's interpretation of unusual values changes.

    ``PYTORCH_MPS_PREFER_METAL`` and ``PYTORCH_ENABLE_MPS_FALLBACK`` are
    deliberately not keyed: both select among faithful float32 implementations,
    the same tolerance class as the intentional CPU/MPS shared key space.

    :param device: Requested device string (``auto``, ``cpu``, ``cuda``, ``mps``),
        ``None`` meaning the default request.
    :return: Math-policy fingerprint, empty when MPS cannot execute or fast math is off.
    """
    normalized_device = (device or DEFAULT_SEMANTIC_DEVICE).strip().lower()
    if normalized_device != "mps" and not (
        normalized_device == "auto" and sys.platform == "darwin"
    ):
        return ""
    raw = os.environ.get("PYTORCH_MPS_FAST_MATH")
    # Mirror torch's own decision exactly: fast math is enabled whenever the
    # variable is set to anything except the literal string "0" - an empty
    # string and whitespace-wrapped zeros all enable it in the Metal compiler.
    if raw is None or raw == "0":
        return ""
    return f"mpsfm={raw}"


def _fast_math_write_allowed(device: str | None, execution_device: str) -> bool:
    """Decide whether cache writes under a fast-math-keyed variant are valid.

    The fast-math variant derives from the requested device before PyTorch
    loads, so a run that then executes elsewhere (MPS unavailable under
    ``auto``, or an OOM/invalid-output fallback to CPU before or during
    encoding) would publish faithful float32 vectors into the fast-math key
    space. Skipping the write keeps the two spaces unmixed, at the cost of
    recomputing those vectors on the next run.

    :param device: Requested device string as given to the public API.
    :param execution_device: Normalized device the model actually executed on,
        read after encoding so mid-encode CPU fallbacks are reflected.
    :return: ``True`` when writing under the derived variant is representative.
    """
    if not _mps_fast_math_variant(device):
        return True
    return execution_device == "mps"


def _cache_variant_for(
    profile: SemanticModelProfile,
    device: str | None,
    plan: EncodePlan,
    *,
    mps_fallback: bool | None,
    trust_remote_code: bool = False,
    resolved_device: str | None = None,
) -> str:
    """Build the complete vector-affecting cache-key variant for one encode call.

    The variant deliberately excludes the device name itself: CPU and MPS
    float32 share one key space even though their kernels differ at float
    rounding scale (measured ~5e-4 elementwise, ~2e-4 on pair similarity for
    the default profile), so a warm cache may serve vectors computed on the
    other device. That tolerance is accepted to keep device switches cheap;
    ``codedupes cache clear`` restores a single-device baseline when an exact
    reference run matters. Behavior that meaningfully changes vectors (dtype,
    Metal math policy, runtime versions, encode plan, remote-code execution
    path) always splits the key space.

    :param profile: Resolved model profile.
    :param device: Requested device string (``auto``, ``cpu``, ``cuda``, ``mps``).
    :param plan: Resolved encode plan (route and effective prompt).
    :param mps_fallback: MPS unsupported-op CPU fallback behavior.
    :param trust_remote_code: Resolved remote-code trust setting.
    :param resolved_device: Already resolved execution target, when available.
    :return: Variant fingerprint combining plan, dtype, math policy, runtime, and trust.
    """
    dtype_variant = _dtype_variant_for(
        profile,
        device,
        mps_fallback=mps_fallback,
        resolved_device=resolved_device,
    )
    return "\x00".join(
        (
            plan.cache_identity(),
            dtype_variant,
            _mps_fast_math_variant(device),
            _embedding_runtime_fingerprint(),
            f"trc={int(trust_remote_code)}",
        )
    )


def _build_embedding_space_identity(
    profile: SemanticModelProfile,
    resolved_revision: str | None,
    encode_plan: EncodePlan,
    device: str,
    *,
    mps_fallback: bool | None,
    trust_remote_code: bool,
    resolved_device: str | None = None,
) -> EmbeddingSpaceIdentity:
    """Build one corpus identity from already resolved embedding inputs.

    :param profile: Resolved model profile.
    :param resolved_revision: Concrete Hub revision or local-content fingerprint.
    :param encode_plan: Corpus encode route and prompt.
    :param device: Device policy that produced the vectors.
    :param mps_fallback: MPS unsupported-op CPU fallback behavior.
    :param trust_remote_code: Resolved remote-code trust setting.
    :param resolved_device: Already resolved execution target, when available.
    :return: Complete embedding-space identity.
    """
    return EmbeddingSpaceIdentity(
        model_name=profile.canonical_name,
        resolved_revision=resolved_revision,
        runtime_variant=_cache_variant_for(
            profile,
            device,
            encode_plan,
            mps_fallback=mps_fallback,
            trust_remote_code=trust_remote_code,
            resolved_device=resolved_device,
        ),
    )


def resolve_embedding_space_identity(
    model_name: str = DEFAULT_MODEL,
    instruction_prefix: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
    semantic_task: str | None = None,
    device: str = DEFAULT_SEMANTIC_DEVICE,
    mps_fallback: bool | None = None,
    persist_local_model_manifest: bool = True,
    strict_revision_cache: bool = False,
) -> EmbeddingSpaceIdentity:
    """Resolve the vector-space identity for code corpus embeddings.

    :param model_name: Model alias, Hub identifier, or local directory.
    :param instruction_prefix: Optional instruction override for code inputs.
    :param revision: Optional model revision; ``None`` uses the profile default.
    :param trust_remote_code: Optional remote-code trust setting.
    :param semantic_task: Semantic task used to embed the corpus.
    :param device: Requested semantic inference device.
    :param mps_fallback: MPS unsupported-op CPU fallback behavior.
    :param persist_local_model_manifest: Whether local-model digests may be
        read from and saved to the persistent cache manifest.
    :param strict_revision_cache: Whether an unpinned hub revision resolves to a
        concrete commit hash (disabling caching when unmappable) instead of the
        requested revision label, defaults to ``False``.
    :return: Canonical model, concrete revision/fingerprint, and runtime variant.
    """
    # Same contract as compute_embeddings_with_identity: configure
    # import-sensitive runtime variables before anything can import torch.
    _configure_semantic_runtime_env(device, mps_fallback=mps_fallback)
    profile = resolve_model_profile(model_name)
    resolved_task = normalize_semantic_task(
        semantic_task,
        default_task=DEFAULT_CHECK_SEMANTIC_TASK,
    )
    encode_plan = _resolve_encode_plan(profile, "code", resolved_task, instruction_prefix)
    resolved_trust_remote_code = _resolve_trust_remote_code(model_name, trust_remote_code)
    local_model_path = resolve_local_model_path(profile.canonical_name)
    if local_model_path is not None:
        resolved_revision = _fingerprint_local_model_dir(
            local_model_path,
            persist_manifest=persist_local_model_manifest,
        )
    else:
        resolved_revision = _resolve_revision_for_cache(
            model_name,
            revision,
            strict=strict_revision_cache,
        )

    return _build_embedding_space_identity(
        profile,
        resolved_revision,
        encode_plan,
        device,
        mps_fallback=mps_fallback,
        trust_remote_code=resolved_trust_remote_code,
    )


def _require_current_embedding_space(
    expected: EmbeddingSpaceIdentity,
    *,
    model_name: str,
    instruction_prefix: str | None,
    revision: str | None,
    trust_remote_code: bool | None,
    semantic_task: str,
    device: str,
    mps_fallback: bool | None,
    persist_local_model_manifest: bool,
    strict_revision_cache: bool = False,
) -> str:
    """Require the configured corpus vector space to match its stored identity.

    :param expected: Identity captured when the corpus matrix was embedded.
    :param model_name: Current model configuration.
    :param instruction_prefix: Current corpus instruction override.
    :param revision: Current revision configuration.
    :param trust_remote_code: Current remote-code trust configuration.
    :param semantic_task: Task that produced the corpus matrix.
    :param device: Current inference-device request.
    :param mps_fallback: Current MPS fallback configuration.
    :param persist_local_model_manifest: Whether local digests may be persisted.
    :param strict_revision_cache: Whether unpinned hub revisions resolve to a
        concrete commit hash instead of the requested revision label; must
        match the mode used to build ``expected``.
    :return: Device policy that keeps query vectors in the stored corpus space.
    :raises RuntimeError: If the identity changed or cannot be pinned concretely.
    """
    current = resolve_embedding_space_identity(
        model_name=model_name,
        instruction_prefix=instruction_prefix,
        revision=revision,
        trust_remote_code=trust_remote_code,
        semantic_task=semantic_task,
        device=device,
        mps_fallback=mps_fallback,
        persist_local_model_manifest=persist_local_model_manifest,
        strict_revision_cache=strict_revision_cache,
    )
    if current == expected and current.resolved_revision is not None:
        return device

    # Any coherence break - fast-math execution leaving MPS, or an accelerator
    # OOM casting a bfloat16 run to float32 - discards the partial corpus and
    # rebuilds it wholly under the faithful CPU policy, recording the CPU
    # identity. A corpus whose stored identity matches that CPU policy must
    # therefore stay searchable: keep its queries on the recorded CPU space
    # even though the analyzer's requested device remains the accelerator/auto
    # request. The remap only engages when the stored identity is exactly the
    # CPU identity, so a genuine model/revision/runtime change still raises.
    normalized_device = (device or DEFAULT_SEMANTIC_DEVICE).strip().lower()
    if normalized_device != "cpu":
        cpu_identity = resolve_embedding_space_identity(
            model_name=model_name,
            instruction_prefix=instruction_prefix,
            revision=revision,
            trust_remote_code=trust_remote_code,
            semantic_task=semantic_task,
            device="cpu",
            mps_fallback=mps_fallback,
            persist_local_model_manifest=persist_local_model_manifest,
            strict_revision_cache=strict_revision_cache,
        )
        if cpu_identity == expected and cpu_identity.resolved_revision is not None:
            return "cpu"

    raise RuntimeError(
        "The semantic model or embedding runtime changed since this corpus was indexed. "
        "Run index() or analyze() again before search()."
    )


def _safe_package_version(package_name: str) -> str | None:
    """Get installed package version string, returning ``None`` if unavailable.

    :param package_name: Package to inspect.
    :return: Installed version string, or ``None``.
    """
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def get_semantic_runtime_versions() -> dict[str, str]:
    """Return semantic runtime versions for diagnostics.

    :return: Mapping of runtime component names to version strings.
    """
    return {
        "python": sys.version.split()[0],
        "torch": _safe_package_version("torch") or "missing",
        "transformers": _safe_package_version("transformers") or "missing",
        "sentence-transformers": _safe_package_version("sentence-transformers") or "missing",
    }


def _validate_torch_runtime() -> None:
    """Enforce the supported PyTorch range even when installer checks were bypassed."""
    raw = _safe_package_version("torch")
    if raw is None:
        raise SemanticBackendError(
            "Could not determine the installed PyTorch version. "
            "Install a supported runtime with `pip install 'torch>=2.13,<3'`."
        )

    try:
        parsed = Version(raw)
    except InvalidVersion as exc:
        raise SemanticBackendError(f"Could not parse torch version: {raw}") from exc

    # Release tuples are compared instead of Version objects: Version ordering puts
    # 2.13.0.dev1 and 2.13.0rc1 below 2.13, which would reject supported pre-releases.
    if not (_TORCH_MIN_RELEASE <= parsed.release < _TORCH_MAX_EXCLUSIVE_RELEASE):
        raise SemanticBackendError(
            f"Incompatible torch version {raw}. codedupes semantic analysis requires "
            ">=2.13,<3. Run: pip install 'torch>=2.13,<3'."
        )


def _is_known_semantic_backend_error(error: Exception) -> bool:
    """Return True when an exception is likely caused by semantic backend compatibility.

    :param error: Captured exception.
    :return: ``True`` when exception text matches known backend issues.
    """
    text = str(error).lower()
    if isinstance(error, ModuleNotFoundError):
        return True
    if isinstance(error, AttributeError) and "all_tied_weights_keys" in text:
        return True
    keywords = (
        "trust_remote_code",
        "flash_attn",
        "embeddinggemma",
        "auto_map",
        "tokenizer",
    )
    return any(keyword in text for keyword in keywords)


def _wrap_semantic_backend_error(
    error: Exception,
    *,
    model_name: str,
    revision: str | None,
    trust_remote_code: bool,
    stage: str,
) -> SemanticBackendError:
    """Convert backend exceptions into a stable semantic error with remediation guidance.

    :param error: Original exception.
    :param model_name: Model involved in the failure.
    :param revision: Resolved model revision.
    :param trust_remote_code: Trust-remote-code flag used.
    :param stage: Backend stage where failure occurred.
    :return: Wrapped ``SemanticBackendError`` with remediation guidance.
    """
    versions = get_semantic_runtime_versions()
    version_info = ", ".join(f"{key}={value}" for key, value in versions.items())
    revision_text = revision or "default"
    hints = ["run traditional-only mode with '--traditional-only'."]

    message = (
        f"Semantic backend failed during {stage} for model={model_name} revision={revision_text} "
        f"trust_remote_code={trust_remote_code}. "
        f"Versions: {version_info}. "
        "Fix suggestions: " + " ".join(hints)
    )
    wrapped = SemanticBackendError(message)
    wrapped.__cause__ = error
    return wrapped


def _require_dependency(module_name: str, install_hint: str) -> None:
    """Raise a clear error when a required dependency is unavailable.

    :param module_name: Required module name.
    :param install_hint: Suggested installation command.
    :return: ``None``.
    :raises ModuleNotFoundError: When dependency is missing.
    """
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise ModuleNotFoundError(
            f"{module_name} is required for semantic analysis. Install with {install_hint}."
        ) from exc


def _check_semantic_dependencies() -> None:
    """Validate required runtime dependencies before model loading."""
    _require_dependency("sentence_transformers", "pip install codedupes")
    _require_dependency("transformers", "pip install codedupes")
    _require_dependency("torch", "pip install codedupes")

    _validate_torch_runtime()


def _resolve_semantic_device_request(
    device: str | None,
    *,
    mps_fallback: bool | None,
) -> str:
    """Configure the runtime environment and resolve one semantic device request.

    :param device: Requested device name.
    :param mps_fallback: MPS unsupported-op fallback behavior.
    :return: Concrete device name.
    :raises SemanticBackendError: If device configuration fails.
    """
    try:
        _configure_semantic_runtime_env(device, mps_fallback=mps_fallback)
        return resolve_semantic_device(device)
    except (DeviceConfigurationError, ValueError) as exc:
        raise SemanticBackendError(str(exc)) from exc


def _validate_explicit_device_request(
    device: str | None,
    *,
    mps_fallback: bool | None,
) -> None:
    """Validate an explicit accelerator device request regardless of cache state.

    ``cpu`` and ``auto`` can never fail this check -- ``auto`` always has a CPU
    fallback -- so both no-op here, keeping the torch-import-free warm-cache path
    available for the default device. An explicit ``cuda``/``mps`` request is
    resolved through :func:`_resolve_semantic_device_request`, which raises the
    documented :class:`SemanticBackendError` for an unavailable accelerator even
    when every embedding for this call is already cached: an explicit unavailable
    ``--device`` must always be an error, never a silent cache-driven no-op.

    :param device: Requested device name.
    :param mps_fallback: MPS unsupported-op fallback behavior.
    :return: ``None``.
    :raises SemanticBackendError: If an explicitly requested device is invalid or unavailable.
    """
    normalized = (device or DEFAULT_SEMANTIC_DEVICE).strip().lower()
    if normalized in {"cpu", "auto"}:
        return
    _resolve_semantic_device_request(device, mps_fallback=mps_fallback)


def _prepare_semantic_device(
    device: str | None,
    *,
    mps_fallback: bool | None,
    mps_memory_fraction: float | None,
) -> str:
    """Resolve and configure one semantic execution device.

    :param device: Requested device name.
    :param mps_fallback: MPS unsupported-op fallback behavior.
    :param mps_memory_fraction: Optional MPS allocator limit, ignored when the
        request resolves to a non-MPS device.
    :return: Concrete device name.
    :raises SemanticBackendError: If device configuration fails.
    """
    resolved_device = _resolve_semantic_device_request(
        device,
        mps_fallback=mps_fallback,
    )
    try:
        # ``device='auto'`` may legitimately resolve to CPU/CUDA with a fraction set,
        # so the strict low-level check is only applied once MPS is the real target.
        if resolved_device == "mps":
            configure_mps_memory_fraction(resolved_device, mps_memory_fraction)
        else:
            if mps_memory_fraction is not None:
                logger.info(f"mps_memory_fraction ignored: resolved device is {resolved_device}")
            # A prior call may have applied a custom MPS allocator cap; this
            # call resolved away from MPS, so restore the baseline instead of
            # leaving the cap stuck in this process until some later call
            # happens to resolve back to MPS with no fraction set.
            restore_mps_memory_fraction_if_managed()
    except (DeviceConfigurationError, ValueError) as exc:
        raise SemanticBackendError(str(exc)) from exc

    global _warned_mlx_mps_contention
    if resolved_device == "mps" and is_mlx_loaded() and not _warned_mlx_mps_contention:
        logger.warning(
            "MLX is already loaded in this process and may share Apple unified-memory "
            "pressure with PyTorch MPS. codedupes manages only the PyTorch allocator; "
            "the host application remains responsible for MLX cache policy."
        )
        _warned_mlx_mps_contention = True
    return resolved_device


def _coerce_device_name(value: object, fallback: str) -> str:
    """Reduce a torch device-like value to ``cpu``, ``cuda``, or ``mps``.

    :param value: Device-like object or string, for example ``torch.device('mps:0')``.
    :param fallback: Device name to use when the value is ``None`` or unrecognized.
    :return: ``cpu``, ``cuda``, ``mps``, or ``fallback``.
    """
    if value is None:
        return fallback
    candidate = str(value).lower().split(":", 1)[0]
    if candidate in {"cpu", "cuda", "mps"}:
        return candidate
    return fallback


def _get_effective_model_device(model: object, fallback: str) -> str:
    """Return cached or model-reported execution device.

    :param model: Model instance to inspect.
    :param fallback: Device name to use when the model reports nothing usable.
    :return: Tracked execution device when ``model`` is the process-wide cached model,
        otherwise the device coerced from ``model.device``.
    """
    if model is _model and _model_execution_device is not None:
        return _model_execution_device
    return _coerce_device_name(getattr(model, "device", None), fallback)


def _model_parameter_dtype(model: object) -> Any | None:
    """Best-effort read of a loaded model's current parameter dtype.

    :param model: Model instance to inspect.
    :return: Dtype of the model's first parameter, or ``None`` when unavailable.
    """
    try:
        return next(model.parameters()).dtype
    except Exception:  # noqa: BLE001 - defensive introspection of an arbitrary model object
        return None


def _dtype_coherence_broken(
    profile: SemanticModelProfile,
    device: str | None,
    mps_fallback: bool | None,
    resolved_device: str,
    model: object,
) -> bool:
    """Detect whether a keyed bfloat16 dtype variant no longer matches live execution.

    An accelerator OOM can move a model keyed under a non-default (bfloat16)
    dtype variant to CPU and, when this CPU fails the capability gate, cast it
    to float32 (see :func:`_move_model_to_cpu`). The keyed variant then
    describes vectors this run can no longer produce; mixing newly computed
    float32 rows with any cache hits recorded under the bfloat16 key would
    silently corrupt the matrix. The check derives from the model's actual
    live parameter dtype, never from a replayed pre-fallback assumption.

    :param profile: Resolved model profile.
    :param device: Requested device string as given to the public API.
    :param mps_fallback: MPS unsupported-op CPU fallback behavior.
    :param resolved_device: Device resolved before model load.
    :param model: Loaded model instance, possibly moved mid-call.
    :return: ``True`` when the keyed dtype variant is no longer representative.
    """
    dtype_variant = _dtype_variant_for(
        profile,
        device,
        mps_fallback=mps_fallback,
        resolved_device=resolved_device,
    )
    if not dtype_variant:
        return False
    current_dtype = _model_parameter_dtype(model)
    return current_dtype is not None and str(current_dtype) != "torch.bfloat16"


def _cache_write_allowed(
    profile: SemanticModelProfile,
    device: str | None,
    mps_fallback: bool | None,
    resolved_device: str,
    model: object,
) -> bool:
    """Decide whether fresh vectors may be written under their keyed cache variant.

    Generalizes :func:`_fast_math_write_allowed` to also cover dtype
    coherence: a write is safe only when the actual post-encode execution
    state - device and dtype, re-read from the model rather than replayed
    pre-encode assumptions - still matches what the keyed variant promises.
    Skipping an unsafe write costs a cache miss on the next run; it never
    risks mixing vectors from two coordinate systems under one key.

    :param profile: Resolved model profile.
    :param device: Requested device string as given to the public API.
    :param mps_fallback: MPS unsupported-op CPU fallback behavior.
    :param resolved_device: Device resolved before model load.
    :param model: Loaded model instance, possibly moved mid-call.
    :return: ``True`` when writing under the derived variant is representative.
    """
    execution_device = _get_effective_model_device(model, resolved_device)
    if not _fast_math_write_allowed(device, execution_device):
        return False
    return not _dtype_coherence_broken(
        profile,
        device,
        mps_fallback,
        resolved_device,
        model,
    )


class _ExecutableStatementCounter(ast.NodeVisitor):
    """Count executable statements recursively, stopping at nested scopes."""

    def __init__(self) -> None:
        """Initialize the running statement count."""
        self.count = 0

    def generic_visit(self, node: ast.AST) -> None:
        """Count ``node`` when it is a statement, then recurse into its children.

        :param node: AST node being visited.
        :return: ``None``.
        """
        if isinstance(node, ast.stmt):
            self.count += 1
        super().generic_visit(node)

    # Nested scopes count as one declaration; their implementation belongs to a
    # separate CodeUnit and must not inflate the enclosing unit's count.
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Count a nested function as one declaration without descending into it.

        :param node: Nested function definition node.
        :return: ``None``.
        """
        self.count += 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Count a nested async function as one declaration without descending into it.

        :param node: Nested async function definition node.
        :return: ``None``.
        """
        self.count += 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Count a nested class as one declaration without descending into it.

        :param node: Nested class definition node.
        :return: ``None``.
        """
        self.count += 1


def get_code_unit_statement_count(unit: CodeUnit) -> int:
    """Get effective statement count for a unit, excluding docstring.

    Statements are counted recursively through control-flow bodies (``try``,
    ``with``, loops, conditionals, ``match``) so a large function implemented
    inside one outer block is not measured as a single statement. Nested
    function/class definitions count as one declaration each; their bodies
    belong to their own units.

    :param unit: Unit to measure.
    :return: Number of executable statements.
    """
    if not unit.source:
        return 0

    # Extracted nested methods/classes retain their file indentation. Dedent
    # the definition before parsing so it parses at module level.
    text = textwrap.dedent(unit.source).strip()
    if not text:
        return 0

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0

    if not tree.body:
        return 0

    top_node = tree.body[0]
    body = []
    if isinstance(top_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        body = top_node.body
    else:
        body = tree.body

    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    counter = _ExecutableStatementCounter()
    for statement in body:
        counter.visit(statement)
    return counter.count


def _resolve_model_dtype(family: str, device: str) -> Any:
    """Choose the explicitly pinned dtype for one model family and device.

    Transformers 5 loads checkpoints in their config-declared dtype by default
    (``dtype="auto"``), so the default profile's float16 checkpoint would
    otherwise embed in half precision - an order of magnitude slower on CPU and
    outside the documented faithful-float32 tolerance. Every load therefore
    pins an explicit dtype under a capability-gated policy rather than a
    hardcoded machine truth: bfloat16 on CUDA hardware with native support
    (unchanged); bfloat16 on CPU iff the experimental ``CODEDUPES_CPU_BF16=1``
    opt-in is set *and* this machine passes the two-part gate in
    :func:`codedupes.devices.resolve_cpu_bf16_native` - native bf16 ISA *and*
    a GEMM backend (oneDNN/mkldnn) able to exploit it, probed live from torch
    at most once per process and never persisted to disk; float32 everywhere
    else, including every MPS run. The opt-in guard exists because no
    gate-passing machine has yet validated the positive path: the duplicate
    and search thresholds are calibrated under float32, and the gate proves
    fast executability, not decision parity, so automatic CPU bf16 waits for
    that evidence. Measured on an Apple M5 (torch 2.13.0,
    macOS arm64 wheel): ``torch.cpu.get_capabilities()`` reports a native bf16
    ISA (``bf16: true``, ``architecture: "arm64"``) but
    ``torch.backends.mkldnn.is_available()`` is ``False``, so a
    1024x1024x1024 bf16 matmul measured 1015 ms versus 1.207 ms for float32 -
    841x slower with an ISA but no backend to exploit it. The gate exists so a
    future machine with both the ISA and mkldnn gets the real speed and
    memory benefit instead of this machine's negative result being hardcoded
    forever. MPS bfloat16 halves model memory but gains only ~13% runtime
    while drifting pair similarities ~1e-2 - tuned-threshold scale - so MPS
    stays faithful float32 and keeps its shared cache key space with CPU
    float32 runs.

    :param family: Resolved model profile family, reserved for per-family policy.
    :param device: Concrete execution device.
    :return: Dtype object for Torch model loading.
    """
    import torch

    if (
        device == "cuda"
        and hasattr(torch.cuda, "is_bf16_supported")
        # including_emulation=False keeps this a native-support check: torch's
        # default returns True on pre-Ampere GPUs that merely construct bf16
        # tensors through emulation, the failure mode this policy exists to avoid.
        and torch.cuda.is_bf16_supported(including_emulation=False)
    ):
        return torch.bfloat16

    if device == "cpu" and resolve_cpu_bf16_inference():
        return torch.bfloat16

    return torch.float32


def _get_model_unlocked(
    model_name: str = DEFAULT_MODEL,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
    device: str = DEFAULT_SEMANTIC_DEVICE,
    mps_fallback: bool | None = None,
    mps_memory_fraction: float | None = None,
    persist_local_model_manifest: bool = True,
) -> object:
    """Lazy-load the embedding model on an explicit resolved device.

    :param model_name: Model alias or identifier.
    :param revision: Optional model revision.
    :param trust_remote_code: Optional remote code trust setting.
    :param device: ``auto``, ``cpu``, ``cuda``, or ``mps``.
    :param mps_fallback: MPS unsupported-op CPU fallback behavior.
    :param mps_memory_fraction: Optional MPS allocator limit in ``(0, 2]``.
    :param persist_local_model_manifest: Whether local-model file digests may be
        read from and saved to the persistent cache manifest.
    :return: Loaded model instance.
    """
    global _model, _model_name, _model_revision, _model_trust_remote_code
    global _model_local_fingerprint
    global _model_device_key, _model_execution_device, _warned_cpu_fallback_reuse

    requested_local_path = resolve_local_model_path(model_name)
    if requested_local_path is None and is_explicit_local_model_path(model_name):
        raise SemanticBackendError(
            "Local model directory does not exist or is not a directory: "
            f"{Path(model_name).expanduser()}"
        )

    profile = resolve_model_profile(model_name)
    resolved_model_name = profile.canonical_name
    local_model_path = resolve_local_model_path(resolved_model_name)
    if local_model_path is not None:
        _validate_local_model_directory(local_model_path)
    local_model_fingerprint = (
        _fingerprint_local_model_dir(
            local_model_path,
            persist_manifest=persist_local_model_manifest,
        )
        if local_model_path is not None
        else None
    )
    resolved_revision = _resolve_model_revision(model_name, revision)
    if resolved_revision is not None and local_model_path is not None:
        logger.warning(
            f"Ignoring revision {resolved_revision!r} for local model directory "
            f"{resolved_model_name}; on-disk weights are unpinned"
        )
        resolved_revision = None
    resolved_trust_remote_code = _resolve_trust_remote_code(model_name, trust_remote_code)

    # Configure MPS environment variables before dependency checks import torch
    # through sentence-transformers/transformers.
    _configure_semantic_runtime_env(device, mps_fallback=mps_fallback)
    _check_semantic_dependencies()
    resolved_device = _prepare_semantic_device(
        device,
        mps_fallback=mps_fallback,
        mps_memory_fraction=mps_memory_fraction,
    )

    cache_miss = any(
        (
            _model is None,
            _model_name != resolved_model_name,
            _model_revision != resolved_revision,
            _model_trust_remote_code != resolved_trust_remote_code,
            _model_local_fingerprint != local_model_fingerprint,
            _model_device_key != resolved_device,
            # Without a reliable fingerprint, reloading is the only safe way to
            # avoid retaining stale weights from a mutable local directory.
            local_model_path is not None and local_model_fingerprint is None,
        )
    )

    if cache_miss:
        if _model is not None:
            _clear_model_cache_unlocked()

        logger.info(f"Loading embedding model {resolved_model_name} on {resolved_device}")
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            if exc.name == "sentence_transformers":
                raise ModuleNotFoundError(
                    "sentence-transformers is not installed. Install it with `pip install codedupes`."
                ) from exc
            raise

        st_kwargs: dict[str, object] = {
            "trust_remote_code": resolved_trust_remote_code,
            "device": resolved_device,
        }
        if local_model_path is not None:
            st_kwargs["local_files_only"] = True
        if resolved_revision is not None:
            st_kwargs["revision"] = resolved_revision

        model_kwargs: dict[str, object] = {}
        processor_kwargs: dict[str, object] = {}
        config_kwargs: dict[str, object] = {}

        selected_dtype = _resolve_model_dtype(profile.family, resolved_device)
        model_kwargs["dtype"] = selected_dtype
        logger.info(f"Pinning torch dtype on {resolved_device}: {selected_dtype}")

        if resolved_revision is not None:
            model_kwargs["revision"] = resolved_revision
            processor_kwargs["revision"] = resolved_revision
            config_kwargs["revision"] = resolved_revision

        if resolved_trust_remote_code:
            model_kwargs["trust_remote_code"] = True
            processor_kwargs["trust_remote_code"] = True
            config_kwargs["trust_remote_code"] = True

        if model_kwargs:
            st_kwargs["model_kwargs"] = model_kwargs
        if processor_kwargs:
            st_kwargs["processor_kwargs"] = processor_kwargs
        if config_kwargs:
            st_kwargs["config_kwargs"] = config_kwargs

        for reload_attempt in range(2):
            load_device = resolved_device
            try:
                loaded_model = SentenceTransformer(resolved_model_name, **st_kwargs)
            except RuntimeError as exc:
                oom_device = _classify_oom_device(exc, resolved_device)
                accelerator_load_oom = (
                    resolved_device in {"cuda", "mps"} and oom_device == resolved_device
                )
                if accelerator_load_oom:
                    exc.__traceback__ = None
                    exc.__context__ = None
                    memory_context = (
                        f" ({format_mps_memory_snapshot()})" if resolved_device == "mps" else ""
                    )
                    cache_label = "Metal cache" if resolved_device == "mps" else "CUDA cache"
                    logger.warning(
                        f"{resolved_device.upper()} OOM while loading {resolved_model_name}"
                        f"{memory_context}; clearing {cache_label} and retrying on CPU"
                    )
                    clear_device_cache(resolved_device, synchronize=True, collect=True)
                    cpu_kwargs = dict(st_kwargs)
                    cpu_kwargs["device"] = "cpu"
                    if resolved_device == "cuda":
                        # An accelerator-resolved dtype (bfloat16 on CUDA) must
                        # never be blindly inherited by the CPU retry: re-pin
                        # through the same capability gate a fresh CPU load
                        # would use. MPS already always resolves float32, so
                        # its inherited model_kwargs dtype needs no re-pin.
                        cpu_model_kwargs = dict(
                            cast(dict[str, object], cpu_kwargs.get("model_kwargs") or {})
                        )
                        cpu_model_kwargs["dtype"] = _resolve_model_dtype(profile.family, "cpu")
                        cpu_kwargs["model_kwargs"] = cpu_model_kwargs
                    try:
                        loaded_model = SentenceTransformer(resolved_model_name, **cpu_kwargs)
                    except Exception as retry_exc:
                        if _is_known_semantic_backend_error(retry_exc):
                            raise _wrap_semantic_backend_error(
                                retry_exc,
                                model_name=resolved_model_name,
                                revision=resolved_revision,
                                trust_remote_code=resolved_trust_remote_code,
                                stage=(
                                    f"CPU model-loading retry after {resolved_device.upper()} OOM"
                                ),
                            )
                        raise
                    load_device = "cpu"
                elif _is_known_semantic_backend_error(exc):
                    raise _wrap_semantic_backend_error(
                        exc,
                        model_name=resolved_model_name,
                        revision=resolved_revision,
                        trust_remote_code=resolved_trust_remote_code,
                        stage=f"model loading on {resolved_device}",
                    )
                else:
                    raise
            except Exception as exc:
                if _is_known_semantic_backend_error(exc):
                    raise _wrap_semantic_backend_error(
                        exc,
                        model_name=resolved_model_name,
                        revision=resolved_revision,
                        trust_remote_code=resolved_trust_remote_code,
                        stage=f"model loading on {resolved_device}",
                    )
                raise

            # Without a pre-load fingerprint there is nothing to verify against;
            # persistent reuse is already disabled and every call reloads.
            if local_model_path is None or local_model_fingerprint is None:
                break
            post_load_fingerprint = _fingerprint_local_model_dir(
                local_model_path,
                persist_manifest=persist_local_model_manifest,
            )
            if post_load_fingerprint == local_model_fingerprint:
                break
            if reload_attempt == 0:
                logger.warning(
                    f"Local model directory {resolved_model_name} changed while loading; "
                    "reloading from the current on-disk state"
                )
                del loaded_model
                local_model_fingerprint = post_load_fingerprint
                continue
            raise SemanticBackendError(
                f"Local model directory changed twice while loading: {resolved_model_name}. "
                "Retry once the directory contents are stable."
            )

        _model = loaded_model
        _model_name = resolved_model_name
        _model_revision = resolved_revision
        _model_trust_remote_code = resolved_trust_remote_code
        _model_local_fingerprint = local_model_fingerprint
        _model_device_key = resolved_device
        _model_execution_device = _coerce_device_name(
            getattr(loaded_model, "device", None),
            load_device,
        )
        _warned_cpu_fallback_reuse = False
    elif _model_execution_device != resolved_device and not _warned_cpu_fallback_reuse:
        logger.warning(
            f"Reusing cached model on {_model_execution_device} after an earlier "
            f"{resolved_device}-to-CPU OOM fallback; call clear_model_cache() to force a fresh "
            f"{resolved_device} load"
        )
        _warned_cpu_fallback_reuse = True

    return _model


def get_model(
    model_name: str = DEFAULT_MODEL,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
    device: str = DEFAULT_SEMANTIC_DEVICE,
    mps_fallback: bool | None = None,
    mps_memory_fraction: float | None = None,
    persist_local_model_manifest: bool = True,
) -> object:
    """Thread-safe wrapper around the process-wide semantic model cache.

    :param model_name: Model alias or identifier, defaults to ``DEFAULT_MODEL``.
    :param revision: Optional model revision; ``None`` uses the profile default.
    :param trust_remote_code: Optional remote-code trust setting; ``None`` uses the
        profile default.
    :param device: ``auto``, ``cpu``, ``cuda``, or ``mps``, defaults to
        ``DEFAULT_SEMANTIC_DEVICE``.
    :param mps_fallback: MPS unsupported-op CPU fallback behavior; ``None`` keeps the
        automatic behavior.
    :param mps_memory_fraction: Optional MPS allocator limit in ``(0, 2]``.
    :param persist_local_model_manifest: Whether local-model file digests may be
        read from and saved to the persistent cache manifest.
    :return: Cached model instance, reloaded when any cache key changed.
    """
    # Same contract as compute_embeddings_with_identity: configure
    # import-sensitive runtime variables before anything can import torch.
    _configure_semantic_runtime_env(device, mps_fallback=mps_fallback)
    with _model_lock:
        return _get_model_unlocked(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            device=device,
            mps_fallback=mps_fallback,
            mps_memory_fraction=mps_memory_fraction,
            persist_local_model_manifest=persist_local_model_manifest,
        )


def _clear_model_cache_unlocked() -> None:
    """Release the cached model and its accelerator allocator cache."""
    global _model, _model_name, _model_revision, _model_trust_remote_code
    global _model_local_fingerprint
    global _model_device_key, _model_execution_device, _warned_cpu_fallback_reuse

    model = _model
    execution_device = _model_execution_device

    # Do not migrate a model to CPU merely to release it. On unified-memory
    # systems that can transiently increase pressure; dropping references and
    # collecting first lets the owning allocator reclaim storage directly.
    _model = None
    _model_name = None
    _model_revision = None
    _model_trust_remote_code = None
    _model_local_fingerprint = None
    _model_device_key = None
    _model_execution_device = None
    _warned_cpu_fallback_reuse = False

    del model
    clear_device_cache(execution_device, synchronize=True, collect=True)


def clear_model_cache() -> None:
    """Thread-safe release of the cached model and accelerator cache."""
    with _model_lock:
        _clear_model_cache_unlocked()


def _classify_oom_device(error: RuntimeError, active_device: str) -> str | None:
    """Classify accelerator/CPU out-of-memory errors without CUDA-only assumptions.

    :param error: Runtime error thrown by model execution.
    :param active_device: Device used for the failed attempt.
    :return: ``cuda``, ``mps``, ``cpu``, or ``None`` when this is not an OOM.
    """
    message = str(error).lower()
    if "cuda out of memory" in message or ("cuda" in message and "out of memory" in message):
        return "cuda"
    if (
        "mps backend out of memory" in message
        or "mps out of memory" in message
        or ("mps" in message and "out of memory" in message)
        or ("metal" in message and "out of memory" in message)
        # The MPS allocator rejects a single buffer above Metal's per-buffer cap
        # with "Invalid buffer size: <n>" - no "out of memory" phrase - and batch
        # halving is the correct recovery for that failure too.
        or "invalid buffer size" in message
    ):
        return "mps"
    if "out of memory" in message or "cannot allocate memory" in message:
        return active_device
    return None


def _move_model_to_cpu(model: object) -> None:
    """Move a model to CPU, re-checking the CPU bf16 inference policy on the way down.

    Every accelerator-to-CPU fallback re-checks the same policy that governs
    load-time dtype selection (:func:`_resolve_model_dtype`): a model loaded
    in bfloat16 must not silently keep executing bf16 on a CPU where the
    policy resolves float32 - either the experimental ``CODEDUPES_CPU_BF16=1``
    opt-in is absent, or the capability gate fails and bf16 measures up to
    841x slower than float32 with no GEMM backend to exploit the ISA
    (measured on an Apple M5). When the policy enables bf16, it is kept: it
    halves memory pressure on this last-resort path at native CPU speed.

    :param model: Model to move, mutated in place.
    :return: ``None``.
    """
    global _model_execution_device
    if hasattr(model, "to"):
        current_dtype = _model_parameter_dtype(model)
        if current_dtype is not None and str(current_dtype) == "torch.bfloat16":
            # The live probe is memoized per process and torch is already
            # imported on this path, so re-checking here is cheap and never
            # touches disk regardless of cache enablement.
            if resolve_cpu_bf16_inference():
                logger.info(
                    "CPU fallback keeps bfloat16: CODEDUPES_CPU_BF16=1 is set and this CPU "
                    "has a native bf16 GEMM backend (mkldnn), halving memory pressure at "
                    "native speed"
                )
                model.to("cpu")
            else:
                logger.warning(
                    "CPU fallback casts bfloat16 to float32: CPU bfloat16 inference is not "
                    "enabled (requires CODEDUPES_CPU_BF16=1 and a native bf16 GEMM backend)"
                )
                import torch

                model.to(device="cpu", dtype=torch.float32)
        else:
            model.to("cpu")
    if model is _model:
        _model_execution_device = "cpu"


def _truncate_code_if_needed(text: str, unit_name: str, model: Any) -> str:
    """Truncate code input to the model max token length with best-effort safety.

    :param text: Source text to truncate.
    :param unit_name: Unit name for logging context.
    :param model: Model object with tokenizer metadata.
    :return: Possibly truncated source text.
    """
    max_tokens = getattr(model, "max_seq_length", None)
    tokenizer = getattr(model, "tokenizer", None)

    if not max_tokens or not tokenizer:
        return text

    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        logger.debug(
            f"Tokenization failed while preparing '{unit_name}'; using full text", exc_info=True
        )
        return text

    token_count = len(token_ids)
    if token_count <= max_tokens:
        return text

    logger.warning(
        f"Code unit '{unit_name}' is long ({token_count} tokens), truncating to {max_tokens} "
        "tokens for semantic embedding"
    )
    try:
        truncated_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
        )
        return tokenizer.decode(truncated_ids, skip_special_tokens=True)
    except Exception:
        logger.debug(
            f"Token decode failed while truncating '{unit_name}'; using char fallback",
            exc_info=True,
        )
        return text[: max_tokens * 4]


def _embedding_cache_namespace(mode: str, variant: str) -> str:
    """Build a namespace grouping equivalent embedding inputs.

    :param mode: Embedding input mode, such as ``code`` or ``query``.
    :param variant: Vector-affecting cache variant, including the encode plan.
    :return: Stable namespace identifier.
    """
    payload = f"{mode}\x00{variant}".encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _prepare_cache_context(
    mode: Literal["code", "query"],
    profile: SemanticModelProfile,
    model_name: str,
    revision: str | None,
    device: str | None,
    encode_plan: EncodePlan,
    *,
    mps_fallback: bool | None,
    trust_remote_code: bool,
    use_cache: bool,
    cache_scope: Path | None,
    strict_revision_cache: bool = False,
) -> tuple[EmbeddingCache | None, str | None, str, str]:
    """Resolve the shared embedding-cache addressing context for one encode call.

    :param mode: Embedding input mode the cache namespace is derived from.
    :param profile: Resolved model profile.
    :param model_name: Requested model identifier.
    :param revision: Explicit model revision request.
    :param device: Requested inference device.
    :param encode_plan: Resolved encode route and prompt.
    :param mps_fallback: Explicit MPS unsupported-op fallback request.
    :param trust_remote_code: Resolved remote-code trust decision.
    :param use_cache: Whether the caller enabled the persistent cache.
    :param cache_scope: Corpus root addressing the cache shard; ``None`` disables caching.
    :param strict_revision_cache: Whether an unpinned hub revision resolves to a
        concrete commit hash (disabling caching when unmappable) instead of the
        requested revision label, defaults to ``False``.
    :return: ``(cache, cache_revision, cache_variant, cache_namespace)``.
    """
    cache = get_embedding_cache() if (use_cache and cache_scope is not None) else None
    cache_revision = (
        _resolve_revision_for_cache(model_name, revision, strict=strict_revision_cache)
        if cache is not None
        else None
    )
    cache_variant = (
        _cache_variant_for(
            profile,
            device,
            encode_plan,
            mps_fallback=mps_fallback,
            trust_remote_code=trust_remote_code,
        )
        if cache is not None
        else ""
    )
    return cache, cache_revision, cache_variant, _embedding_cache_namespace(mode, cache_variant)


def resolve_encode_plan(
    model_name: str = DEFAULT_MODEL,
    mode: Literal["code", "query"] = "code",
    instruction_prefix: str | None = None,
    semantic_task: str | None = None,
) -> EncodePlan:
    """Resolve the encode route and prompt used for one embedding input mode.

    :param model_name: Model identifier.
    :param mode: Embedding mode.
    :param instruction_prefix: Optional explicit prompt override.
    :param semantic_task: Optional task override.
    :return: Encode plan applied exactly once at the backend call.
    """
    task_default = DEFAULT_SEARCH_SEMANTIC_TASK if mode == "query" else DEFAULT_CHECK_SEMANTIC_TASK
    resolved_task = normalize_semantic_task(
        semantic_task,
        default_task=task_default,
    )
    profile = resolve_model_profile(model_name)
    return _resolve_encode_plan(profile, mode, resolved_task, instruction_prefix)


def _encode_texts(
    encode_fn: Callable[..., np.ndarray],
    texts: list[str],
    *,
    batch_size: int,
    show_progress_bar: bool,
    convert_to_numpy: bool,
    normalize_embeddings: bool,
    prompt: str | None = None,
    device: str | None = None,
) -> np.ndarray:
    """Encode texts with defensive kwargs handling across model backends.

    The fallback call is intentionally limited to an explicit "unexpected
    keyword argument: device" failure. Broadly retrying every ``TypeError`` can
    execute inference twice and hide a genuine model/backend bug.

    :param encode_fn: Backend encode callable, for example ``model.encode``.
    :param texts: Inputs to embed.
    :param batch_size: Encode batch size passed to the backend.
    :param show_progress_bar: Whether the backend should render a progress bar.
    :param convert_to_numpy: Whether the backend should return NumPy arrays.
    :param normalize_embeddings: Whether the backend should L2-normalize outputs.
    :param prompt: Explicit prompt for the backend to prepend exactly once, or
        ``None`` to keep the encode function's default prompt selection.
    :param device: Per-call device override, or ``None`` to leave the model device
        untouched, defaults to ``None``.
    :return: Embedding matrix returned by ``encode_fn``.
    """
    kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "show_progress_bar": show_progress_bar,
        "convert_to_numpy": convert_to_numpy,
        "normalize_embeddings": normalize_embeddings,
    }
    if prompt is not None:
        kwargs["prompt"] = prompt
    if device is not None:
        kwargs["device"] = device

    try:
        return encode_fn(texts, **kwargs)
    except TypeError as exc:
        message = str(exc).lower()
        unexpected_device = "device" in message and (
            "unexpected keyword" in message or "unexpected argument" in message
        )
        if device is None or not unexpected_device:
            raise
        kwargs.pop("device", None)
        return encode_fn(texts, **kwargs)


def _encode_with_retries(
    model: object,
    encode_fn: Callable[..., np.ndarray],
    texts: list[str],
    *,
    batch_size: int,
    show_progress_bar: bool,
    initial_device: str,
    model_name: str,
    revision: str | None,
    trust_remote_code: bool,
    stage: str,
    prompt: str | None = None,
) -> np.ndarray:
    """Encode with adaptive OOM recovery for CUDA, MPS, and CPU.

    Recovery first halves the batch until one item remains. Accelerator OOM at
    batch size one then moves the cached model to CPU exactly once, restarting from
    the requested batch size. OOM traceback references are detached before
    synchronization/garbage collection so temporary tensors do not remain live
    during allocator cleanup.

    :param model: Loaded model, moved to CPU in place when accelerator OOM persists.
    :param encode_fn: Backend encode callable bound to ``model``.
    :param texts: Inputs to embed.
    :param batch_size: Requested batch size, also the restart size for the CPU retry.
    :param show_progress_bar: Whether the backend should render a progress bar.
    :param initial_device: Device the model executes on before any CPU fallback.
    :param model_name: Model identifier reported in backend error messages.
    :param revision: Resolved model revision reported in backend error messages.
    :param trust_remote_code: Trust setting reported in backend error messages.
    :param stage: Short stage label used in warnings and error messages.
    :param prompt: Explicit prompt for the backend to prepend exactly once.
    :return: Canonicalized (validated, unit-normalized) embedding matrix for ``texts``.
    :raises SemanticBackendError: If a non-OOM failure matches a known backend issue.
    :raises InvalidEmbeddingError: If encode output violates embedding invariants and
        no CPU retry remains.
    :raises RuntimeError: If OOM persists at batch size one with no fallback left.
    """
    current_batch_size = max(1, batch_size)
    active_device = initial_device
    attempted_cpu_fallback = False

    while True:
        oom_device: str | None = None
        oom_error: RuntimeError | None = None
        result: np.ndarray | None = None

        try:
            result = _encode_texts(
                encode_fn,
                texts,
                batch_size=current_batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=True,
                prompt=prompt,
                device="cpu" if attempted_cpu_fallback else None,
            )
        except RuntimeError as exc:
            oom_device = _classify_oom_device(exc, active_device)
            if oom_device is None:
                if _is_known_semantic_backend_error(exc):
                    raise _wrap_semantic_backend_error(
                        exc,
                        model_name=model_name,
                        revision=revision,
                        trust_remote_code=trust_remote_code,
                        stage=f"{stage} on {active_device}",
                    )
                raise

            oom_error = exc.with_traceback(None)
            oom_error.__context__ = None
        except Exception as exc:
            if _is_known_semantic_backend_error(exc):
                raise _wrap_semantic_backend_error(
                    exc,
                    model_name=model_name,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    stage=f"{stage} on {active_device}",
                )
            raise

        if result is not None:
            try:
                return canonicalize_embeddings(result, expected_rows=len(texts))
            except InvalidEmbeddingError as exc:
                retry_on_cpu = (
                    exc.retryable
                    and active_device in {"cuda", "mps"}
                    and not attempted_cpu_fallback
                )
                if not retry_on_cpu:
                    raise
                logger.warning(
                    f"{active_device.upper()} produced invalid embedding values during {stage} "
                    f"({exc}); clearing the allocator cache and retrying once on CPU"
                )
                del result
                clear_device_cache(active_device, synchronize=True, collect=True)
                _move_model_to_cpu(model)
                active_device = "cpu"
                attempted_cpu_fallback = True
                current_batch_size = max(1, batch_size)
                continue

        # This block runs outside the exception handler so the original traceback
        # no longer retains inference frames while the allocator cache is cleared.
        memory_context = f" ({format_mps_memory_snapshot()})" if oom_device == "mps" else ""

        if current_batch_size > 1:
            next_batch_size = max(1, current_batch_size // 2)
            logger.warning(
                f"{oom_device.upper()} OOM during {stage} at batch_size={current_batch_size}"
                f"{memory_context}; retrying with batch_size={next_batch_size}"
            )
            current_batch_size = next_batch_size
            clear_device_cache(oom_device, synchronize=True, collect=True)
            continue

        source_device = oom_device if oom_device in {"cuda", "mps"} else active_device
        if source_device in {"cuda", "mps"} and not attempted_cpu_fallback:
            logger.warning(
                f"{source_device.upper()} OOM during {stage} at batch_size=1{memory_context}; "
                f"moving the model to CPU and retrying from batch_size={max(1, batch_size)}"
            )
            clear_device_cache(source_device, synchronize=True, collect=True)
            _move_model_to_cpu(model)
            active_device = "cpu"
            attempted_cpu_fallback = True
            # Host memory has different limits than the accelerator, so the CPU retry
            # restarts at the requested batch size instead of inheriting batch_size=1.
            current_batch_size = max(1, batch_size)
            continue

        logger.warning(f"OOM persisted during {stage} at batch_size=1 on {active_device}; aborting")
        raise oom_error


def _compute_embeddings_unlocked(
    units: list[CodeUnit],
    model_name: str = DEFAULT_MODEL,
    instruction_prefix: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
    semantic_task: str | None = None,
    device: str = DEFAULT_SEMANTIC_DEVICE,
    mps_fallback: bool | None = None,
    mps_memory_fraction: float | None = None,
    use_cache: bool = True,
    cache_scope: Path | None = None,
    strict_revision_cache: bool = False,
) -> tuple[np.ndarray, EmbeddingSpaceIdentity]:
    """Compute normalized NumPy embeddings for all code units.

    Embeddings are converted to NumPy immediately, keeping pairwise similarity
    computation on CPU and avoiding long-lived Metal tensors beyond model weights.

    Cache keys are derived from the pre-truncation prepared text so they can be
    computed without loading the model; when every unit hits the on-disk cache,
    the model is never loaded at all. On a cache miss, only the miss texts are
    truncated and encoded through the existing OOM-retry ladder.

    :param units: Code units to embed, preserved in input order.
    :param model_name: Model alias or identifier, defaults to ``DEFAULT_MODEL``.
    :param instruction_prefix: Optional instruction override for embedding inputs.
    :param batch_size: Initial encode batch size, defaults to ``DEFAULT_BATCH_SIZE``.
    :param revision: Optional model revision; ``None`` uses the profile default.
    :param trust_remote_code: Optional remote-code trust setting; ``None`` uses the
        profile default.
    :param semantic_task: Optional task override; ``None`` uses
        ``DEFAULT_CHECK_SEMANTIC_TASK``.
    :param device: ``auto``, ``cpu``, ``cuda``, or ``mps``, defaults to
        ``DEFAULT_SEMANTIC_DEVICE``.
    :param mps_fallback: MPS unsupported-op CPU fallback behavior.
    :param mps_memory_fraction: Optional MPS allocator limit in ``(0, 2]``.
    :param use_cache: Whether to consult/update the persistent embedding cache.
    :param cache_scope: Analyzed corpus root path used to address the cache shard;
        ``None`` disables caching for this call regardless of ``use_cache``.
    :param strict_revision_cache: Whether an unpinned hub revision resolves to a
        concrete commit hash (disabling caching when unmappable) instead of the
        requested revision label, defaults to ``False``.
    :return: Normalized embedding matrix and its effective vector-space identity.
    :raises ValueError: If ``batch_size`` is not positive.
    :raises SemanticBackendError: If an explicitly requested device is unavailable,
        even when every embedding is already cached.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not units:
        return np.zeros((0, 0), dtype=np.float32), resolve_embedding_space_identity(
            model_name=model_name,
            instruction_prefix=instruction_prefix,
            revision=revision,
            trust_remote_code=trust_remote_code,
            semantic_task=semantic_task,
            device=device,
            mps_fallback=mps_fallback,
            persist_local_model_manifest=use_cache and cache_scope is not None,
            strict_revision_cache=strict_revision_cache,
        )

    _validate_explicit_device_request(device, mps_fallback=mps_fallback)

    profile = resolve_model_profile(model_name)
    resolved_task = normalize_semantic_task(
        semantic_task,
        default_task=DEFAULT_CHECK_SEMANTIC_TASK,
    )
    encode_plan = _resolve_encode_plan(profile, "code", resolved_task, instruction_prefix)
    resolved_trust_remote_code = _resolve_trust_remote_code(model_name, trust_remote_code)
    identity_local_model_path = resolve_local_model_path(profile.canonical_name)
    identity_revision = (
        _fingerprint_local_model_dir(
            identity_local_model_path,
            persist_manifest=use_cache and cache_scope is not None,
        )
        if identity_local_model_path is not None
        else _resolve_revision_for_cache(model_name, revision, strict=strict_revision_cache)
    )
    prepared_texts = [unit.source.strip() for unit in units]

    def _effective_identity(
        effective_device: str,
        resolved_identity_revision: str | None = identity_revision,
        concrete_device: str | None = None,
    ) -> EmbeddingSpaceIdentity:
        """Build the identity for the policy that produced the returned matrix.

        :param effective_device: Device policy used for every matrix row.
        :param resolved_identity_revision: Concrete revision/fingerprint for the rows.
        :param concrete_device: Already resolved execution target, when available.
        :return: Effective corpus embedding-space identity.
        """
        return _build_embedding_space_identity(
            profile,
            resolved_identity_revision,
            encode_plan,
            effective_device,
            mps_fallback=mps_fallback,
            trust_remote_code=resolved_trust_remote_code,
            resolved_device=concrete_device,
        )

    def _restart_faithfully_on_cpu(reason: str) -> tuple[np.ndarray, EmbeddingSpaceIdentity]:
        """Restart corpus assembly under one faithful CPU cache and dtype/math policy.

        :param reason: Short human-readable cause logged once before the restart.
        :return: CPU-faithful matrix and identity.
        """
        logger.warning(
            f"{reason}; discarding cache hits and restarting the complete matrix "
            "under one faithful CPU policy"
        )
        return _compute_embeddings_unlocked(
            units,
            model_name=model_name,
            instruction_prefix=instruction_prefix,
            batch_size=batch_size,
            revision=revision,
            trust_remote_code=trust_remote_code,
            semantic_task=resolved_task,
            device="cpu",
            mps_fallback=mps_fallback,
            mps_memory_fraction=None,
            use_cache=use_cache,
            cache_scope=cache_scope,
            strict_revision_cache=strict_revision_cache,
        )

    def _coherence_break_reason(current_model: object) -> str | None:
        """Return why this run cannot stay coherent, or ``None`` when it still can.

        Checks both preconditions that force a discard-and-restart: fast-math
        corpus execution that left MPS (see :func:`_mps_fast_math_variant`),
        and a keyed bfloat16 dtype variant whose live execution can no longer
        produce bfloat16 (see :func:`_dtype_coherence_broken`).

        :param current_model: Loaded model instance, re-inspected for live state.
        :return: Human-readable cause, or ``None``.
        """
        current_execution_device = _get_effective_model_device(current_model, resolved_device)
        if _mps_fast_math_variant(device) and current_execution_device != "mps":
            return "Fast-math corpus execution left MPS"
        if _dtype_coherence_broken(
            profile,
            device,
            mps_fallback,
            resolved_device,
            current_model,
        ):
            return "An accelerator OOM fallback cast this run's keyed bfloat16 vectors to float32"
        return None

    cache, cache_revision, cache_variant, cache_namespace = _prepare_cache_context(
        "code",
        profile,
        model_name,
        revision,
        device,
        encode_plan,
        mps_fallback=mps_fallback,
        trust_remote_code=resolved_trust_remote_code,
        use_cache=use_cache,
        cache_scope=cache_scope,
        strict_revision_cache=strict_revision_cache,
    )
    cache_keys = (
        [
            compute_cache_key(profile.canonical_name, cache_revision, text, variant=cache_variant)
            for text in prepared_texts
        ]
        if cache is not None and cache_revision is not None
        else None
    )
    hits: dict[str, np.ndarray] = (
        cache.get_many(cache_scope, profile.canonical_name, cache_revision, cache_keys)
        if cache is not None and cache_keys is not None
        else {}
    )

    # Duplicate code units share one cache key, so compare against the covered
    # keys rather than the unique-hit count: len(hits) undercounts coverage.
    if cache_keys is not None and all(key in hits for key in cache_keys):
        return _assemble_cached_matrix(cache_keys, hits), _effective_identity(
            device,
            cache_revision,
        )

    resolved_revision = _resolve_load_revision(model_name, revision)
    resolved_device = _prepare_semantic_device(
        device,
        mps_fallback=mps_fallback,
        mps_memory_fraction=mps_memory_fraction,
    )
    model = get_model(
        model_name,
        revision=resolved_revision,
        trust_remote_code=resolved_trust_remote_code,
        device=resolved_device,
        mps_fallback=mps_fallback,
        mps_memory_fraction=mps_memory_fraction,
        persist_local_model_manifest=use_cache and cache_scope is not None,
    )
    execution_device = _get_effective_model_device(model, resolved_device)

    coherence_break_reason = _coherence_break_reason(model)
    if coherence_break_reason is not None:
        return _restart_faithfully_on_cpu(coherence_break_reason)

    confirmed_revision = _confirm_cache_revision_after_load(
        model,
        model_name,
        resolved_revision,
        strict=strict_revision_cache,
    )
    if confirmed_revision is not None:
        identity_revision = confirmed_revision

    if cache is not None:
        if confirmed_revision is None:
            logger.debug(
                "Could not tie the loaded model to a concrete revision; "
                f"bypassing persistent embeddings assumed under {cache_revision}"
            )
            cache = None
            cache_revision = None
            cache_keys = None
            hits = {}
        elif confirmed_revision != cache_revision:
            # Never mix vectors cached under a different resolved revision into
            # this matrix: discard any pre-load hits and re-key under the truth.
            cache_revision = confirmed_revision
            cache_keys = [
                compute_cache_key(
                    profile.canonical_name, cache_revision, text, variant=cache_variant
                )
                for text in prepared_texts
            ]
            hits = cache.get_many(cache_scope, profile.canonical_name, cache_revision, cache_keys)
            if all(key in hits for key in cache_keys):
                return _assemble_cached_matrix(cache_keys, hits), _effective_identity(
                    device,
                    confirmed_revision,
                    resolved_device,
                )

    corpus_source_commit: str | None = None
    if (
        cache is not None
        and cache_revision is not None
        and _revision_is_mutable_label(model_name, cache_revision)
    ):
        # Loose label keying cannot see an upstream branch move on the warm
        # no-load path, but this run loaded the model and knows its commit.
        # Mixing old-commit hits with new-commit misses is only possible here,
        # so a drifted (or provenance-less) shard is purged and every pre-load
        # hit discarded. An unknown loaded commit stays fail-open by design:
        # loose mode keeps serving warm even when a backend cannot report its
        # checkpoint, and the provenance-less rows it writes are purged by the
        # first commit-reporting load.
        loaded_commit = _get_loaded_model_commit_hash(model)
        if loaded_commit is not None:
            corpus_source_commit = loaded_commit
            if not cache.confirm_source_commit(
                cache_scope, profile.canonical_name, cache_revision, loaded_commit
            ):
                logger.warning(
                    f"Cached vectors for branch {cache_revision!r} cannot be tied to loaded "
                    f"commit {loaded_commit[:12]}; discarding {len(hits)} of them and "
                    "re-embedding so one matrix never mixes two checkpoints"
                )
                hits = {}

    miss_indices = _select_cache_miss_indices(cache_keys, hits, len(units))
    miss_texts = [
        _truncate_code_if_needed(prepared_texts[i], units[i].qualified_name, model)
        for i in miss_indices
    ]
    cache_covered_rows = (
        sum(1 for key in cache_keys if key in hits) if cache_keys is not None else 0
    )
    reused_duplicate_rows = len(units) - cache_covered_rows - len(miss_indices)

    logger.info(
        f"Computing embeddings for {len(miss_texts)} unique inputs on {execution_device} "
        f"({cache_covered_rows} cache-covered rows, {reused_duplicate_rows} duplicate rows reused)"
    )
    encode_fn = _select_encode_fn(model, encode_plan.route)

    def _encode_miss_texts(texts: list[str]) -> np.ndarray:
        """Encode prepared miss texts through the shared OOM-retry ladder.

        :param texts: Truncated embedding inputs to encode.
        :return: Normalized embedding matrix row-aligned with ``texts``.
        """
        return _encode_with_retries(
            model,
            encode_fn,
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            initial_device=execution_device,
            model_name=model_name,
            revision=resolved_revision,
            trust_remote_code=resolved_trust_remote_code,
            stage="embedding inference",
            prompt=encode_plan.prompt,
        )

    miss_vectors = _encode_miss_texts(miss_texts)
    coherence_break_reason = _coherence_break_reason(model)
    if coherence_break_reason is not None:
        return _restart_faithfully_on_cpu(coherence_break_reason)

    dim = miss_vectors.shape[1]
    if hits:
        hit_dim = int(next(iter(hits.values())).shape[-1])
        if hit_dim != dim:
            # A shard can be self-consistent on disk yet disagree with the live
            # model's dimensionality; trusting it would corrupt the matrix.
            logger.warning(
                f"Discarding {len(hits)} cached embeddings whose dimensionality ({hit_dim}) does "
                f"not match the loaded model ({dim}); re-embedding all units."
            )
            hits = {}
            miss_indices = _select_cache_miss_indices(cache_keys, hits, len(units))
            # Re-read the live device: a mid-encode accelerator fallback during
            # the first encode call is not reflected by replaying the stale
            # execution_device captured before that call, and this closure
            # variable is what _encode_miss_texts passes as initial_device.
            execution_device = _get_effective_model_device(model, resolved_device)
            miss_vectors = _encode_miss_texts(
                [
                    _truncate_code_if_needed(prepared_texts[i], units[i].qualified_name, model)
                    for i in miss_indices
                ]
            )
            coherence_break_reason = _coherence_break_reason(model)
            if coherence_break_reason is not None:
                return _restart_faithfully_on_cpu(coherence_break_reason)
    if cache_keys is None:
        matrix = np.empty((len(units), dim), dtype=np.float32)
        for local_idx, global_idx in enumerate(miss_indices):
            matrix[global_idx] = miss_vectors[local_idx]
    else:
        vectors_by_key = dict(hits)
        vectors_by_key.update(
            (cache_keys[global_idx], miss_vectors[local_idx])
            for local_idx, global_idx in enumerate(miss_indices)
        )
        matrix = _assemble_cached_matrix(cache_keys, vectors_by_key)

    if (
        cache is not None
        and cache_keys is not None
        and cache_revision is not None
        and miss_indices
        # Re-read live device and dtype: the retry ladder may have moved the
        # model to CPU (or cast it to float32) mid-encode, and those vectors
        # must not enter a key space (fast-math or bfloat16) they can no
        # longer represent.
        and _cache_write_allowed(profile, device, mps_fallback, resolved_device, model)
    ):
        cache.put_many(
            cache_scope,
            profile.canonical_name,
            cache_revision,
            [
                (cache_keys[global_idx], miss_vectors[local_idx])
                for local_idx, global_idx in enumerate(miss_indices)
            ],
            namespace=cache_namespace,
            expected_source_commit=corpus_source_commit,
        )

    return matrix, _effective_identity(device, identity_revision, resolved_device)


def compute_embeddings_with_identity(
    units: list[CodeUnit],
    model_name: str = DEFAULT_MODEL,
    instruction_prefix: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
    semantic_task: str | None = None,
    device: str = DEFAULT_SEMANTIC_DEVICE,
    mps_fallback: bool | None = None,
    mps_memory_fraction: float | None = None,
    use_cache: bool = True,
    cache_scope: Path | None = None,
    strict_revision_cache: bool = False,
) -> tuple[np.ndarray, EmbeddingSpaceIdentity]:
    """Compute embeddings and identity under the shared model lock.

    :param units: Code units to embed, preserved in input order.
    :param model_name: Model alias or identifier, defaults to ``DEFAULT_MODEL``.
    :param instruction_prefix: Optional instruction override for embedding inputs.
    :param batch_size: Initial encode batch size, defaults to ``DEFAULT_BATCH_SIZE``.
    :param revision: Optional model revision; ``None`` uses the profile default.
    :param trust_remote_code: Optional remote-code trust setting; ``None`` uses the
        profile default.
    :param semantic_task: Optional task override; ``None`` uses
        ``DEFAULT_CHECK_SEMANTIC_TASK``.
    :param device: ``auto``, ``cpu``, ``cuda``, or ``mps``, defaults to
        ``DEFAULT_SEMANTIC_DEVICE``.
    :param mps_fallback: MPS unsupported-op CPU fallback behavior.
    :param mps_memory_fraction: Optional MPS allocator limit in ``(0, 2]``.
    :param use_cache: Whether to consult/update the persistent embedding cache.
    :param cache_scope: Analyzed corpus root path used to address the cache shard;
        ``None`` disables caching for this call regardless of ``use_cache``.
    :param strict_revision_cache: Whether an unpinned hub revision resolves to a
        concrete commit hash (disabling caching when unmappable) instead of the
        requested revision label, defaults to ``False``.
    :return: Normalized embedding matrix and its effective vector-space identity.
    """
    # Import-sensitive runtime variables (MPS operator fallback above all) must
    # be set before any path below can import torch - cache-variant derivation
    # may probe CPU capabilities, which is already too late.
    _configure_semantic_runtime_env(device, mps_fallback=mps_fallback)
    with _model_lock:
        return _compute_embeddings_unlocked(
            units,
            model_name=model_name,
            instruction_prefix=instruction_prefix,
            batch_size=batch_size,
            revision=revision,
            trust_remote_code=trust_remote_code,
            semantic_task=semantic_task,
            device=device,
            use_cache=use_cache,
            cache_scope=cache_scope,
            mps_fallback=mps_fallback,
            mps_memory_fraction=mps_memory_fraction,
            strict_revision_cache=strict_revision_cache,
        )


def compute_embeddings(
    units: list[CodeUnit],
    model_name: str = DEFAULT_MODEL,
    instruction_prefix: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
    semantic_task: str | None = None,
    device: str = DEFAULT_SEMANTIC_DEVICE,
    mps_fallback: bool | None = None,
    mps_memory_fraction: float | None = None,
    use_cache: bool = True,
    cache_scope: Path | None = None,
    strict_revision_cache: bool = False,
) -> np.ndarray:
    """Compute embeddings while serializing shared-model lifecycle and inference.

    :param units: Code units to embed, preserved in input order.
    :param model_name: Model alias or identifier, defaults to ``DEFAULT_MODEL``.
    :param instruction_prefix: Optional instruction override for embedding inputs.
    :param batch_size: Initial encode batch size, defaults to ``DEFAULT_BATCH_SIZE``.
    :param revision: Optional model revision; ``None`` uses the profile default.
    :param trust_remote_code: Optional remote-code trust setting; ``None`` uses the
        profile default.
    :param semantic_task: Optional task override; ``None`` uses
        ``DEFAULT_CHECK_SEMANTIC_TASK``.
    :param device: ``auto``, ``cpu``, ``cuda``, or ``mps``, defaults to
        ``DEFAULT_SEMANTIC_DEVICE``.
    :param mps_fallback: MPS unsupported-op CPU fallback behavior.
    :param mps_memory_fraction: Optional MPS allocator limit in ``(0, 2]``.
    :param use_cache: Whether to consult/update the persistent embedding cache.
    :param cache_scope: Analyzed corpus root path used to address the cache shard;
        ``None`` disables caching for this call regardless of ``use_cache``.
    :param strict_revision_cache: Whether an unpinned hub revision resolves to a
        concrete commit hash (disabling caching when unmappable) instead of the
        requested revision label, defaults to ``False``.
    :return: Normalized embedding matrix row-aligned with ``units``.
    """
    embeddings, _identity = compute_embeddings_with_identity(
        units,
        model_name=model_name,
        instruction_prefix=instruction_prefix,
        batch_size=batch_size,
        revision=revision,
        trust_remote_code=trust_remote_code,
        semantic_task=semantic_task,
        device=device,
        mps_fallback=mps_fallback,
        mps_memory_fraction=mps_memory_fraction,
        use_cache=use_cache,
        cache_scope=cache_scope,
        strict_revision_cache=strict_revision_cache,
    )
    return embeddings


def find_semantic_duplicates(
    units: list[CodeUnit],
    embeddings: np.ndarray,
    threshold: float,
    exclude_exact: set[tuple[str, str]] | None = None,
) -> list[DuplicatePair]:
    """Find semantically similar code units via embedding cosine similarity.

    :param units: Candidate units in the same order as ``embeddings``.
    :param embeddings: Embedding matrix.
    :param threshold: Minimum cosine similarity.
    :param exclude_exact: Pairs to exclude from semantic output.
    :return: Similar pairs sorted by confidence.
    """
    exclude_exact = exclude_exact or set()
    n = len(units)

    logger.info(f"Computing pairwise similarities for {n} units")

    duplicates = []

    def _types_compatible(unit_a: CodeUnit, unit_b: CodeUnit) -> bool:
        """Check whether unit kinds are compatible for semantic comparison.

        :param unit_a: First unit.
        :param unit_b: Second unit.
        :return: ``True`` when types are comparable.
        """
        if unit_a.unit_type == unit_b.unit_type:
            return True
        function_like = {"function", "method"}
        return (
            unit_a.unit_type.name.lower() in function_like
            and unit_b.unit_type.name.lower() in function_like
        )

    chunk_size = 500
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        chunk_embeddings = embeddings[i:end_i]

        similarities = chunk_embeddings @ embeddings.T

        for local_idx in range(end_i - i):
            global_idx = i + local_idx
            unit_a = units[global_idx]

            for j in range(global_idx + 1, n):
                sim = float(similarities[local_idx, j])

                # NaN fails every comparison, so `sim < threshold` alone would
                # let a corrupted similarity through as a reported duplicate.
                if not np.isfinite(sim) or sim < threshold:
                    continue

                unit_b = units[j]

                if not _types_compatible(unit_a, unit_b):
                    continue

                if unit_a.file_path == unit_b.file_path and not (
                    unit_a.end_lineno < unit_b.lineno or unit_b.end_lineno < unit_a.lineno
                ):
                    continue

                pair_key = ordered_pair_key(unit_a, unit_b)
                if pair_key in exclude_exact:
                    continue

                duplicates.append(
                    DuplicatePair(
                        unit_a=unit_a,
                        unit_b=unit_b,
                        similarity=float(sim),
                        method="semantic",
                    )
                )

    duplicates.sort(key=lambda x: x.similarity, reverse=True)

    logger.info(f"Found {len(duplicates)} semantic duplicates above threshold {threshold}")
    return duplicates


def _find_similar_to_query_unlocked(
    query: str,
    units: list[CodeUnit],
    embeddings: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    instruction_prefix: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
    threshold: float | None = None,
    semantic_task: str | None = None,
    device: str = DEFAULT_SEMANTIC_DEVICE,
    mps_fallback: bool | None = None,
    mps_memory_fraction: float | None = None,
    use_cache: bool = True,
    cache_scope: Path | None = None,
    corpus_identity: EmbeddingSpaceIdentity | None = None,
    strict_revision_cache: bool = False,
) -> list[tuple[CodeUnit, float]]:
    """Find code units most similar to a natural-language query.

    The query embedding is cached under the same shard as its corpus, keyed on the
    prepared query text. Combined with a fully cached corpus, a repeated identical
    search needs no model load at all.

    :param query: Natural-language search text.
    :param units: Code units row-aligned with ``embeddings``.
    :param embeddings: Precomputed normalized embedding matrix.
    :param model_name: Model alias or identifier, defaults to ``DEFAULT_MODEL``.
    :param instruction_prefix: Optional instruction override for the query text.
    :param top_k: Maximum number of matches to return, defaults to ``DEFAULT_TOP_K``.
    :param revision: Optional model revision; ``None`` uses the profile default.
    :param trust_remote_code: Optional remote-code trust setting; ``None`` uses the
        profile default.
    :param threshold: Minimum cosine similarity; ``None`` uses the model profile
        search default.
    :param semantic_task: Optional task override; ``None`` uses
        ``DEFAULT_SEARCH_SEMANTIC_TASK``.
    :param device: ``auto``, ``cpu``, ``cuda``, or ``mps``, defaults to
        ``DEFAULT_SEMANTIC_DEVICE``.
    :param mps_fallback: MPS unsupported-op CPU fallback behavior.
    :param mps_memory_fraction: Optional MPS allocator limit in ``(0, 2]``.
    :param use_cache: Whether to consult/update the persistent embedding cache.
    :param cache_scope: Analyzed corpus root path used to address the cache shard;
        ``None`` disables caching for this call regardless of ``use_cache``.
    :param corpus_identity: Optional identity captured with ``embeddings``; when
        provided, model/revision/runtime drift requires rebuilding the corpus.
    :param strict_revision_cache: Whether an unpinned hub revision resolves to a
        concrete commit hash (disabling caching when unmappable) instead of the
        requested revision label, defaults to ``False``. Must match the mode
        used to build ``corpus_identity``.
    :return: Up to ``top_k`` ``(unit, similarity)`` pairs at or above the threshold,
        sorted by descending similarity.
    :raises SemanticBackendError: If an explicitly requested device is unavailable,
        even when the query embedding is already cached.
    """
    _validate_explicit_device_request(device, mps_fallback=mps_fallback)

    # After the explicit-device contract above: an empty corpus can match
    # nothing, so return before embedding the query (or loading the model).
    if not units:
        return []

    profile = resolve_model_profile(model_name)
    resolved_threshold = (
        threshold if threshold is not None else get_default_search_threshold(model_name)
    )
    resolved_task = normalize_semantic_task(
        semantic_task,
        default_task=DEFAULT_SEARCH_SEMANTIC_TASK,
    )
    resolved_trust_remote_code = _resolve_trust_remote_code(model_name, trust_remote_code)
    embedding_device = device
    if corpus_identity is not None:
        embedding_device = _require_current_embedding_space(
            corpus_identity,
            model_name=model_name,
            instruction_prefix=instruction_prefix,
            revision=revision,
            trust_remote_code=trust_remote_code,
            semantic_task=resolved_task,
            device=device,
            mps_fallback=mps_fallback,
            persist_local_model_manifest=use_cache and cache_scope is not None,
            strict_revision_cache=strict_revision_cache,
        )

    encode_plan = _resolve_encode_plan(profile, "query", resolved_task, instruction_prefix)
    query_text = query

    cache, cache_revision, cache_variant, cache_namespace = _prepare_cache_context(
        "query",
        profile,
        model_name,
        revision,
        embedding_device,
        encode_plan,
        mps_fallback=mps_fallback,
        trust_remote_code=resolved_trust_remote_code,
        use_cache=use_cache,
        cache_scope=cache_scope,
        strict_revision_cache=strict_revision_cache,
    )
    cache_key = (
        compute_cache_key(
            profile.canonical_name, cache_revision, query_text, mode="query", variant=cache_variant
        )
        if cache is not None and cache_revision is not None
        else None
    )

    def _validated_query_hit(candidate: np.ndarray | None) -> np.ndarray | None:
        """Reject a cached query vector whose dimensionality cannot match the corpus.

        :param candidate: Cached query embedding, or ``None`` on a miss.
        :return: The candidate when usable, else ``None`` to force a fresh encode.
        """
        if candidate is None:
            return None
        if embeddings.size and candidate.shape[-1] != embeddings.shape[1]:
            logger.warning(
                "Discarding a cached query embedding whose dimensionality "
                f"({candidate.shape[-1]}) does not match the corpus matrix "
                f"({embeddings.shape[1]}); re-encoding the query."
            )
            return None
        return candidate

    query_embedding: np.ndarray | None = None
    if cache_key is not None:
        hit = cache.get_many(cache_scope, profile.canonical_name, cache_revision, [cache_key])
        query_embedding = _validated_query_hit(hit.get(cache_key))

    if query_embedding is None:
        resolved_revision = _resolve_load_revision(model_name, revision)
        resolved_device = _prepare_semantic_device(
            embedding_device,
            mps_fallback=mps_fallback,
            mps_memory_fraction=(None if embedding_device == "cpu" else mps_memory_fraction),
        )
        model = get_model(
            model_name,
            revision=resolved_revision,
            trust_remote_code=resolved_trust_remote_code,
            device=resolved_device,
            mps_fallback=mps_fallback,
            mps_memory_fraction=(None if embedding_device == "cpu" else mps_memory_fraction),
            persist_local_model_manifest=use_cache and cache_scope is not None,
        )
        execution_device = _get_effective_model_device(model, resolved_device)

        confirmed_revision = _confirm_cache_revision_after_load(
            model,
            model_name,
            resolved_revision,
            strict=strict_revision_cache,
        )

        query_source_commit: str | None = None
        if (
            cache is not None
            and cache_revision is not None
            and _revision_is_mutable_label(model_name, cache_revision)
        ):
            # A query miss loaded the model, so the shard's source-commit
            # guard can run. Drift here means the corpus matrix was assembled
            # from old-commit cached vectors on the warm no-load path while
            # this query would embed under the new commit: the shard is
            # purged and the comparison must not happen. An unknown loaded
            # commit stays fail-open by design (see the corpus-side guard).
            loaded_commit = _get_loaded_model_commit_hash(model)
            if loaded_commit is not None:
                query_source_commit = loaded_commit
                if not cache.confirm_source_commit(
                    cache_scope, profile.canonical_name, cache_revision, loaded_commit
                ):
                    raise RuntimeError(
                        f"Model branch {cache_revision!r} moved to a different commit since "
                        "this corpus was indexed; its cached vectors were purged. Run index() "
                        "or analyze() again before search()."
                    )

        corpus_encode_plan = _resolve_encode_plan(
            profile,
            "code",
            resolved_task,
            instruction_prefix,
        )

        def _require_compatible_query_execution() -> None:
            """Reject an execution policy or live dtype that cannot match the corpus vectors.

            :raises RuntimeError: If query execution left the corpus math or dtype policy.
            """
            current_execution_device = _get_effective_model_device(model, resolved_device)
            # The identity rebuilt below derives its dtype from the requested
            # device policy, which cannot see a mid-encode accelerator OOM
            # that cast the live model to float32. Check the live parameter
            # dtype directly: a query vector that can no longer be produced
            # under the corpus's keyed bfloat16 policy must never reach the
            # dot product, even though the cache write is already suppressed.
            if _dtype_coherence_broken(
                profile,
                embedding_device,
                mps_fallback,
                resolved_device,
                model,
            ):
                raise RuntimeError(
                    "An accelerator fallback cast query execution to float32, but the "
                    "corpus vectors are keyed under a bfloat16 policy. Rebuild the "
                    "corpus with index() or analyze() before search()."
                )
            effective_policy_device = embedding_device
            if _mps_fast_math_variant(embedding_device) and current_execution_device != "mps":
                effective_policy_device = "cpu"

            if corpus_identity is None:
                if effective_policy_device != embedding_device:
                    raise RuntimeError(
                        "Fast-math query execution left MPS before the similarity comparison. "
                        "Rebuild the corpus and search with device='cpu' so both sides use one "
                        "embedding policy."
                    )
                return

            loaded_identity = _build_embedding_space_identity(
                profile,
                confirmed_revision,
                corpus_encode_plan,
                effective_policy_device,
                mps_fallback=mps_fallback,
                trust_remote_code=resolved_trust_remote_code,
            )
            if loaded_identity != corpus_identity:
                raise RuntimeError(
                    "The semantic model or execution policy changed while preparing this "
                    "search query. Run index() or analyze() again before search()."
                )

        _require_compatible_query_execution()

        if cache is not None:
            if confirmed_revision is None:
                logger.debug(
                    "Could not tie the loaded model to a concrete revision; "
                    f"bypassing persistent query embedding assumed under {cache_revision}"
                )
                cache = None
                cache_revision = None
                cache_key = None
            elif confirmed_revision != cache_revision:
                cache_revision = confirmed_revision
                cache_key = compute_cache_key(
                    profile.canonical_name,
                    cache_revision,
                    query_text,
                    mode="query",
                    variant=cache_variant,
                )
                hit = cache.get_many(
                    cache_scope, profile.canonical_name, cache_revision, [cache_key]
                )
                query_embedding = _validated_query_hit(hit.get(cache_key))

        if query_embedding is None:
            encode_fn = _select_encode_fn(model, encode_plan.route)

            query_embeddings = _encode_with_retries(
                model,
                encode_fn,
                [query_text],
                batch_size=1,
                show_progress_bar=False,
                initial_device=execution_device,
                model_name=model_name,
                revision=resolved_revision,
                trust_remote_code=resolved_trust_remote_code,
                stage="query embedding",
                prompt=encode_plan.prompt,
            )
            _require_compatible_query_execution()
            query_embedding = query_embeddings[0]

            if (
                cache is not None
                and cache_key is not None
                and cache_revision is not None
                # Re-read live device and dtype: the retry ladder may have
                # moved the model to CPU (or cast it to float32) mid-encode,
                # and that query vector must not be persisted into a key
                # space (fast-math or bfloat16) it can no longer represent.
                and _cache_write_allowed(
                    profile, embedding_device, mps_fallback, resolved_device, model
                )
            ):
                cache.put_many(
                    cache_scope,
                    profile.canonical_name,
                    cache_revision,
                    [(cache_key, query_embedding)],
                    namespace=cache_namespace,
                    max_namespace_keys=_MAX_CACHED_QUERY_KEYS,
                    expected_source_commit=query_source_commit,
                )

    similarities = embeddings @ query_embedding
    sorted_indices = np.argsort(similarities)[::-1]
    filtered_indices = [idx for idx in sorted_indices if similarities[idx] >= resolved_threshold]
    top_indices = filtered_indices[:top_k]

    return [(units[i], float(similarities[i])) for i in top_indices]


def find_similar_to_query(
    query: str,
    units: list[CodeUnit],
    embeddings: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    instruction_prefix: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
    threshold: float | None = None,
    semantic_task: str | None = None,
    device: str = DEFAULT_SEMANTIC_DEVICE,
    mps_fallback: bool | None = None,
    mps_memory_fraction: float | None = None,
    use_cache: bool = True,
    cache_scope: Path | None = None,
    corpus_identity: EmbeddingSpaceIdentity | None = None,
    strict_revision_cache: bool = False,
) -> list[tuple[CodeUnit, float]]:
    """Search embeddings while serializing shared-model lifecycle and inference.

    :param query: Natural-language search text.
    :param units: Code units row-aligned with ``embeddings``.
    :param embeddings: Precomputed normalized embedding matrix.
    :param model_name: Model alias or identifier, defaults to ``DEFAULT_MODEL``.
    :param instruction_prefix: Optional instruction override for the query text.
    :param top_k: Maximum number of matches to return, defaults to ``DEFAULT_TOP_K``.
    :param revision: Optional model revision; ``None`` uses the profile default.
    :param trust_remote_code: Optional remote-code trust setting; ``None`` uses the
        profile default.
    :param threshold: Minimum cosine similarity; ``None`` uses the model profile
        search default.
    :param semantic_task: Optional task override; ``None`` uses
        ``DEFAULT_SEARCH_SEMANTIC_TASK``.
    :param device: ``auto``, ``cpu``, ``cuda``, or ``mps``, defaults to
        ``DEFAULT_SEMANTIC_DEVICE``.
    :param mps_fallback: MPS unsupported-op CPU fallback behavior.
    :param mps_memory_fraction: Optional MPS allocator limit in ``(0, 2]``.
    :param use_cache: Whether to consult/update the persistent embedding cache.
    :param cache_scope: Analyzed corpus root path used to address the cache shard;
        ``None`` disables caching for this call regardless of ``use_cache``.
    :param corpus_identity: Optional identity captured with ``embeddings``; when
        provided, model/revision/runtime drift requires rebuilding the corpus.
    :param strict_revision_cache: Whether an unpinned hub revision resolves to a
        concrete commit hash (disabling caching when unmappable) instead of the
        requested revision label, defaults to ``False``. Must match the mode
        used to build ``corpus_identity``.
    :return: Up to ``top_k`` ``(unit, similarity)`` pairs at or above the threshold,
        sorted by descending similarity.
    """
    # Same contract as compute_embeddings_with_identity: configure
    # import-sensitive runtime variables before anything can import torch.
    _configure_semantic_runtime_env(device, mps_fallback=mps_fallback)
    with _model_lock:
        return _find_similar_to_query_unlocked(
            query,
            units,
            embeddings,
            model_name=model_name,
            instruction_prefix=instruction_prefix,
            top_k=top_k,
            revision=revision,
            trust_remote_code=trust_remote_code,
            threshold=threshold,
            semantic_task=semantic_task,
            device=device,
            mps_fallback=mps_fallback,
            mps_memory_fraction=mps_memory_fraction,
            use_cache=use_cache,
            cache_scope=cache_scope,
            corpus_identity=corpus_identity,
            strict_revision_cache=strict_revision_cache,
        )


def run_semantic_analysis_with_identity(
    units: list[CodeUnit],
    model_name: str = DEFAULT_MODEL,
    instruction_prefix: str | None = None,
    threshold: float | None = None,
    exclude_pairs: set[tuple[str, str]] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
    semantic_task: str | None = None,
    device: str = DEFAULT_SEMANTIC_DEVICE,
    mps_fallback: bool | None = None,
    mps_memory_fraction: float | None = None,
    use_cache: bool = True,
    cache_scope: Path | None = None,
    strict_revision_cache: bool = False,
) -> tuple[np.ndarray, list[DuplicatePair], EmbeddingSpaceIdentity]:
    """Run semantic duplicate detection and return the corpus identity.

    :param units: Code units to embed and compare.
    :param model_name: Model alias or identifier, defaults to ``DEFAULT_MODEL``.
    :param instruction_prefix: Optional instruction override for embedding inputs.
    :param threshold: Minimum cosine similarity; ``None`` uses the model profile default.
    :param exclude_pairs: Ordered pair keys to omit from the semantic results.
    :param batch_size: Initial encode batch size, defaults to ``DEFAULT_BATCH_SIZE``.
    :param revision: Optional model revision; ``None`` uses the profile default.
    :param trust_remote_code: Optional remote-code trust setting; ``None`` uses the
        profile default.
    :param semantic_task: Optional task override; ``None`` uses
        ``DEFAULT_CHECK_SEMANTIC_TASK``.
    :param device: ``auto``, ``cpu``, ``cuda``, or ``mps``, defaults to
        ``DEFAULT_SEMANTIC_DEVICE``.
    :param mps_fallback: MPS unsupported-op CPU fallback behavior.
    :param mps_memory_fraction: Optional MPS allocator limit in ``(0, 2]``.
    :param use_cache: Whether to consult/update the persistent embedding cache.
    :param cache_scope: Analyzed corpus root path used to address the cache shard;
        ``None`` disables caching for this call regardless of ``use_cache``.
    :param strict_revision_cache: Whether an unpinned hub revision resolves to a
        concrete commit hash (disabling caching when unmappable) instead of the
        requested revision label, defaults to ``False``.
    :return: ``(embeddings, duplicates, identity)``.
    """
    resolved_threshold = (
        threshold if threshold is not None else get_default_semantic_threshold(model_name)
    )

    embeddings, identity = compute_embeddings_with_identity(
        units,
        model_name=model_name,
        instruction_prefix=instruction_prefix,
        batch_size=batch_size,
        revision=revision,
        trust_remote_code=trust_remote_code,
        semantic_task=semantic_task,
        device=device,
        mps_fallback=mps_fallback,
        mps_memory_fraction=mps_memory_fraction,
        use_cache=use_cache,
        cache_scope=cache_scope,
        strict_revision_cache=strict_revision_cache,
    )
    if not units:
        return embeddings, [], identity

    duplicates = find_semantic_duplicates(
        units,
        embeddings,
        threshold=resolved_threshold,
        exclude_exact=exclude_pairs,
    )

    return embeddings, duplicates, identity


def run_semantic_analysis(
    units: list[CodeUnit],
    model_name: str = DEFAULT_MODEL,
    instruction_prefix: str | None = None,
    threshold: float | None = None,
    exclude_pairs: set[tuple[str, str]] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
    semantic_task: str | None = None,
    device: str = DEFAULT_SEMANTIC_DEVICE,
    mps_fallback: bool | None = None,
    mps_memory_fraction: float | None = None,
    use_cache: bool = True,
    cache_scope: Path | None = None,
    strict_revision_cache: bool = False,
) -> tuple[np.ndarray, list[DuplicatePair]]:
    """Run full semantic duplicate detection.

    :param units: Code units to embed and compare.
    :param model_name: Model alias or identifier, defaults to ``DEFAULT_MODEL``.
    :param instruction_prefix: Optional instruction override for embedding inputs.
    :param threshold: Minimum cosine similarity; ``None`` uses the model profile default.
    :param exclude_pairs: Ordered pair keys to omit from the semantic results.
    :param batch_size: Initial encode batch size, defaults to ``DEFAULT_BATCH_SIZE``.
    :param revision: Optional model revision; ``None`` uses the profile default.
    :param trust_remote_code: Optional remote-code trust setting; ``None`` uses the
        profile default.
    :param semantic_task: Optional task override; ``None`` uses
        ``DEFAULT_CHECK_SEMANTIC_TASK``.
    :param device: ``auto``, ``cpu``, ``cuda``, or ``mps``, defaults to
        ``DEFAULT_SEMANTIC_DEVICE``.
    :param mps_fallback: MPS unsupported-op CPU fallback behavior.
    :param mps_memory_fraction: Optional MPS allocator limit in ``(0, 2]``.
    :param use_cache: Whether to consult/update the persistent embedding cache.
    :param cache_scope: Analyzed corpus root path used to address the cache shard;
        ``None`` disables caching for this call regardless of ``use_cache``.
    :param strict_revision_cache: Whether an unpinned hub revision resolves to a
        concrete commit hash (disabling caching when unmappable) instead of the
        requested revision label, defaults to ``False``.
    :return: ``(embeddings, duplicates)``; both are empty when ``units`` is empty.
    """
    embeddings, duplicates, _identity = run_semantic_analysis_with_identity(
        units,
        model_name=model_name,
        instruction_prefix=instruction_prefix,
        threshold=threshold,
        exclude_pairs=exclude_pairs,
        batch_size=batch_size,
        revision=revision,
        trust_remote_code=trust_remote_code,
        semantic_task=semantic_task,
        device=device,
        mps_fallback=mps_fallback,
        mps_memory_fraction=mps_memory_fraction,
        use_cache=use_cache,
        cache_scope=cache_scope,
        strict_revision_cache=strict_revision_cache,
    )
    return embeddings, duplicates
