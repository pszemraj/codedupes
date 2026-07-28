"""Semantic duplicate detection using embedding similarity."""

from __future__ import annotations

import ast
import hashlib
import importlib
import logging
import os
import sys
import threading
from collections.abc import Callable
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
    DEFAULT_SEMANTIC_THRESHOLD,
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
    resolve_semantic_device,
)
from codedupes.embedding_cache import compute_cache_key, get_embedding_cache
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
_model_device_key: str | None = None
_model_execution_device: str | None = None
_model_lock = threading.RLock()
_warned_mlx_mps_contention = False
_warned_cpu_fallback_reuse = False

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


class SemanticBackendError(RuntimeError):
    """Raised when semantic model loading or inference backend is incompatible."""


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


def _resolve_hf_cached_revision(canonical_model: str) -> str | None:
    """Resolve the locally cached commit hash for an unpinned model, without downloading.

    :param canonical_model: Canonical HuggingFace model identifier.
    :return: Resolved commit hash, or ``None`` when it cannot be determined offline.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        cached = try_to_load_from_cache(canonical_model, "config.json", revision="main")
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


def _fingerprint_local_model_dir(model_dir: Path) -> str | None:
    """Fingerprint a local model directory for use as a cache revision.

    The fingerprint hashes each model file's relative path, size, and mtime,
    excluding Hugging Face's ``--local-dir`` download metadata. Replacing or
    retraining the weights in place therefore changes the cache revision
    without invalidating embeddings when only download timestamps change. It
    reads no file contents, keeping warm-path key derivation cheap enough to
    run before any model import.

    :param model_dir: Resolved local model directory.
    :return: ``"dir-<hex>"`` fingerprint, or ``None`` when the walk fails.
    """
    entries: list[tuple[str, int, int]] = []
    hf_download_metadata = model_dir / ".cache" / "huggingface"
    try:
        for file_path in sorted(model_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.is_relative_to(hf_download_metadata):
                continue
            stat = file_path.stat()
            relative = file_path.relative_to(model_dir).as_posix()
            entries.append((relative, stat.st_size, stat.st_mtime_ns))
    except OSError:
        return None
    if not entries:
        return None
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


def _resolve_revision_for_cache(model_name: str, explicit_revision: str | None) -> str | None:
    """Resolve a concrete revision usable as a cache key component, without loading the model.

    Pinned profiles (or an explicit override) resolve immediately. Local model
    directories resolve to a content fingerprint of the directory (an explicit
    revision is ignored for them because nothing pins on-disk weights). Unpinned
    hub models fall back to reading the locally cached HuggingFace commit hash so
    cache keys stay stable across runs even before the model is loaded.

    :param model_name: Requested model identifier.
    :param explicit_revision: Optional explicit revision override.
    :return: Concrete revision string, or ``None`` when it cannot be resolved offline.
    """
    canonical_model = resolve_model_profile(model_name).canonical_name
    local_dir = resolve_local_model_path(canonical_model)
    if local_dir is not None:
        return _fingerprint_local_model_dir(local_dir)
    profile_revision = _resolve_model_revision(model_name, explicit_revision)
    if profile_revision is not None:
        return profile_revision
    return _resolve_hf_cached_revision(canonical_model)


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


_DEVICE_DTYPE_FAMILIES = frozenset({"embeddinggemma"})


def _cache_variant_for(
    profile: SemanticModelProfile,
    device: str,
    *,
    mps_fallback: bool | None,
) -> str:
    """Build the vector-affecting cache-key variant for one model family.

    EmbeddingGemma selects its torch dtype from the execution device (bfloat16 vs
    float32), so its cache identity records only a non-default dtype. CPU and MPS
    both use float32 and therefore share a key space without importing PyTorch to
    resolve the device. On macOS, ``auto`` can only select MPS or CPU, so it shares
    that same model-free warm path.
    Families that embed identically across devices share one key space.

    :param profile: Resolved model profile.
    :param device: Requested device string (``auto``, ``cpu``, ``cuda``, ``mps``).
    :param mps_fallback: MPS unsupported-op CPU fallback behavior.
    :return: Variant fingerprint, empty for device-independent families.
    """
    if profile.family not in _DEVICE_DTYPE_FAMILIES:
        return ""

    normalized_device = device.strip().lower()
    if normalized_device in {"cpu", "mps"} or (
        normalized_device == "auto" and sys.platform == "darwin"
    ):
        return ""

    resolved_device = _resolve_semantic_device_request(
        device,
        mps_fallback=mps_fallback,
    )
    selected_dtype = _resolve_embeddinggemma_torch_dtype(resolved_device)
    dtype_name = str(selected_dtype) if selected_dtype is not None else "default"
    if dtype_name in {"float32", "fp32", "torch.float32"}:
        return ""
    return f"dtype={dtype_name}"


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
    _configure_semantic_runtime_env(device, mps_fallback=mps_fallback)
    try:
        return resolve_semantic_device(device)
    except (DeviceConfigurationError, ValueError) as exc:
        raise SemanticBackendError(str(exc)) from exc


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
        elif mps_memory_fraction is not None:
            logger.info("mps_memory_fraction ignored: resolved device is %s", resolved_device)
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


def get_code_unit_statement_count(unit: CodeUnit) -> int:
    """Get effective statement count for a unit, excluding docstring.

    :param unit: Unit to measure.
    :return: Number of executable statements.
    """
    if not unit.source:
        return 0

    text = unit.source.strip()
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

    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]

    return len(body)


def _resolve_embeddinggemma_torch_dtype(device: str) -> Any:
    """Choose a stable EmbeddingGemma dtype for the selected device.

    :param device: Concrete execution device.
    :return: Suggested dtype object for Torch models, or ``None``.
    """
    try:
        import torch
    except ModuleNotFoundError:
        return None

    if (
        device == "cuda"
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16

    return torch.float32


def _get_model_unlocked(
    model_name: str = DEFAULT_MODEL,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
    device: str = DEFAULT_SEMANTIC_DEVICE,
    mps_fallback: bool | None = None,
    mps_memory_fraction: float | None = None,
) -> object:
    """Lazy-load the embedding model on an explicit resolved device.

    :param model_name: Model alias or identifier.
    :param revision: Optional model revision.
    :param trust_remote_code: Optional remote code trust setting.
    :param device: ``auto``, ``cpu``, ``cuda``, or ``mps``.
    :param mps_fallback: MPS unsupported-op CPU fallback behavior.
    :param mps_memory_fraction: Optional MPS allocator limit in ``(0, 2]``.
    :return: Loaded model instance.
    """
    global _model, _model_name, _model_revision, _model_trust_remote_code
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
    resolved_revision = _resolve_model_revision(model_name, revision)
    if resolved_revision is not None and local_model_path is not None:
        logger.warning(
            "Ignoring revision %r for local model directory %s; on-disk weights are unpinned",
            resolved_revision,
            resolved_model_name,
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
            _model_device_key != resolved_device,
        )
    )

    if cache_miss:
        if _model is not None:
            _clear_model_cache_unlocked()

        logger.info("Loading embedding model %s on %s", resolved_model_name, resolved_device)
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
        tokenizer_kwargs: dict[str, object] = {}
        config_kwargs: dict[str, object] = {}

        if profile.family == "embeddinggemma":
            selected_dtype = _resolve_embeddinggemma_torch_dtype(resolved_device)
            if selected_dtype is not None:
                model_kwargs["torch_dtype"] = selected_dtype
                logger.info(
                    "Using EmbeddingGemma torch dtype on %s: %s",
                    resolved_device,
                    selected_dtype,
                )

        if resolved_revision is not None:
            model_kwargs["revision"] = resolved_revision
            tokenizer_kwargs["revision"] = resolved_revision
            config_kwargs["revision"] = resolved_revision

        if resolved_trust_remote_code:
            model_kwargs["trust_remote_code"] = True
            tokenizer_kwargs["trust_remote_code"] = True
            config_kwargs["trust_remote_code"] = True

        if model_kwargs:
            st_kwargs["model_kwargs"] = model_kwargs
        if tokenizer_kwargs:
            st_kwargs["tokenizer_kwargs"] = tokenizer_kwargs
        if config_kwargs:
            st_kwargs["config_kwargs"] = config_kwargs

        load_device = resolved_device
        try:
            loaded_model = SentenceTransformer(resolved_model_name, **st_kwargs)
        except RuntimeError as exc:
            oom_device = _classify_oom_device(exc, resolved_device)
            if resolved_device == "mps" and oom_device == "mps":
                exc.__traceback__ = None
                exc.__context__ = None
                logger.warning(
                    "MPS OOM while loading %s (%s); clearing Metal cache and retrying on CPU",
                    resolved_model_name,
                    format_mps_memory_snapshot(),
                )
                clear_device_cache("mps", synchronize=True, collect=True)
                cpu_kwargs = dict(st_kwargs)
                cpu_kwargs["device"] = "cpu"
                try:
                    loaded_model = SentenceTransformer(resolved_model_name, **cpu_kwargs)
                except Exception as retry_exc:
                    if _is_known_semantic_backend_error(retry_exc):
                        raise _wrap_semantic_backend_error(
                            retry_exc,
                            model_name=resolved_model_name,
                            revision=resolved_revision,
                            trust_remote_code=resolved_trust_remote_code,
                            stage="CPU model-loading retry after MPS OOM",
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

        _model = loaded_model
        _model_name = resolved_model_name
        _model_revision = resolved_revision
        _model_trust_remote_code = resolved_trust_remote_code
        _model_device_key = resolved_device
        _model_execution_device = _coerce_device_name(
            getattr(loaded_model, "device", None),
            load_device,
        )
        _warned_cpu_fallback_reuse = False
    elif _model_execution_device != resolved_device and not _warned_cpu_fallback_reuse:
        logger.warning(
            "Reusing cached model on %s after an earlier %s-to-CPU OOM fallback; "
            "call clear_model_cache() to force a fresh %s load",
            _model_execution_device,
            resolved_device,
            resolved_device,
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
    :return: Cached model instance, reloaded when any cache key changed.
    """
    with _model_lock:
        return _get_model_unlocked(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            device=device,
            mps_fallback=mps_fallback,
            mps_memory_fraction=mps_memory_fraction,
        )


def _clear_model_cache_unlocked() -> None:
    """Release the cached model and its accelerator allocator cache."""
    global _model, _model_name, _model_revision, _model_trust_remote_code
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
    ):
        return "mps"
    if "out of memory" in message or "cannot allocate memory" in message:
        return active_device
    return None


def _move_model_to_cpu(model: object) -> None:
    """Move a model to CPU and update cached execution-device state."""
    global _model_execution_device
    if hasattr(model, "to"):
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
            "Tokenization failed while preparing '%s'; using full text", unit_name, exc_info=True
        )
        return text

    token_count = len(token_ids)
    if token_count <= max_tokens:
        return text

    logger.warning(
        "Code unit '%s' is long (%d tokens), truncating to %d tokens for semantic embedding",
        unit_name,
        token_count,
        max_tokens,
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
            "Token decode failed while truncating '%s'; using char fallback",
            unit_name,
            exc_info=True,
        )
        return text[: max_tokens * 4]


def _get_embeddinggemma_prefix(task: SemanticTask, mode: Literal["code", "query"]) -> str:
    """Get task-aware prompt prefixes for EmbeddingGemma.

    :param task: Normalized task.
    :param mode: Input mode.
    :return: Instruction prefix.
    """
    if mode == "query":
        return EMBEDDINGGEMMA_QUERY_PREFIXES[task]

    if task in {"retrieval", "code-retrieval"}:
        return EMBEDDINGGEMMA_DOCUMENT_PREFIX

    return EMBEDDINGGEMMA_QUERY_PREFIXES[task]


def _get_instruction(
    profile: SemanticModelProfile,
    mode: Literal["code", "query"],
    semantic_task: SemanticTask,
) -> str:
    """Get default instruction prefix for model/task/mode.

    :param profile: Resolved model profile.
    :param mode: Input mode.
    :param semantic_task: Normalized task.
    :return: Instruction prefix.
    """
    if profile.family == "embeddinggemma":
        return _get_embeddinggemma_prefix(semantic_task, mode)

    return ""


def _resolve_instruction_prefix(
    profile: SemanticModelProfile,
    mode: Literal["code", "query"],
    instruction_prefix: str | None,
    *,
    semantic_task: SemanticTask,
) -> str:
    """Resolve instruction prefix override for embedding inputs.

    :param profile: Resolved model profile.
    :param mode: Input mode.
    :param instruction_prefix: Optional override.
    :param semantic_task: Resolved task.
    :return: Instruction prefix.
    """
    if instruction_prefix is not None:
        return instruction_prefix
    return _get_instruction(profile, mode, semantic_task)


def _prefix_embedding_text(instruction: str, text: str) -> str:
    """Join a resolved instruction with one embedding input.

    :param instruction: Resolved model instruction prefix.
    :param text: Source or query text, already normalized for its input mode.
    :return: Exact pre-truncation text used for cache identity and inference.
    """
    return f"{instruction}{text}"


def prepare_code_for_embedding(
    unit: CodeUnit,
    model_name: str = DEFAULT_MODEL,
    mode: Literal["code", "query"] = "code",
    instruction_prefix: str | None = None,
    semantic_task: str | None = None,
) -> str:
    """Prepare code unit for embedding.

    :param unit: Source unit to embed.
    :param model_name: Model identifier.
    :param mode: Embedding mode.
    :param instruction_prefix: Optional explicit instruction.
    :param semantic_task: Optional task override.
    :return: Prefixed source payload.
    """
    source = unit.source.strip()
    task_default = DEFAULT_SEARCH_SEMANTIC_TASK if mode == "query" else DEFAULT_CHECK_SEMANTIC_TASK
    resolved_task = normalize_semantic_task(
        semantic_task,
        default_task=task_default,
    )
    profile = resolve_model_profile(model_name)
    instruction = _resolve_instruction_prefix(
        profile,
        mode,
        instruction_prefix,
        semantic_task=resolved_task,
    )
    return _prefix_embedding_text(instruction, source)


def _encode_texts(
    encode_fn: Callable[..., np.ndarray],
    texts: list[str],
    *,
    batch_size: int,
    show_progress_bar: bool,
    convert_to_numpy: bool,
    normalize_embeddings: bool,
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
    :return: Embedding matrix for ``texts``.
    :raises SemanticBackendError: If a non-OOM failure matches a known backend issue.
    :raises RuntimeError: If OOM persists at batch size one with no fallback left.
    """
    current_batch_size = max(1, batch_size)
    active_device = initial_device
    attempted_cpu_fallback = False

    while True:
        oom_device: str | None = None
        oom_error: RuntimeError | None = None

        try:
            return _encode_texts(
                encode_fn,
                texts,
                batch_size=current_batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=True,
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

        # This block runs outside the exception handler so the original traceback
        # no longer retains inference frames while the allocator cache is cleared.
        if oom_device == "mps":
            logger.warning(
                "MPS OOM during %s at batch_size=%d (%s)",
                stage,
                current_batch_size,
                format_mps_memory_snapshot(),
            )

        if current_batch_size > 1:
            next_batch_size = max(1, current_batch_size // 2)
            logger.warning(
                "%s OOM during %s at batch_size=%d; retrying with batch_size=%d",
                oom_device.upper(),
                stage,
                current_batch_size,
                next_batch_size,
            )
            current_batch_size = next_batch_size
            clear_device_cache(oom_device, synchronize=True, collect=True)
            continue

        source_device = oom_device if oom_device in {"cuda", "mps"} else active_device
        if source_device in {"cuda", "mps"} and not attempted_cpu_fallback:
            logger.warning(
                "%s OOM during %s at batch_size=1; moving the model to CPU and retrying "
                "from batch_size=%d",
                source_device.upper(),
                stage,
                max(1, batch_size),
            )
            clear_device_cache(source_device, synchronize=True, collect=True)
            _move_model_to_cpu(model)
            active_device = "cpu"
            attempted_cpu_fallback = True
            # Host memory has different limits than the accelerator, so the CPU retry
            # restarts at the requested batch size instead of inheriting batch_size=1.
            current_batch_size = max(1, batch_size)
            continue

        logger.warning(
            "OOM persisted during %s at batch_size=1 on %s; aborting",
            stage,
            active_device,
        )
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
) -> np.ndarray:
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
    :return: Normalized embedding matrix row-aligned with ``units``.
    :raises ValueError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not units:
        return np.zeros((0, 0), dtype=np.float32)

    profile = resolve_model_profile(model_name)
    resolved_task = normalize_semantic_task(
        semantic_task,
        default_task=DEFAULT_CHECK_SEMANTIC_TASK,
    )
    instruction = _resolve_instruction_prefix(
        profile,
        "code",
        instruction_prefix,
        semantic_task=resolved_task,
    )
    prepared_texts = [_prefix_embedding_text(instruction, unit.source.strip()) for unit in units]

    cache = get_embedding_cache() if (use_cache and cache_scope is not None) else None
    cache_revision = (
        _resolve_revision_for_cache(model_name, revision) if cache is not None else None
    )
    cache_variant = (
        _cache_variant_for(profile, device, mps_fallback=mps_fallback) if cache is not None else ""
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
        return _assemble_cached_matrix(cache_keys, hits)

    resolved_revision = _resolve_load_revision(model_name, revision)
    resolved_trust_remote_code = _resolve_trust_remote_code(model_name, trust_remote_code)
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
    )
    execution_device = _get_effective_model_device(model, resolved_device)

    if cache is not None:
        confirmed_revision = _get_loaded_model_commit_hash(model) or resolved_revision
        if confirmed_revision is None:
            # Unconfirmable is not the same as different: keep the pre-load
            # resolution rather than silently discarding valid hits.
            logger.debug(
                "Could not confirm the loaded model revision; keeping cache revision %s",
                cache_revision,
            )
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
                return _assemble_cached_matrix(cache_keys, hits)

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
        "Computing embeddings for %d unique inputs on %s "
        "(%d cache-covered rows, %d duplicate rows reused)",
        len(miss_texts),
        execution_device,
        cache_covered_rows,
        reused_duplicate_rows,
    )
    encode_fn = model.encode
    if profile.family == "embeddinggemma" and hasattr(model, "encode_document"):
        encode_fn = model.encode_document

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
        )

    miss_vectors = _encode_miss_texts(miss_texts)

    dim = miss_vectors.shape[1]
    if hits:
        hit_dim = int(next(iter(hits.values())).shape[-1])
        if hit_dim != dim:
            # A shard can be self-consistent on disk yet disagree with the live
            # model's dimensionality; trusting it would corrupt the matrix.
            logger.warning(
                "Discarding %d cached embeddings whose dimensionality (%d) does not "
                "match the loaded model (%d); re-embedding all units.",
                len(hits),
                hit_dim,
                dim,
            )
            hits = {}
            miss_indices = _select_cache_miss_indices(cache_keys, hits, len(units))
            miss_vectors = _encode_miss_texts(
                [
                    _truncate_code_if_needed(prepared_texts[i], units[i].qualified_name, model)
                    for i in miss_indices
                ]
            )
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

    if cache is not None and cache_keys is not None and cache_revision is not None and miss_indices:
        cache.put_many(
            cache_scope,
            profile.canonical_name,
            cache_revision,
            [
                (cache_keys[global_idx], miss_vectors[local_idx])
                for local_idx, global_idx in enumerate(miss_indices)
            ],
        )

    return matrix


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
    :return: Normalized embedding matrix row-aligned with ``units``.
    """
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
        )


def find_semantic_duplicates(
    units: list[CodeUnit],
    embeddings: np.ndarray,
    threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
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

    logger.info("Computing pairwise similarities for %d units", n)

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
                sim = similarities[local_idx, j]

                if sim < threshold:
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

    logger.info("Found %d semantic duplicates above threshold %s", len(duplicates), threshold)
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
    :return: Up to ``top_k`` ``(unit, similarity)`` pairs at or above the threshold,
        sorted by descending similarity.
    """
    profile = resolve_model_profile(model_name)
    resolved_threshold = (
        threshold if threshold is not None else get_default_search_threshold(model_name)
    )
    resolved_task = normalize_semantic_task(
        semantic_task,
        default_task=DEFAULT_SEARCH_SEMANTIC_TASK,
    )
    instruction = _resolve_instruction_prefix(
        profile,
        "query",
        instruction_prefix,
        semantic_task=resolved_task,
    )
    query_text = _prefix_embedding_text(instruction, query)

    cache = get_embedding_cache() if (use_cache and cache_scope is not None) else None
    cache_revision = (
        _resolve_revision_for_cache(model_name, revision) if cache is not None else None
    )
    cache_variant = (
        _cache_variant_for(profile, device, mps_fallback=mps_fallback) if cache is not None else ""
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
                "Discarding a cached query embedding whose dimensionality (%d) does "
                "not match the corpus matrix (%d); re-encoding the query.",
                candidate.shape[-1],
                embeddings.shape[1],
            )
            return None
        return candidate

    query_embedding: np.ndarray | None = None
    if cache_key is not None:
        hit = cache.get_many(cache_scope, profile.canonical_name, cache_revision, [cache_key])
        query_embedding = _validated_query_hit(hit.get(cache_key))

    if query_embedding is None:
        resolved_revision = _resolve_load_revision(model_name, revision)
        resolved_trust_remote_code = _resolve_trust_remote_code(model_name, trust_remote_code)
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
        )
        execution_device = _get_effective_model_device(model, resolved_device)

        if cache is not None:
            confirmed_revision = _get_loaded_model_commit_hash(model) or resolved_revision
            if confirmed_revision is None:
                logger.debug(
                    "Could not confirm the loaded model revision; keeping cache revision %s",
                    cache_revision,
                )
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
            encode_fn = model.encode
            if profile.family == "embeddinggemma" and hasattr(model, "encode_query"):
                encode_fn = model.encode_query

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
            )
            query_embedding = query_embeddings[0]

            if cache is not None and cache_key is not None and cache_revision is not None:
                cache.put_many(
                    cache_scope,
                    profile.canonical_name,
                    cache_revision,
                    [(cache_key, query_embedding)],
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
    :return: Up to ``top_k`` ``(unit, similarity)`` pairs at or above the threshold,
        sorted by descending similarity.
    """
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
        )


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
    :return: ``(embeddings, duplicates)``; both are empty when ``units`` is empty.
    """
    if not units:
        return np.array([]), []
    resolved_threshold = (
        threshold if threshold is not None else get_default_semantic_threshold(model_name)
    )

    embeddings = compute_embeddings(
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
    )
    duplicates = find_semantic_duplicates(
        units,
        embeddings,
        threshold=resolved_threshold,
        exclude_exact=exclude_pairs,
    )

    return embeddings, duplicates
