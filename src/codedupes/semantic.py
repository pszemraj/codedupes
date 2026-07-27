"""Semantic duplicate detection using embedding similarity."""

from __future__ import annotations

import ast
import importlib
import logging
import os
import sys
import threading
from collections.abc import Callable
from importlib import metadata as importlib_metadata
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
    SEMANTIC_TASK_CHOICES,
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
from codedupes.models import CodeUnit, DuplicatePair
from codedupes.pairs import ordered_pair_key
from codedupes.semantic_profiles import (
    get_default_search_threshold,
    get_default_semantic_threshold,
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
_DEFAULT_TRANSFORMERS_MIN = Version("4.51")
_DEFAULT_TRANSFORMERS_MAX_EXCLUSIVE = Version("5")
_DEFAULT_ST_MIN = Version("5")
_DEFAULT_ST_MAX_EXCLUSIVE = Version("6")

SemanticTask = Literal[
    "semantic-similarity",
    "code-retrieval",
    "retrieval",
    "question-answering",
    "fact-verification",
    "classification",
    "clustering",
]

# C2LLM task-specific instruction prefixes.
C2LLM_INSTRUCTIONS: dict[str, str] = {
    "code": "Represent this code for finding similar code: ",
    "query": "Represent this query for searching relevant code: ",
}

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
_DEEPSPEED_REQUIRED_MESSAGE = (
    "deepspeed is required for C2LLM models. "
    "Install with `pip install codedupes[gpu]` or `pip install deepspeed`."
)


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
        "deepspeed": _safe_package_version("deepspeed") or "missing",
    }


def _validate_version_range(
    package_name: str,
    min_version: Version,
    max_exclusive: Version,
) -> None:
    """Validate that a package version is within an inclusive/exclusive range.

    :param package_name: Package to validate.
    :param min_version: Lower bound inclusive.
    :param max_exclusive: Upper bound exclusive.
    :return: ``None``.
    :raises SemanticBackendError: If package version is invalid or incompatible.
    """
    raw = _safe_package_version(package_name)
    if raw is None:
        raise SemanticBackendError(
            f"{package_name} is not installed. Install compatible dependencies before semantic runs."
        )

    try:
        parsed = Version(raw)
    except InvalidVersion as exc:
        raise SemanticBackendError(f"Could not parse {package_name} version: {raw}") from exc

    if not (min_version <= parsed < max_exclusive):
        raise SemanticBackendError(
            f"Incompatible {package_name} version {raw} for C2LLM models. "
            f"Supported range is >={min_version},<{max_exclusive}. "
            "Run: pip install 'transformers>=4.51,<5' 'sentence-transformers>=5,<6'."
        )


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


def _check_c2llm_model_compatibility(model_name: str) -> None:
    """Check dependency compatibility for C2LLM-family models."""
    profile = resolve_model_profile(model_name)
    if profile.family != "c2llm":
        return

    _validate_version_range(
        "transformers",
        _DEFAULT_TRANSFORMERS_MIN,
        _DEFAULT_TRANSFORMERS_MAX_EXCLUSIVE,
    )
    _validate_version_range(
        "sentence-transformers",
        _DEFAULT_ST_MIN,
        _DEFAULT_ST_MAX_EXCLUSIVE,
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
        "deepspeed",
        "flash_attn",
        "c2llm",
        "embeddinggemma",
        "auto_map",
        "tokenizer",
        "modeling_c2llm",
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
    profile = resolve_model_profile(model_name)
    hints: list[str] = []
    if profile.requires_deepspeed:
        hints.append(
            "install C2LLM-compatible deps via "
            '\'pip install "transformers>=4.51,<5" "sentence-transformers>=5,<6"\'.'
        )
    hints.append("or run traditional-only mode with '--traditional-only'.")

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


def _check_semantic_dependencies(model_name: str) -> None:
    """Validate required runtime dependencies before model loading."""
    _require_dependency("sentence_transformers", "pip install codedupes")
    _require_dependency("transformers", "pip install codedupes")
    _require_dependency("torch", "pip install codedupes")

    profile = resolve_model_profile(model_name)
    if profile.requires_deepspeed:
        _require_dependency(
            "deepspeed",
            "pip install codedupes[gpu] or pip install deepspeed",
        )

    _validate_torch_runtime()
    _check_c2llm_model_compatibility(model_name)


def _raise_missing_deepspeed(exc: ModuleNotFoundError) -> None:
    """Raise a stable deepspeed dependency error for C2LLM-family loads."""
    raise ModuleNotFoundError(_DEEPSPEED_REQUIRED_MESSAGE) from exc


def _prepare_semantic_device(
    device: str | None,
    *,
    mps_fallback: bool | None,
    mps_memory_fraction: float | None,
) -> str:
    """Configure and resolve one semantic execution device.

    :param device: Requested device name.
    :param mps_fallback: MPS unsupported-op fallback behavior.
    :param mps_memory_fraction: Optional MPS allocator limit, ignored when the
        request resolves to a non-MPS device.
    :return: Concrete device name.
    :raises SemanticBackendError: If device configuration fails.
    """
    _configure_semantic_runtime_env(device, mps_fallback=mps_fallback)
    try:
        resolved_device = resolve_semantic_device(device)
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


def _resolve_c2llm_torch_dtype(device: str) -> Any:
    """Choose a conservative C2LLM dtype for the selected device.

    CUDA uses bfloat16 only when the hardware reports support. CPU preserves the
    existing bfloat16 profile behavior. MPS uses float32 because remote C2LLM
    code and several Metal operators remain less predictable in reduced precision.

    :param device: Concrete execution device.
    :return: Suggested dtype object for Torch models, or ``None``.
    """
    try:
        import torch
    except ModuleNotFoundError:
        return None

    if device == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return None
    if device == "mps":
        return torch.float32
    return torch.bfloat16


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


def _patch_c2llm_runtime_compat() -> None:
    """Patch known C2LLM remote-code symbol gaps for newer runtimes."""
    import builtins

    if not hasattr(builtins, "is_torch_npu_available"):
        builtins.is_torch_npu_available = lambda: False


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

    profile = resolve_model_profile(model_name)
    resolved_model_name = profile.canonical_name
    resolved_revision = _resolve_model_revision(model_name, revision)
    resolved_trust_remote_code = _resolve_trust_remote_code(model_name, trust_remote_code)

    # Configure MPS environment variables before dependency checks import torch
    # through sentence-transformers/transformers.
    _configure_semantic_runtime_env(device, mps_fallback=mps_fallback)
    _check_semantic_dependencies(resolved_model_name)
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
        if profile.family == "c2llm":
            _patch_c2llm_runtime_compat()

        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            if exc.name == "sentence_transformers":
                raise ModuleNotFoundError(
                    "sentence-transformers is not installed. Install it with `pip install codedupes`."
                ) from exc
            if exc.name == "deepspeed":
                _raise_missing_deepspeed(exc)
            raise

        st_kwargs: dict[str, object] = {
            "trust_remote_code": resolved_trust_remote_code,
            "device": resolved_device,
        }
        if resolved_revision is not None:
            st_kwargs["revision"] = resolved_revision

        model_kwargs: dict[str, object] = {}
        tokenizer_kwargs: dict[str, object] = {}
        config_kwargs: dict[str, object] = {}

        if profile.left_padding:
            tokenizer_kwargs["padding_side"] = "left"
        if profile.low_cpu_mem_usage:
            model_kwargs["low_cpu_mem_usage"] = True

        if profile.family == "c2llm":
            selected_dtype = _resolve_c2llm_torch_dtype(resolved_device)
            if selected_dtype is not None:
                model_kwargs["torch_dtype"] = selected_dtype
                logger.info("Using C2LLM torch dtype on %s: %s", resolved_device, selected_dtype)
        elif profile.family == "embeddinggemma":
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
        except ModuleNotFoundError as exc:
            if exc.name == "deepspeed":
                _raise_missing_deepspeed(exc)
            raise
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
                except ModuleNotFoundError as retry_exc:
                    if retry_exc.name == "deepspeed":
                        _raise_missing_deepspeed(retry_exc)
                    raise
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


def _normalize_semantic_task(
    semantic_task: str | None,
    *,
    default_task: SemanticTask,
) -> SemanticTask:
    """Validate and normalize semantic task names.

    :param semantic_task: Candidate task value.
    :param default_task: Fallback task when no value is provided.
    :return: Normalized task enum.
    """
    if semantic_task is None:
        return default_task

    normalized = semantic_task.strip().lower()
    if normalized not in SEMANTIC_TASK_CHOICES:
        allowed = ", ".join(SEMANTIC_TASK_CHOICES)
        raise ValueError(f"Unknown semantic task '{semantic_task}'. Expected one of: {allowed}")
    return normalized  # type: ignore[return-value]


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
    model_name: str,
    mode: Literal["code", "query"],
    semantic_task: SemanticTask,
) -> str:
    """Get default instruction prefix for model/task/mode.

    :param model_name: Model identifier.
    :param mode: Input mode.
    :param semantic_task: Normalized task.
    :return: Instruction prefix.
    """
    profile = resolve_model_profile(model_name)

    if profile.family == "c2llm":
        if mode == "query":
            return C2LLM_INSTRUCTIONS["query"]
        return C2LLM_INSTRUCTIONS["code"]

    if profile.family == "embeddinggemma":
        return _get_embeddinggemma_prefix(semantic_task, mode)

    return ""


def _resolve_instruction_prefix(
    model_name: str,
    mode: Literal["code", "query"],
    instruction_prefix: str | None,
    *,
    semantic_task: SemanticTask,
) -> str:
    """Resolve instruction prefix override for embedding inputs.

    :param model_name: Model identifier.
    :param mode: Input mode.
    :param instruction_prefix: Optional override.
    :param semantic_task: Resolved task.
    :return: Instruction prefix.
    """
    if instruction_prefix is not None:
        return instruction_prefix
    return _get_instruction(model_name, mode, semantic_task)


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
    resolved_task = _normalize_semantic_task(
        semantic_task,
        default_task=task_default,  # type: ignore[arg-type]
    )
    instruction = _resolve_instruction_prefix(
        model_name,
        mode,
        instruction_prefix,
        semantic_task=resolved_task,
    )
    return f"{instruction}{source}"


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
        if oom_device is None or oom_error is None:
            raise RuntimeError("unreachable: OOM state lost during retry handling")

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
) -> np.ndarray:
    """Compute normalized NumPy embeddings for all code units.

    Embeddings are converted to NumPy immediately, keeping pairwise similarity
    computation on CPU and avoiding long-lived Metal tensors beyond model weights.

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
    :return: Normalized embedding matrix row-aligned with ``units``.
    :raises ValueError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    resolved_revision = _resolve_model_revision(model_name, revision)
    resolved_trust_remote_code = _resolve_trust_remote_code(model_name, trust_remote_code)
    profile = resolve_model_profile(model_name)
    resolved_task = _normalize_semantic_task(
        semantic_task,
        default_task=DEFAULT_CHECK_SEMANTIC_TASK,
    )
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

    texts = []
    for unit in units:
        prepared = prepare_code_for_embedding(
            unit,
            model_name=model_name,
            instruction_prefix=instruction_prefix,
            semantic_task=resolved_task,
        )
        texts.append(_truncate_code_if_needed(prepared, unit.qualified_name, model))

    logger.info(
        "Computing embeddings for %d code units on %s",
        len(texts),
        execution_device,
    )
    encode_fn = model.encode
    if profile.family == "embeddinggemma" and hasattr(model, "encode_document"):
        encode_fn = model.encode_document

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
) -> list[tuple[CodeUnit, float]]:
    """Find code units most similar to a natural-language query.

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
    :return: Up to ``top_k`` ``(unit, similarity)`` pairs at or above the threshold,
        sorted by descending similarity.
    """
    resolved_revision = _resolve_model_revision(model_name, revision)
    resolved_trust_remote_code = _resolve_trust_remote_code(model_name, trust_remote_code)
    profile = resolve_model_profile(model_name)
    resolved_threshold = (
        threshold if threshold is not None else get_default_search_threshold(model_name)
    )
    resolved_task = _normalize_semantic_task(
        semantic_task,
        default_task=DEFAULT_SEARCH_SEMANTIC_TASK,
    )
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

    instruction = _resolve_instruction_prefix(
        model_name,
        "query",
        instruction_prefix,
        semantic_task=resolved_task,
    )
    query_text = f"{instruction}{query}"

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
    )
    duplicates = find_semantic_duplicates(
        units,
        embeddings,
        threshold=resolved_threshold,
        exclude_exact=exclude_pairs,
    )

    return embeddings, duplicates
