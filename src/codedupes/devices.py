"""Semantic inference device selection and accelerator memory management.

The module intentionally imports PyTorch lazily so traditional-only workflows do
not initialize an accelerator runtime. It also keeps PyTorch MPS and MLX memory
management separate: they use independent allocators even though both consume
Apple unified memory.
"""

from __future__ import annotations

import gc
import importlib
import logging
import math
import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from codedupes.constants import DEFAULT_SEMANTIC_DEVICE, SEMANTIC_DEVICE_CHOICES

logger = logging.getLogger(__name__)

SemanticDeviceRequest = Literal["auto", "cpu", "cuda", "mps"]
ResolvedSemanticDevice = Literal["cpu", "cuda", "mps"]


class DeviceConfigurationError(RuntimeError):
    """Raised when a requested semantic device cannot be configured."""


@dataclass(frozen=True)
class DeviceDiagnostics:
    """Runtime device capabilities without loading an embedding model."""

    requested: str
    resolved: str | None
    torch_available: bool
    cuda_available: bool
    mps_built: bool
    mps_available: bool
    mps_fallback_env: str
    mlx_loaded: bool
    mps_memory_bytes: dict[str, int] = field(default_factory=dict)
    error: str | None = None


def is_mlx_loaded() -> bool:
    """Return whether MLX has already been imported in this process.

    :return: ``True`` when ``mlx`` or ``mlx.core`` is present in ``sys.modules``.
    """
    return "mlx" in sys.modules or "mlx.core" in sys.modules


def normalize_semantic_device(device: str | None) -> SemanticDeviceRequest:
    """Normalize and validate a semantic device request.

    :param device: Requested device name, or ``None`` for automatic selection.
    :return: One of ``auto``, ``cpu``, ``cuda``, or ``mps``.
    :raises ValueError: If the device name is unsupported.
    """
    normalized = (device or DEFAULT_SEMANTIC_DEVICE).strip().lower()
    if normalized not in SEMANTIC_DEVICE_CHOICES:
        allowed = ", ".join(SEMANTIC_DEVICE_CHOICES)
        raise ValueError(f"Unsupported semantic device '{device}'. Expected one of: {allowed}")
    return cast(SemanticDeviceRequest, normalized)


def validate_mps_memory_fraction(fraction: float | None) -> float | None:
    """Validate an optional PyTorch MPS allocator limit.

    PyTorch accepts values from 0 through 2, but zero means unlimited allocation
    and can cause a system-wide out-of-memory failure. ``codedupes`` therefore
    accepts only the safer open/closed interval ``(0, 2]``.

    :param fraction: Requested allocator fraction.
    :return: Normalized float or ``None``.
    :raises ValueError: If the value is non-finite or outside ``(0, 2]``.
    """
    if fraction is None:
        return None

    value = float(fraction)
    if not math.isfinite(value) or not 0.0 < value <= 2.0:
        raise ValueError("mps_memory_fraction must be finite and in the interval (0.0, 2.0]")
    return value


def configure_mps_environment(
    requested_device: str | None,
    *,
    fallback: bool | None,
) -> None:
    """Configure MPS operator fallback before PyTorch is imported.

    ``fallback=None`` means automatic behavior: on a possible MPS host, enable
    CPU fallback only when the user has not already set
    ``PYTORCH_ENABLE_MPS_FALLBACK``. Explicit ``True``/``False`` values override
    the environment.

    :param requested_device: Semantic device request.
    :param fallback: Explicit fallback setting, or ``None`` for automatic.
    :return: ``None``.
    """
    normalized = normalize_semantic_device(requested_device)
    possible_mps_run = normalized == "mps" or (
        normalized == "auto" and platform.system() == "Darwin"
    )
    if not possible_mps_run:
        return

    env_name = "PYTORCH_ENABLE_MPS_FALLBACK"
    previous = os.environ.get(env_name)
    if fallback is None:
        os.environ.setdefault(env_name, "1")
    else:
        os.environ[env_name] = "1" if fallback else "0"

    current = os.environ.get(env_name)
    if "torch" in sys.modules and previous != current:
        logger.warning(
            "%s changed after PyTorch was imported; restart the process if the setting is not "
            "honored by the existing MPS runtime",
            env_name,
        )


def _load_torch() -> Any:
    """Import PyTorch lazily.

    :return: Imported ``torch`` module.
    :raises DeviceConfigurationError: If PyTorch is unavailable.
    """
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise DeviceConfigurationError(
            "PyTorch is required for semantic device selection. Install codedupes with its "
            "semantic dependencies."
        ) from exc


def _safe_bool_call(owner: Any, name: str) -> bool:
    """Call an optional boolean capability function safely.

    :param owner: Object that may expose the capability function.
    :param name: Attribute name of the zero-argument capability function.
    :return: Result coerced to ``bool``, or ``False`` when the attribute is missing,
        not callable, or raises.
    """
    callback = getattr(owner, name, None)
    if not callable(callback):
        return False
    try:
        return bool(callback())
    except Exception:
        logger.debug(
            "Device capability check %s.%s failed", type(owner).__name__, name, exc_info=True
        )
        return False


def _cuda_available(torch_module: Any) -> bool:
    """Return whether CUDA is available in a torch runtime.

    :param torch_module: Imported ``torch`` module or a compatible test double.
    :return: ``True`` when ``torch.cuda.is_available()`` reports availability.
    """
    cuda = getattr(torch_module, "cuda", None)
    return cuda is not None and _safe_bool_call(cuda, "is_available")


def _mps_capabilities(torch_module: Any) -> tuple[bool, bool]:
    """Return ``(built, available)`` for the PyTorch MPS backend.

    :param torch_module: Imported ``torch`` module or a compatible test double.
    :return: ``(built, available)`` flags, where ``built`` is inferred as ``True`` for
        runtimes that report availability without exposing ``is_built``.
    """
    backends = getattr(torch_module, "backends", None)
    backend_mps = getattr(backends, "mps", None) if backends is not None else None
    torch_mps = getattr(torch_module, "mps", None)

    built = backend_mps is not None and _safe_bool_call(backend_mps, "is_built")
    available = False
    if torch_mps is not None:
        available = _safe_bool_call(torch_mps, "is_available")
    if not available and backend_mps is not None:
        available = _safe_bool_call(backend_mps, "is_available")

    # Some test doubles and older runtimes expose is_available without is_built.
    if available and backend_mps is not None and not hasattr(backend_mps, "is_built"):
        built = True
    return built, available


def _resolve_semantic_device_with_torch(
    requested_device: str | None,
    torch_module: Any,
) -> ResolvedSemanticDevice:
    """Resolve a semantic device using an already-imported torch module.

    :param requested_device: Requested device name, or ``None`` for the default.
    :param torch_module: Imported ``torch`` module or a compatible test double.
    :return: Concrete device name; ``auto`` prefers CUDA, then MPS, then CPU.
    :raises DeviceConfigurationError: If an explicitly requested accelerator is
        unavailable or unsupported by this PyTorch build.
    """
    requested = normalize_semantic_device(requested_device)
    cuda_available = _cuda_available(torch_module)
    mps_built, mps_available = _mps_capabilities(torch_module)

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not cuda_available:
            raise DeviceConfigurationError(
                "CUDA was requested for semantic inference, but torch.cuda.is_available() is false. "
                "Use --device cpu or install a CUDA-enabled PyTorch build."
            )
        return "cuda"
    if requested == "mps":
        if not mps_built:
            raise DeviceConfigurationError(
                "MPS was requested for semantic inference, but this PyTorch build has no MPS "
                "support. Use an official macOS PyTorch wheel on Apple Silicon with macOS 14.0+ "
                "or select --device cpu."
            )
        if not mps_available:
            raise DeviceConfigurationError(
                "MPS was requested for semantic inference, but torch reports it unavailable. "
                "Verify Apple Silicon and macOS 14.0+ support, or select --device cpu."
            )
        return "mps"

    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def resolve_semantic_device(
    requested_device: str | None = DEFAULT_SEMANTIC_DEVICE,
) -> ResolvedSemanticDevice:
    """Resolve ``auto`` or validate an explicit semantic inference device.

    Automatic priority is CUDA, then Apple MPS, then CPU.

    :param requested_device: Requested device name.
    :return: Concrete device name.
    :raises DeviceConfigurationError: If an explicit accelerator is unavailable.
    """
    return _resolve_semantic_device_with_torch(requested_device, _load_torch())


def configure_mps_memory_fraction(
    resolved_device: str,
    fraction: float | None,
) -> None:
    """Apply an optional PyTorch MPS per-process allocator limit.

    :param resolved_device: Concrete semantic execution device.
    :param fraction: Allocator fraction in ``(0, 2]`` or ``None``.
    :return: ``None``.
    :raises DeviceConfigurationError: If used without MPS or unsupported by torch.
    """
    value = validate_mps_memory_fraction(fraction)
    if value is None:
        return
    if resolved_device != "mps":
        raise DeviceConfigurationError(
            "mps_memory_fraction was set, but semantic inference did not resolve to MPS. "
            "Remove the setting or select --device mps."
        )

    torch_module = _load_torch()
    mps = getattr(torch_module, "mps", None)
    setter = getattr(mps, "set_per_process_memory_fraction", None)
    if not callable(setter):
        raise DeviceConfigurationError(
            "This PyTorch build does not expose torch.mps.set_per_process_memory_fraction(), "
            "which usually means it was built without MPS support. Install a macOS PyTorch "
            "wheel with MPS support or remove the mps_memory_fraction setting."
        )

    if value > 1.0:
        logger.warning(
            "MPS memory fraction %.3f exceeds the device recommended working-set size; this can "
            "increase system-wide memory pressure",
            value,
        )
    try:
        setter(value)
    except Exception as exc:
        raise DeviceConfigurationError(
            f"Could not set the PyTorch MPS memory fraction to {value}: {exc}"
        ) from exc


def _safe_int_call(owner: Any, name: str) -> int | None:
    """Call an optional integer-returning runtime function safely.

    :param owner: Object that may expose the query function.
    :param name: Attribute name of the zero-argument query function.
    :return: Result coerced to ``int``, or ``None`` when the attribute is missing,
        not callable, or raises.
    """
    callback = getattr(owner, name, None)
    if not callable(callback):
        return None
    try:
        return int(callback())
    except Exception:
        logger.debug("Device memory query %s failed", name, exc_info=True)
        return None


def get_mps_memory_snapshot() -> dict[str, int]:
    """Return available PyTorch MPS allocator statistics in bytes.

    :return: A possibly empty mapping with ``current_allocated``,
        ``driver_allocated``, and ``recommended_max`` keys.
    """
    try:
        torch_module = _load_torch()
    except DeviceConfigurationError:
        return {}

    _, available = _mps_capabilities(torch_module)
    if not available:
        return {}

    mps = getattr(torch_module, "mps", None)
    if mps is None:
        return {}

    mapping = {
        "current_allocated": _safe_int_call(mps, "current_allocated_memory"),
        "driver_allocated": _safe_int_call(mps, "driver_allocated_memory"),
        "recommended_max": _safe_int_call(mps, "recommended_max_memory"),
    }
    return {key: value for key, value in mapping.items() if value is not None}


def format_bytes(value: int | None) -> str:
    """Format byte counts for compact diagnostics.

    :param value: Byte count, or ``None`` when the statistic is unavailable.
    :return: Binary-unit string such as ``1.5 GiB``, or ``unknown`` for ``None``.
    """
    if value is None:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} TiB"


def format_mps_memory_snapshot(snapshot: dict[str, int] | None = None) -> str:
    """Format an MPS memory snapshot for logs.

    :param snapshot: Pre-collected allocator statistics, or ``None`` to query the live
        PyTorch MPS allocator, defaults to ``None``.
    :return: Comma-separated ``tensor``/``driver``/``recommended`` summary, or a notice
        that MPS memory statistics are unavailable.
    """
    values = snapshot if snapshot is not None else get_mps_memory_snapshot()
    if not values:
        return "MPS memory statistics unavailable"
    return ", ".join(
        (
            f"tensor={format_bytes(values.get('current_allocated'))}",
            f"driver={format_bytes(values.get('driver_allocated'))}",
            f"recommended={format_bytes(values.get('recommended_max'))}",
        )
    )


def clear_device_cache(
    device: str | None,
    *,
    synchronize: bool = True,
    collect: bool = True,
) -> bool:
    """Best-effort release of unoccupied CUDA/MPS caching-allocator memory.

    Queued accelerator work is synchronized before Python garbage collection and
    allocator cache release. This function does not import or clear MLX: mutating
    another framework's allocator would be an unexpected process-wide side effect.

    :param device: ``cuda``, ``mps``, ``cpu``, or ``None`` to inspect both accelerators.
    :param synchronize: Wait for queued accelerator work first.
    :param collect: Run Python garbage collection before allocator cache release.
    :return: Whether at least one accelerator cache-clear call succeeded.
    """
    try:
        torch_module = _load_torch()
    except DeviceConfigurationError:
        if collect:
            gc.collect()
        return False

    normalized = device.lower() if isinstance(device, str) else None
    targets = ("cuda", "mps") if normalized is None else (normalized,)
    available_targets: list[tuple[str, Any]] = []

    for target in targets:
        if target == "cuda":
            backend = getattr(torch_module, "cuda", None)
            if backend is None or not _cuda_available(torch_module):
                continue
        elif target == "mps":
            backend = getattr(torch_module, "mps", None)
            _, available = _mps_capabilities(torch_module)
            if backend is None or not available:
                continue
        else:
            continue

        available_targets.append((target, backend))
        if not synchronize:
            continue

        sync_fn = getattr(backend, "synchronize", None)
        if callable(sync_fn):
            try:
                sync_fn()
            except Exception:
                # An OOM can leave a failed command in flight. Cache release is
                # still worth attempting after synchronization fails.
                logger.debug(
                    "%s synchronization failed during cache cleanup",
                    target.upper(),
                    exc_info=True,
                )

    if collect:
        gc.collect()

    cleared = False
    for target, backend in available_targets:
        empty_fn = getattr(backend, "empty_cache", None)
        if not callable(empty_fn):
            continue
        try:
            empty_fn()
            cleared = True
        except Exception:
            logger.debug(
                "%s cache cleanup failed",
                target.upper(),
                exc_info=True,
            )

    return cleared


def get_device_diagnostics(
    requested_device: str | None = DEFAULT_SEMANTIC_DEVICE,
) -> DeviceDiagnostics:
    """Inspect semantic device capabilities without loading a model.

    :param requested_device: Device request to resolve for the diagnostic.
    :return: Immutable capability and memory summary.
    """
    try:
        requested = normalize_semantic_device(requested_device)
    except ValueError as exc:
        return DeviceDiagnostics(
            requested=str(requested_device),
            resolved=None,
            torch_available=False,
            cuda_available=False,
            mps_built=False,
            mps_available=False,
            mps_fallback_env=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
            mlx_loaded=is_mlx_loaded(),
            error=str(exc),
        )

    try:
        torch_module = _load_torch()
    except DeviceConfigurationError as exc:
        return DeviceDiagnostics(
            requested=requested,
            resolved=None,
            torch_available=False,
            cuda_available=False,
            mps_built=False,
            mps_available=False,
            mps_fallback_env=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
            mlx_loaded=is_mlx_loaded(),
            error=str(exc),
        )

    cuda_available = _cuda_available(torch_module)
    mps_built, mps_available = _mps_capabilities(torch_module)
    try:
        resolved = _resolve_semantic_device_with_torch(requested, torch_module)
        error = None
    except DeviceConfigurationError as exc:
        resolved = None
        error = str(exc)

    memory = get_mps_memory_snapshot() if mps_available else {}
    return DeviceDiagnostics(
        requested=requested,
        resolved=resolved,
        torch_available=True,
        cuda_available=cuda_available,
        mps_built=mps_built,
        mps_available=mps_available,
        mps_fallback_env=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        mlx_loaded=is_mlx_loaded(),
        mps_memory_bytes=memory,
        error=error,
    )
