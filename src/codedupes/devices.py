"""Semantic inference device selection and accelerator memory management.

The module intentionally imports PyTorch lazily so traditional-only workflows do
not initialize an accelerator runtime. It also keeps PyTorch MPS and MLX memory
management separate: they use independent allocators even though both consume
Apple unified memory.
"""

from __future__ import annotations

import contextlib
import gc
import importlib
import json
import logging
import math
import os
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from codedupes.constants import DEFAULT_SEMANTIC_DEVICE, SEMANTIC_DEVICE_CHOICES

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

SemanticDeviceRequest = Literal["auto", "cpu", "cuda", "mps"]
ResolvedSemanticDevice = Literal["cpu", "cuda", "mps"]

_PYTORCH_DEFAULT_MPS_HIGH_WATERMARK_RATIO = 1.7
_mps_memory_fraction_lock = threading.Lock()
_mps_memory_fraction_managed = False
_mps_memory_fraction_restore_value: float | None = None


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
    cpu_name: str | None
    cpu_architecture: str | None
    cpu_bf16_isa: bool
    cpu_mkldnn_available: bool
    cpu_bf16_native: bool
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


def _resolve_mps_memory_fraction_restore_value() -> float:
    """Resolve the allocator limit restored after a codedupes override.

    PyTorch initializes the MPS high-watermark ratio from
    ``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` and otherwise uses ``1.7``. Its public
    setter has no matching getter, so this environment/default value is the
    only recoverable baseline before codedupes applies its first custom cap.

    :return: PyTorch environment override or default high-watermark ratio.
    """
    raw = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    if raw is None:
        return _PYTORCH_DEFAULT_MPS_HIGH_WATERMARK_RATIO

    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if math.isfinite(value) and 0.0 <= value <= 2.0:
        return value

    logger.warning(
        "Ignoring invalid PYTORCH_MPS_HIGH_WATERMARK_RATIO=%r while resolving the "
        "allocator reset target; restoring PyTorch's default %.1f",
        raw,
        _PYTORCH_DEFAULT_MPS_HIGH_WATERMARK_RATIO,
    )
    return _PYTORCH_DEFAULT_MPS_HIGH_WATERMARK_RATIO


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
    possible_mps_run = normalized == "mps" or (normalized == "auto" and sys.platform == "darwin")
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
            f"{env_name} changed after PyTorch was imported; restart the process if the setting "
            "is not honored by the existing MPS runtime"
        )


def describe_mps_fallback_env(value: str | None) -> str:
    """Describe how PyTorch interprets a ``PYTORCH_ENABLE_MPS_FALLBACK`` value.

    PyTorch enables the operator CPU fallback whenever the variable is set to
    anything except the literal string ``0`` - ``false``, ``no``, and even the
    empty string all enable it - so diagnostics must not present the raw value
    as the decision.

    :param value: Raw environment value, or ``None``/``"unset"`` when absent.
    :return: ``"enabled"`` or ``"disabled"``.
    """
    if value is None or value in {"unset", "0"}:
        return "disabled"
    return "enabled"


def _load_torch() -> Any:
    """Import PyTorch lazily.

    A broken install (wrong-arch wheel, missing native dependency) must map to
    the same recoverable error as an absent one: callers such as ``codedupes
    info`` report "torch unavailable" diagnostics instead of dying on the
    import traceback.

    :return: Imported ``torch`` module.
    :raises DeviceConfigurationError: If PyTorch is missing or fails to import.
    """
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise DeviceConfigurationError(
                "PyTorch is required for semantic device selection. Install codedupes with its "
                "semantic dependencies."
            ) from exc
        if (exc.name or "").startswith("torch."):
            raise DeviceConfigurationError(
                f"PyTorch is installed but incomplete ({exc}). Reinstall a build matching "
                "this platform and architecture."
            ) from exc
        raise DeviceConfigurationError(
            f"PyTorch import failed on a missing dependency ({exc}). Reinstall codedupes "
            "with its semantic dependencies."
        ) from exc
    except (ImportError, OSError, RuntimeError) as exc:
        raise DeviceConfigurationError(
            f"PyTorch is installed but failed to import ({exc}). Reinstall a build matching "
            "this platform and architecture."
        ) from exc


def _safe_call(owner: Any, name: str, coerce: Callable[[Any], _T], default: _T) -> _T:
    """Call an optional zero-argument runtime query function safely.

    :param owner: Object that may expose the query function.
    :param name: Attribute name of the zero-argument query function.
    :param coerce: Conversion applied to the call result.
    :param default: Returned when the attribute is missing, not callable, or raises.
    :return: Coerced result, or ``default`` on any failure.
    """
    callback = getattr(owner, name, None)
    if not callable(callback):
        return default
    try:
        return coerce(callback())
    except Exception:
        logger.debug(f"Device runtime query {type(owner).__name__}.{name} failed", exc_info=True)
        return default


def _cuda_available(torch_module: Any) -> bool:
    """Return whether CUDA is available in a torch runtime.

    :param torch_module: Imported ``torch`` module or a compatible test double.
    :return: ``True`` when ``torch.cuda.is_available()`` reports availability.
    """
    cuda = getattr(torch_module, "cuda", None)
    return cuda is not None and _safe_call(cuda, "is_available", bool, False)


def _mps_capabilities(torch_module: Any) -> tuple[bool, bool]:
    """Return ``(built, available)`` for the PyTorch MPS backend.

    :param torch_module: Imported ``torch`` module or a compatible test double.
    :return: ``(built, available)`` flags, where ``built`` is inferred as ``True`` for
        runtimes that report availability without exposing ``is_built``.
    """
    backends = getattr(torch_module, "backends", None)
    backend_mps = getattr(backends, "mps", None) if backends is not None else None
    torch_mps = getattr(torch_module, "mps", None)

    built = backend_mps is not None and _safe_call(backend_mps, "is_built", bool, False)
    available = False
    if torch_mps is not None:
        available = _safe_call(torch_mps, "is_available", bool, False)
    if not available and backend_mps is not None:
        available = _safe_call(backend_mps, "is_available", bool, False)

    # Some test doubles and older runtimes expose is_available without is_built.
    if available and backend_mps is not None and not hasattr(backend_mps, "is_built"):
        built = True
    return built, available


def _read_cpu_capabilities(torch_module: Any) -> Mapping[str, Any]:
    """Read torch's live CPU capability probe, defensively.

    ``torch.cpu.get_capabilities()`` returns a read-only ``mappingproxy``, not
    a plain ``dict``, so callers must check against :class:`~collections.abc.Mapping`.

    :param torch_module: Imported ``torch`` module or a compatible test double.
    :return: Capability mapping from ``torch.cpu.get_capabilities()``, or an
        empty mapping when the probe is unavailable or fails.
    """
    cpu = getattr(torch_module, "cpu", None)
    caps_fn = getattr(cpu, "get_capabilities", None)
    if not callable(caps_fn):
        return {}
    try:
        caps = caps_fn()
    except Exception:
        logger.debug("torch.cpu.get_capabilities() failed", exc_info=True)
        return {}
    return caps if isinstance(caps, Mapping) else {}


def _cpu_bf16_isa_present(caps: Mapping[str, Any], architecture: str) -> bool:
    """Return whether the CPU ISA itself supports bfloat16, before any backend check.

    ARM (``bf16``) and x86 (``amx_bf16``/``avx512_bf16``) report native bf16 support
    under different keys; plain ``avx512`` is deliberately excluded, since AVX-512
    alone does not imply a bf16 execution unit.

    :param caps: Capability mapping from ``torch.cpu.get_capabilities()``.
    :param architecture: Reported ``architecture`` value, for example ``"arm64"``.
    :return: ``True`` when the reported ISA includes native bf16.
    """
    normalized_arch = architecture.lower()
    if "arm" in normalized_arch or "aarch64" in normalized_arch:
        return bool(caps.get("bf16"))
    return bool(caps.get("amx_bf16")) or bool(caps.get("avx512_bf16"))


def cpu_bf16_capability(torch_module: Any) -> bool:
    """Probe whether this CPU can execute native, fast bfloat16 GEMM.

    A native bf16 ISA alone is not enough: without a backend (oneDNN/mkldnn)
    able to exploit it, bf16 GEMM falls back to a slow reference path. Measured
    on an Apple M5 (torch 2.13.0, macOS arm64 wheel): ``get_capabilities()``
    reports ``bf16: true`` and ``architecture: "arm64"``, but
    ``torch.backends.mkldnn.is_available()`` is ``False`` (no oneDNN backend),
    and a 1024x1024x1024 bf16 matmul measured 1015 ms versus 1.207 ms for
    float32 - 841x slower with no backend to exploit the ISA. The gate is
    therefore native ISA *and* a usable GEMM backend. This is a pure, stateless
    probe; use :func:`resolve_cpu_bf16_native` for the record-backed accessor
    that avoids importing torch on a warm path.

    ``torch.backends.cpu.get_cpu_capability()`` is deliberately never consulted
    here: it reports the build-tier baseline (for example ``"DEFAULT"``) the
    wheel was compiled with, not what the running CPU can actually execute.

    :param torch_module: Imported ``torch`` module or a compatible test double.
    :return: ``True`` iff native bf16 ISA is present and a GEMM backend can use
        it; ``False`` on any probe failure (float32 is always safe).
    """
    try:
        caps = _read_cpu_capabilities(torch_module)
        architecture = str(caps.get("architecture", ""))
        has_isa = _cpu_bf16_isa_present(caps, architecture)
        backends = getattr(torch_module, "backends", None)
        mkldnn = getattr(backends, "mkldnn", None) if backends is not None else None
        has_mkldnn = _safe_call(mkldnn, "is_available", bool, False)
        return has_isa and has_mkldnn
    except Exception:
        logger.debug("CPU bf16 capability probe failed", exc_info=True)
        return False


_MACHINE_RECORD_FILENAME = "machine.json"


def _resolve_machine_record_path() -> Path | None:
    """Resolve the on-disk machine-capability record path.

    Imports :mod:`codedupes.embedding_cache` lazily (it has no dependency back
    on this module, so there is no import cycle) and only when a record lookup
    is actually needed. The embedding cache's own ``CODEDUPES_NO_CACHE`` kill
    switch is honored through its :func:`~codedupes.embedding_cache.is_cache_disabled`
    so the two can never drift apart.

    :return: Record path, or ``None`` when caching is disabled or the cache
        root cannot be resolved.
    """
    try:
        from codedupes.embedding_cache import is_cache_disabled, resolve_cache_dir

        if is_cache_disabled():
            return None
        return resolve_cache_dir() / _MACHINE_RECORD_FILENAME
    except Exception:
        logger.debug("Could not resolve the CPU capability record path", exc_info=True)
        return None


def _read_machine_record(record_path: Path) -> bool | None:
    """Read a trustworthy CPU bf16 verdict from an on-disk machine record.

    :param record_path: Candidate record path.
    :return: Recorded verdict, or ``None`` when missing, unreadable, corrupt,
        or stamped with a different installed torch version.
    """
    try:
        current_torch_version = importlib_metadata.version("torch")
    except importlib_metadata.PackageNotFoundError:
        return None
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("torch") != current_torch_version:
        return None
    verdict = payload.get("cpu_bf16_native")
    return verdict if isinstance(verdict, bool) else None


def _persist_machine_record(record_path: Path, verdict: bool) -> None:
    """Best-effort atomic write of the CPU bf16 capability record.

    :param record_path: Destination record path.
    :param verdict: Freshly probed capability verdict to persist.
    :return: ``None``.
    """
    try:
        torch_version = importlib_metadata.version("torch")
    except importlib_metadata.PackageNotFoundError:
        return
    tmp_path = record_path.with_name(f"{record_path.name}.{os.getpid()}.tmp")
    try:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"torch": torch_version, "cpu_bf16_native": verdict})
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, record_path)
    except OSError:
        logger.debug("Could not persist the CPU bf16 capability record", exc_info=True)
    finally:
        with contextlib.suppress(OSError):
            if tmp_path.exists():
                tmp_path.unlink()


def resolve_cpu_bf16_native(*, persist: bool = True) -> bool:
    """Return whether this machine's CPU can execute native, fast bfloat16 GEMM.

    Backed by an on-disk record (``<cache_root>/machine.json``) so repeated,
    warm-path cache-key derivation never imports torch: a machine's CPU
    capabilities cannot change under one installed torch build, so a record
    stamped with the currently installed torch version is trusted without
    re-probing. A missing, unreadable, corrupt, or stale-version record falls
    back to a live :func:`cpu_bf16_capability` probe, which is then persisted
    best-effort for the next call. ``CODEDUPES_NO_CACHE`` disables both the
    read and the write, matching the embedding cache's kill switch; so does
    ``persist=False``, mirroring how callers that disabled the on-disk
    embedding cache for one call (see ``persist_manifest`` on the local-model
    digest manifest) keep that call free of unrelated cache-directory writes.

    :param persist: Whether the on-disk record may be read from and written to,
        defaults to ``True``.
    :return: ``True`` iff native bf16 ISA is present and a GEMM backend can use it.
    """
    record_path = _resolve_machine_record_path() if persist else None

    if record_path is not None:
        cached_verdict = _read_machine_record(record_path)
        if cached_verdict is not None:
            return cached_verdict

    verdict = cpu_bf16_capability(_load_torch())
    if record_path is not None:
        _persist_machine_record(record_path, verdict)
    return verdict


_CPU_BF16_OPT_IN_ENV = "CODEDUPES_CPU_BF16"


def cpu_bf16_opted_in() -> bool:
    """Check the experimental ``CODEDUPES_CPU_BF16=1`` CPU bfloat16 inference opt-in.

    :return: ``True`` iff the environment variable is set to the literal ``1``.
    """
    return os.environ.get(_CPU_BF16_OPT_IN_ENV, "").strip() == "1"


def resolve_cpu_bf16_inference(*, persist: bool = True) -> bool:
    """Decide whether CPU model inference may run in bfloat16.

    Requires both the experimental ``CODEDUPES_CPU_BF16=1`` opt-in and this
    machine's capability gate (:func:`resolve_cpu_bf16_native`). The opt-in
    exists because the positive path is unvalidated: the gate proves the CPU
    can execute bf16 GEMM fast, not that the float32-calibrated duplicate and
    search thresholds survive bfloat16's numeric shift on the built-in
    models. Until a gate-passing machine validates speed and decision parity,
    automatic CPU inference stays float32. The opt-in is checked first so a
    non-opted-in run never reads the capability record or imports torch.

    :param persist: Whether the on-disk capability record may be read from and
        written to, defaults to ``True``.
    :return: ``True`` iff opted in and the capability gate passes.
    """
    if not cpu_bf16_opted_in():
        return False
    return resolve_cpu_bf16_native(persist=persist)


def _mps_backend_built_without_import() -> bool:
    """Check MPS build support from an already-imported torch, without importing it.

    :return: ``True`` when torch is already imported and reports MPS support built.
    """
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return False
    backends = getattr(torch_module, "backends", None)
    backend_mps = getattr(backends, "mps", None) if backends is not None else None
    if backend_mps is None:
        return False
    return _safe_call(backend_mps, "is_built", bool, False)


def restore_mps_memory_fraction_if_managed() -> None:
    """Restore the MPS allocator baseline once resolution moves away from MPS.

    A custom ``--mps-memory-fraction`` applied for an earlier MPS run must not
    stay stuck for the rest of a long-lived process once a later call resolves
    to CPU or CUDA. This never imports torch merely to check: it is a no-op
    when nothing is currently managed, and a no-op when torch is not already
    imported or does not report MPS built, so an unrelated CPU/CUDA run can
    never fail because this build lacks MPS support.

    :return: ``None``.
    """
    if not _mps_memory_fraction_managed:
        return
    if not _mps_backend_built_without_import():
        return
    configure_mps_memory_fraction("mps", None)


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
    """Apply or restore the PyTorch MPS per-process allocator limit.

    :param resolved_device: Concrete semantic execution device.
    :param fraction: Allocator fraction in ``(0, 2]`` or ``None``.
    :return: ``None``.
    :raises DeviceConfigurationError: If used without MPS or unsupported by torch.
    """
    global _mps_memory_fraction_managed, _mps_memory_fraction_restore_value

    value = validate_mps_memory_fraction(fraction)
    if value is None and resolved_device != "mps":
        return
    if value is not None and resolved_device != "mps":
        raise DeviceConfigurationError(
            "mps_memory_fraction was set, but semantic inference did not resolve to MPS. "
            "Remove the setting or select --device mps."
        )

    if value is not None and value > 1.0:
        logger.warning(
            f"MPS memory fraction {value:.3f} exceeds the device recommended working-set size; "
            "this can increase system-wide memory pressure"
        )

    with _mps_memory_fraction_lock:
        if value is None and not _mps_memory_fraction_managed:
            return

        torch_module = _load_torch()
        mps = getattr(torch_module, "mps", None)
        setter = getattr(mps, "set_per_process_memory_fraction", None)
        if not callable(setter):
            raise DeviceConfigurationError(
                "This PyTorch build does not expose "
                "torch.mps.set_per_process_memory_fraction(), which usually means it was "
                "built without MPS support. Install a macOS PyTorch wheel with MPS support "
                "or remove the mps_memory_fraction setting."
            )

        restore_value = _mps_memory_fraction_restore_value
        if not _mps_memory_fraction_managed:
            restore_value = _resolve_mps_memory_fraction_restore_value()
        target = restore_value if value is None else value
        if target is None:
            raise DeviceConfigurationError("Could not resolve the default MPS memory fraction")

        try:
            setter(target)
        except Exception as exc:
            action = "restore" if value is None else "set"
            raise DeviceConfigurationError(
                f"Could not {action} the PyTorch MPS memory fraction to {target}: {exc}"
            ) from exc

        if value is None:
            _mps_memory_fraction_managed = False
            _mps_memory_fraction_restore_value = None
        else:
            _mps_memory_fraction_managed = True
            _mps_memory_fraction_restore_value = restore_value


def _read_mps_memory_snapshot(torch_module: Any) -> dict[str, int]:
    """Read allocator statistics from an available PyTorch MPS runtime.

    :param torch_module: Imported PyTorch module with available MPS support.
    :return: Available MPS allocator statistics in bytes.
    """
    mps = getattr(torch_module, "mps", None)
    if mps is None:
        return {}

    mapping: dict[str, int | None] = {
        "current_allocated": _safe_call(mps, "current_allocated_memory", int, None),
        "driver_allocated": _safe_call(mps, "driver_allocated_memory", int, None),
        "recommended_max": _safe_call(mps, "recommended_max_memory", int, None),
    }
    return {key: value for key, value in mapping.items() if value is not None}


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
    return _read_mps_memory_snapshot(torch_module)


def format_bytes(value: int | None) -> str:
    """Format byte counts for compact diagnostics.

    :param value: Byte count, or ``None`` when the statistic is unavailable.
    :return: Binary-unit string such as ``1.5 GiB``, or ``unknown`` for ``None``.
    """
    if value is None:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units[:-1]:
        if abs(amount) < 1024.0:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} {units[-1]}"


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
                    f"{target.upper()} synchronization failed during cache cleanup", exc_info=True
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
            logger.debug(f"{target.upper()} cache cleanup failed", exc_info=True)

    return cleared


def get_device_diagnostics(
    requested_device: str | None = DEFAULT_SEMANTIC_DEVICE,
) -> DeviceDiagnostics:
    """Inspect semantic device capabilities without loading a model.

    :param requested_device: Device request to resolve for the diagnostic.
    :return: Immutable capability and memory summary.
    """

    def torchless_diagnostics(requested: str, error: Exception) -> DeviceDiagnostics:
        """Build the all-unavailable diagnostics for a pre-torch failure.

        :param requested: Requested device string as far as it was normalized.
        :param error: Failure that prevented device inspection.
        :return: Diagnostics with every capability reported unavailable.
        """
        return DeviceDiagnostics(
            requested=requested,
            resolved=None,
            torch_available=False,
            cuda_available=False,
            mps_built=False,
            mps_available=False,
            mps_fallback_env=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
            mlx_loaded=is_mlx_loaded(),
            cpu_name=None,
            cpu_architecture=None,
            cpu_bf16_isa=False,
            cpu_mkldnn_available=False,
            cpu_bf16_native=False,
            error=str(error),
        )

    try:
        requested = normalize_semantic_device(requested_device)
    except ValueError as exc:
        return torchless_diagnostics(str(requested_device), exc)

    try:
        torch_module = _load_torch()
    except DeviceConfigurationError as exc:
        return torchless_diagnostics(requested, exc)

    cuda_available = _cuda_available(torch_module)
    mps_built, mps_available = _mps_capabilities(torch_module)
    cpu_caps = _read_cpu_capabilities(torch_module)
    cpu_name = cpu_caps.get("cpu_name")
    cpu_architecture_raw = cpu_caps.get("architecture")
    cpu_architecture = cpu_architecture_raw if isinstance(cpu_architecture_raw, str) else None
    cpu_bf16_isa = _cpu_bf16_isa_present(cpu_caps, cpu_architecture or "")
    cpu_backends = getattr(torch_module, "backends", None)
    cpu_mkldnn = getattr(cpu_backends, "mkldnn", None) if cpu_backends is not None else None
    cpu_mkldnn_available = _safe_call(cpu_mkldnn, "is_available", bool, False)
    try:
        resolved = _resolve_semantic_device_with_torch(requested, torch_module)
        error = None
    except DeviceConfigurationError as exc:
        resolved = None
        error = str(exc)

    memory = _read_mps_memory_snapshot(torch_module) if mps_available else {}
    return DeviceDiagnostics(
        requested=requested,
        resolved=resolved,
        torch_available=True,
        cuda_available=cuda_available,
        mps_built=mps_built,
        mps_available=mps_available,
        mps_fallback_env=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        mlx_loaded=is_mlx_loaded(),
        cpu_name=cpu_name if isinstance(cpu_name, str) else None,
        cpu_architecture=cpu_architecture,
        cpu_bf16_isa=cpu_bf16_isa,
        cpu_mkldnn_available=cpu_mkldnn_available,
        cpu_bf16_native=cpu_bf16_isa and cpu_mkldnn_available,
        mps_memory_bytes=memory,
        error=error,
    )
