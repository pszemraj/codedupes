from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import pytest

from codedupes import devices
from codedupes.devices import DeviceConfigurationError


class _FakeCuda:
    def __init__(self, *, available: bool, events: list[str] | None = None) -> None:
        self._available = available
        self._events = events

    def is_available(self) -> bool:
        return self._available

    def synchronize(self) -> None:
        if self._events is not None:
            self._events.append("cuda.sync")

    def empty_cache(self) -> None:
        if self._events is not None:
            self._events.append("cuda.empty")


class _FakeMps:
    def __init__(
        self,
        *,
        available: bool,
        events: list[str] | None = None,
    ) -> None:
        self._available = available
        self._events = events
        self.fractions: list[float] = []

    def is_available(self) -> bool:
        return self._available

    def synchronize(self) -> None:
        if self._events is not None:
            self._events.append("mps.sync")

    def empty_cache(self) -> None:
        if self._events is not None:
            self._events.append("mps.empty")

    def set_per_process_memory_fraction(self, value: float) -> None:
        self.fractions.append(value)

    def current_allocated_memory(self) -> int:
        return 128

    def driver_allocated_memory(self) -> int:
        return 256

    def recommended_max_memory(self) -> int:
        return 1024


class _FakeBackendMps:
    def __init__(self, *, built: bool, available: bool) -> None:
        self._built = built
        self._available = available

    def is_built(self) -> bool:
        return self._built

    def is_available(self) -> bool:
        return self._available


def _fake_torch(
    *,
    cuda_available: bool = False,
    mps_built: bool = False,
    mps_available: bool = False,
    events: list[str] | None = None,
):
    mps = _FakeMps(available=mps_available, events=events)
    return SimpleNamespace(
        cuda=_FakeCuda(available=cuda_available, events=events),
        mps=mps,
        backends=SimpleNamespace(mps=_FakeBackendMps(built=mps_built, available=mps_available)),
    )


def test_normalize_semantic_device() -> None:
    assert devices.normalize_semantic_device(None) == "auto"
    assert devices.normalize_semantic_device(" MPS ") == "mps"

    with pytest.raises(ValueError, match="Unsupported semantic device"):
        devices.normalize_semantic_device("metal")


@pytest.mark.parametrize("value", [0.0, -0.1, 2.1, math.inf, math.nan])
def test_validate_mps_memory_fraction_rejects_unsafe_values(value: float) -> None:
    with pytest.raises(ValueError, match=r"\(0.0, 2.0\]"):
        devices.validate_mps_memory_fraction(value)


def test_validate_mps_memory_fraction_accepts_supported_range() -> None:
    assert devices.validate_mps_memory_fraction(None) is None
    assert devices.validate_mps_memory_fraction(0.75) == 0.75
    assert devices.validate_mps_memory_fraction(2.0) == 2.0


def test_configure_mps_environment_auto_respects_existing_override(monkeypatch) -> None:
    monkeypatch.setattr(devices.platform, "system", lambda: "Darwin")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")

    devices.configure_mps_environment("auto", fallback=None)

    assert devices.os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"


def test_configure_mps_environment_auto_enables_on_darwin(monkeypatch) -> None:
    monkeypatch.setattr(devices.platform, "system", lambda: "Darwin")
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    devices.configure_mps_environment("auto", fallback=None)

    assert devices.os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_configure_mps_environment_explicit_setting_overrides_environment(monkeypatch) -> None:
    monkeypatch.setattr(devices.platform, "system", lambda: "Darwin")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    devices.configure_mps_environment("mps", fallback=False)

    assert devices.os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"


def test_auto_device_priority_is_cuda_then_mps_then_cpu() -> None:
    both = _fake_torch(cuda_available=True, mps_built=True, mps_available=True)
    mps_only = _fake_torch(cuda_available=False, mps_built=True, mps_available=True)
    cpu_only = _fake_torch()

    assert devices._resolve_semantic_device_with_torch("auto", both) == "cuda"
    assert devices._resolve_semantic_device_with_torch("auto", mps_only) == "mps"
    assert devices._resolve_semantic_device_with_torch("auto", cpu_only) == "cpu"


def test_explicit_unavailable_accelerators_raise_clear_errors() -> None:
    torch_module = _fake_torch()

    with pytest.raises(DeviceConfigurationError, match="CUDA was requested"):
        devices._resolve_semantic_device_with_torch("cuda", torch_module)
    with pytest.raises(DeviceConfigurationError, match="no MPS support"):
        devices._resolve_semantic_device_with_torch("mps", torch_module)


def test_mps_built_but_unavailable_has_distinct_error() -> None:
    torch_module = _fake_torch(mps_built=True, mps_available=False)

    with pytest.raises(DeviceConfigurationError, match="reports it unavailable"):
        devices._resolve_semantic_device_with_torch("mps", torch_module)


def test_configure_mps_memory_fraction_calls_torch_api(monkeypatch, caplog) -> None:
    torch_module = _fake_torch(mps_built=True, mps_available=True)
    monkeypatch.setattr(devices, "_load_torch", lambda: torch_module)

    with caplog.at_level(logging.WARNING):
        devices.configure_mps_memory_fraction("mps", 1.25)

    assert torch_module.mps.fractions == [1.25]
    assert "exceeds the device recommended working-set size" in caplog.text


def test_configure_mps_memory_fraction_reports_build_without_api(monkeypatch) -> None:
    torch_module = SimpleNamespace(mps=SimpleNamespace())
    monkeypatch.setattr(devices, "_load_torch", lambda: torch_module)

    with pytest.raises(DeviceConfigurationError, match="built without MPS support"):
        devices.configure_mps_memory_fraction("mps", 0.5)


def test_mps_memory_fraction_rejected_for_non_mps_device() -> None:
    with pytest.raises(DeviceConfigurationError, match="did not resolve to MPS"):
        devices.configure_mps_memory_fraction("cpu", 0.8)


def test_clear_mps_cache_synchronizes_then_collects_then_empties(monkeypatch) -> None:
    events: list[str] = []
    torch_module = _fake_torch(
        mps_built=True,
        mps_available=True,
        events=events,
    )
    monkeypatch.setattr(devices, "_load_torch", lambda: torch_module)
    monkeypatch.setattr(devices.gc, "collect", lambda: events.append("gc") or 0)

    cleared = devices.clear_device_cache("mps", synchronize=True, collect=True)

    assert cleared is True
    assert events == ["mps.sync", "gc", "mps.empty"]


def test_device_diagnostics_reports_mps_memory(monkeypatch) -> None:
    torch_module = _fake_torch(mps_built=True, mps_available=True)
    monkeypatch.setattr(devices, "_load_torch", lambda: torch_module)

    diagnostics = devices.get_device_diagnostics("mps")

    assert diagnostics.resolved == "mps"
    assert diagnostics.mps_built is True
    assert diagnostics.mps_available is True
    assert diagnostics.mps_memory_bytes == {
        "current_allocated": 128,
        "driver_allocated": 256,
        "recommended_max": 1024,
    }


def test_clear_device_cache_does_not_mutate_mlx_allocator(monkeypatch) -> None:
    events: list[str] = []
    torch_module = _fake_torch(mps_built=True, mps_available=True, events=events)
    fake_mlx_core = SimpleNamespace(clear_cache=lambda: events.append("mlx.empty"))
    monkeypatch.setattr(devices, "_load_torch", lambda: torch_module)
    monkeypatch.setattr(devices.gc, "collect", lambda: events.append("gc") or 0)
    monkeypatch.setitem(devices.sys.modules, "mlx.core", fake_mlx_core)

    devices.clear_device_cache("mps", synchronize=True, collect=True)

    assert events == ["mps.sync", "gc", "mps.empty"]
