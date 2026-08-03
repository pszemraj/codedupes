"""Pure-logic device tests.

Everything that touches live MPS hardware — resolution to a real device,
allocator statistics, memory-fraction application, cache clearing, OOM behavior
— lives in ``tests/test_semantic_mps.py`` and runs against the physical
accelerator. This module keeps only logic that does not depend on accelerator
state, so nothing here simulates MPS.
"""

from __future__ import annotations

import math

import pytest

from codedupes import devices
from codedupes.devices import DeviceConfigurationError


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
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")

    devices.configure_mps_environment("auto", fallback=None)

    assert devices.os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"


def test_configure_mps_environment_auto_enables_on_darwin(monkeypatch) -> None:
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    devices.configure_mps_environment("auto", fallback=None)

    assert devices.os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_configure_mps_environment_explicit_setting_overrides_environment(monkeypatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    devices.configure_mps_environment("mps", fallback=False)

    assert devices.os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"


def test_format_bytes_always_returns_largest_supported_unit() -> None:
    assert devices.format_bytes(1024**5) == "1024.0 TiB"


def test_load_torch_reports_missing_dependency(monkeypatch) -> None:
    def missing_torch(name: str):
        assert name == "torch"
        error = ModuleNotFoundError("No module named 'torch'")
        error.name = "torch"
        raise error

    monkeypatch.setattr(devices.importlib, "import_module", missing_torch)

    with pytest.raises(DeviceConfigurationError, match="PyTorch is required"):
        devices._load_torch()


def test_device_diagnostics_degrades_cleanly_without_torch(monkeypatch) -> None:
    def unavailable_torch():
        raise DeviceConfigurationError("PyTorch unavailable for test")

    monkeypatch.setattr(devices, "_load_torch", unavailable_torch)

    diagnostics = devices.get_device_diagnostics("mps")

    assert diagnostics.requested == "mps"
    assert diagnostics.resolved is None
    assert diagnostics.torch_available is False
    assert diagnostics.cuda_available is False
    assert diagnostics.mps_built is False
    assert diagnostics.mps_available is False
    assert diagnostics.error == "PyTorch unavailable for test"


def test_auto_prefers_cuda_when_available(monkeypatch) -> None:
    # CUDA hardware can never exist on this Apple Silicon machine, so the CUDA
    # half of the auto-priority rule is the one branch checked through a stubbed
    # answer; the resolver short-circuits before consulting any MPS state.
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert devices.resolve_semantic_device("auto") == "cuda"


def test_explicit_cuda_request_fails_for_real_without_cuda() -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("This regression requires a CUDA-less host.")

    with pytest.raises(DeviceConfigurationError, match="CUDA was requested"):
        devices.resolve_semantic_device("cuda")


def test_mps_memory_fraction_rejected_for_non_mps_device() -> None:
    with pytest.raises(DeviceConfigurationError, match="did not resolve to MPS"):
        devices.configure_mps_memory_fraction("cpu", 0.8)
