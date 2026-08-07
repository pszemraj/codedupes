"""Pure-logic device tests.

Everything that touches live MPS hardware — resolution to a real device,
allocator statistics, memory-fraction application, cache clearing, OOM behavior
— lives in ``tests/test_semantic_mps.py`` and runs against the physical
accelerator. This module keeps only logic that does not depend on accelerator
state, so nothing here simulates MPS.
"""

from __future__ import annotations

import logging
import math

import pytest

from codedupes import devices
from codedupes.devices import DeviceConfigurationError


def test_describe_mps_fallback_env_matches_torch_interpretation() -> None:
    assert devices.describe_mps_fallback_env(None) == "disabled"
    assert devices.describe_mps_fallback_env("unset") == "disabled"
    assert devices.describe_mps_fallback_env("0") == "disabled"
    assert devices.describe_mps_fallback_env("1") == "enabled"
    # torch enables the fallback for any set value except the literal "0".
    assert devices.describe_mps_fallback_env("false") == "enabled"
    assert devices.describe_mps_fallback_env("") == "enabled"


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
    monkeypatch.setattr(devices.sys, "platform", "darwin")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")

    devices.configure_mps_environment("auto", fallback=None)

    assert devices.os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"


def test_configure_mps_environment_auto_enables_on_darwin(monkeypatch) -> None:
    monkeypatch.setattr(devices.sys, "platform", "darwin")
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    devices.configure_mps_environment("auto", fallback=None)

    assert devices.os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_configure_mps_environment_auto_is_inert_off_darwin(monkeypatch) -> None:
    monkeypatch.setattr(devices.sys, "platform", "linux")
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    devices.configure_mps_environment("auto", fallback=None)

    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in devices.os.environ


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


def test_resolve_mps_memory_fraction_restore_value_defaults_without_env_override(
    monkeypatch,
) -> None:
    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)

    assert (
        devices._resolve_mps_memory_fraction_restore_value()
        == devices._PYTORCH_DEFAULT_MPS_HIGH_WATERMARK_RATIO
    )


@pytest.mark.parametrize("raw, expected", [("1.2", 1.2), ("0.0", 0.0), ("2.0", 2.0)])
def test_resolve_mps_memory_fraction_restore_value_captures_valid_env_override(
    monkeypatch, raw: str, expected: float
) -> None:
    # 0.0 and 2.0 are captured verbatim here even though validate_mps_memory_fraction
    # rejects both: this helper only recovers PyTorch's own baseline and does not
    # re-apply codedupes' narrower (0.0, 2.0] safety interval to it.
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raw)

    assert devices._resolve_mps_memory_fraction_restore_value() == expected


@pytest.mark.parametrize("raw", ["not-a-number", "-0.1", "2.1", "nan", "inf"])
def test_resolve_mps_memory_fraction_restore_value_falls_back_on_invalid_env_value(
    monkeypatch, caplog, raw: str
) -> None:
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raw)

    with caplog.at_level(logging.WARNING, logger="codedupes.devices"):
        result = devices._resolve_mps_memory_fraction_restore_value()

    assert result == devices._PYTORCH_DEFAULT_MPS_HIGH_WATERMARK_RATIO
    assert "Ignoring invalid PYTORCH_MPS_HIGH_WATERMARK_RATIO" in caplog.text


def test_cpu_bf16_capability_matches_live_isa_and_mkldnn_conjunction() -> None:
    torch = pytest.importorskip("torch")

    caps = torch.cpu.get_capabilities()
    architecture = str(caps.get("architecture", "")).lower()
    if "arm" in architecture or "aarch64" in architecture:
        expected_isa = bool(caps.get("bf16"))
    else:
        expected_isa = bool(caps.get("amx_bf16")) or bool(caps.get("avx512_bf16"))
    expected = expected_isa and torch.backends.mkldnn.is_available()

    assert devices.cpu_bf16_capability(torch) is expected


def test_cpu_bf16_capability_defensive_on_probe_failure() -> None:
    class BrokenTorch:
        class cpu:  # mirrors torch's module-shaped attribute access
            @staticmethod
            def get_capabilities():
                raise RuntimeError("boom")

    assert devices.cpu_bf16_capability(BrokenTorch) is False


def test_resolve_cpu_bf16_inference_requires_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("CODEDUPES_CPU_BF16", raising=False)

    def _fail_if_called() -> bool:
        raise AssertionError("the capability gate must not be consulted without the opt-in")

    monkeypatch.setattr(devices, "resolve_cpu_bf16_native", _fail_if_called)

    assert devices.resolve_cpu_bf16_inference() is False


def test_resolve_cpu_bf16_inference_follows_gate_when_opted_in(monkeypatch) -> None:
    monkeypatch.setenv("CODEDUPES_CPU_BF16", "1")

    monkeypatch.setattr(devices, "resolve_cpu_bf16_native", lambda: True)
    assert devices.resolve_cpu_bf16_inference() is True

    monkeypatch.setattr(devices, "resolve_cpu_bf16_native", lambda: False)
    assert devices.resolve_cpu_bf16_inference() is False


def test_resolve_cpu_bf16_inference_skips_torch_import_without_opt_in(monkeypatch) -> None:
    # The opt-in gate must short-circuit before the live probe: a non-opted-in
    # run must never import torch just to decide CPU dtype.
    monkeypatch.delenv("CODEDUPES_CPU_BF16", raising=False)

    def _fail_if_called() -> object:
        raise AssertionError("torch must not be imported without the opt-in")

    monkeypatch.setattr(devices, "_load_torch", _fail_if_called)

    assert devices.resolve_cpu_bf16_inference() is False


def test_resolve_cpu_bf16_inference_probes_torch_at_most_once_per_process(monkeypatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.setenv("CODEDUPES_CPU_BF16", "1")
    devices._reset_cpu_bf16_probe_cache()

    real_load_torch = devices._load_torch
    calls = {"count": 0}

    def _spy_load_torch():
        calls["count"] += 1
        return real_load_torch()

    monkeypatch.setattr(devices, "_load_torch", _spy_load_torch)

    first = devices.resolve_cpu_bf16_inference()
    second = devices.resolve_cpu_bf16_inference()

    assert first == second
    assert calls["count"] == 1


def test_reset_cpu_bf16_probe_cache_forces_a_fresh_probe(monkeypatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.setenv("CODEDUPES_CPU_BF16", "1")
    devices._reset_cpu_bf16_probe_cache()

    real_load_torch = devices._load_torch
    calls = {"count": 0}

    def _spy_load_torch():
        calls["count"] += 1
        return real_load_torch()

    monkeypatch.setattr(devices, "_load_torch", _spy_load_torch)

    devices.resolve_cpu_bf16_inference()
    devices._reset_cpu_bf16_probe_cache()
    devices.resolve_cpu_bf16_inference()

    assert calls["count"] == 2


def test_restore_mps_memory_fraction_noop_when_nothing_managed(monkeypatch) -> None:
    monkeypatch.setattr(devices, "_mps_memory_fraction_managed", False)
    calls: list[tuple] = []
    monkeypatch.setattr(
        devices, "configure_mps_memory_fraction", lambda *a, **k: calls.append((a, k))
    )

    devices.restore_mps_memory_fraction_if_managed()

    assert calls == []


def test_restore_mps_memory_fraction_skips_when_torch_not_imported(monkeypatch) -> None:
    monkeypatch.setattr(devices, "_mps_memory_fraction_managed", True)
    monkeypatch.delitem(devices.sys.modules, "torch", raising=False)
    calls: list[tuple] = []
    monkeypatch.setattr(
        devices, "configure_mps_memory_fraction", lambda *a, **k: calls.append((a, k))
    )

    devices.restore_mps_memory_fraction_if_managed()

    assert calls == []


def test_restore_mps_memory_fraction_uses_real_torch_when_managed(monkeypatch) -> None:
    # No MPS state is faked here: torch is real and already imported, and the
    # setter itself is stubbed so no allocator call happens; only the branch
    # decision (does this build report MPS built?) is read from live torch.
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(devices, "_mps_memory_fraction_managed", True)
    calls: list[tuple] = []
    monkeypatch.setattr(
        devices, "configure_mps_memory_fraction", lambda *a, **k: calls.append((a, k))
    )

    devices.restore_mps_memory_fraction_if_managed()

    if torch.backends.mps.is_built():
        assert calls == [(("mps", None), {})]
    else:
        assert calls == []
