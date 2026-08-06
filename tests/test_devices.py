"""Pure-logic device tests.

Everything that touches live MPS hardware — resolution to a real device,
allocator statistics, memory-fraction application, cache clearing, OOM behavior
— lives in ``tests/test_semantic_mps.py`` and runs against the physical
accelerator. This module keeps only logic that does not depend on accelerator
state, so nothing here simulates MPS.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

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


def _machine_records_dir(cache_dir: Path) -> Path:
    """Resolve the on-disk machine-capability records directory under a cache dir.

    :param cache_dir: Root cache directory used for this test.
    :return: Expected ``machines/`` directory path.
    """
    return cache_dir / devices._MACHINE_RECORDS_DIRNAME


def _legacy_machine_record_path(cache_dir: Path) -> Path:
    """Resolve the legacy, non-namespaced machine-capability record path.

    :param cache_dir: Root cache directory used for this test.
    :return: Expected legacy ``machine.json`` path.
    """
    return cache_dir / devices._LEGACY_MACHINE_RECORD_FILENAME


def test_resolve_cpu_bf16_native_persists_a_fresh_probe(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.setenv("CODEDUPES_CACHE_DIR", str(tmp_path))

    verdict = devices.resolve_cpu_bf16_native()

    environment = devices.CpuCapabilityEnvironment.current()
    record_path = _machine_records_dir(tmp_path) / f"{environment.digest()}.json"
    assert record_path.is_file()
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["schema"] == devices.CPU_CAPABILITY_RECORD_SCHEMA
    assert payload["environment"] == environment.as_dict()
    assert payload["cpu_bf16_native"] == verdict
    assert isinstance(payload["capabilities"], dict)
    assert isinstance(payload["mkldnn_available"], bool)


def test_resolve_cpu_bf16_native_trusts_a_valid_record_without_probing(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("torch")
    monkeypatch.setenv("CODEDUPES_CACHE_DIR", str(tmp_path))
    environment = devices.CpuCapabilityEnvironment.current()
    record_path = _machine_records_dir(tmp_path) / f"{environment.digest()}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(
            {
                "schema": devices.CPU_CAPABILITY_RECORD_SCHEMA,
                "environment": environment.as_dict(),
                "capabilities": {},
                "mkldnn_available": True,
                "cpu_bf16_native": True,
            }
        ),
        encoding="utf-8",
    )

    def _fail_if_probed() -> None:
        raise AssertionError("a valid record must be trusted without a live probe")

    monkeypatch.setattr(devices, "_load_torch", _fail_if_probed)

    assert devices.resolve_cpu_bf16_native() is True


def test_resolve_cpu_bf16_native_reprobes_on_environment_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("torch")
    monkeypatch.setenv("CODEDUPES_CACHE_DIR", str(tmp_path))
    # Write a record at the CURRENT identity's digest path, but with a payload
    # environment that does not match: this proves payload validation is
    # enforced independent of the (correct) filename.
    environment = devices.CpuCapabilityEnvironment.current()
    record_path = _machine_records_dir(tmp_path) / f"{environment.digest()}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    mismatched_environment = dict(environment.as_dict())
    mismatched_environment["hostname"] = f"{mismatched_environment['hostname']}-stale"
    record_path.write_text(
        json.dumps(
            {
                "schema": devices.CPU_CAPABILITY_RECORD_SCHEMA,
                "environment": mismatched_environment,
                "capabilities": {},
                "mkldnn_available": True,
                "cpu_bf16_native": True,
            }
        ),
        encoding="utf-8",
    )

    # A payload/identity mismatch is untrusted, so this falls back to the live
    # probe and overwrites the record with the current environment identity.
    devices.resolve_cpu_bf16_native()

    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["environment"] == environment.as_dict()


def test_resolve_cpu_bf16_native_reprobes_on_schema_mismatch(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.setenv("CODEDUPES_CACHE_DIR", str(tmp_path))
    environment = devices.CpuCapabilityEnvironment.current()
    record_path = _machine_records_dir(tmp_path) / f"{environment.digest()}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(
            {
                "schema": devices.CPU_CAPABILITY_RECORD_SCHEMA - 1,
                "environment": environment.as_dict(),
                "capabilities": {},
                "mkldnn_available": True,
                "cpu_bf16_native": True,
            }
        ),
        encoding="utf-8",
    )

    devices.resolve_cpu_bf16_native()

    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["schema"] == devices.CPU_CAPABILITY_RECORD_SCHEMA


def test_resolve_cpu_bf16_native_reprobes_on_corrupt_json(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.setenv("CODEDUPES_CACHE_DIR", str(tmp_path))
    environment = devices.CpuCapabilityEnvironment.current()
    record_path = _machine_records_dir(tmp_path) / f"{environment.digest()}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text("{not valid json", encoding="utf-8")

    probed = {"called": False}
    real_load_torch = devices._load_torch

    def _tracking_load_torch():
        probed["called"] = True
        return real_load_torch()

    monkeypatch.setattr(devices, "_load_torch", _tracking_load_torch)

    devices.resolve_cpu_bf16_native()

    assert probed["called"] is True


def test_resolve_cpu_bf16_native_skips_cache_when_disabled(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.setenv("CODEDUPES_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CODEDUPES_NO_CACHE", "1")

    devices.resolve_cpu_bf16_native()

    assert not _machine_records_dir(tmp_path).exists()


def test_resolve_cpu_bf16_native_namespaces_records_by_environment(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("torch")
    monkeypatch.setenv("CODEDUPES_CACHE_DIR", str(tmp_path))

    devices.resolve_cpu_bf16_native()
    first_environment = devices.CpuCapabilityEnvironment.current()
    first_record_path = _machine_records_dir(tmp_path) / f"{first_environment.digest()}.json"
    assert first_record_path.is_file()
    first_payload = first_record_path.read_text(encoding="utf-8")

    monkeypatch.setattr(devices.platform, "node", lambda: "a-different-host")
    devices.resolve_cpu_bf16_native()
    second_environment = devices.CpuCapabilityEnvironment.current()
    second_record_path = _machine_records_dir(tmp_path) / f"{second_environment.digest()}.json"

    assert second_record_path != first_record_path
    assert second_record_path.is_file()
    # The first environment's record is untouched by the second probe/write.
    assert first_record_path.read_text(encoding="utf-8") == first_payload


def test_resolve_cpu_bf16_native_removes_legacy_record_on_persist(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("torch")
    monkeypatch.setenv("CODEDUPES_CACHE_DIR", str(tmp_path))
    legacy_path = _legacy_machine_record_path(tmp_path)
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        json.dumps({"torch": "0.0.0-does-not-exist", "cpu_bf16_native": True}), encoding="utf-8"
    )

    devices.resolve_cpu_bf16_native()

    assert not legacy_path.exists()


def test_resolve_cpu_bf16_inference_requires_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("CODEDUPES_CPU_BF16", raising=False)

    def _fail_if_called(**_kwargs):
        raise AssertionError("the capability gate must not be consulted without the opt-in")

    monkeypatch.setattr(devices, "resolve_cpu_bf16_native", _fail_if_called)

    assert devices.resolve_cpu_bf16_inference() is False


def test_resolve_cpu_bf16_inference_follows_gate_when_opted_in(monkeypatch) -> None:
    monkeypatch.setenv("CODEDUPES_CPU_BF16", "1")

    monkeypatch.setattr(devices, "resolve_cpu_bf16_native", lambda **_kwargs: True)
    assert devices.resolve_cpu_bf16_inference() is True

    monkeypatch.setattr(devices, "resolve_cpu_bf16_native", lambda **_kwargs: False)
    assert devices.resolve_cpu_bf16_inference() is False


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
