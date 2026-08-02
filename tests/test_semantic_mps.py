"""Real-hardware Apple MPS tests.

Every test in this module runs against the live Metal device: real model loads,
real allocator state, and real MPS out-of-memory errors provoked through
``torch.mps.set_per_process_memory_fraction``. There is deliberately no
simulated MPS anywhere in the test suite — if this module skips, the hardware
is genuinely absent (non-Mac, or a sandbox that blocks Metal device access),
and the run does not count as MPS validation.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from codedupes import devices, semantic
from codedupes.constants import DEFAULT_MODEL
from tests.conftest import extract_arithmetic_units

torch = pytest.importorskip("torch")

if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
    pytest.skip(
        "Real Apple MPS hardware is required and there is no simulated fallback; "
        "run unsandboxed/escalated on Apple Silicon.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.mps

# Small enough that any tensor allocation exceeds the Metal allocator ceiling,
# which turns the allocator's genuine OOM path into a deterministic fixture.
_TINY_MEMORY_FRACTION = 0.0001
# torch's default high-watermark ratio (PYTORCH_MPS_HIGH_WATERMARK_RATIO).
_DEFAULT_MEMORY_FRACTION = 1.7


@pytest.fixture(autouse=True)
def _reset_real_mps_state():
    semantic.clear_model_cache()
    semantic._warned_mlx_mps_contention = False
    yield
    semantic.clear_model_cache()
    semantic._warned_mlx_mps_contention = False
    torch.mps.set_per_process_memory_fraction(_DEFAULT_MEMORY_FRACTION)


def test_auto_and_explicit_requests_resolve_to_real_mps() -> None:
    assert devices.resolve_semantic_device("auto") == "mps"
    assert devices.resolve_semantic_device("mps") == "mps"


def test_device_diagnostics_report_real_mps_memory() -> None:
    diagnostics = devices.get_device_diagnostics("mps")

    assert diagnostics.resolved == "mps"
    assert diagnostics.error is None
    assert diagnostics.torch_available is True
    assert diagnostics.mps_built is True
    assert diagnostics.mps_available is True
    assert set(diagnostics.mps_memory_bytes) == {
        "current_allocated",
        "driver_allocated",
        "recommended_max",
    }
    assert diagnostics.mps_memory_bytes["recommended_max"] > 0


def test_configure_mps_memory_fraction_applies_and_warns_above_recommended(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="codedupes.devices"):
        devices.configure_mps_memory_fraction("mps", 1.25)
    assert "exceeds the device recommended working-set size" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="codedupes.devices"):
        devices.configure_mps_memory_fraction("mps", 0.5)
    assert "exceeds the device recommended working-set size" not in caplog.text


def test_clear_device_cache_synchronizes_then_collects_then_empties(monkeypatch) -> None:
    events: list[str] = []
    real_synchronize = torch.mps.synchronize
    real_empty_cache = torch.mps.empty_cache
    real_collect = gc.collect

    def recording_synchronize() -> None:
        events.append("mps.sync")
        real_synchronize()

    def recording_empty_cache() -> None:
        events.append("mps.empty")
        real_empty_cache()

    def recording_collect() -> int:
        events.append("gc")
        return real_collect()

    monkeypatch.setattr(torch.mps, "synchronize", recording_synchronize)
    monkeypatch.setattr(torch.mps, "empty_cache", recording_empty_cache)
    monkeypatch.setattr(devices.gc, "collect", recording_collect)

    cleared = devices.clear_device_cache("mps", synchronize=True, collect=True)

    assert cleared is True
    assert events == ["mps.sync", "gc", "mps.empty"]


def test_clear_device_cache_does_not_mutate_mlx_allocator(monkeypatch) -> None:
    # The MLX module is a stand-in recorder: importing real MLX would initialize a
    # second Metal allocator for the remainder of the test process, which is the
    # exact contention this guarantee protects users from. The PyTorch side of the
    # call runs against the real allocator.
    events: list[str] = []
    fake_mlx_core = SimpleNamespace(clear_cache=lambda: events.append("mlx.empty"))
    monkeypatch.setitem(devices.sys.modules, "mlx.core", fake_mlx_core)

    cleared = devices.clear_device_cache("mps", synchronize=True, collect=True)

    assert cleared is True
    assert events == []


def test_mlx_contention_warning_is_emitted_once(monkeypatch, caplog) -> None:
    # Only the MLX-loaded flag is stubbed (importing real MLX would poison the
    # process allocator); device resolution and configuration run for real.
    monkeypatch.setattr(semantic, "is_mlx_loaded", lambda: True)

    with caplog.at_level(logging.WARNING, logger="codedupes.semantic"):
        resolved_first = semantic._prepare_semantic_device(
            "mps",
            mps_fallback=None,
            mps_memory_fraction=None,
        )
        resolved_second = semantic._prepare_semantic_device(
            "mps",
            mps_fallback=None,
            mps_memory_fraction=None,
        )

    assert resolved_first == resolved_second == "mps"
    assert caplog.text.count("MLX is already loaded") == 1


def test_embeddinggemma_dtype_on_mps_is_float32() -> None:
    assert semantic._resolve_embeddinggemma_torch_dtype("mps") is torch.float32


def test_model_loads_and_encodes_on_mps(tmp_path: Path) -> None:
    units = extract_arithmetic_units(tmp_path)

    embeddings = semantic.compute_embeddings(units, device="mps", batch_size=2)
    model = semantic.get_model(DEFAULT_MODEL, device="mps")
    torch.mps.synchronize()

    assert str(getattr(model, "device", "")).startswith("mps")
    assert semantic._model_execution_device == "mps"
    assert embeddings.shape[0] == len(units)
    assert np.isfinite(embeddings).all()
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)


def test_model_cache_is_keyed_by_resolved_device() -> None:
    first = semantic.get_model(DEFAULT_MODEL, device="cpu")
    again = semantic.get_model(DEFAULT_MODEL, device="cpu")
    mps_model = semantic.get_model(DEFAULT_MODEL, device="mps")

    assert first is again
    assert mps_model is not first
    assert str(getattr(mps_model, "device", "")).startswith("mps")


def test_mps_and_cpu_embeddings_agree(tmp_path: Path) -> None:
    units = extract_arithmetic_units(tmp_path)

    cpu_embeddings = semantic.compute_embeddings(units, device="cpu", use_cache=False)
    mps_embeddings = semantic.compute_embeddings(units, device="mps", use_cache=False)

    assert cpu_embeddings.shape == mps_embeddings.shape
    np.testing.assert_allclose(cpu_embeddings, mps_embeddings, atol=5e-3)
    cpu_similarity = float(cpu_embeddings[0] @ cpu_embeddings[1])
    mps_similarity = float(mps_embeddings[0] @ mps_embeddings[1])
    assert abs(cpu_similarity - mps_similarity) < 2e-3


def test_warm_cache_serves_explicit_mps_without_model_load(tmp_path: Path, monkeypatch) -> None:
    units = extract_arithmetic_units(tmp_path)
    first = semantic.compute_embeddings(units, device="cpu", cache_scope=tmp_path)
    semantic.clear_model_cache()

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("the model must not load when every embedding is cached")

    monkeypatch.setattr(semantic, "get_model", _fail_if_called)

    # The explicit-device availability check still runs for real and passes on
    # this hardware; float32 CPU and MPS runs share one cache key space.
    second = semantic.compute_embeddings(units, device="mps", cache_scope=tmp_path)

    np.testing.assert_array_equal(first, second)


def test_clear_model_cache_drops_mps_model_without_cpu_migration() -> None:
    model = semantic.get_model(DEFAULT_MODEL, device="mps")
    moves: list[str] = []
    original_to = model.to

    def recording_to(device, *args, **kwargs):
        moves.append(str(device))
        return original_to(device, *args, **kwargs)

    model.to = recording_to

    semantic.clear_model_cache()

    assert moves == []
    assert semantic._model is None


def test_load_oom_falls_back_to_cpu_and_reuse_warns_once(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="codedupes.semantic"):
        model = semantic.get_model(
            DEFAULT_MODEL,
            device="mps",
            mps_memory_fraction=_TINY_MEMORY_FRACTION,
        )
        semantic.get_model(DEFAULT_MODEL, device="mps")
        semantic.get_model(DEFAULT_MODEL, device="mps")

    assert "clearing Metal cache and retrying on CPU" in caplog.text
    assert semantic._model_device_key == "mps"
    assert semantic._model_execution_device == "cpu"
    assert not str(getattr(model, "device", "")).startswith("mps")
    assert caplog.text.count("after an earlier mps-to-CPU OOM fallback") == 1


def test_encode_oom_halves_batch_then_falls_back_to_cpu(tmp_path: Path) -> None:
    units = extract_arithmetic_units(tmp_path)
    model = semantic.get_model(DEFAULT_MODEL, device="mps")

    attempts: list[tuple[int, str | None]] = []
    original_encode = model.encode

    def recording_encode(texts, **kwargs):
        attempts.append((kwargs.get("batch_size"), kwargs.get("device")))
        return original_encode(texts, **kwargs)

    model.encode = recording_encode

    # The model's weights already exceed the lowered ceiling, so every further
    # MPS allocation raises the allocator's genuine OOM error.
    torch.mps.set_per_process_memory_fraction(_TINY_MEMORY_FRACTION)

    embeddings = semantic.compute_embeddings(units, device="mps", batch_size=8)

    assert attempts == [(8, None), (4, None), (2, None), (1, None), (8, "cpu")]
    assert semantic._model_execution_device == "cpu"
    assert embeddings.shape[0] == len(units)
    assert np.isfinite(embeddings).all()


def test_query_oom_recovers_on_cpu(tmp_path: Path) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = semantic.compute_embeddings(units, device="cpu", use_cache=False)
    model = semantic.get_model(DEFAULT_MODEL, device="mps")

    attempts: list[tuple[int, str | None]] = []
    original_encode = model.encode

    def recording_encode(texts, **kwargs):
        attempts.append((kwargs.get("batch_size"), kwargs.get("device")))
        return original_encode(texts, **kwargs)

    model.encode = recording_encode
    torch.mps.set_per_process_memory_fraction(_TINY_MEMORY_FRACTION)

    results = semantic.find_similar_to_query(
        "addition",
        units,
        embeddings,
        device="mps",
        threshold=0.0,
        use_cache=False,
    )

    # Query embedding runs at batch size one, so the ladder is a single real MPS
    # OOM followed by the CPU retry.
    assert attempts == [(1, None), (1, "cpu")]
    assert len(results) == len(units)
    assert [score for _unit, score in results] == sorted(
        (score for _unit, score in results), reverse=True
    )
