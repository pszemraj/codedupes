"""Real-hardware CUDA tests.

Every test in this module runs against a live CUDA device: real model loads,
real allocator state, and real CUDA out-of-memory errors provoked through
``torch.cuda.set_per_process_memory_fraction``. Like the MPS suite there is no
simulated backend anywhere - if this module skips, the hardware is genuinely
absent and the run does not count as CUDA validation.

This is the executable form of the release checklist for CUDA hosts:

1. Native CUDA bfloat16 is selected only on natively capable hardware -
   ``test_bf16_policy_follows_native_capability``.
2. Cold CUDA inference completes and agrees with CPU -
   ``test_model_loads_and_encodes_on_cuda``,
   ``test_cuda_and_cpu_embeddings_agree``.
3. Corpus OOM fallback rebuilds a complete, coherent, searchable matrix on CPU -
   ``test_corpus_oom_completes_on_cpu_and_stays_searchable``,
   ``test_bf16_corpus_oom_rebuilds_one_float32_matrix``.
4. A query whose fallback changes the vector policy aborts before the dot
   product - ``test_bf16_query_fallback_aborts_before_similarity``.
5. A warm rerun serves the expected cache namespace -
   ``test_warm_cache_serves_explicit_cuda_without_model_load``.

The default model must already be cached locally (any prior ``codedupes
check`` or ``hf download`` does this).
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pytest

from codedupes import devices, semantic
from codedupes.constants import DEFAULT_MODEL
from codedupes.semantic import CPU_FALLBACK_MAX_BATCH_SIZE
from codedupes.semantic_profiles import resolve_model_profile
from tests.conftest import extract_arithmetic_units

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip(
        "Real CUDA hardware is required and there is no simulated fallback.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.gpu

# Small enough that any further allocation exceeds the caching allocator's
# ceiling, which turns a genuine CUDA OOM into a deterministic fixture. Already
# resident weights are unaffected: the cap applies to new allocations.
_TINY_MEMORY_FRACTION = 0.0001
_UNCAPPED_MEMORY_FRACTION = 1.0

_PROFILE_FAMILY = resolve_model_profile(DEFAULT_MODEL).family
_NATIVE_BF16 = torch.cuda.is_bf16_supported(including_emulation=False)
_requires_bf16 = pytest.mark.skipif(
    not _NATIVE_BF16,
    reason="This GPU has no native bfloat16 support, so CUDA runs key as float32.",
)


@pytest.fixture(autouse=True)
def _reset_real_cuda_state():
    torch.cuda.set_per_process_memory_fraction(_UNCAPPED_MEMORY_FRACTION)
    semantic.clear_model_cache()
    yield
    semantic.clear_model_cache()
    torch.cuda.set_per_process_memory_fraction(_UNCAPPED_MEMORY_FRACTION)


def _recording_encode(model: object) -> list[tuple[int | None, str | None]]:
    """Record every ``(batch_size, device)`` the OOM ladder encodes with.

    :param model: Loaded model whose ``encode`` is wrapped in place.
    :return: Live list of attempts appended to as encoding proceeds.
    """
    attempts: list[tuple[int | None, str | None]] = []
    original_encode = model.encode

    def recording(texts, **kwargs):
        attempts.append((kwargs.get("batch_size"), kwargs.get("device")))
        return original_encode(texts, **kwargs)

    model.encode = recording
    return attempts


def test_auto_and_explicit_requests_resolve_to_real_cuda() -> None:
    assert devices.resolve_semantic_device("auto") == "cuda"
    assert devices.resolve_semantic_device("cuda") == "cuda"


def test_device_diagnostics_report_real_cuda() -> None:
    diagnostics = devices.get_device_diagnostics("cuda")

    assert diagnostics.resolved == "cuda"
    assert diagnostics.error is None
    assert diagnostics.torch_available is True
    assert diagnostics.cuda_available is True


def test_clear_device_cache_synchronizes_then_collects_then_empties(monkeypatch) -> None:
    events: list[str] = []
    real_synchronize = torch.cuda.synchronize
    real_empty_cache = torch.cuda.empty_cache
    real_collect = gc.collect

    def recording_synchronize(*args, **kwargs) -> None:
        events.append("cuda.sync")
        real_synchronize(*args, **kwargs)

    def recording_empty_cache() -> None:
        events.append("cuda.empty")
        real_empty_cache()

    def recording_collect() -> int:
        events.append("gc")
        return real_collect()

    monkeypatch.setattr(torch.cuda, "synchronize", recording_synchronize)
    monkeypatch.setattr(torch.cuda, "empty_cache", recording_empty_cache)
    monkeypatch.setattr(devices.gc, "collect", recording_collect)

    cleared = devices.clear_device_cache("cuda", synchronize=True, collect=True)

    assert cleared is True
    assert events == ["cuda.sync", "gc", "cuda.empty"]


def test_bf16_policy_follows_native_capability() -> None:
    """Gate item 1: bfloat16 is pinned only where the GPU supports it natively.

    ``torch.cuda.is_bf16_supported()`` defaults to reporting emulated support on
    pre-Ampere hardware; the policy passes ``including_emulation=False``
    deliberately, so this test asserts against that same native-only question
    and the live parameter dtype of a real load must agree with it.
    """
    expected_dtype = torch.bfloat16 if _NATIVE_BF16 else torch.float32

    assert semantic._resolve_model_dtype(_PROFILE_FAMILY, "cuda") is expected_dtype
    assert semantic._resolve_model_dtype("embeddinggemma", "cuda") is expected_dtype

    model = semantic.get_model(DEFAULT_MODEL, device="cuda")

    assert semantic._model_execution_device == "cuda"
    assert semantic._model_parameter_dtype(model) is expected_dtype


def test_model_loads_and_encodes_on_cuda(tmp_path: Path) -> None:
    """Gate item 2: a cold CUDA run loads, encodes, and returns unit vectors."""
    units = extract_arithmetic_units(tmp_path)

    embeddings = semantic.compute_embeddings(units, device="cuda", batch_size=2)
    model = semantic.get_model(DEFAULT_MODEL, device="cuda")
    torch.cuda.synchronize()

    assert str(getattr(model, "device", "")).startswith("cuda")
    assert semantic._model_execution_device == "cuda"
    assert embeddings.shape[0] == len(units)
    assert np.isfinite(embeddings).all()
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)


def test_cuda_and_cpu_embeddings_agree(tmp_path: Path) -> None:
    """Gate item 2: CUDA and CPU stay inside the tolerance their dtypes allow.

    A bfloat16 CUDA run is not expected to match float32 CPU elementwise - that
    is exactly why the two key into different cache namespaces. What must hold
    is that pair similarity, the quantity every tuned threshold is compared
    against, does not move at threshold scale.
    """
    units = extract_arithmetic_units(tmp_path)

    cpu_embeddings = semantic.compute_embeddings(units, device="cpu", use_cache=False)
    cuda_embeddings = semantic.compute_embeddings(units, device="cuda", use_cache=False)

    assert cpu_embeddings.shape == cuda_embeddings.shape
    assert np.isfinite(cuda_embeddings).all()
    elementwise_tolerance = 5e-2 if _NATIVE_BF16 else 5e-3
    similarity_tolerance = 2e-2 if _NATIVE_BF16 else 2e-3
    np.testing.assert_allclose(cpu_embeddings, cuda_embeddings, atol=elementwise_tolerance)
    cpu_similarity = float(cpu_embeddings[0] @ cpu_embeddings[1])
    cuda_similarity = float(cuda_embeddings[0] @ cuda_embeddings[1])
    assert abs(cpu_similarity - cuda_similarity) < similarity_tolerance


def test_model_cache_is_keyed_by_resolved_device() -> None:
    first = semantic.get_model(DEFAULT_MODEL, device="cpu")
    again = semantic.get_model(DEFAULT_MODEL, device="cpu")
    cuda_model = semantic.get_model(DEFAULT_MODEL, device="cuda")

    assert first is again
    assert cuda_model is not first
    assert str(getattr(cuda_model, "device", "")).startswith("cuda")


def test_load_oom_falls_back_to_cpu_and_repins_the_cpu_dtype(caplog) -> None:
    """A load-time CUDA OOM retries on CPU without inheriting the CUDA dtype.

    The re-pin is CUDA-specific and has no MPS equivalent to validate it: MPS
    always resolves float32, so only this path can carry a bfloat16 dtype onto
    a host that may not execute it well.
    """
    torch.cuda.set_per_process_memory_fraction(_TINY_MEMORY_FRACTION)

    with caplog.at_level(logging.WARNING, logger="codedupes.semantic"):
        model = semantic.get_model(DEFAULT_MODEL, device="cuda")
        semantic.get_model(DEFAULT_MODEL, device="cuda")
        semantic.get_model(DEFAULT_MODEL, device="cuda")

    assert "clearing CUDA cache and retrying on CPU" in caplog.text
    assert semantic._model_device_key == "cuda"
    assert semantic._model_execution_device == "cpu"
    assert not str(getattr(model, "device", "")).startswith("cuda")
    assert caplog.text.count("after an earlier cuda-to-CPU OOM fallback") == 1

    expected_cpu_dtype = torch.bfloat16 if devices.resolve_cpu_bf16_inference() else torch.float32
    assert semantic._model_parameter_dtype(model) is expected_cpu_dtype

    semantic.clear_model_cache()
    torch.cuda.set_per_process_memory_fraction(_UNCAPPED_MEMORY_FRACTION)
    fresh_model = semantic.get_model(DEFAULT_MODEL, device="cuda")

    assert semantic._model_execution_device == "cuda"
    assert str(getattr(fresh_model, "device", "")).startswith("cuda")


def test_encode_oom_halves_batch_then_restarts_on_cpu_at_the_cap(tmp_path: Path, caplog) -> None:
    """The ladder halves to one, then restarts on CPU at the documented cap.

    The requested batch (64) is deliberately above
    ``CPU_FALLBACK_MAX_BATCH_SIZE`` so the CPU restart proves the cap rather
    than merely inheriting the request: a host OOM can arrive as an uncatchable
    OOM-killer kill that this ladder would never see.
    """
    units = extract_arithmetic_units(tmp_path)
    model = semantic.get_model(DEFAULT_MODEL, device="cuda")
    attempts = _recording_encode(model)

    torch.cuda.set_per_process_memory_fraction(_TINY_MEMORY_FRACTION)

    with caplog.at_level(logging.WARNING, logger="codedupes.semantic"):
        embeddings = semantic.compute_embeddings(units, device="cuda", batch_size=64)

    # Prefix, not equality: on a bfloat16 GPU the CPU landing also breaks dtype
    # coherence, so the corpus legitimately restarts and encodes again after the
    # ladder finishes. The ladder itself is what this test pins.
    assert attempts[:8] == [
        (64, None),
        (32, None),
        (16, None),
        (8, None),
        (4, None),
        (2, None),
        (1, None),
        (CPU_FALLBACK_MAX_BATCH_SIZE, "cpu"),
    ]
    assert semantic._model_execution_device == "cpu"
    assert embeddings.shape[0] == len(units)
    assert np.isfinite(embeddings).all()
    oom_warnings = [
        message
        for message in caplog.messages
        if message.startswith("CUDA OOM during embedding inference")
    ]
    assert len(oom_warnings) == 7


def test_corpus_oom_completes_on_cpu_and_stays_searchable(tmp_path: Path) -> None:
    """Gate item 3: an OOM mid-corpus still yields a complete, searchable matrix."""
    units = extract_arithmetic_units(tmp_path)
    semantic.get_model(DEFAULT_MODEL, device="cuda")

    torch.cuda.set_per_process_memory_fraction(_TINY_MEMORY_FRACTION)

    embeddings, identity = semantic.compute_embeddings_with_identity(
        units, device="cuda", batch_size=8, cache_scope=tmp_path
    )

    assert semantic._model_execution_device == "cpu"
    assert embeddings.shape[0] == len(units)
    assert np.isfinite(embeddings).all()

    results = semantic.find_similar_to_query(
        "addition",
        units,
        embeddings,
        device="cpu",
        threshold=0.0,
        use_cache=False,
        corpus_identity=identity,
    )

    assert len(results) == len(units)
    assert [score for _unit, score in results] == sorted(
        (score for _unit, score in results), reverse=True
    )


@_requires_bf16
def test_bf16_corpus_oom_rebuilds_one_float32_matrix(tmp_path: Path) -> None:
    """Gate item 3: a keyed-bfloat16 run that lands on CPU rebuilds coherently.

    The partially warm bfloat16 cache must not be blended with float32 rows
    encoded after the fallback: the run discards its hits and rebuilds the
    complete matrix under the faithful CPU identity.
    """
    units = extract_arithmetic_units(tmp_path)
    _warm, warm_identity = semantic.compute_embeddings_with_identity(
        units[:1],
        device="cuda",
        cache_scope=tmp_path,
    )
    assert "dtype=torch.bfloat16" in warm_identity.runtime_variant

    torch.cuda.set_per_process_memory_fraction(_TINY_MEMORY_FRACTION)

    embeddings, identity = semantic.compute_embeddings_with_identity(
        units, device="cuda", batch_size=8, cache_scope=tmp_path
    )

    assert semantic._model_execution_device == "cpu"
    assert embeddings.shape[0] == len(units)
    assert "dtype=torch.bfloat16" not in identity.runtime_variant

    cpu_reference = semantic.compute_embeddings(units, device="cpu", use_cache=False)
    np.testing.assert_array_equal(embeddings, cpu_reference)


@_requires_bf16
def test_bf16_query_fallback_aborts_before_similarity(tmp_path: Path) -> None:
    """Gate item 4: the correctness boundary is the dot product, not the cache key."""
    units = extract_arithmetic_units(tmp_path)
    embeddings, identity = semantic.compute_embeddings_with_identity(
        units,
        device="cuda",
        use_cache=False,
    )
    assert "dtype=torch.bfloat16" in identity.runtime_variant

    torch.cuda.set_per_process_memory_fraction(_TINY_MEMORY_FRACTION)

    with pytest.raises(RuntimeError, match="keyed under a bfloat16 policy"):
        semantic.find_similar_to_query(
            "addition",
            units,
            embeddings,
            device="cuda",
            threshold=0.0,
            use_cache=False,
            corpus_identity=identity,
        )


def test_warm_cache_serves_explicit_cuda_without_model_load(tmp_path: Path, monkeypatch) -> None:
    """Gate item 5: a warm rerun reads the same namespace and never loads the model."""
    units = extract_arithmetic_units(tmp_path)
    first, first_identity = semantic.compute_embeddings_with_identity(
        units, device="cuda", cache_scope=tmp_path
    )
    semantic.clear_model_cache()

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("the model must not load when every embedding is cached")

    monkeypatch.setattr(semantic, "get_model", _fail_if_called)

    # The explicit-device availability check still runs for real and passes on
    # this hardware.
    second, second_identity = semantic.compute_embeddings_with_identity(
        units, device="cuda", cache_scope=tmp_path
    )

    np.testing.assert_array_equal(first, second)
    assert second_identity == first_identity


@_requires_bf16
def test_bf16_cuda_and_float32_cpu_key_into_different_namespaces(tmp_path: Path) -> None:
    """A bfloat16 CUDA corpus must not be served to a float32 CPU run.

    Unlike MPS - which pins float32 and therefore shares CPU's key space - a
    natively bf16-capable CUDA host produces vectors CPU cannot reproduce, so
    the dtype fingerprint has to split the namespace.
    """
    units = extract_arithmetic_units(tmp_path)
    _cuda_embeddings, cuda_identity = semantic.compute_embeddings_with_identity(
        units, device="cuda", cache_scope=tmp_path
    )
    _cpu_embeddings, cpu_identity = semantic.compute_embeddings_with_identity(
        units, device="cpu", cache_scope=tmp_path
    )

    assert "dtype=torch.bfloat16" in cuda_identity.runtime_variant
    assert "dtype=torch.bfloat16" not in cpu_identity.runtime_variant
    assert cuda_identity != cpu_identity
