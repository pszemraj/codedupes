"""Device and dtype fallback: OOM retries, CPU restarts, and warm-cache device resolution."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import sentence_transformers
import torch

from codedupes import devices, semantic
from codedupes.constants import CPU_FALLBACK_MAX_BATCH_SIZE
from codedupes.embedding_cache import EmbeddingCache
from codedupes.models import CodeUnit
from codedupes.semantic import (
    SemanticBackendError,
    compute_embeddings,
    find_similar_to_query,
)
from tests.conftest import extract_arithmetic_units
from tests.semantic_helpers import (
    FULL_REVISION,
    WarmCacheModel,
    WhitespaceTokenizer,
    constant_embeddings,
    fail_if_called,
)


class _StaticEmbeddingModel:
    """Model stub that returns a precomputed embedding matrix."""

    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def encode(self, _texts, **_kwargs):
        return self.output


@pytest.mark.parametrize(
    ("output_factory", "match"),
    [
        pytest.param(
            lambda row_count: constant_embeddings(row_count, (np.nan, 0.0)),
            "NaN or infinity",
            id="nonfinite",
        ),
        pytest.param(
            lambda row_count: np.zeros((row_count, 2), dtype=np.float32),
            "zero or invalid vector",
            id="zero-vector",
        ),
        pytest.param(
            lambda _row_count: np.array([[1.0, 0.0]], dtype=np.float32),
            "rows",
            id="wrong-row-count",
        ),
        pytest.param(
            lambda row_count: np.empty((row_count, 0), dtype=np.float32),
            "zero columns",
            id="zero-width",
        ),
    ],
)
def test_compute_embeddings_rejects_invalid_model_output(
    tmp_path: Path, monkeypatch, output_factory, match: str
) -> None:
    units = extract_arithmetic_units(tmp_path)

    model = _StaticEmbeddingModel(output_factory(len(units)))
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: model)

    with pytest.raises(semantic.InvalidEmbeddingError, match=match):
        compute_embeddings(units, device="cpu")


def test_accelerator_nonfinite_output_retries_once_on_cpu(tmp_path: Path, monkeypatch) -> None:
    units = extract_arithmetic_units(tmp_path)
    devices_seen: list[str | None] = []

    class FlakyAcceleratorModel:
        device = "cuda"

        def encode(self, texts, **kwargs):
            devices_seen.append(kwargs.get("device"))
            if kwargs.get("device") != "cpu":
                return np.array([[np.nan, 0.0]] * len(texts), dtype=np.float32)
            return np.array(
                [[1.0, 0.0] if i == 0 else [0.0, 1.0] for i in range(len(texts))],
                dtype=np.float32,
            )

    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: FlakyAcceleratorModel())
    monkeypatch.setattr(semantic, "_prepare_semantic_device", lambda *_args, **_kwargs: "cuda")
    monkeypatch.setattr(semantic, "validate_explicit_device_request", lambda *_a, **_k: None)

    embeddings = compute_embeddings(units, device="cuda")

    assert devices_seen == [None, "cpu"]
    assert embeddings.shape == (2, 2)
    assert np.isfinite(embeddings).all()


def test_invalid_output_cpu_retry_restarts_at_capped_batch(tmp_path: Path, monkeypatch) -> None:
    units = extract_arithmetic_units(tmp_path)
    seen_batches: list[tuple[int, str | None]] = []

    class FlakyAcceleratorModel:
        device = "cuda"

        def encode(self, texts, **kwargs):
            seen_batches.append((kwargs.get("batch_size"), kwargs.get("device")))
            if kwargs.get("device") != "cpu":
                return np.array([[np.nan, 0.0]] * len(texts), dtype=np.float32)
            return np.array(
                [[1.0, 0.0] if i == 0 else [0.0, 1.0] for i in range(len(texts))],
                dtype=np.float32,
            )

    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: FlakyAcceleratorModel())
    monkeypatch.setattr(semantic, "_prepare_semantic_device", lambda *_args, **_kwargs: "cuda")
    monkeypatch.setattr(semantic, "validate_explicit_device_request", lambda *_a, **_k: None)

    embeddings = compute_embeddings(units, device="cuda", batch_size=512)

    assert seen_batches == [(512, None), (CPU_FALLBACK_MAX_BATCH_SIZE, "cpu")]
    assert embeddings.shape == (2, 2)


def test_fresh_embeddings_are_renormalized_centrally(tmp_path: Path, monkeypatch) -> None:
    units = extract_arithmetic_units(tmp_path)
    model = _StaticEmbeddingModel(constant_embeddings(len(units), (3.0, 4.0)))
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: model)

    embeddings = compute_embeddings(units, device="cpu")

    np.testing.assert_allclose(embeddings, [[0.6, 0.8]] * len(units), atol=1e-6)


def test_compute_embeddings_retries_with_reduced_batch_before_cpu(monkeypatch, tmp_path) -> None:
    units = extract_arithmetic_units(tmp_path)
    seen_batch_sizes: list[int] = []

    class OomThenRecoverModel:
        def encode(self, texts, **kwargs):
            seen_batch_sizes.append(kwargs["batch_size"])
            if kwargs["batch_size"] > 2:
                raise RuntimeError("CUDA out of memory")
            return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: OomThenRecoverModel())

    embeddings = compute_embeddings(units, batch_size=8)

    assert embeddings.shape == (2, 2)
    assert seen_batch_sizes[:3] == [8, 4, 2]


def test_compute_embeddings_cpu_fallback_retries_once_and_bails_on_persistent_oom(
    monkeypatch, tmp_path
) -> None:
    units = extract_arithmetic_units(tmp_path)
    seen_batches: list[tuple[int, str | None]] = []

    class PersistentCpuOomModel:
        def encode(self, texts, **kwargs):
            seen_batches.append((kwargs["batch_size"], kwargs.get("device")))
            if kwargs["batch_size"] >= 2:
                raise RuntimeError("CUDA out of memory")
            if kwargs["batch_size"] >= 1:
                raise RuntimeError("CUDA out of memory")
            return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: PersistentCpuOomModel())

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        compute_embeddings(units, batch_size=8)

    assert seen_batches == [
        (8, None),
        (4, None),
        (2, None),
        (1, None),
        (8, "cpu"),
        (4, "cpu"),
        (2, "cpu"),
        (1, "cpu"),
    ]


def test_cpu_fallback_restart_batch_size_is_capped(monkeypatch, tmp_path) -> None:
    """The CPU retry after an exhausted accelerator ladder must not inherit a huge
    requested batch size: host OOM can be an uncatchable OOM-killer SIGKILL
    (observed live on WSL2 with batch_size=512), so the restart is capped at
    ``CPU_FALLBACK_MAX_BATCH_SIZE``.
    """
    units = extract_arithmetic_units(tmp_path)
    seen_batches: list[tuple[int, str | None]] = []

    class OomUntilCpuModel:
        def encode(self, texts, **kwargs):
            seen_batches.append((kwargs["batch_size"], kwargs.get("device")))
            if kwargs.get("device") != "cpu":
                raise RuntimeError("CUDA out of memory")
            return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: OomUntilCpuModel())

    embeddings = compute_embeddings(units, batch_size=512)

    assert embeddings.shape == (2, 2)
    cuda_batches = [size for size, device in seen_batches if device != "cpu"]
    assert cuda_batches == [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    assert seen_batches[-1] == (CPU_FALLBACK_MAX_BATCH_SIZE, "cpu")


def test_get_model_load_time_accelerator_oom_falls_back_to_cpu(monkeypatch) -> None:
    """An OOM raised while constructing the model on an accelerator (not while
    encoding) must retry the load on CPU rather than propagate.

    Mirrors the encode-time OOM ladder tests above
    (``test_compute_embeddings_retries_with_reduced_batch_before_cpu`` and
    ``test_compute_embeddings_cpu_fallback_retries_once_and_bails_on_persistent_oom``),
    but exercises the model-*load* fallback inside ``_get_model_unlocked``,
    which previously had no offline coverage.
    """
    calls: list[dict] = []

    class LoadTimeOomThenRecoverSentenceTransformer:
        def __init__(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            if kwargs.get("device") != "cpu":
                raise RuntimeError("CUDA out of memory. Tried to allocate 20 MiB")

    monkeypatch.setattr(semantic, "_check_semantic_dependencies", lambda: None)
    monkeypatch.setattr(semantic, "_prepare_semantic_device", lambda *_args, **_kwargs: "cuda")
    monkeypatch.setattr(
        sentence_transformers,
        "SentenceTransformer",
        LoadTimeOomThenRecoverSentenceTransformer,
    )
    semantic.clear_model_cache()

    try:
        model = semantic.get_model("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

        assert isinstance(model, LoadTimeOomThenRecoverSentenceTransformer)
        assert [call["kwargs"]["device"] for call in calls] == ["cuda", "cpu"]
        # The sticky-reuse cache key stays the *requested* device so an
        # identical later request hits this CPU-fallback instance instead of
        # retrying the accelerator load; the tracked execution device is what
        # actually ran the model.
        assert semantic._model_device_key == "cuda"
        assert semantic._model_execution_device == "cpu"
    finally:
        semantic.clear_model_cache()


@pytest.mark.parametrize(
    ("message", "active_device", "expected"),
    [
        pytest.param("CUDA out of memory. Tried to allocate 20 MiB", "cpu", "cuda", id="cuda-oom"),
        pytest.param("cuda runtime error: out of memory", "mps", "cuda", id="cuda-oom-word-order"),
        pytest.param(
            "MPS backend out of memory (MPS allocated: 1 GB)", "cpu", "mps", id="mps-oom-backend"
        ),
        pytest.param("Invalid buffer size: 123456", "mps", "mps", id="mps-invalid-buffer-size"),
        pytest.param("Metal error: out of memory", "cpu", "mps", id="mps-oom-metal-word"),
        pytest.param(
            "RuntimeError: out of memory", "cpu", "cpu", id="generic-out-of-memory-active-device"
        ),
        pytest.param(
            "cannot allocate memory", "cuda", "cuda", id="generic-cannot-allocate-active-device"
        ),
        pytest.param("some unrelated failure", "cpu", None, id="non-oom-returns-none"),
    ],
)
def test_classify_oom_device_covers_all_branches(
    message: str, active_device: str, expected: str | None
) -> None:
    assert semantic._classify_oom_device(RuntimeError(message), active_device) == expected


def test_move_model_to_cpu_casts_bf16_when_inference_policy_is_float32(monkeypatch) -> None:
    # Without the experimental opt-in the CPU inference policy is float32 on
    # every machine, so an accelerator bf16 model is always cast on the way down.
    monkeypatch.delenv("CODEDUPES_CPU_BF16", raising=False)
    module = torch.nn.Linear(4, 4).to(dtype=torch.bfloat16)

    semantic._move_model_to_cpu(module)

    assert next(module.parameters()).dtype is torch.float32
    assert str(next(module.parameters()).device) == "cpu"


def test_move_model_to_cpu_keeps_bf16_when_inference_policy_allows(monkeypatch) -> None:
    monkeypatch.setattr(semantic, "resolve_cpu_bf16_inference", lambda: True)
    module = torch.nn.Linear(4, 4).to(dtype=torch.bfloat16)

    semantic._move_model_to_cpu(module)

    assert next(module.parameters()).dtype is torch.bfloat16
    assert str(next(module.parameters()).device) == "cpu"


def test_move_model_to_cpu_leaves_float32_models_untouched() -> None:
    module = torch.nn.Linear(4, 4)

    semantic._move_model_to_cpu(module)

    assert next(module.parameters()).dtype is torch.float32


def _warm_corpus_cache(
    tmp_path: Path, monkeypatch, model_name: str = "gte-modernbert-base"
) -> list[CodeUnit]:
    """Populate the on-disk embedding cache so the corpus is fully covered.

    :param tmp_path: Per-test cache scope and corpus directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    :param model_name: Model alias to warm the cache under.
    :return: Extracted units whose embeddings are now fully cached under ``tmp_path``.
    """
    units = extract_arithmetic_units(tmp_path)
    model = WarmCacheModel()
    monkeypatch.setattr(semantic, "get_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)
    compute_embeddings(
        units,
        model_name=model_name,
        revision=FULL_REVISION,
        device="cpu",
        cache_scope=tmp_path,
    )
    return units


@pytest.mark.parametrize("model_name", ["gte-modernbert-base", "embeddinggemma-300m"])
def test_compute_embeddings_warm_cache_raises_for_explicit_unavailable_device(
    tmp_path: Path, monkeypatch, model_name: str
) -> None:
    units = _warm_corpus_cache(tmp_path, monkeypatch, model_name=model_name)

    def _raise_unavailable(*_args, **_kwargs):
        raise SemanticBackendError("cuda is not available in this environment")

    monkeypatch.setattr(semantic, "_resolve_semantic_device_request", _raise_unavailable)
    monkeypatch.setattr(semantic, "get_model", fail_if_called)

    with pytest.raises(SemanticBackendError):
        compute_embeddings(
            units,
            model_name=model_name,
            revision=FULL_REVISION,
            device="cuda",
            cache_scope=tmp_path,
        )


def test_compute_embeddings_empty_corpus_raises_for_explicit_unavailable_device(
    tmp_path: Path, monkeypatch
) -> None:
    def _raise_unavailable(*_args, **_kwargs):
        raise SemanticBackendError("mps is not available in this environment")

    monkeypatch.setattr(semantic, "_resolve_semantic_device_request", _raise_unavailable)
    monkeypatch.setattr(semantic, "get_model", fail_if_called)

    with pytest.raises(SemanticBackendError):
        compute_embeddings([], device="mps", cache_scope=tmp_path)


def test_find_similar_to_query_warm_cache_raises_for_explicit_unavailable_device(
    tmp_path: Path, monkeypatch
) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    model = WarmCacheModel()
    monkeypatch.setattr(semantic, "get_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    find_similar_to_query(
        "find addition",
        units,
        embeddings,
        model_name="gte-modernbert-base",
        revision=FULL_REVISION,
        device="cpu",
        threshold=0.0,
        cache_scope=tmp_path,
    )

    def _raise_unavailable(*_args, **_kwargs):
        raise SemanticBackendError("cuda is not available in this environment")

    monkeypatch.setattr(semantic, "_resolve_semantic_device_request", _raise_unavailable)
    monkeypatch.setattr(semantic, "get_model", fail_if_called)

    with pytest.raises(SemanticBackendError):
        find_similar_to_query(
            "find addition",
            units,
            embeddings,
            model_name="gte-modernbert-base",
            revision=FULL_REVISION,
            device="cuda",
            cache_scope=tmp_path,
        )


def test_warm_cache_returns_restore_managed_mps_cap_without_allocator_work(
    tmp_path: Path, monkeypatch
) -> None:
    units = _warm_corpus_cache(tmp_path, monkeypatch)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    find_similar_to_query(
        "find addition",
        units,
        embeddings,
        model_name="gte-modernbert-base",
        revision=FULL_REVISION,
        device="cpu",
        threshold=0.0,
        cache_scope=tmp_path,
    )

    restore_calls: list[bool] = []
    monkeypatch.setattr(
        semantic,
        "restore_mps_memory_fraction_if_managed",
        lambda: restore_calls.append(True),
    )
    monkeypatch.setattr(semantic, "get_model", fail_if_called)
    monkeypatch.setattr(semantic, "_prepare_semantic_device", fail_if_called)

    # Fully cache-covered corpus run: the warm return must still restore a
    # previously managed allocator cap when this run leaves the fraction unset.
    compute_embeddings(
        units,
        model_name="gte-modernbert-base",
        revision=FULL_REVISION,
        device="cpu",
        cache_scope=tmp_path,
    )
    assert restore_calls == [True]

    # Warm query hit: same contract on the search path.
    restore_calls.clear()
    find_similar_to_query(
        "find addition",
        units,
        embeddings,
        model_name="gte-modernbert-base",
        revision=FULL_REVISION,
        device="cpu",
        threshold=0.0,
        cache_scope=tmp_path,
    )
    assert restore_calls == [True]

    # A CPU run's supplied fraction is ignored. Because the warm return performs
    # no allocator work, it must still restore a cap left by an earlier run.
    restore_calls.clear()
    compute_embeddings(
        units,
        model_name="gte-modernbert-base",
        revision=FULL_REVISION,
        device="cpu",
        mps_memory_fraction=0.9,
        cache_scope=tmp_path,
    )
    assert restore_calls == [True]

    restore_calls.clear()
    find_similar_to_query(
        "find addition",
        units,
        embeddings,
        model_name="gte-modernbert-base",
        revision=FULL_REVISION,
        device="cpu",
        mps_memory_fraction=0.9,
        threshold=0.0,
        cache_scope=tmp_path,
    )
    assert restore_calls == [True]


@pytest.mark.parametrize(
    ("platform_name", "device", "expects_resolution"),
    [
        ("darwin", "auto", False),
        ("darwin", "cpu", False),
        ("linux", "cpu", False),
        ("linux", "auto", True),
    ],
)
def test_warm_cache_device_resolution_matches_dtype_policy(
    tmp_path: Path,
    monkeypatch,
    platform_name: str,
    device: str,
    expects_resolution: bool,
) -> None:
    """Warm-cache keying resolves a concrete device only when the dtype may differ.

    On darwin, ``auto`` can only pick MPS or CPU and both share the float32
    key space, so no resolution (and no torch import) is needed. Off darwin,
    ``auto`` may select CUDA and its bfloat16 dtype namespace, so the device
    must be resolved before the cache key is trustworthy.
    """
    units = _warm_corpus_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(semantic.sys, "platform", platform_name)

    resolution_calls = {"count": 0}

    def _count_and_resolve(*_args, **_kwargs) -> str:
        resolution_calls["count"] += 1
        return "cpu"

    monkeypatch.setattr(semantic, "_resolve_semantic_device_request", _count_and_resolve)
    monkeypatch.setattr(semantic, "get_model", fail_if_called)

    result = compute_embeddings(
        units,
        model_name="gte-modernbert-base",
        revision=FULL_REVISION,
        device=device,
        cache_scope=tmp_path,
    )

    assert result.shape == (len(units), 2)
    assert (resolution_calls["count"] > 0) == expects_resolution


def test_runtime_env_configured_before_capability_probe_can_import_torch(
    tmp_path: Path, monkeypatch
) -> None:
    """The MPS fallback variable is set before any torch-importing probe runs.

    The first darwin ``auto`` invocation with no machine capability record
    derives a cache variant, which can probe CPU capabilities and import
    torch. ``PYTORCH_ENABLE_MPS_FALLBACK`` must already be configured at that
    moment - this is a pure initialization-order check; real fallback
    behavior stays in the live MPS suite.
    """
    monkeypatch.setattr(semantic.sys, "platform", "darwin")
    monkeypatch.setenv("CODEDUPES_CACHE_DIR", str(tmp_path / "cache"))
    # The opt-in makes darwin-auto variant derivation consult the capability
    # gate, which is the torch-importing probe this ordering test exists for.
    monkeypatch.setenv("CODEDUPES_CPU_BF16", "1")
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    env_at_torch_probe: list[str | None] = []
    real_load_torch = devices._load_torch

    def _spying_load_torch():
        env_at_torch_probe.append(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
        return real_load_torch()

    monkeypatch.setattr(devices, "_load_torch", _spying_load_torch)

    semantic.resolve_embedding_space_identity(device="auto")

    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    assert env_at_torch_probe, "expected the capability probe to require torch"
    assert all(value == "1" for value in env_at_torch_probe)


class _BfloatAcceleratorOomModel:
    """Fake bf16 model with configurable successful encodes before accelerator OOM."""

    def __init__(self, *, successful_encodes_before_oom: int = 0) -> None:
        self._dtype = torch.bfloat16
        self._successful_encodes_before_oom = successful_encodes_before_oom

    def parameters(self):
        yield torch.zeros(1, dtype=self._dtype)

    def to(self, device=None, dtype=None):
        if dtype is not None:
            self._dtype = dtype
        return self

    def encode(self, texts, **kwargs):
        if self._successful_encodes_before_oom:
            self._successful_encodes_before_oom -= 1
        elif kwargs.get("device") != "cpu":
            raise RuntimeError("CUDA out of memory")
        return constant_embeddings(len(texts), (1.0, 0.0))


def test_dtype_diverging_accelerator_fallback_skips_bf16_keyed_cache_write(
    tmp_path: Path, monkeypatch
) -> None:
    # No MPS is touched anywhere here: torch.cuda.is_bf16_supported is the
    # only stub, matching the repo's existing convention for exercising
    # CUDA-only branches on a CUDA-less host (see
    # test_cuda_bf16_selection_excludes_emulated_support).
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **_kwargs: True)
    monkeypatch.setattr(semantic, "_resolve_semantic_device_request", lambda *_a, **_k: "cuda")
    units = extract_arithmetic_units(tmp_path)
    model = _BfloatAcceleratorOomModel()
    monkeypatch.setattr(semantic, "get_model", lambda *_a, **_k: model)

    profile = semantic.resolve_model_profile("gte-modernbert-base")
    plan = semantic.resolve_encode_plan("gte-modernbert-base", mode="code")
    bf16_variant = semantic._cache_variant_for(
        profile,
        "cuda",
        plan,
        mps_fallback=None,
        trust_remote_code=False,
        resolved_device="cuda",
    )
    assert "dtype=torch.bfloat16" in bf16_variant
    bf16_namespace = semantic._embedding_cache_namespace("code", bf16_variant)

    put_calls: list[dict] = []
    original_put_many = EmbeddingCache.put_many

    def _recording_put_many(self, *args, **kwargs):
        put_calls.append({"args": args, "kwargs": kwargs})
        return original_put_many(self, *args, **kwargs)

    monkeypatch.setattr(EmbeddingCache, "put_many", _recording_put_many)

    embeddings = compute_embeddings(
        units,
        model_name="gte-modernbert-base",
        revision=FULL_REVISION,
        device="cuda",
        batch_size=1,
        cache_scope=tmp_path,
    )

    assert embeddings.shape[0] == len(units)
    # A dtype-diverging fallback (bf16 CUDA -> float32 CPU under the default
    # no-opt-in policy) must never write float32 vectors under the bf16-keyed
    # namespace: the coherence-restart discards that run and recomputes under
    # a fresh, correctly-keyed identity instead, so *some* write is expected -
    # just never one landing in the original bf16 key space.
    bf16_writes = [call for call in put_calls if call["kwargs"].get("namespace") == bf16_namespace]
    assert bf16_writes == []
    assert len(put_calls) == 1


def test_cpu_restarted_accelerator_corpus_stays_searchable(tmp_path: Path, monkeypatch) -> None:
    """A CUDA-bf16 corpus that restarted faithfully on CPU keeps working for search.

    Reviewer repro: the coherence restart used to record a CPU identity that
    only the MPS fast-math branch of the query-space check could rediscover,
    so a CUDA-fallback corpus raised "reindex" forever. The CPU-policy retry
    must now engage for every accelerator request.
    """
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **_kwargs: True)
    monkeypatch.setattr(
        semantic,
        "_resolve_semantic_device_request",
        lambda device, **_k: "cpu" if device == "cpu" else "cuda",
    )
    units = extract_arithmetic_units(tmp_path)
    model = _BfloatAcceleratorOomModel()
    monkeypatch.setattr(semantic, "get_model", lambda *_a, **_k: model)

    embeddings, identity = semantic.compute_embeddings_with_identity(
        units,
        model_name="gte-modernbert-base",
        revision=FULL_REVISION,
        device="cuda",
        batch_size=1,
        cache_scope=tmp_path,
    )
    # The OOM fallback cast bf16 to float32, so the whole corpus restarted
    # under the faithful CPU identity.
    assert "dtype=torch.bfloat16" not in identity.runtime_variant

    query_device = semantic._require_current_embedding_space(
        identity,
        model_name="gte-modernbert-base",
        instruction_prefix=None,
        revision=FULL_REVISION,
        trust_remote_code=None,
        semantic_task=semantic.DEFAULT_CHECK_SEMANTIC_TASK,
        device="cuda",
        mps_fallback=None,
        persist_local_model_manifest=True,
    )
    assert query_device == "cpu"

    hits = find_similar_to_query(
        "add two numbers",
        units,
        embeddings,
        model_name="gte-modernbert-base",
        revision=FULL_REVISION,
        semantic_task=semantic.DEFAULT_CHECK_SEMANTIC_TASK,
        device=query_device,
        threshold=0.0,
        cache_scope=tmp_path,
        corpus_identity=identity,
    )
    assert hits


def test_query_dtype_fallback_never_reaches_the_dot_product(tmp_path: Path, monkeypatch) -> None:
    """A query cast to float32 mid-encode must not be compared with a bf16 corpus.

    Reviewer repro: the corpus embeds successfully under CUDA-bf16, the query
    encode OOMs down to a CPU float32 cast, and the similarity comparison
    used to proceed anyway because the compatibility check rebuilt the
    identity from the requested device policy. The live-dtype check must
    abort before the dot product.
    """
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **_kwargs: True)
    monkeypatch.setattr(
        semantic,
        "_resolve_semantic_device_request",
        lambda device, **_k: "cpu" if device == "cpu" else "cuda",
    )
    units = extract_arithmetic_units(tmp_path)
    model = _BfloatAcceleratorOomModel(successful_encodes_before_oom=1)
    monkeypatch.setattr(semantic, "get_model", lambda *_a, **_k: model)

    embeddings, identity = semantic.compute_embeddings_with_identity(
        units,
        model_name="gte-modernbert-base",
        revision=FULL_REVISION,
        device="cuda",
        batch_size=8,
        cache_scope=tmp_path,
    )
    assert "dtype=torch.bfloat16" in identity.runtime_variant

    with pytest.raises(RuntimeError, match="bfloat16 policy"):
        find_similar_to_query(
            "add two numbers",
            units,
            embeddings,
            model_name="gte-modernbert-base",
            revision=FULL_REVISION,
            semantic_task=semantic.DEFAULT_CHECK_SEMANTIC_TASK,
            device="cuda",
            threshold=0.0,
            cache_scope=tmp_path,
            corpus_identity=identity,
        )
    assert str(next(iter(model.parameters())).dtype) == "torch.float32"


def test_dimension_mismatch_reencode_reads_live_device(tmp_path: Path, monkeypatch) -> None:
    """Regression test: the dimension-mismatch re-encode must observe the live
    effective device, not the value captured before the first encode call.

    Previously ``execution_device`` was captured once and only read again
    inside a fast-math-specific check; a mid-encode accelerator fallback that
    changed the model's real device was not reflected for the second
    ``_encode_miss_texts`` call, which could misclassify a later CPU
    allocator failure as an accelerator OOM.
    """
    units = extract_arithmetic_units(tmp_path)

    class DriftingDeviceModel:
        """Model whose reported ``.device`` flips to cpu mid-first-encode."""

        max_seq_length = 3
        tokenizer = WhitespaceTokenizer()

        def __init__(self) -> None:
            self.device = "cuda"

        def encode(self, texts, **kwargs):
            self.device = "cpu"
            return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)

    model = DriftingDeviceModel()
    monkeypatch.setattr(semantic, "get_model", lambda *_a, **_k: model)
    monkeypatch.setattr(semantic, "_resolve_semantic_device_request", lambda *_a, **_k: "cuda")

    recorded_initial_devices: list[str] = []
    original_encode_with_retries = semantic._encode_with_retries

    def _recording_encode_with_retries(*args, **kwargs):
        recorded_initial_devices.append(kwargs["initial_device"])
        return original_encode_with_retries(*args, **kwargs)

    monkeypatch.setattr(semantic, "_encode_with_retries", _recording_encode_with_retries)

    profile = semantic.resolve_model_profile("gte-modernbert-base")
    plan = semantic.resolve_encode_plan("gte-modernbert-base", mode="code")
    cache, cache_revision, cache_variant, cache_namespace = semantic._prepare_cache_context(
        "code",
        profile,
        "gte-modernbert-base",
        FULL_REVISION,
        "cuda",
        plan,
        mps_fallback=None,
        trust_remote_code=False,
        use_cache=True,
        cache_scope=tmp_path,
    )
    assert cache is not None
    assert cache_revision is not None
    prepared_texts = [unit.source.strip() for unit in units]
    cache_keys = [
        semantic.compute_cache_key(
            profile.canonical_name, cache_revision, text, variant=cache_variant
        )
        for text in prepared_texts
    ]
    # Seed a mismatched-dimensionality hit for the first unit so the live
    # model's real 2-dim output forces the dimension-mismatch re-encode.
    cache.put_many(
        tmp_path,
        profile.canonical_name,
        cache_revision,
        [(cache_keys[0], np.array([1.0, 0.0, 0.0], dtype=np.float32))],
        namespace=cache_namespace,
    )

    diagnostics = []
    embeddings = compute_embeddings(
        units,
        model_name="gte-modernbert-base",
        revision=FULL_REVISION,
        device="cuda",
        cache_scope=tmp_path,
        diagnostics=diagnostics,
    )

    assert recorded_initial_devices == ["cuda", "cpu"]
    assert embeddings.shape == (2, 2)
    # Recovery re-encodes the previously cached unit as well as the initial
    # miss. Both overflow warnings must appear exactly once.
    assert len(diagnostics) == 2
    assert {diagnostic.lineno for diagnostic in diagnostics} == {unit.lineno for unit in units}
    assert all(diagnostic.code == "semantic-context-overflow" for diagnostic in diagnostics)
