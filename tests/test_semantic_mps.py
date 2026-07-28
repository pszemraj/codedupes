from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import sentence_transformers

from codedupes import semantic
from tests.conftest import extract_arithmetic_units


def _reset_model_cache_state() -> None:
    semantic._model = None
    semantic._model_name = None
    semantic._model_revision = None
    semantic._model_trust_remote_code = None
    semantic._model_device_key = None
    semantic._model_execution_device = None
    semantic._warned_mlx_mps_contention = False
    semantic._warned_cpu_fallback_reuse = False


def _prepare_device(device, *, mps_fallback, mps_memory_fraction):
    del mps_fallback, mps_memory_fraction
    return "cpu" if device == "auto" else device


@pytest.fixture(autouse=True)
def _isolate_model_cache():
    _reset_model_cache_state()
    yield
    _reset_model_cache_state()


def test_model_cache_is_keyed_by_resolved_device(monkeypatch) -> None:
    calls: list[str] = []
    moved: list[str] = []

    class FakeSentenceTransformer:
        def __init__(self, _model_name, **kwargs) -> None:
            self.device = kwargs["device"]
            calls.append(self.device)

        def to(self, device: str) -> None:
            self.device = device
            moved.append(device)

    monkeypatch.setattr(semantic, "_configure_semantic_runtime_env", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(semantic, "_check_semantic_dependencies", lambda: None)
    monkeypatch.setattr(semantic, "_prepare_semantic_device", _prepare_device)
    monkeypatch.setattr(semantic, "clear_device_cache", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)

    first = semantic.get_model("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    again = semantic.get_model("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    mps_model = semantic.get_model("sentence-transformers/all-MiniLM-L6-v2", device="mps")

    assert first is again
    assert mps_model is not first
    assert calls == ["cpu", "mps"]
    assert moved == []


def test_mps_model_load_oom_retries_once_on_cpu(monkeypatch) -> None:
    calls: list[str] = []
    cache_clears: list[str] = []

    class FakeModel:
        device = "cpu"

    def fake_constructor(_model_name, **kwargs):
        calls.append(kwargs["device"])
        if kwargs["device"] == "mps":
            raise RuntimeError("MPS backend out of memory")
        return FakeModel()

    monkeypatch.setattr(semantic, "_configure_semantic_runtime_env", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(semantic, "_check_semantic_dependencies", lambda: None)
    monkeypatch.setattr(semantic, "_prepare_semantic_device", _prepare_device)
    monkeypatch.setattr(
        semantic,
        "clear_device_cache",
        lambda device, **_kwargs: cache_clears.append(device) or True,
    )
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fake_constructor)

    model = semantic.get_model("sentence-transformers/all-MiniLM-L6-v2", device="mps")

    assert isinstance(model, FakeModel)
    assert calls == ["mps", "cpu"]
    assert cache_clears == ["mps"]
    assert semantic._model_device_key == "mps"
    assert semantic._model_execution_device == "cpu"


def test_sticky_cpu_fallback_reuse_warns_once(monkeypatch, caplog) -> None:
    class FakeModel:
        device = "cpu"

    def fake_constructor(_model_name, **kwargs):
        if kwargs["device"] == "mps":
            raise RuntimeError("MPS backend out of memory")
        return FakeModel()

    monkeypatch.setattr(semantic, "_configure_semantic_runtime_env", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(semantic, "_check_semantic_dependencies", lambda: None)
    monkeypatch.setattr(semantic, "_prepare_semantic_device", _prepare_device)
    monkeypatch.setattr(semantic, "clear_device_cache", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fake_constructor)

    with caplog.at_level("WARNING"):
        semantic.get_model("sentence-transformers/all-MiniLM-L6-v2", device="mps")
        semantic.get_model("sentence-transformers/all-MiniLM-L6-v2", device="mps")
        semantic.get_model("sentence-transformers/all-MiniLM-L6-v2", device="mps")

    assert caplog.text.count("after an earlier mps-to-CPU OOM fallback") == 1


def test_mps_embedding_oom_halves_batch_then_moves_model_to_cpu(
    monkeypatch,
    tmp_path: Path,
) -> None:
    units = extract_arithmetic_units(tmp_path)
    attempts: list[tuple[int, str | None]] = []
    cache_clears: list[str] = []
    recovery_events: list[str] = []

    class OomModel:
        def __init__(self) -> None:
            self.device = "mps"
            self.moves: list[str] = []

        def to(self, device: str) -> None:
            self.device = device
            self.moves.append(device)
            recovery_events.append(f"move:{device}")

        def encode(self, texts, **kwargs):
            selected_device = kwargs.get("device")
            attempts.append((kwargs["batch_size"], selected_device))
            if selected_device != "cpu":
                raise RuntimeError("MPS backend out of memory")
            return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    model = OomModel()
    monkeypatch.setattr(semantic, "_prepare_semantic_device", _prepare_device)
    monkeypatch.setattr(semantic, "get_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(semantic, "format_mps_memory_snapshot", lambda: "test-memory")
    monkeypatch.setattr(
        semantic,
        "clear_device_cache",
        lambda device, **_kwargs: (
            cache_clears.append(device),
            recovery_events.append(f"clear:{device}"),
            True,
        )[-1],
    )

    embeddings = semantic.compute_embeddings(units, device="mps", batch_size=8)

    assert embeddings.shape == (2, 2)
    assert attempts == [(8, None), (4, None), (2, None), (1, None), (8, "cpu")]
    assert model.moves == ["cpu"]
    assert cache_clears == ["mps", "mps", "mps", "mps"]
    assert recovery_events[-2:] == ["clear:mps", "move:cpu"]


def test_mps_query_oom_uses_shared_cpu_recovery(monkeypatch, tmp_path: Path) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    attempts: list[str | None] = []

    class QueryModel:
        def __init__(self) -> None:
            self.device = "mps"

        def to(self, device: str) -> None:
            self.device = device

        def encode(self, texts, **kwargs):
            del texts
            attempts.append(kwargs.get("device"))
            if kwargs.get("device") != "cpu":
                raise RuntimeError("Metal out of memory")
            return np.array([[1.0, 0.0]], dtype=np.float32)

    model = QueryModel()
    monkeypatch.setattr(semantic, "_prepare_semantic_device", _prepare_device)
    monkeypatch.setattr(semantic, "get_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(semantic, "clear_device_cache", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(semantic, "format_mps_memory_snapshot", lambda: "test-memory")

    results = semantic.find_similar_to_query(
        "addition",
        units,
        embeddings,
        device="mps",
        threshold=0.0,
    )

    assert len(results) == 2
    assert attempts == [None, "cpu"]


def test_mps_profile_dtypes_are_float32(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_bf16_supported() -> bool:
            return True

    class FakeTorch:
        bfloat16 = "bf16"
        float32 = "fp32"
        cuda = FakeCuda()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    assert semantic._resolve_embeddinggemma_torch_dtype("mps") == "fp32"


def test_encode_texts_does_not_hide_unrelated_type_error() -> None:
    calls = 0

    def broken_encode(_texts, **_kwargs):
        nonlocal calls
        calls += 1
        raise TypeError("internal tensor type mismatch")

    try:
        semantic._encode_texts(
            broken_encode,
            ["code"],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device="cpu",
        )
    except TypeError as exc:
        assert "tensor type mismatch" in str(exc)
    else:
        raise AssertionError("Expected TypeError")

    assert calls == 1


def test_clear_model_cache_drops_mps_model_without_cpu_migration(monkeypatch) -> None:
    events: list[str] = []

    class FakeModel:
        def to(self, device: str) -> None:
            events.append(f"move:{device}")

    semantic._model = FakeModel()
    semantic._model_name = "fake"
    semantic._model_revision = None
    semantic._model_trust_remote_code = False
    semantic._model_device_key = "mps"
    semantic._model_execution_device = "mps"
    monkeypatch.setattr(
        semantic,
        "clear_device_cache",
        lambda device, **_kwargs: events.append(f"clear:{device}") or True,
    )

    semantic.clear_model_cache()

    assert events == ["clear:mps"]
    assert semantic._model is None


def test_mps_mlx_contention_warning_is_emitted_once(monkeypatch, caplog) -> None:
    monkeypatch.setattr(semantic, "_configure_semantic_runtime_env", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(semantic, "resolve_semantic_device", lambda _device: "mps")
    monkeypatch.setattr(semantic, "configure_mps_memory_fraction", lambda *_args: None)
    monkeypatch.setattr(semantic, "is_mlx_loaded", lambda: True)

    with caplog.at_level("WARNING"):
        semantic._prepare_semantic_device(
            "mps",
            mps_fallback=None,
            mps_memory_fraction=None,
        )
        semantic._prepare_semantic_device(
            "mps",
            mps_fallback=None,
            mps_memory_fraction=None,
        )

    assert caplog.text.count("MLX is already loaded") == 1
