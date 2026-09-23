"""Model loading, dtype resolution, and local-directory reload behavior for get_model()."""

from __future__ import annotations

import inspect
import logging
import stat
from pathlib import Path

import numpy as np
import pytest
import sentence_transformers
import torch

from codedupes import devices, semantic
from codedupes.semantic import (
    SemanticBackendError,
    compute_embeddings,
)
from tests.conftest import extract_arithmetic_units


def _recording_sentence_transformer(calls: list[dict]) -> type:
    """Build a SentenceTransformer double that records constructor invocations.

    :param calls: Sink receiving one ``{"args", "kwargs"}`` entry per construction.
    :return: Recording stand-in class.
    """

    class RecordingSentenceTransformer:
        def __init__(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})

    return RecordingSentenceTransformer


def test_cuda_bf16_selection_excludes_emulated_support(monkeypatch) -> None:
    recorded_kwargs: dict = {}

    def fake_is_bf16_supported(**kwargs):
        recorded_kwargs.update(kwargs)
        return True

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", fake_is_bf16_supported)

    assert semantic._resolve_model_dtype("test-model", "cuda") is torch.bfloat16
    # Pre-Ampere GPUs pass torch's default emulation probe; the policy must ask
    # for native support only.
    assert recorded_kwargs == {"including_emulation": False}


def test_resolve_model_dtype_cpu_follows_inference_policy(monkeypatch) -> None:
    monkeypatch.setattr(semantic, "resolve_cpu_bf16_inference", lambda: True)
    assert semantic._resolve_model_dtype("test-model", "cpu") is torch.bfloat16

    monkeypatch.setattr(semantic, "resolve_cpu_bf16_inference", lambda: False)
    assert semantic._resolve_model_dtype("test-model", "cpu") is torch.float32


def test_model_cache_reloads_when_dtype_policy_changes(monkeypatch) -> None:
    """Hardening (round-2 review): flipping the CPU bf16 policy mid-process is not
    a supported lifecycle, but if it happens the process model cache must reload
    under the newly pinned dtype rather than serve the stale instance - stale
    reuse is what could answer a float32 key space with bfloat16 weights, or send
    the coherence restart into unbounded recursion against the same cached model."""
    calls: list[dict] = []
    monkeypatch.setattr(semantic, "_check_semantic_dependencies", lambda: None)
    monkeypatch.setattr(semantic, "_prepare_semantic_device", lambda *_args, **_kwargs: "cpu")
    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", _recording_sentence_transformer(calls)
    )
    semantic.clear_model_cache()

    monkeypatch.setattr(semantic, "resolve_cpu_bf16_inference", lambda: False)
    first = semantic.get_model("sentence-transformers/all-MiniLM-L6-v2")
    assert semantic.get_model("sentence-transformers/all-MiniLM-L6-v2") is first
    assert len(calls) == 1
    assert calls[0]["kwargs"]["model_kwargs"]["dtype"] is torch.float32

    monkeypatch.setattr(semantic, "resolve_cpu_bf16_inference", lambda: True)
    second = semantic.get_model("sentence-transformers/all-MiniLM-L6-v2")
    assert second is not first
    assert len(calls) == 2
    assert calls[1]["kwargs"]["model_kwargs"]["dtype"] is torch.bfloat16

    # An unchanged policy keeps hitting: no reload churn on the supported path.
    assert semantic.get_model("sentence-transformers/all-MiniLM-L6-v2") is second
    assert len(calls) == 2
    semantic.clear_model_cache()


def test_resolve_model_dtype_cpu_stays_float32_without_opt_in(monkeypatch) -> None:
    # Even on a gate-passing machine, automatic CPU bf16 is unvalidated: the
    # experimental CODEDUPES_CPU_BF16=1 opt-in is required for the positive path.
    monkeypatch.delenv("CODEDUPES_CPU_BF16", raising=False)
    monkeypatch.setattr(devices, "resolve_cpu_bf16_native", lambda: True)

    assert semantic._resolve_model_dtype("test-model", "cpu") is torch.float32

    monkeypatch.setenv("CODEDUPES_CPU_BF16", "1")
    assert semantic._resolve_model_dtype("test-model", "cpu") is torch.bfloat16


def test_resolve_model_dtype_cpu_opted_in_never_writes_cache_root(tmp_path, monkeypatch) -> None:
    # The CPU bf16 capability gate is a live, per-process, in-memory probe with
    # no on-disk record (third-party review Issue 3): even an opted-in dtype
    # resolution that loads a model on CPU must leave the cache root untouched.
    pytest.importorskip("torch")
    monkeypatch.setenv("CODEDUPES_CPU_BF16", "1")
    monkeypatch.setenv("CODEDUPES_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CODEDUPES_NO_CACHE", raising=False)
    devices._reset_cpu_bf16_probe_cache()

    semantic._resolve_model_dtype("test-model", "cpu")

    assert not (tmp_path / "cache").exists()


def test_resolve_model_dtype_mps_always_float32_regardless_of_cpu_policy(monkeypatch) -> None:
    # MPS is never CPU: the CPU inference policy must not leak into the MPS branch.
    monkeypatch.setattr(semantic, "resolve_cpu_bf16_inference", lambda: True)
    assert semantic._resolve_model_dtype("test-model", "mps") is torch.float32


def test_dtype_variant_for_mps_is_always_empty(monkeypatch) -> None:
    monkeypatch.setattr(semantic, "resolve_cpu_bf16_inference", lambda: True)
    profile = semantic.resolve_model_profile("gte-modernbert-base")

    assert semantic._dtype_variant_for(profile, "mps", mps_fallback=None) == ""


def test_dtype_variant_for_cpu_follows_inference_policy(monkeypatch) -> None:
    profile = semantic.resolve_model_profile("gte-modernbert-base")

    monkeypatch.setattr(semantic, "resolve_cpu_bf16_inference", lambda: False)
    assert semantic._dtype_variant_for(profile, "cpu", mps_fallback=None) == ""

    monkeypatch.setattr(semantic, "resolve_cpu_bf16_inference", lambda: True)
    assert semantic._dtype_variant_for(profile, "cpu", mps_fallback=None) == "dtype=torch.bfloat16"


def test_dtype_variant_for_auto_on_darwin_skips_resolution_when_policy_float32(
    monkeypatch,
) -> None:
    profile = semantic.resolve_model_profile("gte-modernbert-base")
    monkeypatch.setattr(semantic, "resolve_cpu_bf16_inference", lambda: False)
    monkeypatch.setattr(semantic.sys, "platform", "darwin")

    def fail_if_called(*_a, **_k):
        raise AssertionError("must not resolve a concrete device when the CPU gate is false")

    monkeypatch.setattr(semantic, "_resolve_semantic_device_request", fail_if_called)

    assert semantic._dtype_variant_for(profile, "auto", mps_fallback=None) == ""


def test_dtype_variant_matches_pre_capability_gate_baseline_without_opt_in(
    monkeypatch,
) -> None:
    # Without the experimental CODEDUPES_CPU_BF16 opt-in, cpu/mps/darwin-auto
    # must key byte-identically to the pre-capability-gate policy (empty
    # variant) on every machine, gate-passing or not:
    # The CPU opt-in itself does not split the faithful float32 baseline, so
    # old and new code must agree here or warm caches would silently miss.
    monkeypatch.delenv("CODEDUPES_CPU_BF16", raising=False)
    monkeypatch.setattr(semantic.sys, "platform", "darwin")
    profile = semantic.resolve_model_profile("gte-modernbert-base")

    assert semantic._dtype_variant_for(profile, "cpu", mps_fallback=None) == ""
    assert semantic._dtype_variant_for(profile, "mps", mps_fallback=None) == ""
    assert semantic._dtype_variant_for(profile, "auto", mps_fallback=None) == ""


# --- Finding 2: canonical "could resolve to MPS" predicate -----------------


@pytest.mark.parametrize(
    # The full priority matrix is authoritative in
    # test_devices.py::test_could_resolve_to_mps_mirrors_auto_resolution_priority;
    # these two rows only prove delegation (one true, one false case).
    ("device", "platform_name", "expect_mps_possible"),
    [
        ("auto", "darwin", True),
        ("cpu", "darwin", False),
    ],
)
def test_mps_fast_math_variant_matches_could_resolve_to_mps(
    monkeypatch, device, platform_name, expect_mps_possible
) -> None:
    """``_mps_fast_math_variant`` must gate on the same predicate as devices.py."""
    monkeypatch.setattr(devices.sys, "platform", platform_name)
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "1")

    variant = semantic._mps_fast_math_variant(device)

    assert bool(variant) is expect_mps_possible
    assert devices.could_resolve_to_mps(device) is expect_mps_possible


@pytest.mark.parametrize(
    ("platform_name", "expect_mps_possible"),
    [
        ("darwin", True),
        ("linux", False),
        ("win32", False),
    ],
)
def test_dtype_variant_for_auto_branch_matches_could_resolve_to_mps(
    monkeypatch, platform_name, expect_mps_possible
) -> None:
    """The "auto" dtype shortcut fires exactly when could_resolve_to_mps("auto") does."""
    profile = semantic.resolve_model_profile("gte-modernbert-base")
    monkeypatch.setattr(semantic, "resolve_cpu_bf16_inference", lambda: False)
    monkeypatch.setattr(semantic.sys, "platform", platform_name)
    assert devices.could_resolve_to_mps("auto") is expect_mps_possible

    if expect_mps_possible:

        def fail_if_called(*_a, **_k):
            raise AssertionError("must not resolve a concrete device when the MPS shortcut applies")

        monkeypatch.setattr(semantic, "_resolve_semantic_device_request", fail_if_called)
        assert semantic._dtype_variant_for(profile, "auto", mps_fallback=None) == ""
    else:
        # Falls through to concrete-device resolution instead of short-circuiting.
        monkeypatch.setattr(semantic, "_resolve_semantic_device_request", lambda *_a, **_k: "cpu")
        monkeypatch.setattr(semantic, "_resolve_model_dtype", lambda *_a, **_k: torch.bfloat16)
        assert (
            semantic._dtype_variant_for(profile, "auto", mps_fallback=None)
            == "dtype=torch.bfloat16"
        )


@pytest.mark.parametrize(
    ("revision", "trust_remote_code"),
    [
        pytest.param(None, None, id="safe-defaults"),
        pytest.param("test-revision", True, id="trusted-revision"),
        pytest.param("test-revision", False, id="untrusted-revision"),
    ],
)
def test_get_model_passes_revision_and_trust_options(
    monkeypatch,
    revision: str | None,
    trust_remote_code: bool | None,
) -> None:
    calls: list[dict] = []

    monkeypatch.setattr(semantic, "_check_semantic_dependencies", lambda: None)
    monkeypatch.setattr(semantic, "_prepare_semantic_device", lambda *_args, **_kwargs: "cpu")
    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", _recording_sentence_transformer(calls)
    )
    semantic.clear_model_cache()

    try:
        semantic.get_model(
            "sentence-transformers/all-MiniLM-L6-v2",
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        assert len(calls) == 1
        kwargs = calls[0]["kwargs"]
        expected_trust = trust_remote_code is True
        assert kwargs["trust_remote_code"] is expected_trust

        # Every load pins an explicit dtype so checkpoint-declared float16 configs
        # cannot leak into inference (transformers 5 defaults dtype="auto").
        assert kwargs["model_kwargs"]["dtype"] is torch.float32

        if revision is None:
            assert "revision" not in kwargs
            assert kwargs["model_kwargs"] == {"dtype": torch.float32}
            assert "processor_kwargs" not in kwargs
            assert "config_kwargs" not in kwargs
            return

        assert kwargs["revision"] == revision
        for nested_name in ("model_kwargs", "processor_kwargs", "config_kwargs"):
            nested = kwargs[nested_name]
            assert nested["revision"] == revision
            if expected_trust:
                assert nested["trust_remote_code"] is True
            else:
                assert "trust_remote_code" not in nested
    finally:
        semantic.clear_model_cache()


def test_constructor_kwargs_bind_to_real_sentence_transformer_signature() -> None:
    # The recording double above swallows **kwargs, so nothing else binds the
    # kwarg names to the installed SentenceTransformer, whose __init__ takes no
    # **kwargs; a renamed or misspelled key would otherwise only fail on a real
    # model load.
    parameters = inspect.signature(sentence_transformers.SentenceTransformer.__init__).parameters
    for name in (
        "revision",
        "trust_remote_code",
        "model_kwargs",
        "processor_kwargs",
        "config_kwargs",
    ):
        assert name in parameters


def test_get_model_loads_local_directory_without_hub_revision(tmp_path: Path, monkeypatch) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}')
    (model_dir / "model.safetensors").write_text("weights")
    calls: list[dict] = []

    monkeypatch.setattr(semantic, "_check_semantic_dependencies", lambda: None)
    monkeypatch.setattr(semantic, "_prepare_semantic_device", lambda *_args, **_kwargs: "cpu")
    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", _recording_sentence_transformer(calls)
    )
    semantic.clear_model_cache()

    try:
        semantic.get_model(str(model_dir), revision="ignored-local-revision")

        assert calls == [
            {
                "args": (str(model_dir.resolve()),),
                "kwargs": {
                    "trust_remote_code": False,
                    "device": "cpu",
                    "local_files_only": True,
                    "model_kwargs": {"dtype": torch.float32},
                },
            }
        ]
    finally:
        semantic.clear_model_cache()


def test_local_model_manifest_persists_only_after_cache_enabled_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}')
    (model_dir / "model.safetensors").write_text("weights")
    units = extract_arithmetic_units(tmp_path)

    class FakeSentenceTransformer:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def encode(self, texts, **_kwargs):
            return np.stack(
                [
                    np.array([1.0, float(index + 1)], dtype=np.float32)
                    for index, _ in enumerate(texts)
                ]
            )

    monkeypatch.setattr(semantic, "_check_semantic_dependencies", lambda: None)
    monkeypatch.setattr(semantic, "_prepare_semantic_device", lambda *_args, **_kwargs: "cpu")
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    semantic.clear_model_cache()

    try:
        embeddings = compute_embeddings(
            units,
            model_name=str(model_dir),
            device="cpu",
            use_cache=False,
            cache_scope=tmp_path,
        )
    finally:
        semantic.clear_model_cache()

    assert embeddings.shape[0] == len(units)
    manifest_path = semantic._local_model_manifest_path(model_dir)
    assert not manifest_path.exists()

    assert semantic._fingerprint_local_model_dir(model_dir, persist_manifest=True) is not None
    assert manifest_path.is_file()
    assert stat.S_IMODE(manifest_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600


def test_get_model_reloads_local_directory_after_weights_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}')
    weights_path = model_dir / "model.safetensors"
    weights_path.write_text("weights-v1")
    loaded_models: list[object] = []

    class FakeSentenceTransformer:
        def __init__(self, *_args, **_kwargs):
            loaded_models.append(self)

    monkeypatch.setattr(semantic, "_check_semantic_dependencies", lambda: None)
    monkeypatch.setattr(semantic, "_prepare_semantic_device", lambda *_args, **_kwargs: "cpu")
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    semantic.clear_model_cache()

    try:
        first = semantic.get_model(str(model_dir))
        unchanged = semantic.get_model(str(model_dir))
        weights_path.write_text("weights-v2-longer")
        changed = semantic.get_model(str(model_dir))

        assert first is unchanged
        assert changed is not first
        assert loaded_models == [first, changed]
    finally:
        semantic.clear_model_cache()


def test_get_model_reloads_once_when_local_dir_changes_during_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}')
    weights_path = model_dir / "model.safetensors"
    weights_path.write_text("weights-v1")
    loaded_models: list[object] = []

    class MidLoadSwapSentenceTransformer:
        def __init__(self, *_args, **_kwargs):
            if not loaded_models:
                weights_path.write_text("weights-v2-swapped-mid-load")
            loaded_models.append(self)

    monkeypatch.setattr(semantic, "_check_semantic_dependencies", lambda: None)
    monkeypatch.setattr(semantic, "_prepare_semantic_device", lambda *_args, **_kwargs: "cpu")
    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", MidLoadSwapSentenceTransformer
    )
    semantic.clear_model_cache()

    try:
        model = semantic.get_model(str(model_dir))

        # The first load raced the swap and was discarded; the kept model was
        # verified against a stable post-swap fingerprint.
        assert len(loaded_models) == 2
        assert model is loaded_models[1]
        assert semantic._model_local_fingerprint == semantic._fingerprint_local_model_dir(model_dir)
    finally:
        semantic.clear_model_cache()


def test_get_model_fails_when_local_dir_keeps_changing_during_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}')
    weights_path = model_dir / "model.safetensors"
    weights_path.write_text("weights-v0")
    load_count = {"count": 0}

    class AlwaysMutatingSentenceTransformer:
        def __init__(self, *_args, **_kwargs):
            load_count["count"] += 1
            weights_path.write_text(f"weights-mutated-{load_count['count']}")

    monkeypatch.setattr(semantic, "_check_semantic_dependencies", lambda: None)
    monkeypatch.setattr(semantic, "_prepare_semantic_device", lambda *_args, **_kwargs: "cpu")
    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", AlwaysMutatingSentenceTransformer
    )
    semantic.clear_model_cache()

    with pytest.raises(SemanticBackendError, match="changed twice while loading"):
        semantic.get_model(str(model_dir))
    assert load_count["count"] == 2


def test_get_model_rejects_missing_explicit_local_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing-model"
    semantic.clear_model_cache()

    with pytest.raises(SemanticBackendError, match="does not exist"):
        semantic.get_model(str(missing))


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({"model.safetensors": "weights"}, "missing config.json"),
        ({"config.json": "{}"}, "contains no safetensors or PyTorch model weights"),
    ],
)
def test_get_model_rejects_incomplete_local_directory(
    tmp_path: Path,
    files: dict[str, str],
    message: str,
) -> None:
    model_dir = tmp_path / "incomplete-model"
    model_dir.mkdir()
    for filename, content in files.items():
        (model_dir / filename).write_text(content)
    semantic.clear_model_cache()

    with pytest.raises(SemanticBackendError, match=message):
        semantic.get_model(str(model_dir))


def test_prepare_semantic_device_ignores_fraction_on_non_mps(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="codedupes.semantic"):
        resolved = semantic._prepare_semantic_device(
            "cpu",
            mps_fallback=None,
            mps_memory_fraction=0.9,
        )

    assert resolved == "cpu"
    assert "mps_memory_fraction ignored: resolved device is cpu" in caplog.text


def test_get_model_wraps_known_backend_error(monkeypatch) -> None:
    def fake_ctor(*args, **kwargs):
        raise RuntimeError("EmbeddingGemma tokenizer backend is incompatible")

    monkeypatch.setattr(semantic, "_check_semantic_dependencies", lambda: None)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fake_ctor)
    semantic.clear_model_cache()

    with pytest.raises(SemanticBackendError, match="Semantic backend failed"):
        semantic.get_model("embeddinggemma-300m")


@pytest.mark.parametrize(
    ("missing_module", "expected_snippet"),
    [
        ("sentence_transformers", "sentence_transformers"),
        ("transformers", "transformers"),
        ("torch", "torch"),
    ],
)
def test_get_model_reports_missing_core_dependency(
    monkeypatch, missing_module: str, expected_snippet: str
) -> None:
    original_import = semantic.importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == missing_module:
            e = ModuleNotFoundError(f"No module named '{name}'")
            e.name = name
            raise e
        return original_import(name, package)

    monkeypatch.setattr(semantic.importlib, "import_module", fake_import_module)
    semantic.clear_model_cache()

    with pytest.raises(ModuleNotFoundError) as excinfo:
        semantic.get_model("gte-modernbert-base")

    assert expected_snippet in str(excinfo.value).lower()
