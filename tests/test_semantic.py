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
from codedupes.embedding_cache import EmbeddingCache
from codedupes.models import CodeUnit, CodeUnitType
from codedupes.semantic import (
    SemanticBackendError,
    compute_embeddings,
    find_semantic_duplicates,
    find_similar_to_query,
    get_code_unit_statement_count,
    run_semantic_analysis,
)
from tests.conftest import extract_arithmetic_units, extract_units


class FakeModel:
    """Simple deterministic embedding model stub."""

    def __init__(self) -> None:
        self.codes = 0

    def encode(self, texts, **kwargs):
        self.codes += 1
        if len(texts) == 2:
            return np.array(
                [
                    [1.0, 0.0],
                    [0.97, 0.243],
                ],
                dtype=np.float32,
            )
        return np.array([[1.0, 0.0]], dtype=np.float32)


def test_run_semantic_analysis_with_mock_model(tmp_path, monkeypatch):
    units = extract_arithmetic_units(tmp_path)
    fake = FakeModel()
    monkeypatch.setattr(semantic, "_model", None)
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: fake)

    _, duplicates = run_semantic_analysis(units, threshold=0.9)

    assert len(duplicates) == 1
    assert duplicates[0].method == "semantic"
    assert duplicates[0].similarity > 0.9


def test_query_search_with_mocked_semantic_model(tmp_path, monkeypatch):
    units = extract_arithmetic_units(tmp_path)
    fake = FakeModel()
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: fake)

    embeddings = fake.encode([u.source for u in units], convert_to_numpy=True)
    results = find_similar_to_query(
        query="find addition",
        units=units,
        embeddings=embeddings,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0][0] in units


def test_code_unit_statement_count_ignores_docstring(tmp_path: Path) -> None:
    source = """
    def sample(a, b):
        \"\"\"doc\"\"\"
        x = 1
        return a + b + x
    """
    unit = extract_arithmetic_units(tmp_path)[0]
    unit.source = source
    assert get_code_unit_statement_count(unit) == 2


def test_statement_count_dedents_decorated_method_source(tmp_path: Path) -> None:
    units = extract_units(
        tmp_path,
        """
        class Widget:
            @property
            def area(self):
                width = self.width
                height = self.height
                scale = self.scale
                return width * height * scale

            def perimeter(self):
                width = self.width
                height = self.height
                scale = self.scale
                return (width + height) * 2 * scale
        """,
        include_private=True,
    )
    by_name = {unit.name: unit for unit in units}

    assert get_code_unit_statement_count(by_name["area"]) == 4
    assert get_code_unit_statement_count(by_name["perimeter"]) == 4


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            """
            def guarded():
                try:
                    a = 1
                    b = 2
                    c = 3
                    return a + b + c
                except ValueError:
                    return 0
            """,
            # try + 4 body statements + handler return; the except clause itself
            # is an ast.excepthandler, not a statement.
            6,
            id="single-try",
        ),
        pytest.param(
            """
            def managed(path):
                with open(path) as handle:
                    first = handle.readline()
                    second = handle.readline()
                    return first + second
            """,
            4,
            id="single-with",
        ),
        pytest.param(
            """
            def looped(items):
                for item in items:
                    if item:
                        yield item
                    else:
                        continue
            """,
            4,
            id="single-loop",
        ),
    ],
)
def test_statement_count_recurses_into_control_flow(
    tmp_path: Path, source: str, expected: int
) -> None:
    unit = extract_arithmetic_units(tmp_path)[0]
    unit.source = source
    assert get_code_unit_statement_count(unit) == expected


def test_statement_count_stops_at_nested_scopes(tmp_path: Path) -> None:
    source = """
    def outer():
        def inner():
            a = 1
            b = 2
            return a + b

        class Helper:
            def method(self):
                return 1

        return inner
    """
    unit = extract_arithmetic_units(tmp_path)[0]
    unit.source = source
    # inner (1) + Helper (1) + return (1); nested bodies belong to their own units.
    assert get_code_unit_statement_count(unit) == 3


def test_resolve_encode_plan_default_model_symmetric_no_prompt() -> None:
    for mode in ("code", "query"):
        plan = semantic.resolve_encode_plan(mode=mode)
        assert plan == semantic.EncodePlan(route="symmetric", prompt=None)


def test_resolve_encode_plan_custom_prefix_replaces_prompt_and_keeps_route() -> None:
    plan = semantic.resolve_encode_plan(
        model_name="embeddinggemma-300m",
        mode="code",
        instruction_prefix="Represent this code as vector: ",
        semantic_task="code-retrieval",
    )
    assert plan == semantic.EncodePlan(route="document", prompt="Represent this code as vector: ")


def test_query_search_uses_custom_instruction_prefix(tmp_path: Path, monkeypatch) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    captured: dict[str, object] = {}

    class QueryModel:
        def encode(self, texts, **kwargs):
            captured["texts"] = list(texts)
            captured["prompt"] = kwargs.get("prompt")
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: QueryModel())

    results = find_similar_to_query(
        query="find addition",
        units=units,
        embeddings=embeddings,
        instruction_prefix="CUSTOM_QUERY_PREFIX: ",
        top_k=1,
    )

    assert len(results) == 1
    # The prompt travels as backend configuration; the input text stays raw.
    assert captured["texts"] == ["find addition"]
    assert captured["prompt"] == "CUSTOM_QUERY_PREFIX: "


# Saved prompts exactly as they appear in EmbeddingGemma's
# config_sentence_transformers.json; the fake below composes them the same way
# SentenceTransformers does, so these tests assert the *effective* model input.
EMBEDDINGGEMMA_SAVED_PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "STS": "task: sentence similarity | query: ",
    "InstructionRetrieval": "task: code retrieval | query: ",
}


class PromptAwareGemmaModel:
    """Fake EmbeddingGemma emulating SentenceTransformers prompt composition.

    ``encode_query``/``encode_document`` fall back to the saved query/document
    prompt whenever the caller provides no explicit ``prompt``/``prompt_name``,
    exactly like the real backend, so a manually prefixed input would surface
    here as a double prompt.
    """

    def __init__(self) -> None:
        self.prompts = dict(EMBEDDINGGEMMA_SAVED_PROMPTS)
        self.calls: list[tuple[str, list[str]]] = []

    def _run(
        self,
        method: str,
        texts,
        prompt: str | None,
        prompt_name: str | None,
        default_prompt_name: str | None,
    ) -> np.ndarray:
        if prompt is None:
            name = prompt_name if prompt_name is not None else default_prompt_name
            prompt = self.prompts.get(name, "") if name is not None else ""
        effective = [f"{prompt}{text}" for text in texts]
        self.calls.append((method, effective))
        return np.array(
            [[1.0, 0.0] if i == 0 else [0.0, 1.0] for i in range(len(texts))],
            dtype=np.float32,
        )

    def encode(self, texts, prompt=None, prompt_name=None, **kwargs):
        return self._run("encode", texts, prompt, prompt_name, None)

    def encode_query(self, texts, prompt=None, prompt_name=None, **kwargs):
        return self._run("encode_query", texts, prompt, prompt_name, "query")

    def encode_document(self, texts, prompt=None, prompt_name=None, **kwargs):
        return self._run("encode_document", texts, prompt, prompt_name, "document")


def test_embeddinggemma_duplicate_mode_symmetric_route_single_sts_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    units = extract_arithmetic_units(tmp_path)
    model = PromptAwareGemmaModel()
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: model)

    embeddings = compute_embeddings(units, model_name="embeddinggemma-300m", batch_size=2)

    assert embeddings.shape == (2, 2)
    ((method, effective),) = model.calls
    assert method == "encode"
    assert effective == [
        f"task: sentence similarity | query: {unit.source.strip()}" for unit in units
    ]


def test_embeddinggemma_search_corpus_document_route_single_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    units = extract_arithmetic_units(tmp_path)
    model = PromptAwareGemmaModel()
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: model)

    compute_embeddings(
        units,
        model_name="embeddinggemma-300m",
        semantic_task="code-retrieval",
        batch_size=2,
    )

    ((method, effective),) = model.calls
    assert method == "encode_document"
    assert effective == [f"title: none | text: {unit.source.strip()}" for unit in units]


def test_embeddinggemma_query_route_single_task_prompt(tmp_path: Path, monkeypatch) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    model = PromptAwareGemmaModel()
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: model)

    results = find_similar_to_query(
        query="find addition",
        units=units,
        embeddings=embeddings,
        model_name="embeddinggemma-300m",
        top_k=2,
    )

    assert len(results) == 1
    ((method, effective),) = model.calls
    assert method == "encode_query"
    assert effective == ["task: code retrieval | query: find addition"]


def test_embeddinggemma_custom_instruction_replaces_saved_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    model = PromptAwareGemmaModel()
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: model)

    find_similar_to_query(
        query="find addition",
        units=units,
        embeddings=embeddings,
        model_name="embeddinggemma-300m",
        instruction_prefix="CUSTOM: ",
        top_k=2,
    )

    ((method, effective),) = model.calls
    assert method == "encode_query"
    assert effective == ["CUSTOM: find addition"]


def test_find_similar_to_query_applies_threshold_filter(tmp_path: Path, monkeypatch) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)

    class QueryModel:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: QueryModel())

    results = find_similar_to_query(
        query="find addition",
        units=units,
        embeddings=embeddings,
        top_k=5,
        threshold=0.9,
    )

    assert len(results) == 1


def test_find_similar_to_query_default_threshold_is_search_default(
    tmp_path: Path, monkeypatch
) -> None:
    units = extract_arithmetic_units(tmp_path)
    # First row scores 0.6: above the search default (0.50) but far below the
    # duplicate-detection default (0.96); second row scores 0.3 and is dropped.
    embeddings = np.array([[0.6, 0.8], [0.3, 0.9539392]], dtype=np.float32)

    class QueryModel:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: QueryModel())

    results = find_similar_to_query(
        query="find addition",
        units=units,
        embeddings=embeddings,
        top_k=5,
    )

    assert [unit for unit, _score in results] == [units[0]]
    assert results[0][1] == pytest.approx(0.6, abs=1e-6)


def test_find_semantic_duplicates_ignores_nan_similarity(tmp_path: Path) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array(
        [
            [np.nan, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    duplicates = find_semantic_duplicates(units, embeddings, threshold=0.5)

    assert duplicates == []


def test_compute_embeddings_rejects_nonfinite_model_output(tmp_path: Path, monkeypatch) -> None:
    units = extract_arithmetic_units(tmp_path)

    class NanModel:
        def encode(self, texts, **kwargs):
            return np.array([[np.nan, 0.0]] * len(texts), dtype=np.float32)

    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: NanModel())

    with pytest.raises(semantic.InvalidEmbeddingError, match="NaN or infinity"):
        compute_embeddings(units, device="cpu")


def test_compute_embeddings_rejects_zero_vector_output(tmp_path: Path, monkeypatch) -> None:
    units = extract_arithmetic_units(tmp_path)

    class ZeroModel:
        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: ZeroModel())

    with pytest.raises(semantic.InvalidEmbeddingError, match="zero or invalid vector"):
        compute_embeddings(units, device="cpu")


def test_compute_embeddings_rejects_wrong_row_count(tmp_path: Path, monkeypatch) -> None:
    units = extract_arithmetic_units(tmp_path)

    class ShortModel:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: ShortModel())

    with pytest.raises(semantic.InvalidEmbeddingError, match="rows"):
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
    monkeypatch.setattr(semantic, "_validate_explicit_device_request", lambda *_a, **_k: None)

    embeddings = compute_embeddings(units, device="cuda")

    assert devices_seen == [None, "cpu"]
    assert embeddings.shape == (2, 2)
    assert np.isfinite(embeddings).all()


def test_fresh_embeddings_are_renormalized_centrally(tmp_path: Path, monkeypatch) -> None:
    units = extract_arithmetic_units(tmp_path)

    class UnnormalizedModel:
        def encode(self, texts, **kwargs):
            return np.array([[3.0, 4.0]] * len(texts), dtype=np.float32)

    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: UnnormalizedModel())

    embeddings = compute_embeddings(units, device="cpu")

    np.testing.assert_allclose(embeddings, [[0.6, 0.8]] * len(units), atol=1e-6)


def test_find_semantic_duplicates_skips_incompatible_unit_types(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.py"
    source_path.write_text("class C:\n    pass\n\ndef f():\n    return 1\n")

    class_unit = CodeUnit(
        name="C",
        qualified_name="sample.C",
        unit_type=CodeUnitType.CLASS,
        file_path=source_path,
        lineno=1,
        end_lineno=2,
        source="class C:\n    pass",
        is_public=True,
        is_exported=False,
    )
    function_unit = CodeUnit(
        name="f",
        qualified_name="sample.f",
        unit_type=CodeUnitType.FUNCTION,
        file_path=source_path,
        lineno=4,
        end_lineno=5,
        source="def f():\n    return 1",
        is_public=True,
        is_exported=False,
    )
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    duplicates = find_semantic_duplicates(
        units=[class_unit, function_unit],
        embeddings=embeddings,
        threshold=0.9,
    )

    assert duplicates == []


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


def test_resolve_model_dtype_cpu_follows_capability_gate(monkeypatch) -> None:
    monkeypatch.setattr(semantic, "resolve_cpu_bf16_native", lambda: True)
    assert semantic._resolve_model_dtype("test-model", "cpu") is torch.bfloat16

    monkeypatch.setattr(semantic, "resolve_cpu_bf16_native", lambda: False)
    assert semantic._resolve_model_dtype("test-model", "cpu") is torch.float32


def test_resolve_model_dtype_mps_always_float32_regardless_of_cpu_gate(monkeypatch) -> None:
    # MPS is never CPU: the gate must not leak into the MPS branch.
    monkeypatch.setattr(semantic, "resolve_cpu_bf16_native", lambda: True)
    assert semantic._resolve_model_dtype("test-model", "mps") is torch.float32


def test_dtype_variant_for_mps_is_always_empty(monkeypatch) -> None:
    monkeypatch.setattr(semantic, "resolve_cpu_bf16_native", lambda: True)
    profile = semantic.resolve_model_profile("gte-modernbert-base")

    assert semantic._dtype_variant_for(profile, "mps", mps_fallback=None) == ""


def test_dtype_variant_for_cpu_follows_capability_gate(monkeypatch) -> None:
    profile = semantic.resolve_model_profile("gte-modernbert-base")

    monkeypatch.setattr(semantic, "resolve_cpu_bf16_native", lambda **_kwargs: False)
    assert semantic._dtype_variant_for(profile, "cpu", mps_fallback=None) == ""

    monkeypatch.setattr(semantic, "resolve_cpu_bf16_native", lambda **_kwargs: True)
    assert semantic._dtype_variant_for(profile, "cpu", mps_fallback=None) == "dtype=torch.bfloat16"


def test_dtype_variant_for_auto_on_darwin_skips_resolution_when_gate_false(monkeypatch) -> None:
    profile = semantic.resolve_model_profile("gte-modernbert-base")
    monkeypatch.setattr(semantic, "resolve_cpu_bf16_native", lambda **_kwargs: False)
    monkeypatch.setattr(semantic.sys, "platform", "darwin")

    def _fail_if_called(*_a, **_k):
        raise AssertionError("must not resolve a concrete device when the CPU gate is false")

    monkeypatch.setattr(semantic, "_resolve_semantic_device_request", _fail_if_called)

    assert semantic._dtype_variant_for(profile, "auto", mps_fallback=None) == ""


def test_dtype_variant_matches_pre_capability_gate_baseline_when_gate_is_false() -> None:
    # On a machine without a CPU bf16 GEMM backend, cpu/mps/darwin-auto must
    # key byte-identically to the pre-capability-gate policy (empty variant):
    # EMBEDDING_PIPELINE_SCHEMA is not bumped, so old and new code must agree
    # here or warm caches on every non-mkldnn machine would silently miss.
    if devices.resolve_cpu_bf16_native():
        pytest.skip("This machine's CPU passes the bf16 capability gate.")
    profile = semantic.resolve_model_profile("gte-modernbert-base")

    assert semantic._dtype_variant_for(profile, "cpu", mps_fallback=None) == ""
    assert semantic._dtype_variant_for(profile, "mps", mps_fallback=None) == ""
    assert semantic._dtype_variant_for(profile, "auto", mps_fallback=None) == ""


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

    first = semantic.get_model(str(model_dir))
    unchanged = semantic.get_model(str(model_dir))
    weights_path.write_text("weights-v2-longer")
    changed = semantic.get_model(str(model_dir))

    assert first is unchanged
    assert changed is not first
    assert loaded_models == [first, changed]


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

    model = semantic.get_model(str(model_dir))

    # The first load raced the swap and was discarded; the kept model was
    # verified against a stable post-swap fingerprint.
    assert len(loaded_models) == 2
    assert model is loaded_models[1]
    assert semantic._model_local_fingerprint == semantic._fingerprint_local_model_dir(model_dir)


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


@pytest.mark.parametrize("version", ["2.12.9", "3.0.0", "3.0.0.dev1"])
def test_torch_runtime_rejects_unsupported_versions(monkeypatch, version: str) -> None:
    monkeypatch.setattr(semantic, "_safe_package_version", lambda _name: version)

    with pytest.raises(SemanticBackendError, match="requires >=2.13,<3"):
        semantic._validate_torch_runtime()


@pytest.mark.parametrize("version", ["2.13.0", "2.13.0.dev20260101", "2.13.0rc1", "2.14.1"])
def test_torch_runtime_accepts_supported_versions(monkeypatch, version: str) -> None:
    monkeypatch.setattr(semantic, "_safe_package_version", lambda _name: version)

    semantic._validate_torch_runtime()


def test_prepare_semantic_device_ignores_fraction_on_non_mps(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="codedupes.semantic"):
        resolved = semantic._prepare_semantic_device(
            "cpu",
            mps_fallback=None,
            mps_memory_fraction=0.9,
        )

    assert resolved == "cpu"
    assert "mps_memory_fraction ignored: resolved device is cpu" in caplog.text


def test_encode_texts_does_not_hide_unrelated_type_error() -> None:
    calls = 0

    def broken_encode(_texts, **_kwargs):
        nonlocal calls
        calls += 1
        raise TypeError("internal tensor type mismatch")

    with pytest.raises(TypeError, match="tensor type mismatch"):
        semantic._encode_texts(
            broken_encode,
            ["code"],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device="cpu",
        )

    assert calls == 1


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


def test_move_model_to_cpu_casts_bf16_only_when_gate_is_false() -> None:
    module = torch.nn.Linear(4, 4).to(dtype=torch.bfloat16)

    semantic._move_model_to_cpu(module)

    expected_dtype = torch.bfloat16 if devices.resolve_cpu_bf16_native() else torch.float32
    assert next(module.parameters()).dtype is expected_dtype
    assert str(next(module.parameters()).device) == "cpu"


def test_move_model_to_cpu_leaves_float32_models_untouched() -> None:
    module = torch.nn.Linear(4, 4)

    semantic._move_model_to_cpu(module)

    assert next(module.parameters()).dtype is torch.float32


_FULL_REVISION = "1" * 40


class _WarmCacheModel:
    """Deterministic embedding model stub used to populate a warm on-disk cache."""

    def __init__(self, dim: int = 2) -> None:
        self.dim = dim
        self.encode_calls = 0

    def encode(self, texts, **_kwargs):
        self.encode_calls += 1
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)


def _fail_if_called(*_args, **_kwargs):
    """Fail the test whenever the mocked callable it replaces is invoked."""
    raise AssertionError("this callable must not run on a fully warm cache hit")


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
    model = _WarmCacheModel()
    monkeypatch.setattr(semantic, "get_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)
    compute_embeddings(
        units,
        model_name=model_name,
        revision=_FULL_REVISION,
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
    monkeypatch.setattr(semantic, "get_model", _fail_if_called)

    with pytest.raises(SemanticBackendError):
        compute_embeddings(
            units,
            model_name=model_name,
            revision=_FULL_REVISION,
            device="cuda",
            cache_scope=tmp_path,
        )


def test_find_similar_to_query_warm_cache_raises_for_explicit_unavailable_device(
    tmp_path: Path, monkeypatch
) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    model = _WarmCacheModel()
    monkeypatch.setattr(semantic, "get_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    find_similar_to_query(
        "find addition",
        units,
        embeddings,
        model_name="gte-modernbert-base",
        revision=_FULL_REVISION,
        device="cpu",
        cache_scope=tmp_path,
    )

    def _raise_unavailable(*_args, **_kwargs):
        raise SemanticBackendError("cuda is not available in this environment")

    monkeypatch.setattr(semantic, "_resolve_semantic_device_request", _raise_unavailable)
    monkeypatch.setattr(semantic, "get_model", _fail_if_called)

    with pytest.raises(SemanticBackendError):
        find_similar_to_query(
            "find addition",
            units,
            embeddings,
            model_name="gte-modernbert-base",
            revision=_FULL_REVISION,
            device="cuda",
            cache_scope=tmp_path,
        )


def test_query_embedding_cache_put_is_fifo_capped(tmp_path: Path, monkeypatch) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    model = _WarmCacheModel()
    monkeypatch.setattr(semantic, "get_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    captured: dict[str, object] = {}
    original_put_many = EmbeddingCache.put_many

    def _recording_put_many(self, *args, **kwargs):
        captured["max_namespace_keys"] = kwargs.get("max_namespace_keys")
        return original_put_many(self, *args, **kwargs)

    monkeypatch.setattr(EmbeddingCache, "put_many", _recording_put_many)

    find_similar_to_query(
        "find addition",
        units,
        embeddings,
        model_name="gte-modernbert-base",
        revision=_FULL_REVISION,
        device="cpu",
        cache_scope=tmp_path,
    )

    assert captured["max_namespace_keys"] == semantic._MAX_CACHED_QUERY_KEYS


def test_compute_embeddings_warm_cache_auto_and_cpu_skip_device_validation(
    tmp_path: Path, monkeypatch
) -> None:
    units = _warm_corpus_cache(tmp_path, monkeypatch)

    validation_calls = {"count": 0}

    def _count_and_raise(*_args, **_kwargs):
        validation_calls["count"] += 1
        raise SemanticBackendError("must not be called for auto/cpu on a warm cache")

    monkeypatch.setattr(semantic, "_resolve_semantic_device_request", _count_and_raise)
    monkeypatch.setattr(semantic, "get_model", _fail_if_called)

    for device in ("auto", "cpu"):
        result = compute_embeddings(
            units,
            model_name="gte-modernbert-base",
            revision=_FULL_REVISION,
            device=device,
            cache_scope=tmp_path,
        )
        assert result.shape == (len(units), 2)

    assert validation_calls["count"] == 0


class _BfloatAcceleratorFallbackModel:
    """Fake bf16 accelerator model whose encode OOMs down to a real CPU dtype cast."""

    def __init__(self) -> None:
        self._dtype = torch.bfloat16

    def parameters(self):
        yield torch.zeros(1, dtype=self._dtype)

    def to(self, device=None, dtype=None):
        if dtype is not None:
            self._dtype = dtype
        return self

    def encode(self, texts, **kwargs):
        if kwargs.get("device") != "cpu":
            raise RuntimeError("CUDA out of memory")
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)


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
    model = _BfloatAcceleratorFallbackModel()
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
        revision=_FULL_REVISION,
        device="cuda",
        batch_size=1,
        cache_scope=tmp_path,
    )

    assert embeddings.shape[0] == len(units)
    # A dtype-diverging fallback (bf16 CUDA -> float32 CPU on this gate-false
    # machine) must never write float32 vectors under the bf16-keyed
    # namespace: the coherence-restart discards that run and recomputes under
    # a fresh, correctly-keyed identity instead, so *some* write is expected -
    # just never one landing in the original bf16 key space.
    bf16_writes = [call for call in put_calls if call["kwargs"].get("namespace") == bf16_namespace]
    assert bf16_writes == []
    if not devices.resolve_cpu_bf16_native():
        assert len(put_calls) == 1


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
        _FULL_REVISION,
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

    compute_embeddings(
        units,
        model_name="gte-modernbert-base",
        revision=_FULL_REVISION,
        device="cuda",
        cache_scope=tmp_path,
    )

    assert recorded_initial_devices == ["cuda", "cpu"]


def test_fingerprint_local_model_dir_follows_symlinked_subdirectories(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    real_shards_dir = tmp_path / "real-shards"
    real_shards_dir.mkdir()
    shard_path = real_shards_dir / "model-00001.safetensors"
    shard_path.write_text("weights-v1")
    (model_dir / "shards").symlink_to(real_shards_dir, target_is_directory=True)

    before = semantic._fingerprint_local_model_dir(model_dir, persist_manifest=False)

    shard_path.write_text("weights-v2-changed")

    after = semantic._fingerprint_local_model_dir(model_dir, persist_manifest=False)

    assert before is not None
    assert after is not None
    assert before != after


def test_fingerprint_local_model_dir_handles_symlink_cycles(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "loop").symlink_to(model_dir, target_is_directory=True)

    fingerprint = semantic._fingerprint_local_model_dir(model_dir, persist_manifest=False)

    assert fingerprint is not None


# --- T7: loose-by-default cache revision keying, strict opt-in -------------


def test_resolve_revision_for_cache_loose_default_labels_unpinned_model(monkeypatch) -> None:
    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("loose mode must never consult the offline hub-cache lookup")

    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", _fail_if_called)

    assert semantic._resolve_revision_for_cache("some-generic-model", None) == "main"
    assert (
        semantic._resolve_revision_for_cache("some-generic-model", "feature-branch")
        == "feature-branch"
    )


def test_resolve_revision_for_cache_explicit_commit_hash_keys_as_is_either_mode() -> None:
    commit_hash = "a" * 40

    assert semantic._resolve_revision_for_cache("some-generic-model", commit_hash) == commit_hash
    assert (
        semantic._resolve_revision_for_cache("some-generic-model", commit_hash, strict=True)
        == commit_hash
    )


def test_resolve_revision_for_cache_strict_resolves_commit_and_disables_on_unmappable(
    monkeypatch,
) -> None:
    def _fake_resolve(_canonical_model, revision="main"):
        return "resolved-hash" if revision == "main" else None

    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", _fake_resolve)

    assert (
        semantic._resolve_revision_for_cache("some-generic-model", None, strict=True)
        == "resolved-hash"
    )
    # A branch/tag that cannot be mapped offline disables caching in strict mode
    # (never in loose mode, where the same input keys by its label instead).
    assert (
        semantic._resolve_revision_for_cache("some-generic-model", "feature-branch", strict=True)
        is None
    )
    assert (
        semantic._resolve_revision_for_cache("some-generic-model", "feature-branch")
        == "feature-branch"
    )


def test_confirm_cache_revision_after_load_loose_default_trusts_pre_load_label() -> None:
    # Loose mode never inspects the model at all: an arbitrary object without
    # the introspection surface _get_loaded_model_commit_hash expects proves
    # that no post-load reconciliation happens.
    sentinel_model = object()

    assert (
        semantic._confirm_cache_revision_after_load(sentinel_model, "some-generic-model", "main")
        == "main"
    )
    assert (
        semantic._confirm_cache_revision_after_load(sentinel_model, "some-generic-model", None)
        == "main"
    )
    commit_hash = "b" * 40
    assert (
        semantic._confirm_cache_revision_after_load(
            sentinel_model, "some-generic-model", commit_hash
        )
        == commit_hash
    )


def test_confirm_cache_revision_after_load_strict_requires_loaded_commit(monkeypatch) -> None:
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    assert (
        semantic._confirm_cache_revision_after_load(
            object(), "some-generic-model", "main", strict=True
        )
        is None
    )

    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "loaded-hash")

    assert (
        semantic._confirm_cache_revision_after_load(
            object(), "some-generic-model", "main", strict=True
        )
        == "loaded-hash"
    )


def test_loose_default_cache_survives_simulated_branch_move(tmp_path: Path, monkeypatch) -> None:
    """A warm loose-mode cache must not invalidate when an upstream ref moves.

    Uses an unpinned (generic-profile) model name so the default request
    genuinely goes through the symbolic-revision path rather than a built-in
    profile's pinned commit hash. The label ("main") is the whole key; there
    is no hub-cache lookup to go stale, so re-pointing what a branch would
    resolve to has no effect.
    """
    units = extract_arithmetic_units(tmp_path)
    model = _WarmCacheModel()
    monkeypatch.setattr(semantic, "get_model", lambda *_a, **_k: model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("loose mode must never consult the offline hub-cache lookup")

    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", _fail_if_called)

    first = compute_embeddings(
        units, model_name="some-generic-model", device="cpu", cache_scope=tmp_path
    )
    assert model.encode_calls == 1

    second = compute_embeddings(
        units, model_name="some-generic-model", device="cpu", cache_scope=tmp_path
    )

    assert model.encode_calls == 1
    np.testing.assert_array_equal(first, second)


def test_strict_revision_cache_reencodes_after_simulated_branch_move(
    tmp_path: Path, monkeypatch
) -> None:
    """A moved branch changes the resolved commit hash, invalidating a strict-mode warm cache.

    Uses an unpinned (generic-profile) model name with no explicit revision,
    so the resolved commit is entirely a function of the (stubbed) offline
    hub-cache lookup and the (stubbed) post-load reported commit - both are
    moved together to simulate a real branch move.
    """
    units = extract_arithmetic_units(tmp_path)
    model = _WarmCacheModel()
    monkeypatch.setattr(semantic, "get_model", lambda *_a, **_k: model)
    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", lambda *_a, **_k: "commit-a")
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "commit-a")

    compute_embeddings(
        units,
        model_name="some-generic-model",
        device="cpu",
        cache_scope=tmp_path,
        strict_revision_cache=True,
    )
    assert model.encode_calls == 1

    # Same resolved commit: a warm hit needs no model load at all.
    compute_embeddings(
        units,
        model_name="some-generic-model",
        device="cpu",
        cache_scope=tmp_path,
        strict_revision_cache=True,
    )
    assert model.encode_calls == 1

    # Simulate an upstream branch move to a new commit.
    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", lambda *_a, **_k: "commit-b")
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "commit-b")

    compute_embeddings(
        units,
        model_name="some-generic-model",
        device="cpu",
        cache_scope=tmp_path,
        strict_revision_cache=True,
    )

    assert model.encode_calls == 2


def test_strict_revision_cache_disables_caching_for_unmappable_symbolic_ref(
    tmp_path: Path, monkeypatch
) -> None:
    units = extract_arithmetic_units(tmp_path)
    model = _WarmCacheModel()
    monkeypatch.setattr(semantic, "get_model", lambda *_a, **_k: model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)
    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", lambda *_a, **_k: None)

    for _ in range(2):
        compute_embeddings(
            units,
            model_name="gte-modernbert-base",
            revision="unmappable-branch",
            device="cpu",
            cache_scope=tmp_path,
            strict_revision_cache=True,
        )

    assert model.encode_calls == 2
