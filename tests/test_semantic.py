from __future__ import annotations

import logging
import stat
from pathlib import Path

import numpy as np
import pytest
import sentence_transformers

from codedupes import semantic
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

    if revision is None:
        assert "revision" not in kwargs
        assert "model_kwargs" not in kwargs
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
