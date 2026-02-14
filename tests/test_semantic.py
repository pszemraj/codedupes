from __future__ import annotations

import sys
import numpy as np
import pytest
import sentence_transformers
from pathlib import Path

from codedupes import semantic
from codedupes.constants import DEFAULT_C2LLM_REVISION, DEFAULT_MODEL
from codedupes.models import CodeUnit, CodeUnitType
from codedupes.semantic import (
    SemanticBackendError,
    compute_embeddings,
    find_semantic_duplicates,
    find_similar_to_query,
    get_code_unit_statement_count,
    run_semantic_analysis,
)
from tests.conftest import extract_arithmetic_units


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


def _extract_units(tmp_path: Path) -> list[CodeUnit]:
    return extract_arithmetic_units(tmp_path, include_private=True, exclude_patterns=[])


def test_run_semantic_analysis_with_mock_model(tmp_path, monkeypatch):
    units = _extract_units(tmp_path)
    fake = FakeModel()
    monkeypatch.setattr(semantic, "_model", None)
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: fake)

    _, duplicates = run_semantic_analysis(units, threshold=0.9)

    assert len(duplicates) == 1
    assert duplicates[0].method == "semantic"
    assert duplicates[0].similarity > 0.9


def test_query_search_with_mocked_semantic_model(tmp_path, monkeypatch):
    units = _extract_units(tmp_path)
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
    unit = extract_arithmetic_units(tmp_path, include_private=True)[0]
    unit.source = source
    assert get_code_unit_statement_count(unit) == 2


def test_prepare_code_for_embedding_prefixes_query(tmp_path: Path) -> None:
    units = _extract_units(tmp_path)
    prepared = semantic.prepare_code_for_embedding(units[0], mode="query")
    assert prepared.startswith(semantic.C2LLM_INSTRUCTIONS["query"])
    assert prepared.endswith(units[0].source.strip())


def test_prepare_code_for_embedding_uses_custom_prefix(tmp_path: Path) -> None:
    units = _extract_units(tmp_path)
    prepared = semantic.prepare_code_for_embedding(
        units[0],
        mode="code",
        instruction_prefix="Represent this code as vector: ",
    )
    assert prepared.startswith("Represent this code as vector: ")
    assert prepared.endswith(units[0].source.strip())


def test_query_search_uses_custom_instruction_prefix(tmp_path: Path, monkeypatch) -> None:
    units = _extract_units(tmp_path)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    captured: dict[str, list[str]] = {}

    class QueryModel:
        def encode(self, texts, **kwargs):
            captured["texts"] = list(texts)
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
    assert captured["texts"][0].startswith("CUSTOM_QUERY_PREFIX: ")


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


def test_get_model_reports_deepspeed_guidance(monkeypatch) -> None:
    def fake_ctor(*args, **kwargs):
        e = ModuleNotFoundError("No module named 'deepspeed'")
        e.name = "deepspeed"
        raise e

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fake_ctor)
    semantic.clear_model_cache()

    with pytest.raises(ModuleNotFoundError) as excinfo:
        semantic.get_model("codefuse-ai/C2LLM-0.5B")
    assert "deepspeed is required" in str(excinfo.value)


def test_get_model_does_not_trust_remote_code_for_non_c2llm(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(
        semantic,
        "_check_semantic_dependencies",
        lambda model_name: None,
    )
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    semantic.clear_model_cache()

    semantic.get_model("sentence-transformers/all-MiniLM-L6-v2")
    assert len(calls) == 1
    assert calls[0]["kwargs"]["trust_remote_code"] is False


def test_get_model_trusts_remote_code_for_c2llm_variants(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(semantic, "_check_semantic_dependencies", lambda model_name: None)
    monkeypatch.setattr(semantic, "_resolve_c2llm_torch_dtype", lambda: "bf16")
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    semantic.clear_model_cache()

    semantic.get_model("codefuse-ai/C2LLM-7B")

    assert len(calls) == 1
    kwargs = calls[0]["kwargs"]
    assert kwargs["trust_remote_code"] is True
    assert kwargs["model_kwargs"]["trust_remote_code"] is True
    assert kwargs["tokenizer_kwargs"]["trust_remote_code"] is True
    assert kwargs["config_kwargs"]["trust_remote_code"] is True


def test_get_model_passes_revision_and_trust_kwargs(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(semantic, "_check_semantic_dependencies", lambda model_name: None)
    monkeypatch.setattr(semantic, "_resolve_c2llm_torch_dtype", lambda: "bf16")
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    semantic.clear_model_cache()

    semantic.get_model(DEFAULT_MODEL, revision=DEFAULT_C2LLM_REVISION, trust_remote_code=True)

    assert len(calls) == 1
    kwargs = calls[0]["kwargs"]
    assert kwargs["trust_remote_code"] is True
    assert kwargs["revision"] == DEFAULT_C2LLM_REVISION
    assert kwargs["model_kwargs"]["trust_remote_code"] is True
    assert kwargs["model_kwargs"]["revision"] == DEFAULT_C2LLM_REVISION
    assert kwargs["model_kwargs"]["dtype"] == "bf16"
    assert kwargs["model_kwargs"]["low_cpu_mem_usage"] is True
    assert kwargs["tokenizer_kwargs"]["trust_remote_code"] is True
    assert kwargs["tokenizer_kwargs"]["revision"] == DEFAULT_C2LLM_REVISION
    assert kwargs["config_kwargs"]["trust_remote_code"] is True
    assert kwargs["config_kwargs"]["revision"] == DEFAULT_C2LLM_REVISION


def test_get_model_keeps_explicit_revision_for_non_default_model(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(semantic, "_check_semantic_dependencies", lambda model_name: None)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    semantic.clear_model_cache()

    semantic.get_model(
        "sentence-transformers/all-MiniLM-L6-v2",
        revision=DEFAULT_C2LLM_REVISION,
        trust_remote_code=False,
    )

    assert len(calls) == 1
    kwargs = calls[0]["kwargs"]
    assert kwargs["trust_remote_code"] is False
    assert kwargs["revision"] == DEFAULT_C2LLM_REVISION
    assert kwargs["model_kwargs"]["revision"] == DEFAULT_C2LLM_REVISION
    assert kwargs["tokenizer_kwargs"]["revision"] == DEFAULT_C2LLM_REVISION
    assert kwargs["config_kwargs"]["revision"] == DEFAULT_C2LLM_REVISION


def test_get_model_rejects_incompatible_default_model_versions(monkeypatch) -> None:
    def fake_safe_package_version(package_name: str) -> str | None:
        fake_versions = {
            "transformers": "4.49.0",
            "sentence-transformers": "5.1.0",
            "torch": "2.5.0",
            "deepspeed": "0.16.0",
        }
        return fake_versions.get(package_name)

    monkeypatch.setattr(semantic, "_safe_package_version", fake_safe_package_version)
    semantic.clear_model_cache()

    with pytest.raises(SemanticBackendError, match="Incompatible transformers version"):
        semantic._check_default_model_compatibility(DEFAULT_MODEL)


def test_get_model_wraps_known_backend_error(monkeypatch) -> None:
    def fake_ctor(*args, **kwargs):
        raise AttributeError("'C2LLMForEmbedding' object has no attribute 'all_tied_weights_keys'")

    monkeypatch.setattr(semantic, "_check_semantic_dependencies", lambda model_name: None)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fake_ctor)
    semantic.clear_model_cache()

    with pytest.raises(SemanticBackendError, match="Semantic backend failed"):
        semantic.get_model(DEFAULT_MODEL, trust_remote_code=True)


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
        semantic.get_model("codefuse-ai/C2LLM-0.5B")

    assert expected_snippet in str(excinfo.value).lower()


def test_compute_embeddings_retries_with_reduced_batch_before_cpu(monkeypatch, tmp_path) -> None:
    units = _extract_units(tmp_path)
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
    units = _extract_units(tmp_path)
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

    assert seen_batches == [(8, None), (4, None), (2, None), (1, None), (1, "cpu")]


def test_resolve_c2llm_torch_dtype_prefers_cpu_bf16(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        bfloat16 = "bf16"
        cuda = FakeCuda()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    assert semantic._resolve_c2llm_torch_dtype() == "bf16"
