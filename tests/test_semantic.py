from __future__ import annotations

import numpy as np
from pathlib import Path
import pytest
import sentence_transformers

from codedupes import semantic
from codedupes.models import CodeUnit
from codedupes.semantic import (
    get_code_unit_statement_count,
    find_similar_to_query,
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
    assert "trust_remote_code" not in calls[0]["kwargs"]


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
