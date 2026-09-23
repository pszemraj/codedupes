"""Encode plan resolution, EmbeddingGemma prompt routing, and search-document mode validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codedupes import semantic
from codedupes.semantic import (
    compute_embeddings,
    find_similar_to_query,
)
from tests.conftest import extract_arithmetic_units
from tests.semantic_helpers import PromptAwareGemmaModel


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
    identity = semantic.resolve_embedding_space_identity(
        instruction_prefix="CUSTOM_QUERY_PREFIX: ",
        semantic_task=semantic.DEFAULT_SEARCH_SEMANTIC_TASK,
    )

    results = find_similar_to_query(
        query="find addition",
        units=units,
        embeddings=embeddings,
        instruction_prefix="CUSTOM_QUERY_PREFIX: ",
        top_k=1,
        threshold=0.0,
        corpus_identity=identity,
    )

    assert len(results) == 1
    # The prompt travels as backend configuration; the input text stays raw.
    assert captured["texts"] == ["find addition"]
    assert captured["prompt"] == "CUSTOM_QUERY_PREFIX: "


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
    identity = semantic.resolve_embedding_space_identity(
        model_name="embeddinggemma-300m",
        semantic_task=semantic.DEFAULT_SEARCH_SEMANTIC_TASK,
    )

    results = find_similar_to_query(
        query="find addition",
        units=units,
        embeddings=embeddings,
        model_name="embeddinggemma-300m",
        top_k=2,
        corpus_identity=identity,
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
    identity = semantic.resolve_embedding_space_identity(
        model_name="embeddinggemma-300m",
        instruction_prefix="CUSTOM: ",
        semantic_task=semantic.DEFAULT_SEARCH_SEMANTIC_TASK,
    )

    find_similar_to_query(
        query="find addition",
        units=units,
        embeddings=embeddings,
        model_name="embeddinggemma-300m",
        instruction_prefix="CUSTOM: ",
        top_k=2,
        threshold=0.0,
        corpus_identity=identity,
    )

    ((method, effective),) = model.calls
    assert method == "encode_query"
    assert effective == ["CUSTOM: find addition"]


def test_prompt_sensitive_search_requires_corpus_identity(tmp_path: Path) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="corpus_identity is required"):
        find_similar_to_query(
            "find addition",
            units,
            embeddings,
            model_name="embeddinggemma-300m",
            threshold=0.0,
            use_cache=False,
        )


@pytest.mark.parametrize(
    "compute",
    [semantic.compute_embeddings, semantic.compute_embeddings_with_identity],
)
def test_embedding_apis_reject_invalid_search_document_before_model_loading(
    tmp_path: Path, monkeypatch, compute
) -> None:
    units = extract_arithmetic_units(tmp_path)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("invalid search_document must fail before model loading")

    monkeypatch.setattr(semantic, "get_model", fail_if_called)

    with pytest.raises(ValueError, match="search_document must be 'source' or 'contextual'"):
        compute(
            units,
            document_texts=[f"path: arithmetic.py\n{unit.source}" for unit in units],
            search_document="contextual-typo",
            device="cpu",
            use_cache=False,
        )


def test_embedding_identity_rejects_invalid_search_document() -> None:
    with pytest.raises(ValueError, match="search_document must be 'source' or 'contextual'"):
        semantic.EmbeddingSpaceIdentity(
            model_name="model",
            resolved_revision="revision",
            runtime_variant="variant",
            search_document="contextual-typo",
        )


def test_array_only_embedding_api_rejects_contextual_documents(tmp_path: Path, monkeypatch) -> None:
    units = extract_arithmetic_units(tmp_path)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("contextual array-only embedding must fail before model loading")

    monkeypatch.setattr(semantic, "get_model", fail_if_called)

    with pytest.raises(ValueError, match="require compute_embeddings_with_identity"):
        compute_embeddings(
            units,
            document_texts=[f"path: arithmetic.py\n{unit.source}" for unit in units],
            search_document="contextual",
            device="cpu",
            use_cache=False,
        )


@pytest.mark.parametrize(
    ("mode", "input_count", "isatty", "expected"),
    [
        ("never", 1000, True, False),
        ("always", 1, False, True),
        ("auto", 100, True, False),
        ("auto", 101, True, True),
        ("auto", 101, False, False),
    ],
)
def test_embedding_progress_policy(monkeypatch, mode, input_count, isatty, expected) -> None:
    monkeypatch.setattr(semantic.sys.stderr, "isatty", lambda: isatty)

    assert semantic._should_show_progress(mode, input_count) is expected


@pytest.mark.parametrize(("progress", "expected"), [("always", True), ("never", False)])
def test_compute_embeddings_forwards_progress_policy(
    monkeypatch, tmp_path, progress, expected
) -> None:
    units = extract_arithmetic_units(tmp_path)
    captured: list[bool] = []

    class RecordingModel:
        def encode(self, texts, **kwargs):
            captured.append(kwargs["show_progress_bar"])
            return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: RecordingModel())

    compute_embeddings(units, use_cache=False, progress=progress)

    assert captured == [expected]
