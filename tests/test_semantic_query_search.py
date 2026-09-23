"""find_similar_to_query threshold, top_k, and query-validation behavior."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import get_type_hints

import numpy as np
import pytest

from codedupes import semantic
from codedupes.analyzer import CodeAnalyzer
from codedupes.semantic import (
    find_semantic_duplicates,
    find_similar_to_query,
)
from tests.conftest import extract_arithmetic_units
from tests.semantic_helpers import FakeModel, PromptAwareGemmaModel, RecordingModel


def test_query_search_with_mocked_semantic_model(tmp_path, monkeypatch):
    units = extract_arithmetic_units(tmp_path)
    fake = FakeModel()
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: fake)

    embeddings = fake.encode([u.source for u in units], convert_to_numpy=True)
    execution = []
    results = find_similar_to_query(
        query="find addition",
        units=units,
        embeddings=embeddings,
        top_k=1,
        device="cpu",
        use_cache=False,
        execution=execution,
    )

    assert len(results) == 1
    assert results[0][0] in units
    assert execution == [
        semantic.QueryExecution(
            execution_device="cpu",
            cache_hit=False,
            threshold=semantic.resolve_search_threshold("gte-modernbert-base", None),
        )
    ]

    find_similar_to_query(
        query="find addition",
        units=units,
        embeddings=embeddings,
        top_k=1,
        threshold=0.9,
        device="cpu",
        use_cache=False,
        execution=execution,
    )
    assert execution[-1].threshold == 0.9


@pytest.mark.parametrize("model_kind", ["gte", "gemma", "local", "hub"])
@pytest.mark.parametrize("selection", ["auto", "generic", "embeddinggemma-300m", "numeric"])
def test_search_threshold_notices_do_not_repeat_on_warm_queries(
    tmp_path, monkeypatch, caplog, model_kind, selection
) -> None:
    units = extract_arithmetic_units(tmp_path)
    local = tmp_path / "embeddinggemma-copy"
    local.mkdir()
    (local / "config.json").write_text("{}", encoding="utf-8")
    (local / "model.safetensors").write_bytes(b"weights")
    model_name = {
        "gte": "gte-modernbert-base",
        "gemma": "embeddinggemma-300m",
        "local": str(local),
        "hub": "someone/embeddinggemma-300m-code-ft",
    }[model_kind]
    model = PromptAwareGemmaModel()
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: model)
    revision = "f" * 40 if model_kind == "hub" else None
    identity = semantic.resolve_embedding_space_identity(
        model_name=model_name,
        revision=revision,
        device="cpu",
        semantic_task=semantic.DEFAULT_SEARCH_SEMANTIC_TASK,
    )
    with caplog.at_level(logging.INFO):
        for _ in range(2):
            find_similar_to_query(
                "find addition",
                units,
                np.eye(2, dtype=np.float32),
                model_name=model_name,
                revision=revision,
                device="cpu",
                corpus_identity=identity,
                cache_scope=tmp_path,
                threshold=0.5 if selection == "numeric" else None,
                threshold_profile="auto" if selection == "numeric" else selection,
            )
    assert len(model.calls) == 1
    assert "Search threshold:" not in caplog.text  # Per-query detail is DEBUG in the API.
    assert caplog.text.count("Use --threshold-profile generic") == int(
        selection == "auto" and model_kind in {"local", "hub"}
    )
    assert caplog.text.count("score distribution may differ") == int(
        selection == "auto" and model_kind == "hub"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"semantic_task": "classification"},
        {"instruction_prefix": "CUSTOM: "},
        {"revision": "f" * 40},
        {"trust_remote_code": True},
        {"search_document": "contextual"},
    ],
)
@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("threshold_profile", ["auto", "generic", "embeddinggemma-300m"])
def test_uncalibrated_search_context_requires_explicit_threshold(
    tmp_path: Path, monkeypatch, kwargs: dict[str, object], use_cache: bool, threshold_profile: str
) -> None:
    units = extract_arithmetic_units(tmp_path)
    model = RecordingModel()
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: model)
    query_options = dict(kwargs)
    search_document = query_options.pop("search_document", "source")
    options = {
        "model_name": "embeddinggemma-300m",
        "device": "cpu",
        "use_cache": use_cache,
        "cache_scope": tmp_path,
        "semantic_task": semantic.DEFAULT_SEARCH_SEMANTIC_TASK,
        **query_options,
    }
    # Exercise identities returned by both cold encoding and warm cache reads.
    for iteration in range(2):
        corpus_calls_before = len(model.encoded)
        embeddings, identity = semantic.compute_embeddings_with_identity(
            units,
            document_texts=[f"path: arithmetic.py\n{unit.source}" for unit in units]
            if search_document == "contextual"
            else None,
            search_document=search_document,
            **options,
        )
        assert identity.search_document == search_document
        assert len(model.encoded) - corpus_calls_before == (0 if use_cache and iteration else 1)
        calls_before = len(model.encoded)
        with pytest.raises(ValueError, match=r"find_similar_to_query\(threshold=\.\.\.\)"):
            find_similar_to_query(
                "find addition",
                units,
                embeddings,
                corpus_identity=identity,
                threshold_profile=threshold_profile,
                **options,
            )
        assert len(model.encoded) == calls_before
        assert len(
            find_similar_to_query(
                "find addition",
                units,
                embeddings,
                threshold=0.0,
                threshold_profile=threshold_profile,
                corpus_identity=identity,
                **options,
            )
        ) == len(units)


def test_empty_search_still_rejects_uncalibrated_default() -> None:
    with pytest.raises(ValueError, match="explicit threshold"):
        find_similar_to_query(
            "anything",
            [],
            np.empty((0, 0), dtype=np.float32),
            model_name="gte-modernbert-base",
            revision="f" * 40,
            use_cache=False,
        )
    assert (
        find_similar_to_query(
            "anything",
            [],
            np.empty((0, 0), dtype=np.float32),
            model_name="gte-modernbert-base",
            revision="f" * 40,
            threshold=0.0,
            use_cache=False,
        )
        == []
    )
    assert find_semantic_duplicates([], np.empty((0, 0), dtype=np.float32), threshold=0.0) == []


def test_direct_embeddings_are_normalized_before_query_scoring(tmp_path: Path, monkeypatch) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array([[5.0, 0.0], [1.0, 0.5]], dtype=np.float32)

    class QueryModel:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: QueryModel())

    results = find_similar_to_query(
        "find addition",
        units,
        embeddings,
        top_k=np.int64(1),
        threshold=0.95,
        device="cpu",
        use_cache=False,
    )

    assert results == [(units[0], 1.0)]


def test_search_top_k_annotations_are_int() -> None:
    """Expose the conventional static type while accepting NumPy integers at runtime."""
    for search in (
        semantic._find_similar_to_query_unlocked,
        find_similar_to_query,
        CodeAnalyzer.search,
    ):
        assert get_type_hints(search)["top_k"] is int


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_search_requires_positive_integer_top_k(top_k) -> None:
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        find_similar_to_query(
            "anything",
            [],
            np.empty((0, 0), dtype=np.float32),
            top_k=top_k,
            threshold=0.0,
            use_cache=False,
        )


@pytest.mark.parametrize("query", ["", " \t\n"])
def test_search_requires_nonempty_query(query: str) -> None:
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        find_similar_to_query(
            query,
            [],
            np.empty((0, 0), dtype=np.float32),
            threshold=0.0,
            use_cache=False,
        )


@pytest.mark.parametrize(
    "threshold",
    [
        -1.0,
        0.0,
        0.9,
        float("nan"),
        float("inf"),
        -float("inf"),
        pytest.param(0.45, id="decimal-floor"),
        pytest.param(float(np.float32(0.45)), id="exact-score"),
        pytest.param(math.nextafter(float(np.float32(0.45)), -math.inf), id="below-score"),
        pytest.param(math.nextafter(float(np.float32(0.45)), math.inf), id="above-score"),
    ],
)
@pytest.mark.parametrize("use_cache", [False, True])
def test_find_similar_to_query_applies_threshold_filter(
    tmp_path: Path, monkeypatch, threshold: float, use_cache: bool
) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array([[1.0, 0.0], [0.45, math.sqrt(1 - 0.45**2)]], dtype=np.float32)

    model = RecordingModel()
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: model)
    options = {
        "query": "find addition",
        "units": units,
        "embeddings": embeddings,
        "top_k": 5,
        "device": "cpu",
        "use_cache": use_cache,
        "cache_scope": tmp_path,
    }

    # Invalid input must fail before encoding and before consuming a cached query.
    for attempt in range(2):
        calls_before = len(model.encoded)
        if not math.isfinite(threshold):
            with pytest.raises(ValueError, match="threshold must be finite"):
                find_similar_to_query(threshold=threshold, **options)
            assert len(model.encoded) == calls_before
        else:
            results = find_similar_to_query(threshold=threshold, **options)
            assert results == [
                (unit, float(row[0]))
                for unit, row in zip(units, embeddings)
                if float(row[0]) >= threshold
            ]
        if use_cache and attempt == 1:
            assert len(model.encoded) == calls_before
        if attempt == 0:
            assert len(find_similar_to_query(threshold=-1.0, **options)) == len(units)


def test_find_similar_to_query_default_threshold_is_search_default(
    tmp_path: Path, monkeypatch
) -> None:
    units = extract_arithmetic_units(tmp_path)
    # First row scores 0.7: above the search default (0.68) but below every
    # duplicate-detection gate; second row scores 0.3 and is dropped.
    embeddings = np.array([[0.7, 0.71414284], [0.3, 0.9539392]], dtype=np.float32)

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
    assert results[0][1] == pytest.approx(0.7, abs=1e-6)


def test_query_scores_bound_cosine_overshoot_before_thresholding(tmp_path: Path, monkeypatch):
    units = extract_arithmetic_units(tmp_path)
    vector = np.array([-1.2083186, -0.004454133, 0.65647495], dtype=np.float32)
    embeddings = semantic.canonicalize_embeddings([vector, -vector], expected_rows=2)

    class QueryModel:
        def encode(self, texts, **kwargs):
            return vector[None, :]

    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: QueryModel())
    results = find_similar_to_query("entry", units, embeddings, threshold=-1.0, use_cache=False)

    assert results == [(units[0], 1.0), (units[1], -1.0)]


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("auto", 0.56),
        ("generic", 0.35),
        ("embeddinggemma-300m", 0.56),
        ("gte-modernbert-base", 0.68),
    ],
)
def test_search_threshold_profile_defaults_and_numeric_precedence(choice, expected) -> None:
    assert (
        semantic.resolve_search_threshold("embeddinggemma", None, threshold_profile=choice)
        == expected
    )
    assert (
        semantic.resolve_search_threshold("embeddinggemma", 0.62, threshold_profile=choice) == 0.62
    )


@pytest.mark.parametrize("threshold", [math.nan, math.inf, -math.inf])
def test_resolve_search_threshold_rejects_nonfinite_override(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold must be finite"):
        semantic.resolve_search_threshold("embeddinggemma", threshold)
