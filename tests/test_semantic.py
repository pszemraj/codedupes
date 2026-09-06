from __future__ import annotations

import inspect
import logging
import math
import os
import stat
from pathlib import Path

import numpy as np
import pytest
import sentence_transformers
import torch

from codedupes import devices, semantic
from codedupes.constants import CPU_FALLBACK_MAX_BATCH_SIZE
from codedupes.embedding_cache import EmbeddingCache, compute_cache_key
from codedupes.models import CodeUnit, CodeUnitType
from codedupes.pairs import ordered_pair_key
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
    # The precomputed extraction-time count no longer matches the swapped
    # source; force the Python AST fallback these tests exercise.
    unit.statement_count = None
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
    # The precomputed extraction-time count no longer matches the swapped
    # source; force the Python AST fallback these tests exercise.
    unit.statement_count = None
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
    # The precomputed extraction-time count no longer matches the swapped
    # source; force the Python AST fallback these tests exercise.
    unit.statement_count = None
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
def test_uncalibrated_search_context_requires_explicit_threshold(
    tmp_path: Path, monkeypatch, kwargs: dict[str, object], use_cache: bool
) -> None:
    units = extract_arithmetic_units(tmp_path)
    model = _RecordingModel()
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
                "find addition", units, embeddings, corpus_identity=identity, **options
            )
        assert len(model.encoded) == calls_before
        assert len(
            find_similar_to_query(
                "find addition",
                units,
                embeddings,
                threshold=0.0,
                corpus_identity=identity,
                **options,
            )
        ) == len(units)


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

    model = _RecordingModel()
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
    # First row scores 0.6: above the search default (0.50) but below every
    # duplicate-detection gate; second row scores 0.3 and is dropped.
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


def test_find_semantic_duplicates_rejects_nan_and_inf_but_keeps_finite_pair(
    tmp_path: Path,
) -> None:
    # Rounds out the isfinite guard's coverage (see the comment above the
    # pair loop in find_semantic_duplicates): both a NaN similarity and a
    # +inf similarity must be dropped, and neither should suppress a genuine
    # finite above-threshold pair computed in the same run.
    units = extract_units(
        tmp_path,
        """
        def first(x):
            return x + 1

        def second(x):
            return x + 2

        def third(x):
            return x + 3

        def fourth(x):
            return x + 4
        """,
    )
    embeddings = np.array(
        [
            [np.nan, 0.0],  # first: every pair involving this row is NaN.
            [1.0, 0.0],  # second
            [1.0, 0.0],  # third: (second, third) is a legitimate finite pair.
            [np.inf, 0.0],  # fourth: every pair involving this row is +inf.
        ],
        dtype=np.float32,
    )

    duplicates = find_semantic_duplicates(units, embeddings, threshold=0.9)

    assert len(duplicates) == 1
    kept = duplicates[0]
    assert {kept.unit_a.name, kept.unit_b.name} == {"second", "third"}
    assert kept.similarity == pytest.approx(1.0)


@pytest.mark.parametrize("threshold", [0.82, 0.78, 0.70, 0.90])
def test_find_semantic_duplicates_rechecks_threshold_after_numpy_prefilter(
    tmp_path: Path, threshold: float
) -> None:
    units = extract_arithmetic_units(tmp_path)
    rounded_down = np.float32(threshold)
    assert float(rounded_down) < threshold
    embeddings = np.array([[1.0, 0.0], [rounded_down, 0.0]], dtype=np.float32)

    duplicates = find_semantic_duplicates(units, embeddings, threshold=threshold)

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


def test_find_semantic_duplicates_cross_language_requires_opt_in(tmp_path: Path) -> None:
    python_path = tmp_path / "sample.py"
    python_path.write_text("def f():\n    return 1\n")
    rust_path = tmp_path / "sample.rs"
    rust_path.write_text("fn f() -> i64 { 1 }\n")

    python_unit = CodeUnit(
        name="f",
        qualified_name="sample.f",
        unit_type=CodeUnitType.FUNCTION,
        file_path=python_path,
        lineno=1,
        end_lineno=2,
        source="def f():\n    return 1",
        is_public=True,
        is_exported=False,
        language="python",
    )
    rust_unit = CodeUnit(
        name="f",
        qualified_name="sample::f",
        unit_type=CodeUnitType.FUNCTION,
        file_path=rust_path,
        lineno=1,
        end_lineno=1,
        source="fn f() -> i64 { 1 }",
        is_public=True,
        is_exported=False,
        language="rust",
    )
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    same_language_only = find_semantic_duplicates(
        units=[python_unit, rust_unit],
        embeddings=embeddings,
        threshold=0.9,
    )
    assert same_language_only == []

    cross = find_semantic_duplicates(
        units=[python_unit, rust_unit],
        embeddings=embeddings,
        threshold=0.9,
        cross_language=True,
    )
    assert [(pair.unit_a.language, pair.unit_b.language) for pair in cross] == [("python", "rust")]


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

    def _fail_if_called(*_a, **_k):
        raise AssertionError("must not resolve a concrete device when the CPU gate is false")

    monkeypatch.setattr(semantic, "_resolve_semantic_device_request", _fail_if_called)

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
    ("device", "platform_name", "expect_mps_possible"),
    [
        ("mps", "darwin", True),
        ("mps", "linux", True),
        ("mps", "win32", True),
        ("auto", "darwin", True),
        ("auto", "linux", False),
        ("auto", "win32", False),
        ("cpu", "darwin", False),
        ("cuda", "darwin", False),
        (None, "darwin", True),
        (None, "linux", False),
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

        def _fail_if_called(*_a, **_k):
            raise AssertionError("must not resolve a concrete device when the MPS shortcut applies")

        monkeypatch.setattr(semantic, "_resolve_semantic_device_request", _fail_if_called)
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


def test_compute_embeddings_passes_long_code_to_backend(monkeypatch, tmp_path: Path) -> None:
    units = extract_arithmetic_units(tmp_path)
    units[0].qualified_name = "module.long_tail"
    units[0].source = "one two three four five six seven eight changed_tail"
    encode_calls: list[list[str]] = []

    class Tokenizer:
        def encode(self, text, **kwargs):
            return text.split()

    class ShortContextModel:
        max_seq_length = 8
        tokenizer = Tokenizer()

        def encode(self, texts, **kwargs):
            encode_calls.append(list(texts))
            return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: ShortContextModel())

    embeddings = compute_embeddings([units[0]], use_cache=False)

    assert embeddings.shape == (1, 2)
    assert encode_calls == [[units[0].source]]


def test_find_similar_to_query_passes_long_query_to_backend(monkeypatch, tmp_path: Path) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    encode_calls: list[list[str]] = []

    class Tokenizer:
        def encode(self, text, **kwargs):
            return text.split()

    class ShortContextModel:
        max_seq_length = 4
        tokenizer = Tokenizer()

        def encode(self, texts, **kwargs):
            encode_calls.append(list(texts))
            return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: ShortContextModel())

    query = "find code that validates every record"
    results = find_similar_to_query(
        query,
        units,
        embeddings,
        threshold=0.0,
        use_cache=False,
    )

    assert len(results) == len(units)
    assert encode_calls == [[query]]


class _WhitespaceTokenizer:
    """Tokenizer stub whose token count is the whitespace-separated word count."""

    def encode(self, text, **_kwargs):
        return text.split()


class _ShortContextModel:
    """Model stub with a tiny context window that records every encode call."""

    max_seq_length = 8
    tokenizer = _WhitespaceTokenizer()

    def __init__(self) -> None:
        self.encode_calls: list[list[str]] = []
        self.prompts: list[str | None] = []

    def encode(self, texts, **kwargs):
        self.encode_calls.append(list(texts))
        self.prompts.append(kwargs.get("prompt"))
        return np.ones((len(texts), 2), dtype=np.float32)


def test_code_truncation_is_left_to_backend_with_prompt(monkeypatch, tmp_path: Path) -> None:
    # Even when the prompt pushes the input over the context limit, pass both
    # through unchanged so the backend applies its normal tokenization policy.
    units = extract_arithmetic_units(tmp_path)
    units[0].qualified_name = "module.exact_fit"
    units[0].source = "one two three four five six seven eight"
    model = _ShortContextModel()
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: model)

    compute_embeddings([units[0]], use_cache=False)
    assert model.encode_calls == [["one two three four five six seven eight"]]

    embeddings = compute_embeddings([units[0]], instruction_prefix="task: code ", use_cache=False)

    assert embeddings.shape == (1, 2)
    assert model.encode_calls == [[units[0].source], [units[0].source]]
    assert model.prompts == [None, "task: code "]


def test_query_truncation_is_left_to_backend_with_prompt(monkeypatch, tmp_path: Path) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    model = _ShortContextModel()
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: model)
    corpus_identity = semantic.resolve_embedding_space_identity(
        instruction_prefix="task: search ",
    )

    query = "find the code that validates every incoming record"
    results = find_similar_to_query(
        query,
        units,
        embeddings,
        instruction_prefix="task: search ",
        threshold=0.0,
        use_cache=False,
        corpus_identity=corpus_identity,
    )

    assert len(results) == len(units)
    assert model.encode_calls == [[query]]
    assert model.prompts == ["task: search "]


def test_long_duplicate_texts_retain_all_rows_and_reuse_cache(monkeypatch, tmp_path: Path) -> None:
    # Duplicate sources share one encoded input and cache key while each unit
    # keeps its own row in the returned matrix.
    long_source = "one two three four five six seven eight nine"
    units = [
        CodeUnit(
            name="long_a",
            qualified_name="mod.long_a",
            unit_type=CodeUnitType.FUNCTION,
            file_path=tmp_path / "a.py",
            lineno=1,
            end_lineno=2,
            source=long_source,
        ),
        CodeUnit(
            name="long_b",
            qualified_name="mod.long_b",
            unit_type=CodeUnitType.FUNCTION,
            file_path=tmp_path / "b.py",
            lineno=1,
            end_lineno=2,
            source=long_source,
        ),
        CodeUnit(
            name="short",
            qualified_name="mod.short",
            unit_type=CodeUnitType.FUNCTION,
            file_path=tmp_path / "c.py",
            lineno=1,
            end_lineno=2,
            source="one two",
        ),
    ]
    model = _ShortContextModel()
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: model)

    embeddings, _identity = semantic.compute_embeddings_with_identity(
        units,
        cache_scope=tmp_path,
    )

    assert model.encode_calls == [[long_source, "one two"]]
    assert embeddings.shape == (3, 2)
    warm, _ = semantic.compute_embeddings_with_identity(units, cache_scope=tmp_path)
    np.testing.assert_array_equal(warm, embeddings)
    assert len(model.encode_calls) == 1


def test_all_long_inputs_remain_in_corpus(monkeypatch, tmp_path: Path) -> None:
    units = extract_arithmetic_units(tmp_path)
    for index, unit in enumerate(units):
        unit.source = f"one two three four five six seven eight nine {index}"
    model = _ShortContextModel()
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: model)

    embeddings, _identity = semantic.compute_embeddings_with_identity(
        units,
        cache_scope=tmp_path,
    )

    assert model.encode_calls == [[unit.source for unit in units]]
    assert embeddings.shape == (len(units), 2)


class _RecordingModel:
    """Model stub returning one fixed vector per call and recording its inputs."""

    def __init__(self) -> None:
        self.encoded: list[list[str]] = []

    def encode(self, texts, **_kwargs):
        self.encoded.append(list(texts))
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


@pytest.mark.parametrize(
    ("revision", "expect_bypass"),
    [("main", True), ("b" * 40, False)],
)
def test_unreportable_mutable_provenance_bypasses_the_query_cache(
    monkeypatch, tmp_path: Path, revision: str, expect_bypass: bool
) -> None:
    # A corpus that had to bypass its shard (mutable branch, no reportable
    # commit) has no provenance to compare a cached query row against, so the
    # query cache must be bypassed with it.
    units = extract_arithmetic_units(tmp_path)
    model = _RecordingModel()
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: None)

    embeddings, identity = semantic.compute_embeddings_with_identity(
        units,
        model_name="test-model",
        revision=revision,
        cache_scope=tmp_path,
    )
    assert identity.source_commit is None

    for _ in range(2):
        find_similar_to_query(
            "find addition",
            units,
            embeddings,
            model_name="test-model",
            revision=revision,
            cache_scope=tmp_path,
            corpus_identity=identity,
            threshold=0.0,
        )

    query_encodes = [call for call in model.encoded if call == ["find addition"]]
    assert len(query_encodes) == (2 if expect_bypass else 1)


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


def test_compute_embeddings_empty_corpus_raises_for_explicit_unavailable_device(
    tmp_path: Path, monkeypatch
) -> None:
    def _raise_unavailable(*_args, **_kwargs):
        raise SemanticBackendError("mps is not available in this environment")

    monkeypatch.setattr(semantic, "_resolve_semantic_device_request", _raise_unavailable)
    monkeypatch.setattr(semantic, "get_model", _fail_if_called)

    with pytest.raises(SemanticBackendError):
        compute_embeddings([], device="mps", cache_scope=tmp_path)


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
        threshold=0.0,
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


def test_warm_cache_returns_with_unset_fraction_restore_managed_mps_cap(
    tmp_path: Path, monkeypatch
) -> None:
    units = _warm_corpus_cache(tmp_path, monkeypatch)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    find_similar_to_query(
        "find addition",
        units,
        embeddings,
        model_name="gte-modernbert-base",
        revision=_FULL_REVISION,
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
    monkeypatch.setattr(semantic, "get_model", _fail_if_called)
    monkeypatch.setattr(semantic, "_prepare_semantic_device", _fail_if_called)

    # Fully cache-covered corpus run: the warm return must still restore a
    # previously managed allocator cap when this run leaves the fraction unset.
    compute_embeddings(
        units,
        model_name="gte-modernbert-base",
        revision=_FULL_REVISION,
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
        revision=_FULL_REVISION,
        device="cpu",
        threshold=0.0,
        cache_scope=tmp_path,
    )
    assert restore_calls == [True]

    # A run that requests its own fraction is not "unset": the warm corpus
    # return must leave cap management to the next real device preparation.
    restore_calls.clear()
    compute_embeddings(
        units,
        model_name="gte-modernbert-base",
        revision=_FULL_REVISION,
        device="cpu",
        mps_memory_fraction=0.9,
        cache_scope=tmp_path,
    )
    assert restore_calls == []


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
        threshold=0.0,
        cache_scope=tmp_path,
    )

    assert captured["max_namespace_keys"] == semantic._MAX_CACHED_QUERY_KEYS


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
    monkeypatch.setattr(semantic, "get_model", _fail_if_called)

    result = compute_embeddings(
        units,
        model_name="gte-modernbert-base",
        revision=_FULL_REVISION,
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
    model = _BfloatAcceleratorFallbackModel()
    monkeypatch.setattr(semantic, "get_model", lambda *_a, **_k: model)

    embeddings, identity = semantic.compute_embeddings_with_identity(
        units,
        model_name="gte-modernbert-base",
        revision=_FULL_REVISION,
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
        revision=_FULL_REVISION,
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
        revision=_FULL_REVISION,
        semantic_task=semantic.DEFAULT_CHECK_SEMANTIC_TASK,
        device=query_device,
        threshold=0.0,
        cache_scope=tmp_path,
        corpus_identity=identity,
    )
    assert hits


class _QueryOOMBfloatModel:
    """Fake bf16 CUDA model whose corpus encode succeeds but whose query encode OOMs."""

    def __init__(self) -> None:
        self._dtype = torch.bfloat16
        self.corpus_encoded = False

    def parameters(self):
        yield torch.zeros(1, dtype=self._dtype)

    def to(self, device=None, dtype=None):
        if dtype is not None:
            self._dtype = dtype
        return self

    def encode(self, texts, **kwargs):
        if not self.corpus_encoded:
            self.corpus_encoded = True
            return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)
        if kwargs.get("device") != "cpu":
            raise RuntimeError("CUDA out of memory")
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)


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
    model = _QueryOOMBfloatModel()
    monkeypatch.setattr(semantic, "get_model", lambda *_a, **_k: model)

    embeddings, identity = semantic.compute_embeddings_with_identity(
        units,
        model_name="gte-modernbert-base",
        revision=_FULL_REVISION,
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
            revision=_FULL_REVISION,
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


# --- Finding 1: memoized identity-path fingerprint --------------------------


def _counting_fingerprint_stub(monkeypatch, return_value="dir-stub"):
    """Patch the raw walker with a call-counting stub and return the counter list."""
    calls: list[Path] = []

    def _stub(model_dir: Path, *, persist_manifest: bool = True) -> str | None:
        calls.append(model_dir)
        return return_value

    monkeypatch.setattr(semantic, "_fingerprint_local_model_dir", _stub)
    return calls


def test_fingerprint_local_model_dir_cached_walks_once_within_an_open_scope(
    tmp_path: Path, monkeypatch
) -> None:
    model_dir = tmp_path / "model"
    calls = _counting_fingerprint_stub(monkeypatch)

    with semantic._local_model_fingerprint_walk_scope():
        first = semantic._fingerprint_local_model_dir_cached(model_dir)
        second = semantic._fingerprint_local_model_dir_cached(model_dir)

    assert first == "dir-stub"
    assert second == "dir-stub"
    assert len(calls) == 1


def test_fingerprint_local_model_dir_cached_walks_fresh_outside_any_scope(
    tmp_path: Path, monkeypatch
) -> None:
    """No active scope (the default) must match ``_fingerprint_local_model_dir`` exactly.

    This is the safe-by-default behavior every external caller of
    :func:`resolve_embedding_space_identity`/``_resolve_revision_for_cache``
    relies on: without an explicitly opened call scope, every lookup walks.
    """
    assert semantic._local_model_fingerprint_scope is None
    model_dir = tmp_path / "model"
    calls = _counting_fingerprint_stub(monkeypatch)

    semantic._fingerprint_local_model_dir_cached(model_dir)
    semantic._fingerprint_local_model_dir_cached(model_dir)

    assert len(calls) == 2


def test_fingerprint_local_model_dir_cached_recomputes_across_separate_scopes(
    tmp_path: Path, monkeypatch
) -> None:
    """Each top-level scope starts empty: nothing survives from an earlier one."""
    model_dir = tmp_path / "model"
    calls = _counting_fingerprint_stub(monkeypatch)

    with semantic._local_model_fingerprint_walk_scope():
        semantic._fingerprint_local_model_dir_cached(model_dir)
    with semantic._local_model_fingerprint_walk_scope():
        semantic._fingerprint_local_model_dir_cached(model_dir)

    assert len(calls) == 2


def test_fingerprint_local_model_dir_cached_recomputes_for_a_different_path(
    tmp_path: Path, monkeypatch
) -> None:
    calls = _counting_fingerprint_stub(monkeypatch)

    with semantic._local_model_fingerprint_walk_scope():
        semantic._fingerprint_local_model_dir_cached(tmp_path / "model-a")
        semantic._fingerprint_local_model_dir_cached(tmp_path / "model-b")

    assert len(calls) == 2


def test_local_model_fingerprint_walk_scope_is_reentrant(tmp_path: Path, monkeypatch) -> None:
    """A nested scope open (recursion under the same lock) extends the outer one."""
    model_dir = tmp_path / "model"
    calls = _counting_fingerprint_stub(monkeypatch)

    with semantic._local_model_fingerprint_walk_scope():
        semantic._fingerprint_local_model_dir_cached(model_dir)
        with semantic._local_model_fingerprint_walk_scope():
            # Reuses the outer scope's memo instead of starting a fresh one.
            semantic._fingerprint_local_model_dir_cached(model_dir)
        # The inner "with" must not have torn down the outer scope.
        assert semantic._local_model_fingerprint_scope is not None
        semantic._fingerprint_local_model_dir_cached(model_dir)

    assert semantic._local_model_fingerprint_scope is None
    assert len(calls) == 1


def test_remember_local_model_fingerprint_in_scope_seeds_without_a_walk(
    tmp_path: Path, monkeypatch
) -> None:
    model_dir = tmp_path / "model"
    calls = _counting_fingerprint_stub(monkeypatch)

    with semantic._local_model_fingerprint_walk_scope():
        semantic._remember_local_model_fingerprint_in_scope(model_dir, "dir-from-load")
        result = semantic._fingerprint_local_model_dir_cached(model_dir)

    assert result == "dir-from-load"
    assert calls == []


def test_remember_local_model_fingerprint_in_scope_is_a_no_op_without_a_scope(
    tmp_path: Path, monkeypatch
) -> None:
    assert semantic._local_model_fingerprint_scope is None
    model_dir = tmp_path / "model"
    calls = _counting_fingerprint_stub(monkeypatch)

    semantic._remember_local_model_fingerprint_in_scope(model_dir, "dir-from-load")
    result = semantic._fingerprint_local_model_dir_cached(model_dir)

    assert result == "dir-stub"
    assert len(calls) == 1


def test_resolve_embedding_space_identity_shares_one_walk_within_an_open_scope(
    tmp_path: Path, monkeypatch
) -> None:
    """Repeated identity resolution inside one open scope shares a single walk."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}')
    (model_dir / "model.safetensors").write_text("weights")

    walk_calls = _counting_fingerprint_stub(monkeypatch, return_value="dir-once")

    with semantic._local_model_fingerprint_walk_scope():
        first = semantic.resolve_embedding_space_identity(model_name=str(model_dir), device="cpu")
        second = semantic.resolve_embedding_space_identity(model_name=str(model_dir), device="cpu")

    assert first == second
    assert first.resolved_revision == "dir-once"
    assert len(walk_calls) == 1


def test_resolve_embedding_space_identity_detects_disk_changes_across_separate_calls(
    tmp_path: Path,
) -> None:
    """Outside a shared scope, every call must reflect the live on-disk state.

    This is the provenance guarantee finding 1 must preserve: without an
    explicitly opened call scope (the state every caller outside
    search()/compute_embeddings is in), a directory edit between two calls
    must never be masked by a leftover memoized fingerprint.
    """
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}')
    weights_path = model_dir / "model.safetensors"
    weights_path.write_text("weights-v1")

    before = semantic.resolve_embedding_space_identity(model_name=str(model_dir), device="cpu")
    weights_path.write_text("weights-v2-changed")
    after = semantic.resolve_embedding_space_identity(model_name=str(model_dir), device="cpu")

    assert before.resolved_revision != after.resolved_revision


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
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "a" * 40)

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
    commit_a = "a" * 40
    commit_b = "b" * 40
    loaded_revisions: list[str | None] = []

    def fake_get_model(*_args, **kwargs):
        loaded_revisions.append(kwargs.get("revision"))
        return model

    monkeypatch.setattr(semantic, "get_model", fake_get_model)
    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", lambda *_a, **_k: commit_a)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: commit_a)

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
    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", lambda *_a, **_k: commit_b)
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: commit_b)

    compute_embeddings(
        units,
        model_name="some-generic-model",
        device="cpu",
        cache_scope=tmp_path,
        strict_revision_cache=True,
    )

    assert model.encode_calls == 2
    assert loaded_revisions == [commit_a, commit_b]


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


def _scan_fixture() -> tuple[list[CodeUnit], np.ndarray]:
    """Build a deterministic two-language corpus for the pairwise-scan fuzz.

    Every component is +/-0.25 across 16 dimensions, so rows are exactly
    unit-norm and every dot product is an exact multiple of 1/16. Blocking a
    matrix multiply therefore cannot perturb a single similarity, which lets the
    reference below compare bit-exactly, and the resulting ties make the scan's
    emission order observable through the stable final sort.

    :return: Units and their row-aligned embedding matrix.
    """
    count, dimensions = 1040, 16
    rng = np.random.default_rng(20260825)
    embeddings = (
        rng.integers(0, 2, size=(count, dimensions)).astype(np.float32) * 2.0 - 1.0
    ) * 0.25

    # Non-finite rows. Row 0 is all-positive so its product with the +inf row is
    # +inf rather than NaN, exercising the finite guard behind the gate mask.
    embeddings[0] = 0.25
    embeddings[1] = np.inf
    embeddings[4] = np.nan

    # Perfect-similarity rows for the three post-gate filters: an overlapping
    # same-file pair, a surviving cross-file twin, and a class/function pair.
    embeddings[9] = embeddings[8]
    embeddings[13] = embeddings[12]
    embeddings[17] = embeddings[16]

    units: list[CodeUnit] = []
    for index in range(count):
        language = "python" if index % 4 < 2 else "rust"
        if (index + 1) % 17 == 0:
            unit_type = CodeUnitType.CLASS
        elif index % 5 == 0:
            unit_type = CodeUnitType.METHOD
        else:
            unit_type = CodeUnitType.FUNCTION
        units.append(
            CodeUnit(
                name=f"unit_{index}",
                qualified_name=f"mod_{index}.unit_{index}",
                unit_type=unit_type,
                file_path=Path(f"mod_{index}.{'py' if language == 'python' else 'rs'}"),
                lineno=1,
                end_lineno=4,
                source=f"def unit_{index}(): ...",
                language=language,
                start_byte=0,
                end_byte=40,
            )
        )

    units[9].file_path = units[8].file_path
    units[9].start_byte = 20
    units[9].end_byte = 60
    return units, embeddings


def _reference_semantic_duplicates(
    units: list[CodeUnit],
    embeddings: np.ndarray,
    threshold: float,
    *,
    exclude_exact: set[tuple[str, str]] | None = None,
    cross_language: bool = False,
    language_thresholds: dict[str, float] | None = None,
) -> list[tuple[tuple[str, str], float]]:
    """Restate the documented duplicate-scan semantics as a naive O(N^2) loop.

    :param units: Candidate units row-aligned with ``embeddings``.
    :param embeddings: Embedding matrix.
    :param threshold: Fallback gate for languages without a calibrated gate.
    :param exclude_exact: Pairs the scan must not report.
    :param cross_language: Whether to scan one mixed group instead of per-language groups.
    :param language_thresholds: Per-language duplicate gates.
    :return: ``(ordered pair key, similarity)`` in the order the scan must report.
    """
    excluded = exclude_exact or set()
    gates = dict(language_thresholds or {})
    similarity_matrix = embeddings @ embeddings.T

    if cross_language:
        groups = {"*": list(range(len(units)))}
    else:
        groups = {}
        for index, unit in enumerate(units):
            groups.setdefault(unit.language, []).append(index)

    reported: list[tuple[tuple[str, str], float]] = []
    for language, indices in groups.items():
        if cross_language:
            group_gate = min(gates.get(units[index].language, threshold) for index in indices)
        else:
            group_gate = gates.get(language, threshold)
        for position, index_a in enumerate(indices):
            unit_a = units[index_a]
            row = similarity_matrix[index_a].tolist()
            for index_b in indices[position + 1 :]:
                similarity = row[index_b]
                if not math.isfinite(similarity) or similarity < group_gate:
                    continue
                unit_b = units[index_b]
                if cross_language and similarity < min(
                    gates.get(unit_a.language, threshold),
                    gates.get(unit_b.language, threshold),
                ):
                    continue
                kinds = {unit_a.unit_type, unit_b.unit_type}
                if len(kinds) > 1 and kinds != {CodeUnitType.FUNCTION, CodeUnitType.METHOD}:
                    continue
                # Every fixture unit has a real byte range, so overlap is the
                # same-file byte-interval test.
                if (
                    unit_a.file_path == unit_b.file_path
                    and unit_a.start_byte < unit_b.end_byte
                    and unit_b.start_byte < unit_a.end_byte
                ):
                    continue
                key = ordered_pair_key(unit_a, unit_b)
                if key in excluded:
                    continue
                reported.append((key, similarity))
    reported.sort(key=lambda entry: entry[1], reverse=True)
    return reported


def _pair_view(duplicates) -> list[tuple[tuple[str, str], float]]:
    """Reduce reported duplicates to comparable ``(pair key, similarity)`` entries.

    :param duplicates: Duplicate pairs as reported by the scan.
    :return: Pair keys with similarities, in report order.
    """
    return [(ordered_pair_key(pair.unit_a, pair.unit_b), pair.similarity) for pair in duplicates]


def _random_float_scan_fixture() -> tuple[list[CodeUnit], np.ndarray]:
    """Build a random-float fixture that exposes width-dependent BLAS scores.

    :return: Units and their row-aligned embedding matrix.
    """
    count, dimensions = 520, 768
    rng = np.random.default_rng(20260826)
    embeddings = rng.normal(size=(count, dimensions)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Make the final partial row block mutually similar, leaving a small,
    # deterministic above-gate result set whose scores still use all 768 dims.
    base = rng.normal(size=dimensions).astype(np.float32)
    base /= np.linalg.norm(base)
    embeddings[-20:] = base + rng.normal(scale=0.005, size=(20, dimensions)).astype(np.float32)
    embeddings[-20:] /= np.linalg.norm(embeddings[-20:], axis=1, keepdims=True)

    units = [
        CodeUnit(
            name=f"random_{index}",
            qualified_name=f"random_{index}",
            unit_type=CodeUnitType.FUNCTION,
            file_path=Path(f"random_{index}.py"),
            lineno=1,
            end_lineno=2,
            source=f"def random_{index}(): ...",
            language="python",
            start_byte=0,
            end_byte=20,
        )
        for index in range(count)
    ]
    return units, embeddings


def test_vectorized_pair_scan_matches_naive_reference() -> None:
    # The scan multiplies full-width row chunks and thresholds column blocks in
    # numpy; a wrong column offset or mis-ordered candidate walk only shows up
    # past the 500-row chunk boundary, which this 520-per-language corpus crosses
    # in every mode.
    units, embeddings = _scan_fixture()
    gates = {"python": 0.875, "rust": 0.75}

    same_language = find_semantic_duplicates(
        units, embeddings, threshold=0.75, language_thresholds=gates
    )
    expected = _reference_semantic_duplicates(units, embeddings, 0.75, language_thresholds=gates)
    assert _pair_view(same_language) == expected
    assert len(expected) > 100

    reported_keys = {key for key, _ in _pair_view(same_language)}
    assert ordered_pair_key(units[12], units[13]) in reported_keys
    assert ordered_pair_key(units[8], units[9]) not in reported_keys
    assert ordered_pair_key(units[16], units[17]) not in reported_keys
    assert ordered_pair_key(units[0], units[1]) not in reported_keys
    assert all(units[4].uid not in key for key in reported_keys)

    cross = find_semantic_duplicates(
        units, embeddings, threshold=0.9, cross_language=True, language_thresholds=gates
    )
    cross_expected = _reference_semantic_duplicates(
        units, embeddings, 0.9, cross_language=True, language_thresholds=gates
    )
    assert _pair_view(cross) == cross_expected
    assert any(pair.unit_a.language != pair.unit_b.language for pair in cross)

    excluded = {key for key, _ in expected[::37]}
    filtered = find_semantic_duplicates(
        units, embeddings, threshold=0.75, exclude_exact=excluded, language_thresholds=gates
    )
    filtered_expected = _reference_semantic_duplicates(
        units, embeddings, 0.75, exclude_exact=excluded, language_thresholds=gates
    )
    assert _pair_view(filtered) == filtered_expected
    assert len(filtered) == len(expected) - len(excluded)


def test_vectorized_pair_scan_preserves_full_width_float32_scores_and_order() -> None:
    units, embeddings = _random_float_scan_fixture()
    threshold = 0.97

    # This is the product shape used by the original scalar candidate walk.
    full_width_scores = embeddings[500:] @ embeddings.T

    expected: list[tuple[tuple[str, str], float]] = []
    for local_idx, group_i in enumerate(range(500, 520)):
        for group_j in range(group_i + 1, 520):
            similarity = float(full_width_scores[local_idx, group_j])
            if similarity >= threshold:
                expected.append((ordered_pair_key(units[group_i], units[group_j]), similarity))
    expected.sort(key=lambda entry: entry[1], reverse=True)
    assert len(expected) == 190

    matmul_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    class MatmulTracingArray(np.ndarray):
        def __matmul__(self, other):
            matmul_shapes.append((self.shape, other.shape))
            return np.ndarray.__matmul__(self, other)

    reported = find_semantic_duplicates(
        units, embeddings.view(MatmulTracingArray), threshold=threshold
    )

    assert _pair_view(reported) == expected
    assert matmul_shapes == [((500, 768), (768, 520)), ((20, 768), (768, 520))]


def test_vectorized_pair_scan_bounds_candidate_extraction_and_masks_lower_triangle(
    monkeypatch,
) -> None:
    units, _ = _random_float_scan_fixture()
    shared_path = Path("overlapping.py")
    for unit in units:
        unit.file_path = shared_path
    embeddings = np.ones((len(units), 1), dtype=np.float32)

    nonzero_calls: list[tuple[tuple[int, ...], int]] = []
    original_nonzero = np.nonzero

    def recording_nonzero(mask):
        result = original_nonzero(mask)
        nonzero_calls.append((mask.shape, len(result[0])))
        return result

    monkeypatch.setattr(semantic.np, "nonzero", recording_nonzero)

    # Every score clears the gate, while overlap filtering keeps the returned
    # result empty so this test measures intermediate candidate batching only.
    assert find_semantic_duplicates(units, embeddings, threshold=0.0) == []

    block_size = semantic._PAIRWISE_SCAN_BLOCK_SIZE
    assert nonzero_calls == [
        ((500, 500), 500 * 499 // 2),
        ((500, 20), 500 * 20),
        ((20, 20), 20 * 19 // 2),
    ]
    assert max(rows * columns for (rows, columns), _ in nonzero_calls) <= block_size**2


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def f():\r\n    return 1\r\n", "def f():\n    return 1"),
        ("def f():\r    return 1\r", "def f():\n    return 1"),
        ("\r\n  keep  \r\n", "keep"),
        ("already\nnormalized", "already\nnormalized"),
    ],
)
def test_prepare_embedding_text_normalizes_line_endings(source: str, expected: str) -> None:
    assert semantic._prepare_embedding_text(source) == expected


def test_crlf_and_lf_units_share_one_embedding_cache_key(tmp_path: Path) -> None:
    lf_source = "fn f() -> i64 {\n    1\n}\n"
    lf_unit = CodeUnit(
        name="f",
        qualified_name="sample::f",
        unit_type=CodeUnitType.FUNCTION,
        file_path=tmp_path / "lf.rs",
        lineno=1,
        end_lineno=3,
        source=lf_source,
        language="rust",
    )
    crlf_unit = CodeUnit(
        name="f",
        qualified_name="sample::f",
        unit_type=CodeUnitType.FUNCTION,
        file_path=tmp_path / "crlf.rs",
        lineno=1,
        end_lineno=3,
        source=lf_source.replace("\n", "\r\n"),
        language="rust",
    )

    lf_text = semantic._prepare_embedding_text(lf_unit.source)
    crlf_text = semantic._prepare_embedding_text(crlf_unit.source)
    assert lf_text == crlf_text

    lf_key = compute_cache_key("model", "revision", lf_text)
    assert lf_key == compute_cache_key("model", "revision", crlf_text)
    # Without normalization the two checkouts would key - and embed - apart.
    assert lf_key != compute_cache_key("model", "revision", crlf_unit.source.strip())


def test_compute_embeddings_normalizes_crlf_before_encoding(tmp_path: Path, monkeypatch) -> None:
    unit = CodeUnit(
        name="f",
        qualified_name="sample::f",
        unit_type=CodeUnitType.FUNCTION,
        file_path=tmp_path / "sample.rs",
        lineno=1,
        end_lineno=3,
        source="fn f() -> i64 {\r\n    1\r\n}\r\n",
        language="rust",
    )
    model = _RecordingModel()
    monkeypatch.setattr(semantic, "get_model", lambda *_a, **_k: model)

    compute_embeddings([unit], device="cpu")

    assert model.encoded == [["fn f() -> i64 {\n    1\n}"]]
