"""Context-overflow handling: long inputs pass through to the backend unchanged with truncation diagnostics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from codedupes import semantic
from codedupes.models import CodeUnit, CodeUnitType
from codedupes.semantic import (
    compute_embeddings,
    find_similar_to_query,
)
from tests.conftest import extract_arithmetic_units
from tests.semantic_helpers import WhitespaceTokenizer


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


class _ShortContextModel:
    """Model stub with a tiny context window that records every encode call."""

    def __init__(self, *, max_seq_length: int = 8, tokenizer: object | None = None) -> None:
        self.max_seq_length = max_seq_length
        self.tokenizer = WhitespaceTokenizer() if tokenizer is None else tokenizer
        self.encode_calls: list[list[str]] = []
        self.prompts: list[str | None] = []

    def encode(self, texts, **kwargs):
        self.encode_calls.append(list(texts))
        self.prompts.append(kwargs.get("prompt"))
        return np.ones((len(texts), 2), dtype=np.float32)


def test_compute_embeddings_passes_long_code_to_backend(monkeypatch, tmp_path: Path) -> None:
    units = extract_arithmetic_units(tmp_path)
    units[0].qualified_name = "module.long_tail"
    units[0].source = "one two three four five six seven eight changed_tail"
    model = _ShortContextModel()
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: model)

    embeddings = compute_embeddings([units[0]], use_cache=False)

    assert embeddings.shape == (1, 2)
    assert model.encode_calls == [[units[0].source]]


def test_find_similar_to_query_passes_long_query_to_backend(monkeypatch, tmp_path: Path) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    model = _ShortContextModel(max_seq_length=4)
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: model)

    query = "find code that validates every record"
    results = find_similar_to_query(
        query,
        units,
        embeddings,
        threshold=0.0,
        use_cache=False,
    )

    assert len(results) == len(units)
    assert model.encode_calls == [[query]]


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


def test_context_diagnostic_counts_prompt_and_special_tokens(monkeypatch, tmp_path: Path) -> None:
    unit, short_unit = extract_arithmetic_units(tmp_path)
    unit.source = "one two three four five six seven"
    short_unit.source = "one"
    later_unit = replace(unit, qualified_name="later", source=unit.source + " eight")
    tokenizer_calls: list[tuple[list[str], bool]] = []

    class Tokenizer:
        def __call__(
            self,
            texts,
            *,
            add_special_tokens,
            truncation,
            padding,
            return_attention_mask,
            return_token_type_ids,
            verbose,
        ):
            assert truncation is False
            assert padding is False
            assert return_attention_mask is False
            assert return_token_type_ids is False
            assert verbose is False
            tokenizer_calls.append((texts, add_special_tokens))
            return {
                "input_ids": [
                    text.split() + (["special"] if add_special_tokens else []) for text in texts
                ]
            }

    model = _ShortContextModel(tokenizer=Tokenizer())
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: model)

    diagnostics = []
    embeddings = compute_embeddings(
        [unit, short_unit, later_unit],
        batch_size=2,
        instruction_prefix="task: code ",
        diagnostics=diagnostics,
        use_cache=False,
    )

    assert embeddings.shape == (3, 2)
    assert model.encode_calls == [[unit.source, short_unit.source, later_unit.source]]
    assert model.prompts == ["task: code "]
    assert tokenizer_calls == [
        (["task: code " + unit.source, "task: code " + short_unit.source], True),
        (["task: code " + later_unit.source], True),
    ]
    assert len(diagnostics) == 2
    assert "10 tokens including the encode prompt" in diagnostics[0].message
    assert "11 tokens including the encode prompt" in diagnostics[1].message
    assert all(diagnostic.code == "semantic-context-overflow" for diagnostic in diagnostics)
    assert all(diagnostic.severity == "warning" for diagnostic in diagnostics)


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
    diagnostics = []

    embeddings, _identity = semantic.compute_embeddings_with_identity(
        units,
        cache_scope=tmp_path,
        diagnostics=diagnostics,
    )

    assert model.encode_calls == [[long_source, "one two"]]
    assert embeddings.shape == (3, 2)
    assert [diagnostic.file_path for diagnostic in diagnostics] == [
        tmp_path / "a.py",
        tmp_path / "b.py",
    ]
    assert all(diagnostic.code == "semantic-context-overflow" for diagnostic in diagnostics)
    warm_diagnostics = []
    warm, _ = semantic.compute_embeddings_with_identity(
        units,
        cache_scope=tmp_path,
        diagnostics=warm_diagnostics,
    )
    np.testing.assert_array_equal(warm, embeddings)
    assert len(model.encode_calls) == 1
    assert warm_diagnostics == []


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
