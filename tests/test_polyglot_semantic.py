"""Polyglot semantic comparison boundaries that do not require a model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from codedupes.models import CodeUnit, CodeUnitType
from codedupes.semantic import find_semantic_duplicates


def _unit(path: str, name: str, language: str) -> CodeUnit:
    return CodeUnit(
        name=name,
        qualified_name=name,
        unit_type=CodeUnitType.FUNCTION,
        file_path=Path(path),
        lineno=1,
        end_lineno=2,
        source=f"function {name}",
        language=language,
        start_byte=0,
        end_byte=10,
    )


def _pairwise_matrix(cosine: float, count: int) -> np.ndarray:
    """Build unit vectors whose every distinct pair has the given cosine.

    :param cosine: Target cosine similarity between any two distinct rows.
    :param count: Number of rows.
    :return: Row-normalized embedding matrix.
    """
    shared = np.sqrt(cosine)
    matrix = np.zeros((count, count + 1), dtype=np.float32)
    matrix[:, 0] = shared
    for row in range(count):
        matrix[row, row + 1] = np.sqrt(1.0 - cosine)
    return matrix


def _tracing_matrix(values: np.ndarray, fancy_indexes: list[object]) -> np.ndarray:
    """Wrap a matrix so list-style (copying) indexing is recorded.

    :param values: Embedding matrix to wrap.
    :param fancy_indexes: Sink recording every fancy-index operation.
    :return: Matrix view that records fancy indexing.
    """

    class _Tracing(np.ndarray):
        def __getitem__(self, item):
            if isinstance(item, list):
                fancy_indexes.append(item)
            return np.ndarray.__getitem__(self, item)

    return values.view(_Tracing)


def test_per_language_gates_are_applied_inside_the_scan() -> None:
    units = [
        _unit("alpha.py", "alpha_one", "python"),
        _unit("beta.py", "alpha_two", "python"),
        _unit("alpha.js", "betaOne", "javascript"),
        _unit("beta.js", "betaTwo", "javascript"),
    ]
    embeddings = _pairwise_matrix(0.75, len(units))

    duplicates = find_semantic_duplicates(
        units,
        embeddings,
        threshold=0.60,
        language_thresholds={"python": 0.90, "javascript": 0.60},
    )

    assert [(pair.unit_a.name, pair.unit_b.name) for pair in duplicates] == [("betaOne", "betaTwo")]


def test_cross_language_pairs_use_the_looser_of_both_language_gates() -> None:
    units = [
        _unit("alpha.py", "alpha_one", "python"),
        _unit("beta.py", "alpha_two", "python"),
        _unit("alpha.js", "betaOne", "javascript"),
    ]
    embeddings = _pairwise_matrix(0.70, len(units))

    duplicates = find_semantic_duplicates(
        units,
        embeddings,
        # The fallback is intentionally stricter than either calibrated gate:
        # it must not prefilter the mixed group before endpoint gates apply.
        threshold=0.95,
        cross_language=True,
        language_thresholds={"python": 0.90, "javascript": 0.60},
    )

    # Mixed pairs clear min(0.90, 0.60); the python/python pair still needs 0.90.
    assert {(pair.unit_a.name, pair.unit_b.name) for pair in duplicates} == {
        ("alpha_one", "betaOne"),
        ("alpha_two", "betaOne"),
    }


def test_single_language_scan_reuses_the_embedding_matrix() -> None:
    fancy_indexes: list[object] = []
    same_language = [_unit("a.py", "a", "python"), _unit("b.py", "b", "python")]
    find_semantic_duplicates(
        same_language,
        _tracing_matrix(_pairwise_matrix(0.99, 2), fancy_indexes),
        threshold=0.99,
    )
    assert fancy_indexes == []

    mixed = [_unit("a.py", "a", "python"), _unit("b.js", "b", "javascript")]
    find_semantic_duplicates(
        mixed,
        _tracing_matrix(_pairwise_matrix(0.99, 2), fancy_indexes),
        threshold=0.99,
    )
    assert len(fancy_indexes) == 2


def test_semantic_duplicate_matrix_is_partitioned_by_language() -> None:
    units = [
        _unit("first.c", "first", "c"),
        _unit("second.c", "second", "c"),
        _unit("same.rs", "same", "rust"),
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    duplicates = find_semantic_duplicates(units, embeddings, threshold=0.99)

    assert len(duplicates) == 1
    assert {duplicates[0].unit_a.language, duplicates[0].unit_b.language} == {"c"}
