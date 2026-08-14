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
