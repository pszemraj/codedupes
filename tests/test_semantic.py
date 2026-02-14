from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np

from codedupes.extractor import CodeExtractor
from codedupes import semantic
from codedupes.semantic import find_similar_to_query, run_semantic_analysis


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


def _extract_units(tmp_path: Path) -> list:
    source = dedent(
        """
        def first(x):
            return x + 1

        def second(x):
            return x + 2
        """
    ).strip()
    path = tmp_path / "sample.py"
    path.write_text(source)
    extractor = CodeExtractor(tmp_path, exclude_patterns=[], include_private=True)
    return list(extractor.extract_from_file(path))


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
