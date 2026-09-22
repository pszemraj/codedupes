"""Indexing and query search through the analyzer, including over-context units."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

import codedupes.semantic as semantic_module
from codedupes import analyzer as analyzer_module
from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from codedupes.models import AnalysisResult, CodeUnit
from tests.analyzer_helpers import embedding_identity_from_kwargs, make_semantic_runner
from tests.conftest import create_project

_QUERY_KWARG_NAMES = {
    "threshold_profile",
    "cache_scope",
    "corpus_identity",
    "device",
    "execution",
    "instruction_prefix",
    "model_name",
    "mps_fallback",
    "mps_memory_fraction",
    "revision",
    "semantic_task",
    "strict_revision_cache",
    "threshold",
    "top_k",
    "trust_remote_code",
    "use_cache",
}


def _capture_query_runner(
    capture: dict[str, object],
) -> Callable[..., list[tuple[CodeUnit, float]]]:
    """Build a query runner that records and validates forwarded keyword arguments."""

    def fake_find_similar_to_query(query, units, embeddings, **kwargs):
        del query, units, embeddings
        assert set(kwargs) == _QUERY_KWARG_NAMES
        capture.update({f"query_{key}": value for key, value in kwargs.items()})
        return []

    return fake_find_similar_to_query


def test_search_after_analyze_uses_analysis_task_when_unset(tmp_path: Path, monkeypatch) -> None:
    source = "def entry(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(capture=captured),
    )

    monkeypatch.setattr(
        semantic_module,
        "find_similar_to_query",
        _capture_query_runner(captured),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
        )
    )
    analyzer.analyze(project)
    analyzer.search("entry")

    assert captured["semantic_task"] == analyzer_module.DEFAULT_CHECK_SEMANTIC_TASK
    assert captured["query_semantic_task"] == analyzer_module.DEFAULT_CHECK_SEMANTIC_TASK


def test_embeddinggemma_search_after_analyze_requires_explicit_threshold(
    tmp_path: Path, monkeypatch
) -> None:
    project = create_project(tmp_path, "def entry(x):\n    return x + 1\n")
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(),
    )
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            model_name="embeddinggemma-300m",
            embedding_cache=False,
        )
    )
    analyzer.analyze(project)

    with pytest.raises(ValueError, match=r"search\(threshold=\.\.\.\)"):
        analyzer.search("entry")


def test_search_threshold_argument_overrides_the_config_for_one_call(
    tmp_path: Path, monkeypatch
) -> None:
    project = create_project(tmp_path, "def entry(x):\n    return x + 1\n")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(),
    )
    monkeypatch.setattr(
        semantic_module,
        "find_similar_to_query",
        _capture_query_runner(captured),
    )

    # The per-call threshold must not disturb the calibrated per-language
    # duplicate gates, which config.semantic_threshold would flatten.
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            model_name="embeddinggemma-300m",
            embedding_cache=False,
        )
    )
    analyzer.analyze(project)
    analyzer.search("entry", threshold=0.31)

    assert analyzer.config.semantic_threshold is None
    assert captured["query_threshold"] == 0.31


@pytest.mark.parametrize("search_document", ["source", "contextual"])
def test_search_threshold_defaults_to_none_and_honors_explicit_config(
    tmp_path: Path, monkeypatch, search_document
) -> None:
    project = create_project(tmp_path, "def entry(x):\n    return x + 1\n")
    encoded: list[str] = []

    class QueryModel:
        def encode(self, texts, **kwargs):
            encoded.extend(texts)
            return np.array(
                [[0.7, 0.71414284] if text == "entry" else [1.0, 0.0] for text in texts],
                dtype=np.float32,
            )

    monkeypatch.setattr(semantic_module, "get_model", lambda *args, **kwargs: QueryModel())
    base_config = {
        "run_traditional": False,
        "run_semantic": True,
        "run_unused": False,
        "min_semantic_statements": 0,
        "search_document": search_document,
        "device": "cpu",
        "embedding_cache": False,
    }
    analyzer = CodeAnalyzer(AnalyzerConfig(**base_config))
    analyzer.index(project)
    calls_before = len(encoded)
    for invalid_threshold in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="threshold must be finite"):
            analyzer.search("entry", threshold=invalid_threshold)
    assert len(encoded) == calls_before
    if search_document == "contextual":
        with pytest.raises(ValueError, match="contextual.*explicit threshold"):
            analyzer.search("entry")
        assert "entry" not in encoded
        assert len(analyzer.search("entry", threshold=0.0)) == 1
        assert analyzer.config.semantic_threshold is None

        # Changing future indexing options does not change the current corpus.
        analyzer.config.search_document = "source"
        with pytest.raises(ValueError, match="contextual.*explicit threshold"):
            analyzer.search("entry")
        analyzer.config.search_document = "contextual"
    else:
        assert len(analyzer.search("entry")) == 1

    # analyze() replaces the corpus with bare source in either document mode.
    analyzer.analyze(project)
    assert len(analyzer.search("entry")) == 1

    explicit = CodeAnalyzer(AnalyzerConfig(semantic_threshold=0.71, **base_config))
    explicit.index(project)
    assert explicit.search("entry") == []
    assert len(explicit.search("entry", threshold=0.0)) == 1
    assert explicit.config.semantic_threshold == 0.71


def test_index_embeds_corpus_without_mining_duplicates(tmp_path: Path, monkeypatch) -> None:
    source = "def entry(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)
    captured: dict[str, object] = {}
    embedded_units: list[CodeUnit] = []

    def fail_duplicate_mining(*_args, **_kwargs):
        raise AssertionError("index()/search() must never mine duplicate pairs")

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", fail_duplicate_mining)
    monkeypatch.setattr(semantic_module, "find_semantic_duplicates", fail_duplicate_mining)

    def fake_compute_embeddings(units, **kwargs):
        embedded_units.extend(units)
        captured.update(kwargs)
        return (
            np.zeros((len(units), 2), dtype=np.float32),
            embedding_identity_from_kwargs(kwargs),
        )

    monkeypatch.setattr(analyzer_module, "compute_embeddings", fake_compute_embeddings)
    monkeypatch.setattr(
        semantic_module,
        "find_similar_to_query",
        _capture_query_runner(captured),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
        )
    )
    indexed = analyzer.index(project)
    results = analyzer.search("entry")

    assert indexed == 1
    assert [unit.name for unit in embedded_units] == ["entry"]
    assert captured["progress"] == "auto"
    assert results == []
    assert captured["semantic_task"] == analyzer_module.DEFAULT_SEARCH_SEMANTIC_TASK
    assert captured["query_semantic_task"] == analyzer_module.DEFAULT_SEARCH_SEMANTIC_TASK
    assert captured["cache_scope"] == project.resolve()


@pytest.mark.parametrize("search_document", ["source", "contextual"])
def test_index_empty_corpus_yields_empty_search(tmp_path: Path, search_document) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            search_document=search_document,
            device="cpu",
        )
    )

    assert analyzer.index(empty) == 0
    assert analyzer.extracted_unit_count == 0
    for invalid_threshold in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="threshold must be finite"):
            analyzer.search("anything", threshold=invalid_threshold)
    if search_document == "contextual":
        with pytest.raises(ValueError, match="contextual.*explicit threshold"):
            analyzer.search("anything")
    else:
        assert analyzer.search("anything") == []
    assert analyzer.search("anything", threshold=0.0) == []


def test_search_requires_reindex_when_local_model_contents_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = create_project(tmp_path, "def entry(x):\n    return x + 1\n")
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    weights_path = model_dir / "model.safetensors"
    weights_path.write_bytes(b"first weights")

    monkeypatch.setattr(
        analyzer_module,
        "compute_embeddings",
        lambda units, **kwargs: (
            np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(units), 1)),
            embedding_identity_from_kwargs(kwargs),
        ),
    )
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            model_name=str(model_dir),
            device="cpu",
            embedding_cache=False,
            min_semantic_statements=0,
        )
    )
    analyzer.index(project)

    weights_path.write_bytes(b"second weights")

    with pytest.raises(RuntimeError, match=r"changed since this corpus was indexed.*index\(\)"):
        analyzer.search("entry")


def test_search_requires_reindex_when_embedding_runtime_variant_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = create_project(tmp_path, "def entry(x):\n    return x + 1\n")
    monkeypatch.setattr(
        analyzer_module,
        "compute_embeddings",
        lambda units, **kwargs: (
            np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(units), 1)),
            embedding_identity_from_kwargs(kwargs),
        ),
    )
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            device="cpu",
            embedding_cache=False,
            min_semantic_statements=0,
        )
    )
    analyzer.index(project)

    analyzer.config.instruction_prefix = "Represent this code differently: "

    with pytest.raises(RuntimeError, match=r"changed since this corpus was indexed.*index\(\)"):
        analyzer.search("entry")


def test_search_requires_embeddings(tmp_path: Path) -> None:
    source = "def entry():\n    return 1\n"
    create_project(tmp_path, source)
    project = tmp_path / "src"
    analyzer = CodeAnalyzer(
        AnalyzerConfig(run_semantic=False, run_traditional=True, run_unused=False)
    )

    analyzer.analyze(project)
    with pytest.raises(RuntimeError, match="run_semantic=True"):
        analyzer.search("entry")


def test_empty_reanalysis_clears_previous_search_state(tmp_path: Path, monkeypatch) -> None:
    project = create_project(tmp_path, "def entry():\n    return 1\n")
    empty_project = tmp_path / "empty"
    empty_project.mkdir()
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(),
    )
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
        )
    )

    analyzer.analyze(project)
    result = analyzer.analyze(empty_project)

    assert result.analysis_mode == "semantic"
    assert analyzer.search("entry") == []


class _WhitespaceTokenizer:
    """Tokenizer stub whose token count is the whitespace-separated word count."""

    def __call__(self, texts, **_kwargs):
        return {"input_ids": [text.split() for text in texts]}


class _ContextLimitedModel:
    """Model stub that rejects nothing but exposes a tiny context window."""

    max_seq_length = 20
    tokenizer = _WhitespaceTokenizer()

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, texts, **_kwargs):
        self.encoded.extend(texts)
        return np.array(
            [
                [0.0, 1.0]
                if "second_axis" in text
                else [0.5, 0.5]
                if "long_tail" in text
                else [1.0, 0.0]
                for text in texts
            ],
            dtype=np.float32,
        )


_OVERFLOW_PROJECT_SOURCE = dedent(
    """
    def short_one(x):
        y = x + 1
        return y

    def short_two(x):
        total = x * 3
        print(total)
        return total

    def long_tail(x):
        words = "aa bb cc dd ee ff gg hh ii jj kk ll mm nn oo pp qq rr ss tt"
        return words
    """
).strip()


def test_over_context_units_are_embedded_with_backend_truncation(
    tmp_path: Path, monkeypatch
) -> None:
    project = create_project(tmp_path, _OVERFLOW_PROJECT_SOURCE)
    model = _ContextLimitedModel()
    monkeypatch.setattr(semantic_module, "get_model", lambda *args, **kwargs: model)

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            semantic_threshold=0.5,
            embedding_cache=False,
        )
    )
    result = analyzer.analyze(project)

    assert [unit.name for unit in result.units] == ["short_one", "short_two", "long_tail"]
    assert any(
        "long_tail" in (duplicate.unit_a.name, duplicate.unit_b.name)
        for duplicate in result.semantic_duplicates
    )
    assert len(result.semantic_diagnostics) == 1
    diagnostic = result.semantic_diagnostics[0]
    assert diagnostic.code == "semantic-context-overflow"
    assert diagnostic.severity == "warning"
    assert diagnostic.lineno == result.units[-1].lineno
    assert "truncat" in diagnostic.message
    assert any("long_tail" in text for text in model.encoded)


def test_over_context_units_enter_and_reuse_the_embedding_cache(
    tmp_path: Path, monkeypatch
) -> None:
    project = create_project(tmp_path, _OVERFLOW_PROJECT_SOURCE)
    model = _ContextLimitedModel()
    monkeypatch.setattr(semantic_module, "get_model", lambda *args, **kwargs: model)

    def run() -> AnalysisResult:
        analyzer = CodeAnalyzer(
            AnalyzerConfig(
                run_traditional=False,
                run_semantic=True,
                run_unused=False,
                min_semantic_statements=0,
                semantic_threshold=0.5,
                embedding_cache=True,
            )
        )
        return analyzer.analyze(project)

    first = run()
    second = run()

    assert first.embedding_stats is not None
    assert second.embedding_stats is not None
    assert first.embedding_stats.manifest_generation == 1
    assert second.embedding_stats.manifest_generation == 2
    assert first.embedding_stats.encoded_inputs == 3
    assert second.embedding_stats.cache_hit_rows == 3
    assert second.embedding_stats.encoded_inputs == 0
    assert len(first.semantic_diagnostics) == 1
    assert first.semantic_diagnostics[0].code == "semantic-context-overflow"
    assert second.semantic_diagnostics == []
    for result in (first, second):
        assert any(
            "long_tail" in (duplicate.unit_a.name, duplicate.unit_b.name)
            for duplicate in result.semantic_duplicates
        )


def test_index_keeps_over_context_units_searchable(tmp_path: Path, monkeypatch) -> None:
    source = dedent(
        """
        def long_tail(x):
            words = "aa bb cc dd ee ff gg hh ii jj kk ll mm nn oo pp qq rr ss tt"
            return words

        def wanted(x):
            y = x + 1
            return y

        def second_axis(x):
            z = x + 2
            return z
        """
    ).strip()
    project = create_project(tmp_path, source)
    model = _ContextLimitedModel()
    monkeypatch.setattr(semantic_module, "get_model", lambda *args, **kwargs: model)

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            semantic_threshold=0.0,
            embedding_cache=False,
        )
    )
    indexed = analyzer.index(project)
    results = analyzer.search("anything", top_k=3)

    assert indexed == 3
    assert len(analyzer.semantic_diagnostics) == 1
    assert analyzer.semantic_diagnostics[0].code == "semantic-context-overflow"
    assert results[0][0].name == "wanted"
    assert "long_tail" in [unit.name for unit, _score in results]


def test_over_context_search_query_reaches_backend(tmp_path: Path, monkeypatch) -> None:
    project = create_project(tmp_path, "def wanted(x):\n    y = x + 1\n    return y\n")
    model = _ContextLimitedModel()
    monkeypatch.setattr(semantic_module, "get_model", lambda *args, **kwargs: model)

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            embedding_cache=False,
        )
    )
    analyzer.index(project)

    query = " ".join(["word"] * 40)
    results = analyzer.search(query)

    assert [unit.name for unit, _score in results] == ["wanted"]
    assert any(query in text for text in model.encoded)


@pytest.mark.parametrize(
    ("config_overrides", "expected_values"),
    [
        pytest.param(
            {
                "device": "mps",
                "mps_fallback": False,
                "mps_memory_fraction": 0.8,
            },
            {
                "device": "mps",
                "mps_fallback": False,
                "mps_memory_fraction": 0.8,
            },
            id="device-controls",
        ),
        pytest.param(
            {"embedding_cache": False},
            {"use_cache": False},
            id="cache-control",
        ),
        pytest.param(
            {"strict_revision_cache": True},
            {"strict_revision_cache": True},
            id="strict-revision-cache-control",
        ),
    ],
)
def test_analyzer_passes_semantic_controls_to_index_and_query(
    tmp_path: Path,
    monkeypatch,
    config_overrides: dict[str, object],
    expected_values: dict[str, object],
) -> None:
    source = "def entry(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(capture=captured),
    )
    monkeypatch.setattr(
        semantic_module,
        "find_similar_to_query",
        _capture_query_runner(captured),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            **config_overrides,
        )
    )
    analyzer.analyze(project)
    analyzer.search("entry")

    for key, expected in expected_values.items():
        assert captured[key] == expected
        assert captured[f"query_{key}"] == expected
    assert captured["cache_scope"] == project
    assert captured["query_cache_scope"] == project


def test_analyzer_default_embedding_cache_enabled_and_scoped_to_analyzed_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = "def entry(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(capture=captured),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
        )
    )
    analyzer.analyze(project)

    assert captured["use_cache"] is True
    assert captured["cache_scope"] == project
