"""Semantic backend failures: hard failure by default and the traditional fallback."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from codedupes import analyzer as analyzer_module
from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from codedupes.semantic import SemanticBackendError
from tests.analyzer_helpers import make_semantic_runner, traditional_single_jaccard_runner
from tests.conftest import create_project


def test_combined_mode_fails_hard_on_runtime_semantic_error_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    source = "def entry(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(error=RuntimeError("CUDA out of memory")),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            filter_tiny_traditional=False,
        )
    )

    # The wrapper must name the root cause; the CLI only prints str(exc).
    with pytest.raises(
        RuntimeError,
        match=r"Semantic analysis failed in combined mode \(CUDA out of memory\)",
    ) as excinfo:
        analyzer.analyze(project)

    assert "allow-semantic-fallback" in str(excinfo.value)


def test_combined_mode_fallback_keeps_full_scope_traditional_units(
    tmp_path: Path, monkeypatch
) -> None:
    source = dedent(
        """
        class Box:
            def method(self):
                value = 1
                return value

        def short():
            return 1

        def longer():
            value = 2
            return value
        """
    ).strip()
    project = create_project(tmp_path, source)

    traditional_calls: list[tuple[tuple[str, ...], list[str]]] = []

    def fake_traditional(units, jaccard_threshold=0.85):
        traditional_calls.append(
            (tuple(unit.name for unit in units), [unit.name for unit in units])
        )
        return [], []

    monkeypatch.setattr(analyzer_module, "run_traditional_analysis", fake_traditional)
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(error=RuntimeError("CUDA out of memory")),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            allow_semantic_fallback=True,
            run_unused=False,
            min_semantic_statements=2,
            filter_tiny_traditional=False,
        )
    )
    analyzer.analyze(project)

    assert len(traditional_calls) == 1
    assert set(traditional_calls[0][0]) == {"Box", "method", "short", "longer"}


def test_combined_mode_fallback_marks_semantic_degradation(tmp_path: Path, monkeypatch) -> None:
    source = dedent(
        """
        def dead(x):
            return x
        """
    ).strip()
    project = create_project(tmp_path, source)

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(error=RuntimeError("backend unavailable")),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            allow_semantic_fallback=True,
            run_unused=False,
            min_semantic_statements=0,
            filter_tiny_traditional=False,
        )
    )
    result = analyzer.analyze(project)

    assert result.semantic_fallback is True
    assert result.semantic_fallback_reason is not None
    assert "backend unavailable" in result.semantic_fallback_reason


def test_semantic_only_fails_hard_on_runtime_semantic_error(tmp_path: Path, monkeypatch) -> None:
    source = "def entry(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(error=RuntimeError("CUDA out of memory")),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
        )
    )

    with pytest.raises(RuntimeError):
        analyzer.analyze(project)


def test_mixed_mode_semantic_failure_still_builds_hybrid_from_traditional(
    tmp_path: Path, monkeypatch
) -> None:
    source = dedent(
        """
        def one(x):
            return x + 1

        def two(y):
            return y + 2
        """
    ).strip()
    project = create_project(tmp_path, source)

    monkeypatch.setattr(
        analyzer_module,
        "run_traditional_analysis",
        traditional_single_jaccard_runner(0.9),
    )
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(error=SemanticBackendError("semantic backend mismatch")),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            allow_semantic_fallback=True,
            run_unused=False,
            min_semantic_statements=0,
            filter_tiny_traditional=False,
        )
    )
    result = analyzer.analyze(project)

    assert len(result.traditional_duplicates) == 1
    assert len(result.hybrid_duplicates) == 1
    assert result.hybrid_duplicates[0].tier == "traditional_near"


@pytest.mark.parametrize(
    "semantic_error",
    [
        ModuleNotFoundError("No module named 'sentence_transformers'"),
        SemanticBackendError("semantic backend mismatch"),
    ],
)
def test_semantic_failures_fall_back_when_traditional_enabled(
    tmp_path: Path, monkeypatch, caplog, semantic_error
) -> None:
    source = dedent(
        """
        def used():
            return 1

        def unused():
            return 2
        """
    ).strip()
    project = create_project(tmp_path, source)

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(error=semantic_error),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            allow_semantic_fallback=True,
            run_unused=False,
        )
    )

    result = analyzer.analyze(project)
    assert len(result.units) == 2
    assert result.semantic_duplicates == []
    assert "Retry with `codedupes check" in caplog.text


@pytest.mark.parametrize(
    "semantic_error",
    [
        ModuleNotFoundError("No module named 'sentence_transformers'"),
        SemanticBackendError("semantic backend mismatch"),
    ],
)
@pytest.mark.parametrize("run_unused", [False, True])
def test_semantic_failures_raise_when_semantic_required(
    tmp_path: Path, monkeypatch, semantic_error, run_unused
) -> None:
    source = "def only_func():\n    return 1\n"
    project = create_project(tmp_path, source)

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(error=semantic_error),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=run_unused,
            min_semantic_statements=0,
        )
    )

    with pytest.raises(type(semantic_error)):
        analyzer.analyze(project)
