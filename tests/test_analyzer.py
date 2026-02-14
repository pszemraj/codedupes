from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from codedupes.models import DuplicatePair
from codedupes.semantic import SemanticBackendError
from tests.conftest import build_two_function_source, create_project


@pytest.mark.parametrize(
    "analyzer_config, expected_unused",
    [
        (
            AnalyzerConfig(run_traditional=False, run_semantic=False, run_unused=True),
            set(),
        ),
        (
            AnalyzerConfig(
                run_traditional=True,
                run_semantic=False,
                run_unused=False,
                jaccard_threshold=0.5,
            ),
            set(),
        ),
        (
            AnalyzerConfig(
                run_traditional=False,
                run_semantic=False,
                run_unused=True,
                strict_unused=True,
            ),
            {"used", "unused"},
        ),
    ],
)
def test_unused_detection_config_variants(tmp_path: Path, analyzer_config, expected_unused) -> None:
    project = create_project(tmp_path, build_two_function_source())
    analyzer = CodeAnalyzer(analyzer_config)

    result = analyzer.analyze(project)

    assert {unit.name for unit in result.potentially_unused} == expected_unused


def test_integration_on_mixed_project(tmp_path: Path) -> None:
    src_root = tmp_path / "project"
    src_root.mkdir()

    (src_root / "bad.py").write_text("def bad(:\n    pass")
    (src_root / "tests").mkdir()
    (src_root / "tests" / "test_skip.py").write_text("def test_case():\n    return 1")
    (src_root / "util.py").write_text(
        dedent(
            """
            def add(a, b):
                return a + b

            def sum_values(x, y):
                return x + y

            def helper():
                return 2

            def caller():
                return helper()

            class Engine:
                def run(self):
                    return helper()

                def _internal(self):
                    return 0

            def _private_entry():
                return helper()

            def get_value():
                return 3

            def set_value(value):
                return value
            """
        ).strip()
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_semantic=False,
            run_traditional=True,
            run_unused=True,
            include_private=False,
        )
    )
    result = analyzer.analyze(src_root)

    assert len(result.exact_duplicates) >= 1
    assert not any("tests" in str(unit.file_path) for unit in result.units)
    names = {unit.name for unit in result.potentially_unused}
    assert "caller" not in names
    assert "_private_entry" not in names


def test_short_functions_are_skipped_from_semantic(tmp_path: Path) -> None:
    source = dedent(
        """
        def tiny():
            return 1

        def another_tiny():
            return 2
        """
    ).strip()
    project = create_project(tmp_path, source, module="tiny.py")
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_lines=3,
        )
    )
    result = analyzer.analyze(project)
    assert result.semantic_duplicates == []


def test_unused_semantic_pairs_are_filtered(tmp_path: Path, monkeypatch) -> None:
    source = dedent(
        """
        def _a():
            x = 1
            return x + 1

        def _b():
            y = 2
            return y + 2
        """
    ).strip()
    project = create_project(tmp_path, source, module="pairs.py")

    from codedupes import analyzer as analyzer_module

    def fake_run_semantic(
        units,
        model_name="codefuse-ai/C2LLM-0.5B",
        instruction_prefix=None,
        threshold=0.82,
        exclude_pairs=None,
        batch_size=32,
        revision=None,
        trust_remote_code=None,
    ):
        a, b = units
        return np.array([[0.0, 0.0]] * 2, dtype=np.float32), [
            DuplicatePair(unit_a=a, unit_b=b, similarity=0.99, method="semantic")
        ]

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", fake_run_semantic)

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=True,
            min_semantic_lines=0,
            strict_unused=False,
        )
    )

    result = analyzer.analyze(project)
    assert result.semantic_duplicates == []


def test_semantic_only_pre_excludes_exact_hash_pairs(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "src"
    project.mkdir()
    (project / "__init__.py").write_text("")
    (project / "a.py").write_text("def helper(x):\n    return x + 1\n")
    (project / "b.py").write_text("def helper(x):\n    return x + 1\n")

    from codedupes import analyzer as analyzer_module

    captured_exclude_pairs: set[tuple[str, str]] = set()

    def fake_run_semantic(
        units,
        model_name="codefuse-ai/C2LLM-0.5B",
        instruction_prefix=None,
        threshold=0.82,
        exclude_pairs=None,
        batch_size=32,
        revision=None,
        trust_remote_code=None,
    ):
        captured_exclude_pairs.update(exclude_pairs or set())
        return np.zeros((len(units), 2), dtype=np.float32), []

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", fake_run_semantic)

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_lines=0,
        )
    )

    result = analyzer.analyze(project)
    assert result.semantic_duplicates == []
    assert captured_exclude_pairs


def test_suppress_test_semantic_matches_filters_test_named_pairs(
    tmp_path: Path, monkeypatch
) -> None:
    source = dedent(
        """
        def test_alpha():
            return 1

        def test_beta():
            return 2

        def helper_alpha():
            return 3

        def helper_beta():
            return 4
        """
    ).strip()
    project = create_project(tmp_path, source, module="tests_like.py")

    from codedupes import analyzer as analyzer_module

    def fake_run_semantic(
        units,
        model_name="codefuse-ai/C2LLM-0.5B",
        instruction_prefix=None,
        threshold=0.82,
        exclude_pairs=None,
        batch_size=32,
        revision=None,
        trust_remote_code=None,
    ):
        by_name = {unit.name: unit for unit in units}
        return np.zeros((len(units), 2), dtype=np.float32), [
            DuplicatePair(
                unit_a=by_name["test_alpha"],
                unit_b=by_name["test_beta"],
                similarity=0.99,
                method="semantic",
            ),
            DuplicatePair(
                unit_a=by_name["helper_alpha"],
                unit_b=by_name["helper_beta"],
                similarity=0.99,
                method="semantic",
            ),
        ]

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", fake_run_semantic)

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_lines=0,
            suppress_test_semantic_matches=True,
        )
    )

    result = analyzer.analyze(project)

    assert len(result.semantic_duplicates) == 1
    assert {
        result.semantic_duplicates[0].unit_a.name,
        result.semantic_duplicates[0].unit_b.name,
    } == {"helper_alpha", "helper_beta"}


def test_search_requires_embeddings(tmp_path: Path) -> None:
    source = "def entry():\n    return 1\n"
    create_project(tmp_path, source)
    project = tmp_path / "src"
    analyzer = CodeAnalyzer(
        AnalyzerConfig(run_semantic=False, run_traditional=False, run_unused=False)
    )

    analyzer.analyze(project)
    with pytest.raises(RuntimeError, match="run_semantic=True"):
        analyzer.search("entry")


def test_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError, match="jaccard_threshold"):
        AnalyzerConfig(jaccard_threshold=1.5)

    with pytest.raises(ValueError, match="semantic_threshold"):
        AnalyzerConfig(semantic_threshold=-0.1)


def test_empty_directory_analysis(tmp_path: Path) -> None:
    analyzer = CodeAnalyzer()
    result = analyzer.analyze(tmp_path)

    assert result.units == []
    assert result.exact_duplicates == []
    assert result.semantic_duplicates == []
    assert result.potentially_unused == []


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

    from codedupes import analyzer as analyzer_module

    def fake_run_semantic(*args, **kwargs):
        raise semantic_error

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", fake_run_semantic)

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            run_unused=False,
        )
    )

    result = analyzer.analyze(project)
    assert len(result.units) == 2
    assert result.semantic_duplicates == []
    assert "Retry with `codedupes check" in caplog.text


def test_semantic_missing_dependency_raises_when_semantic_required(
    tmp_path: Path, monkeypatch
) -> None:
    source = "def only_func():\n    return 1\n"
    project = create_project(tmp_path, source)

    from codedupes import analyzer as analyzer_module

    def fake_run_semantic(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'sentence_transformers'")

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", fake_run_semantic)

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_lines=0,
        )
    )

    with pytest.raises(ModuleNotFoundError):
        analyzer.analyze(project)
