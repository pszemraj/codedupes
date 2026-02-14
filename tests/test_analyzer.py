from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from codedupes import analyzer as analyzer_module
from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer, analyze_directory
from codedupes.models import AnalysisResult, CodeUnit, CodeUnitType, DuplicatePair
from codedupes.semantic import SemanticBackendError
import codedupes.semantic as semantic_module
from tests.conftest import build_two_function_source, create_project


def _make_unit(
    tmp_path: Path,
    *,
    name: str,
    source: str,
    lineno: int,
    unit_type: CodeUnitType = CodeUnitType.FUNCTION,
) -> CodeUnit:
    return CodeUnit(
        name=name,
        qualified_name=f"sample.{name}",
        unit_type=unit_type,
        file_path=tmp_path / "sample.py",
        lineno=lineno,
        end_lineno=lineno + max(1, len(source.strip().splitlines()) - 1),
        source=source,
        is_public=True,
        is_exported=False,
    )


def _make_semantic_runner(
    *,
    duplicate_factory: Callable[[list[CodeUnit]], list[DuplicatePair]] | None = None,
    capture: dict[str, object] | None = None,
    capture_exclude_pairs: set[tuple[str, str]] | None = None,
    error: Exception | None = None,
) -> Callable[..., tuple[np.ndarray, list[DuplicatePair]]]:
    """Build a reusable semantic-analysis test double."""

    def fake_run_semantic(
        units,
        model_name="gte-modernbert-base",
        instruction_prefix=None,
        threshold=0.82,
        exclude_pairs=None,
        batch_size=32,
        revision=None,
        trust_remote_code=None,
        semantic_task=None,
    ):
        if capture is not None:
            capture["model_name"] = model_name
            capture["instruction_prefix"] = instruction_prefix
            capture["threshold"] = threshold
            capture["exclude_pairs"] = exclude_pairs
            capture["batch_size"] = batch_size
            capture["revision"] = revision
            capture["trust_remote_code"] = trust_remote_code
            capture["semantic_task"] = semantic_task
        if capture_exclude_pairs is not None:
            capture_exclude_pairs.update(exclude_pairs or set())
        if error is not None:
            raise error

        duplicates = duplicate_factory(units) if duplicate_factory is not None else []
        return np.zeros((len(units), 2), dtype=np.float32), duplicates

    return fake_run_semantic


def test_all_duplicates_returns_raw_for_single_method_modes(tmp_path: Path) -> None:
    file_a = tmp_path / "a.py"
    file_b = tmp_path / "b.py"
    file_a.write_text("def foo():\n    return 1\n")
    file_b.write_text("def foo():\n    return 1\n")

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=False,
            run_unused=False,
            filter_tiny_traditional=False,
        )
    )
    result = analyzer.analyze(tmp_path)

    assert result.analysis_mode == "traditional"
    assert result.traditional_duplicates
    assert result.all_duplicates == result.traditional_duplicates

    unit = _make_unit(
        tmp_path,
        name="bar",
        source="def bar():\n    return 1\n",
        lineno=1,
    )
    semantic_duplicate = DuplicatePair(
        unit_a=unit,
        unit_b=unit,
        similarity=0.95,
        method="semantic",
    )
    semantic_result = AnalysisResult(
        units=[unit],
        traditional_duplicates=[],
        semantic_duplicates=[semantic_duplicate],
        hybrid_duplicates=[],
        potentially_unused=[],
        analysis_mode="semantic",
        filtered_raw_duplicates=0,
    )

    assert semantic_result.all_duplicates == [semantic_duplicate]


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
            filter_tiny_traditional=False,
        )
    )
    result = analyzer.analyze(src_root)

    assert len(result.traditional_duplicates) >= 1
    assert result.hybrid_duplicates == []
    assert not any("tests" in str(unit.file_path) for unit in result.units)
    names = {unit.name for unit in result.potentially_unused}
    assert "caller" not in names
    assert "_private_entry" not in names


def test_analyze_directory_uses_auto_revision_for_custom_model(tmp_path: Path, monkeypatch) -> None:
    source = "def add_one(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(capture=captured),
    )

    analyze_directory(
        project,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        min_semantic_lines=0,
        run_unused=False,
    )

    assert captured["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert captured["revision"] is None


def test_combined_mode_preserves_near_dupes_for_semantic_confirmation(
    tmp_path: Path, monkeypatch
) -> None:
    source = dedent(
        """
        def exact_a():
            return 1

        def exact_b():
            return 1

        def near_c():
            return 2
        """
    ).strip()
    project = create_project(tmp_path, source)

    captured_exclude_pairs: set[tuple[str, str]] = set()
    expected_exact_pair: tuple[str, str] = ("", "")

    def fake_traditional(
        units,
        jaccard_threshold=0.85,
        compute_unused=True,
        strict_unused=False,
        project_root=None,
    ):
        first, second, third = units
        nonlocal expected_exact_pair
        expected_exact_pair = tuple(sorted((first.uid, second.uid)))
        return (
            [DuplicatePair(unit_a=first, unit_b=second, similarity=1.0, method="ast_hash")],
            [DuplicatePair(unit_a=second, unit_b=third, similarity=0.9, method="jaccard")],
            [],
        )

    monkeypatch.setattr(analyzer_module, "run_traditional_analysis", fake_traditional)
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(
            capture_exclude_pairs=captured_exclude_pairs,
            duplicate_factory=lambda units: [
                DuplicatePair(
                    unit_a=units[1],
                    unit_b=units[2],
                    similarity=0.95,
                    method="semantic",
                )
            ],
        ),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            run_unused=False,
            min_semantic_lines=0,
            jaccard_threshold=0.85,
            semantic_threshold=0.82,
            filter_tiny_traditional=False,
        )
    )
    result = analyzer.analyze(project)

    assert set(captured_exclude_pairs) == {expected_exact_pair}
    assert len(result.traditional_duplicates) == 2
    assert len(result.semantic_duplicates) == 1
    assert len(result.hybrid_duplicates) == 2
    assert {duplicate.tier for duplicate in result.hybrid_duplicates} == {
        "exact",
        "hybrid_confirmed",
    }


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


def test_semantic_defaults_exclude_class_units(tmp_path: Path, monkeypatch) -> None:
    source = dedent(
        """
        class Box:
            def method(self):
                return 1

        def helper():
            return 2
        """
    ).strip()
    project = create_project(tmp_path, source, module="scope.py")
    captured_types: list[CodeUnitType] = []

    def fake_run_semantic(units, **_kwargs):
        captured_types.extend(unit.unit_type for unit in units)
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
    analyzer.analyze(project)

    assert CodeUnitType.CLASS not in captured_types
    assert CodeUnitType.FUNCTION in captured_types
    assert CodeUnitType.METHOD in captured_types


def test_semantic_class_scope_can_be_enabled_explicitly(tmp_path: Path, monkeypatch) -> None:
    source = dedent(
        """
        class Box:
            def method(self):
                return 1

        def helper():
            return 2
        """
    ).strip()
    project = create_project(tmp_path, source, module="scope.py")
    captured_types: list[CodeUnitType] = []

    def fake_run_semantic(units, **_kwargs):
        captured_types.extend(unit.unit_type for unit in units)
        return np.zeros((len(units), 2), dtype=np.float32), []

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", fake_run_semantic)

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_lines=0,
            semantic_unit_types=("class",),
        )
    )
    analyzer.analyze(project)

    assert captured_types
    assert set(captured_types) == {CodeUnitType.CLASS}


def test_combined_mode_scopes_traditional_duplicates_to_semantic_candidates(
    tmp_path: Path, monkeypatch
) -> None:
    source = dedent(
        """
        class Box:
            def method(self):
                return 1

        def tiny():
            return 2
        """
    ).strip()
    project = create_project(tmp_path, source, module="scope.py")
    captured_traditional_units: list[CodeUnit] = []

    def fake_traditional(
        units,
        jaccard_threshold=0.85,
        compute_unused=True,
        strict_unused=False,
        project_root=None,
    ):
        captured_traditional_units.extend(units)
        return [], [], []

    monkeypatch.setattr(analyzer_module, "run_traditional_analysis", fake_traditional)
    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", _make_semantic_runner())

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            run_unused=False,
            min_semantic_lines=2,
        )
    )
    analyzer.analyze(project)

    assert captured_traditional_units == []


def test_traditional_only_keeps_full_scope_even_with_semantic_filters(
    tmp_path: Path, monkeypatch
) -> None:
    source = dedent(
        """
        class Box:
            def method(self):
                return 1

        def tiny():
            return 2
        """
    ).strip()
    project = create_project(tmp_path, source, module="scope.py")
    captured_traditional_units: list[CodeUnit] = []

    def fake_traditional(
        units,
        jaccard_threshold=0.85,
        compute_unused=True,
        strict_unused=False,
        project_root=None,
    ):
        captured_traditional_units.extend(units)
        return [], [], []

    monkeypatch.setattr(analyzer_module, "run_traditional_analysis", fake_traditional)

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=False,
            run_unused=False,
            min_semantic_lines=2,
        )
    )
    analyzer.analyze(project)

    assert captured_traditional_units
    assert {unit.unit_type for unit in captured_traditional_units} == {
        CodeUnitType.CLASS,
        CodeUnitType.METHOD,
        CodeUnitType.FUNCTION,
    }


def test_tiny_exact_duplicates_are_filtered_by_default(tmp_path: Path) -> None:
    source = dedent(
        """
        def wrapper_a():
            return helper_a()

        def wrapper_b():
            return helper_b()

        def helper_a():
            return 1

        def helper_b():
            return 1
        """
    ).strip()
    project = create_project(tmp_path, source, module="tiny_exact.py")

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=False,
            run_unused=False,
            jaccard_threshold=0.99,
        )
    )
    result = analyzer.analyze(project)

    assert result.traditional_duplicates == []


def test_tiny_exact_duplicates_can_be_restored(tmp_path: Path) -> None:
    source = dedent(
        """
        def wrapper_a():
            return helper_a()

        def wrapper_b():
            return helper_b()

        def helper_a():
            return 1

        def helper_b():
            return 1
        """
    ).strip()
    project = create_project(tmp_path, source, module="tiny_exact.py")

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=False,
            run_unused=False,
            jaccard_threshold=0.99,
            filter_tiny_traditional=False,
        )
    )
    result = analyzer.analyze(project)

    assert any(
        duplicate.method in {"ast_hash", "token_hash"}
        for duplicate in result.traditional_duplicates
    )


@pytest.mark.parametrize(
    ("similarity", "expected_count"),
    [
        (0.90, 0),
        (0.95, 1),
    ],
)
def test_tiny_near_duplicates_use_high_jaccard_floor(
    tmp_path: Path, monkeypatch, similarity: float, expected_count: int
) -> None:
    source = dedent(
        """
        def first():
            return alpha()

        def second():
            return beta()
        """
    ).strip()
    project = create_project(tmp_path, source, module="tiny_near.py")

    def fake_traditional(
        units,
        jaccard_threshold=0.85,
        compute_unused=True,
        strict_unused=False,
        project_root=None,
    ):
        return (
            [],
            [
                DuplicatePair(
                    unit_a=units[0], unit_b=units[1], similarity=similarity, method="jaccard"
                )
            ],
            [],
        )

    monkeypatch.setattr(analyzer_module, "run_traditional_analysis", fake_traditional)

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=False,
            run_unused=False,
            tiny_near_jaccard_min=0.93,
        )
    )
    result = analyzer.analyze(project)

    assert len(result.traditional_duplicates) == expected_count


def test_analyzer_resolves_profile_default_semantic_threshold(tmp_path: Path, monkeypatch) -> None:
    source = "def add_one(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)
    captured: dict[str, object] = {}

    monkeypatch.setattr(analyzer_module, "get_default_semantic_threshold", lambda _model: 0.77)
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(capture=captured),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_lines=0,
            semantic_threshold=None,
        )
    )
    analyzer.analyze(project)

    assert captured["threshold"] == 0.77


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

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(
            duplicate_factory=lambda units: [
                DuplicatePair(
                    unit_a=units[0],
                    unit_b=units[1],
                    similarity=0.99,
                    method="semantic",
                )
            ],
        ),
    )

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

    captured_exclude_pairs: set[tuple[str, str]] = set()

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(capture_exclude_pairs=captured_exclude_pairs),
    )

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
    assert not captured_exclude_pairs


def test_combined_mode_falls_back_on_runtime_semantic_error(tmp_path: Path, monkeypatch) -> None:
    source = "def entry(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(error=RuntimeError("CUDA out of memory")),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            run_unused=False,
            min_semantic_lines=0,
            filter_tiny_traditional=False,
        )
    )

    result = analyzer.analyze(project)
    assert result.traditional_duplicates == []
    assert result.semantic_duplicates == []
    assert result.potentially_unused == []


def test_combined_mode_fallback_runs_traditional_on_full_units(tmp_path: Path, monkeypatch) -> None:
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

    def fake_traditional(
        units,
        jaccard_threshold=0.85,
        compute_unused=True,
        strict_unused=False,
        project_root=None,
    ):
        traditional_calls.append(
            (tuple(unit.name for unit in units), [unit.name for unit in units])
        )
        return [], [], []

    monkeypatch.setattr(analyzer_module, "run_traditional_analysis", fake_traditional)
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(error=RuntimeError("CUDA out of memory")),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            run_unused=False,
            min_semantic_lines=2,
            filter_tiny_traditional=False,
        )
    )
    analyzer.analyze(project)

    assert len(traditional_calls) == 2
    assert set(traditional_calls[0][0]) == {"method", "longer"}
    assert set(traditional_calls[1][0]) == {"Box", "method", "short", "longer"}


def test_semantic_task_resolves_check_default_for_indexing(tmp_path: Path, monkeypatch) -> None:
    source = "def entry(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(capture=captured),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_lines=0,
        )
    )
    analyzer.analyze(project)

    assert captured["semantic_task"] == analyzer_module.DEFAULT_CHECK_SEMANTIC_TASK


def test_search_uses_index_task_when_unset(tmp_path: Path, monkeypatch) -> None:
    source = "def entry(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(capture=captured),
    )

    def fake_find_similar_to_query(
        query: str,
        units,
        embeddings,
        model_name="gte-modernbert-base",
        instruction_prefix=None,
        top_k=10,
        revision=None,
        trust_remote_code=None,
        threshold=None,
        semantic_task=None,
    ):
        captured["query_task"] = semantic_task
        return []

    monkeypatch.setattr(semantic_module, "find_similar_to_query", fake_find_similar_to_query)

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_lines=0,
        )
    )
    analyzer.analyze(project)
    analyzer.search("entry")

    assert captured["semantic_task"] == analyzer_module.DEFAULT_CHECK_SEMANTIC_TASK
    assert captured["query_task"] == analyzer_module.DEFAULT_CHECK_SEMANTIC_TASK


def test_semantic_only_fails_hard_on_runtime_semantic_error(tmp_path: Path, monkeypatch) -> None:
    source = "def entry(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(error=RuntimeError("CUDA out of memory")),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_lines=0,
        )
    )

    with pytest.raises(RuntimeError):
        analyzer.analyze(project)


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

    def fake_run_semantic(
        units,
        model_name="codefuse-ai/C2LLM-0.5B",
        instruction_prefix=None,
        threshold=0.82,
        exclude_pairs=None,
        batch_size=32,
        revision=None,
        trust_remote_code=None,
        semantic_task=None,
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


def test_hybrid_synthesis_exact_only_included(tmp_path: Path) -> None:
    unit_a = _make_unit(tmp_path, name="a", source="def a(x):\n    return x + 1\n", lineno=1)
    unit_b = _make_unit(tmp_path, name="b", source="def b(y):\n    return y + 1\n", lineno=5)
    traditional = [DuplicatePair(unit_a=unit_a, unit_b=unit_b, similarity=1.0, method="ast_hash")]

    hybrid, filtered = analyzer_module._synthesize_hybrid_duplicates(
        traditional,
        [],
        semantic_threshold=0.82,
        jaccard_threshold=0.85,
    )

    assert len(hybrid) == 1
    assert hybrid[0].tier == "exact"
    assert hybrid[0].confidence == 1.0
    assert filtered == 0


def test_hybrid_synthesis_jaccard_only_included(tmp_path: Path) -> None:
    unit_a = _make_unit(tmp_path, name="a", source="def a(x):\n    return x + 1\n", lineno=1)
    unit_b = _make_unit(tmp_path, name="b", source="def b(y):\n    return y + 2\n", lineno=5)
    traditional = [DuplicatePair(unit_a=unit_a, unit_b=unit_b, similarity=0.9, method="jaccard")]

    hybrid, _ = analyzer_module._synthesize_hybrid_duplicates(
        traditional,
        [],
        semantic_threshold=0.82,
        jaccard_threshold=0.85,
    )

    assert len(hybrid) == 1
    assert hybrid[0].tier == "traditional_near"
    assert hybrid[0].jaccard_similarity == pytest.approx(0.9)


def test_hybrid_synthesis_hybrid_confirmed(tmp_path: Path) -> None:
    unit_a = _make_unit(tmp_path, name="a", source="def a(x):\n    return x + 1\n", lineno=1)
    unit_b = _make_unit(tmp_path, name="b", source="def b(y):\n    return y + 1\n", lineno=5)
    traditional = [DuplicatePair(unit_a=unit_a, unit_b=unit_b, similarity=0.88, method="jaccard")]
    semantic = [DuplicatePair(unit_a=unit_a, unit_b=unit_b, similarity=0.93, method="semantic")]

    hybrid, _ = analyzer_module._synthesize_hybrid_duplicates(
        traditional,
        semantic,
        semantic_threshold=0.82,
        jaccard_threshold=0.85,
    )

    assert len(hybrid) == 1
    assert hybrid[0].tier == "hybrid_confirmed"
    assert hybrid[0].confidence == pytest.approx((0.5 * 0.93) + (0.5 * 0.88))


def test_hybrid_synthesis_semantic_only_gate_enforced(tmp_path: Path) -> None:
    unit_a = _make_unit(
        tmp_path, name="a", source="def alpha(v):\n    z = v + 1\n    return z\n", lineno=1
    )
    unit_b = _make_unit(
        tmp_path, name="b", source="def beta(v):\n    q = v + 2\n    return q\n", lineno=6
    )

    low_semantic = [DuplicatePair(unit_a=unit_a, unit_b=unit_b, similarity=0.90, method="semantic")]
    hybrid_low, filtered_low = analyzer_module._synthesize_hybrid_duplicates(
        [],
        low_semantic,
        semantic_threshold=0.82,
        jaccard_threshold=0.85,
    )
    assert hybrid_low == []
    assert filtered_low == 1

    weak_sources_a = _make_unit(
        tmp_path,
        name="c",
        source="def c(a):\n    x = a + 1\n    y = x + 1\n    z = y + 1\n    return z\n",
        lineno=12,
    )
    weak_sources_b = _make_unit(
        tmp_path,
        name="d",
        source="def d(v):\n    return v\n",
        lineno=20,
    )
    weak_semantic = [
        DuplicatePair(
            unit_a=weak_sources_a, unit_b=weak_sources_b, similarity=0.95, method="semantic"
        )
    ]
    hybrid_weak, _ = analyzer_module._synthesize_hybrid_duplicates(
        [],
        weak_semantic,
        semantic_threshold=0.82,
        jaccard_threshold=0.85,
    )
    assert hybrid_weak == []

    strong_semantic = [
        DuplicatePair(unit_a=unit_a, unit_b=unit_b, similarity=0.95, method="semantic")
    ]
    hybrid_strong, _ = analyzer_module._synthesize_hybrid_duplicates(
        [],
        strong_semantic,
        semantic_threshold=0.82,
        jaccard_threshold=0.85,
    )
    assert len(hybrid_strong) == 1
    assert hybrid_strong[0].tier == "semantic_high_confidence"


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

    def fake_traditional(
        units,
        jaccard_threshold=0.85,
        compute_unused=True,
        strict_unused=False,
        project_root=None,
    ):
        first, second = units[:2]
        return (
            [DuplicatePair(unit_a=first, unit_b=second, similarity=0.9, method="jaccard")],
            [],
            [],
        )

    monkeypatch.setattr(analyzer_module, "run_traditional_analysis", fake_traditional)
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(error=SemanticBackendError("semantic backend mismatch")),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            run_unused=False,
            min_semantic_lines=0,
            filter_tiny_traditional=False,
        )
    )
    result = analyzer.analyze(project)

    assert len(result.traditional_duplicates) == 1
    assert len(result.hybrid_duplicates) == 1
    assert result.hybrid_duplicates[0].tier == "traditional_near"


def test_single_method_modes_bypass_hybrid_synthesis(tmp_path: Path, monkeypatch) -> None:
    source = dedent(
        """
        def one(x):
            return x + 1

        def two(y):
            return y + 2
        """
    ).strip()
    project = create_project(tmp_path, source)

    def fake_traditional(
        units,
        jaccard_threshold=0.85,
        compute_unused=True,
        strict_unused=False,
        project_root=None,
    ):
        first, second = units[:2]
        return (
            [DuplicatePair(unit_a=first, unit_b=second, similarity=0.9, method="jaccard")],
            [],
            [],
        )

    monkeypatch.setattr(analyzer_module, "run_traditional_analysis", fake_traditional)
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(
            duplicate_factory=lambda units: [
                DuplicatePair(
                    unit_a=units[0],
                    unit_b=units[1],
                    similarity=0.96,
                    method="semantic",
                )
            ],
        ),
    )

    traditional_result = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=False,
            run_unused=False,
            min_semantic_lines=0,
            filter_tiny_traditional=False,
        )
    ).analyze(project)
    assert len(traditional_result.traditional_duplicates) == 1
    assert traditional_result.hybrid_duplicates == []

    semantic_result = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_lines=0,
        )
    ).analyze(project)
    assert len(semantic_result.semantic_duplicates) == 1
    assert semantic_result.hybrid_duplicates == []


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

    with pytest.raises(ValueError, match="semantic_unit_types"):
        AnalyzerConfig(semantic_unit_types=())

    with pytest.raises(ValueError, match="Invalid semantic_unit_types"):
        AnalyzerConfig(semantic_unit_types=("invalid",))

    with pytest.raises(ValueError, match="Invalid semantic_task"):
        AnalyzerConfig(semantic_task="not-a-task")

    with pytest.raises(ValueError, match="tiny_unit_statement_cutoff"):
        AnalyzerConfig(tiny_unit_statement_cutoff=-1)

    with pytest.raises(ValueError, match="tiny_near_jaccard_min"):
        AnalyzerConfig(tiny_near_jaccard_min=1.1)


def test_empty_directory_analysis(tmp_path: Path) -> None:
    analyzer = CodeAnalyzer()
    result = analyzer.analyze(tmp_path)

    assert result.units == []
    assert result.traditional_duplicates == []
    assert result.semantic_duplicates == []
    assert result.hybrid_duplicates == []
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

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(error=semantic_error),
    )

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
        _make_semantic_runner(error=semantic_error),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=run_unused,
            min_semantic_lines=0,
        )
    )

    with pytest.raises(type(semantic_error)):
        analyzer.analyze(project)
