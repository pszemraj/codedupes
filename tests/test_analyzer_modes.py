"""Analysis modes end to end: combined, single-method, unused, and mode-invariant findings."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from codedupes import analyzer as analyzer_module
from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from codedupes.models import AnalysisResult, CodeUnit, CodeUnitType, DuplicatePair
from codedupes.pairs import ordered_pair_key
from codedupes.report.selection import build_exact_families
from tests.analyzer_helpers import (
    embedding_identity_from_kwargs,
    make_semantic_runner,
    traditional_single_jaccard_runner,
)
from tests.conftest import build_two_function_source, create_project, make_code_unit


def _capture_traditional_units_runner(captured_units: list[CodeUnit]):
    """Build a traditional runner that records incoming units and returns no matches."""

    def fake_traditional(units, jaccard_threshold=0.85):
        captured_units.extend(units)
        return [], []

    return fake_traditional


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

    unit = make_code_unit(
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

    def fake_traditional(units, jaccard_threshold=0.85):
        first, second, third = units
        nonlocal expected_exact_pair
        expected_exact_pair = tuple(sorted((first.uid, second.uid)))
        return (
            [DuplicatePair(unit_a=first, unit_b=second, similarity=1.0, method="structural_hash")],
            [DuplicatePair(unit_a=second, unit_b=third, similarity=0.9, method="jaccard")],
        )

    monkeypatch.setattr(analyzer_module, "run_traditional_analysis", fake_traditional)
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(
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
            min_semantic_statements=0,
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


@pytest.mark.parametrize(
    "run_semantic",
    [True, False],
)
def test_traditional_scope_is_independent_of_semantic_mode(
    tmp_path: Path,
    monkeypatch,
    run_semantic: bool,
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

    monkeypatch.setattr(
        analyzer_module,
        "run_traditional_analysis",
        _capture_traditional_units_runner(captured_traditional_units),
    )
    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", make_semantic_runner())

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=run_semantic,
            run_unused=False,
            min_semantic_statements=2,
        )
    )
    analyzer.analyze(project)

    assert {unit.unit_type for unit in captured_traditional_units} == {
        CodeUnitType.CLASS,
        CodeUnitType.METHOD,
        CodeUnitType.FUNCTION,
    }


def test_traditional_findings_are_mode_invariant(tmp_path: Path, monkeypatch) -> None:
    """Combined mode must retain full-scope deterministic findings."""
    source = dedent(
        """
        class Alpha:
            def render(self, value):
                return value.strip()

        class Beta:
            def render(self, value):
                return value.strip()
        """
    ).strip()
    project = create_project(tmp_path, source, module="scope.py")
    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", make_semantic_runner())

    traditional = CodeAnalyzer(
        AnalyzerConfig(
            run_semantic=False,
            run_unused=False,
            filter_tiny_traditional=False,
        )
    ).analyze(project)
    combined = CodeAnalyzer(
        AnalyzerConfig(run_unused=False, filter_tiny_traditional=False)
    ).analyze(project)

    def pair_keys(result: AnalysisResult) -> set[tuple[str, str, str]]:
        return {
            (*ordered_pair_key(duplicate.unit_a, duplicate.unit_b), duplicate.method)
            for duplicate in result.traditional_duplicates
        }

    assert pair_keys(traditional)
    assert pair_keys(combined) == pair_keys(traditional)


def test_ignore_duplicates_removes_the_pair_and_shrinks_the_family(tmp_path: Path) -> None:
    """``codedupes: ignore[duplicates]`` drops every pair naming the marked unit."""
    source = dedent(
        """
        def alpha(value):
            total = value + 1
            total *= 2
            return total + 3

        def beta(value):
            total = value + 1
            total *= 2
            return total + 3

        def gamma(value):  # codedupes: ignore[duplicates]
            total = value + 1
            total *= 2
            return total + 3
        """
    ).strip()
    project = create_project(tmp_path, source, module="triplet.py")

    result = CodeAnalyzer(
        AnalyzerConfig(
            run_semantic=False,
            run_unused=False,
            filter_tiny_traditional=False,
        )
    ).analyze(project)

    assert len(result.traditional_duplicates) == 1
    assert result.suppressed_duplicates == 2
    names = {
        result.traditional_duplicates[0].unit_a.name,
        result.traditional_duplicates[0].unit_b.name,
    }
    assert names == {"alpha", "beta"}

    families = build_exact_families(result.traditional_duplicates)
    assert len(families) == 1
    assert len(families[0].members) == 2


@pytest.mark.parametrize("run_unused", [True, False])
@pytest.mark.parametrize("run_traditional", [True, False])
def test_unused_analysis_preserves_duplicate_findings(
    tmp_path: Path, monkeypatch, run_unused: bool, run_traditional: bool
) -> None:
    """Keep duplicate evidence for units the unused heuristic reports."""
    source = dedent(
        """
        def _a(value):
            x = value + 1
            x *= 2
            return x + 1

        def _b(value):
            y = value + 2
            y *= 3
            return y + 2
        """
    ).strip()
    project = create_project(tmp_path, source, module="pairs.py")

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(
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
            run_traditional=run_traditional,
            run_semantic=True,
            run_unused=run_unused,
            embedding_cache=False,
            strict_unused=False,
        )
    )

    result = analyzer.analyze(project)
    assert len(result.semantic_duplicates) == 1
    assert result.semantic_duplicates[0].similarity == 0.99
    assert {unit.name for unit in result.potentially_unused} == (
        {"_a", "_b"} if run_unused else set()
    )
    if run_traditional:
        assert len(result.hybrid_duplicates) == 1
        assert result.hybrid_duplicates[0].semantic_similarity == 0.99


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
        make_semantic_runner(capture_exclude_pairs=captured_exclude_pairs),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
        )
    )

    result = analyzer.analyze(project)
    assert result.semantic_duplicates == []
    assert not captured_exclude_pairs


def test_combined_mode_excludes_tiny_filtered_structural_only_exact_pairs(
    tmp_path: Path, monkeypatch
) -> None:
    """Tiny-filtered structural-hash-only exact pairs must stay excluded from semantic scoring."""
    project = tmp_path / "src"
    project.mkdir()
    (project / "__init__.py").write_text("")
    (project / "a.py").write_text(
        "def alpha(x):\n    first = x + 1\n    second = first * 2\n    return second\n"
    )
    (project / "b.py").write_text(
        "def beta(y):\n    one = y + 1\n    two = one * 2\n    return two\n"
    )

    captured_exclude_pairs: set[tuple[str, str]] = set()
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(capture_exclude_pairs=captured_exclude_pairs),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_unused=False,
            # Both functions have 3 statements: semantic candidates at the default
            # min_semantic_statements, and tiny under this raised cutoff.
            tiny_unit_statement_cutoff=4,
        )
    )
    result = analyzer.analyze(project)

    unit_by_name = {unit.name: unit for unit in result.units}
    pair = ordered_pair_key(unit_by_name["alpha"], unit_by_name["beta"])

    # Same normalized structure, different identifiers: a structural_hash-only exact pair.
    assert unit_by_name["alpha"].structural_hash == unit_by_name["beta"].structural_hash
    assert unit_by_name["alpha"].token_hash != unit_by_name["beta"].token_hash
    # The tiny filter strips the pair from traditional output...
    assert result.traditional_duplicates == []
    # ...but semantic scoring must still treat it as an already-known exact pair.
    assert pair in captured_exclude_pairs


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
        model_name="Alibaba-NLP/gte-modernbert-base",
        instruction_prefix=None,
        threshold=0.82,
        exclude_pairs=None,
        batch_size=32,
        revision=None,
        trust_remote_code=None,
        semantic_task=None,
        **_device_kwargs,
    ):
        by_name = {unit.name: unit for unit in units}
        duplicates = [
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
        identity_kwargs = {
            "model_name": model_name,
            "instruction_prefix": instruction_prefix,
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "semantic_task": semantic_task,
            **_device_kwargs,
        }
        return (
            np.zeros((len(units), 2), dtype=np.float32),
            duplicates,
            embedding_identity_from_kwargs(identity_kwargs),
        )

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", fake_run_semantic)

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            suppress_test_semantic_matches=True,
        )
    )

    result = analyzer.analyze(project)

    assert len(result.semantic_duplicates) == 1
    assert {
        result.semantic_duplicates[0].unit_a.name,
        result.semantic_duplicates[0].unit_b.name,
    } == {"helper_alpha", "helper_beta"}


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

    monkeypatch.setattr(
        analyzer_module,
        "run_traditional_analysis",
        traditional_single_jaccard_runner(0.9),
    )
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(
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
            min_semantic_statements=0,
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
            min_semantic_statements=0,
        )
    ).analyze(project)
    assert len(semantic_result.semantic_duplicates) == 1
    assert semantic_result.hybrid_duplicates == []


def test_empty_directory_analysis(tmp_path: Path) -> None:
    analyzer = CodeAnalyzer()
    result = analyzer.analyze(tmp_path)

    assert result.units == []
    assert result.traditional_duplicates == []
    assert result.semantic_duplicates == []
    assert result.hybrid_duplicates == []
    assert result.potentially_unused == []


@pytest.mark.grammar
def test_cowsay_fixture_reports_planted_rust_exact_clone() -> None:
    fixture = Path(__file__).resolve().parents[1] / "test_fixtures" / "cowsay_wasm" / "src"

    result = CodeAnalyzer(
        AnalyzerConfig(run_semantic=False, run_unused=False),
    ).analyze(fixture)

    assert result.extraction_diagnostics == []
    assert len(result.units) == 28
    assert {unit.language for unit in result.units} == {"rust"}
    duplicate_names = {
        frozenset((duplicate.unit_a.qualified_name, duplicate.unit_b.qualified_name))
        for duplicate in result.traditional_duplicates
    }
    assert (
        frozenset(
            ("bubble.speech.make_borders", "bubble.thought.make_borders"),
        )
        in duplicate_names
    )
    assert frozenset(("cow.render", "bubble.render")) not in duplicate_names
