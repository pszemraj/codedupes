"""Semantic candidate selection and the tiny-duplicate filter."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from codedupes import analyzer as analyzer_module
from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from codedupes.models import CodeUnit, CodeUnitType, DuplicatePair
from tests.analyzer_helpers import make_semantic_runner
from tests.conftest import create_project


def _record_unit_types(captured_types: list[CodeUnitType]):
    """Build a duplicate_factory that records unit types and finds no matches."""

    def record(units: list[CodeUnit]) -> list[DuplicatePair]:
        captured_types.extend(unit.unit_type for unit in units)
        return []

    return record


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
            min_semantic_statements=3,
        )
    )
    result = analyzer.analyze(project)
    assert result.semantic_duplicates == []


def test_decorated_methods_survive_semantic_and_tiny_filters(tmp_path: Path, monkeypatch) -> None:
    source = dedent(
        """
        class First:
            @property
            def area(self):
                width = self.width
                height = self.height
                scale = self.scale
                return width * height * scale

        class Second:
            @property
            def area(self):
                width = self.width
                height = self.height
                scale = self.scale
                return width * height * scale
        """
    ).strip()
    project = create_project(tmp_path, source, module="decorated.py")
    semantic_units: list[CodeUnit] = []

    def record_semantic_units(units: list[CodeUnit]) -> list[DuplicatePair]:
        semantic_units.extend(units)
        return []

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(duplicate_factory=record_semantic_units),
    )
    result = CodeAnalyzer(AnalyzerConfig(run_unused=False)).analyze(project)

    assert {unit.qualified_name for unit in semantic_units} == {
        "decorated.First.area",
        "decorated.Second.area",
    }
    assert any(
        {duplicate.unit_a.qualified_name, duplicate.unit_b.qualified_name}
        == {"decorated.First.area", "decorated.Second.area"}
        for duplicate in result.traditional_duplicates
    )


@pytest.mark.parametrize(
    ("semantic_unit_types", "expected_types"),
    [
        (None, {CodeUnitType.FUNCTION, CodeUnitType.METHOD}),
        (("class",), {CodeUnitType.CLASS}),
    ],
)
def test_semantic_unit_scope(
    tmp_path: Path,
    monkeypatch,
    semantic_unit_types: tuple[str, ...] | None,
    expected_types: set[CodeUnitType],
) -> None:
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

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(duplicate_factory=_record_unit_types(captured_types)),
    )

    config_kwargs = {
        "run_traditional": False,
        "run_semantic": True,
        "run_unused": False,
        "min_semantic_statements": 0,
    }
    if semantic_unit_types is not None:
        config_kwargs["semantic_unit_types"] = semantic_unit_types
    analyzer = CodeAnalyzer(AnalyzerConfig(**config_kwargs))
    analyzer.analyze(project)

    assert captured_types
    assert set(captured_types) == expected_types


@pytest.mark.parametrize(
    ("filter_tiny_traditional", "expected_exact_count"),
    [(None, 0), (False, 2)],
)
def test_tiny_exact_duplicate_filter(
    tmp_path: Path,
    caplog,
    filter_tiny_traditional: bool | None,
    expected_exact_count: int,
) -> None:
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

    config_kwargs = {
        "run_traditional": True,
        "run_semantic": False,
        "run_unused": False,
        "jaccard_threshold": 0.99,
    }
    if filter_tiny_traditional is not None:
        config_kwargs["filter_tiny_traditional"] = filter_tiny_traditional
    analyzer = CodeAnalyzer(AnalyzerConfig(**config_kwargs))
    with caplog.at_level("INFO"):
        result = analyzer.analyze(project)

    # Pin the expected count independently of the result under test, so a
    # wrong analysis with a matching log message cannot pass silently.
    exact_count = sum(
        duplicate.method in {"structural_hash", "token_hash"}
        for duplicate in result.traditional_duplicates
    )
    assert exact_count == expected_exact_count
    exact_logs = [
        record.getMessage()
        for record in caplog.records
        if "exact duplicates" in record.getMessage()
    ]
    assert exact_logs == [f"Found {expected_exact_count} exact duplicates"]


@pytest.mark.parametrize("filter_tiny_traditional", [True, False])
def test_tiny_class_duplicates_follow_tiny_filter(
    tmp_path: Path, filter_tiny_traditional: bool
) -> None:
    """Marker classes should follow the same tiny-unit policy as callables."""
    source = dedent(
        """
        class FirstError(RuntimeError):
            \"\"\"First domain error.\"\"\"

        class SecondError(RuntimeError):
            \"\"\"Second domain error.\"\"\"
        """
    ).strip()
    project = create_project(tmp_path, source, module="tiny_classes.py")

    result = CodeAnalyzer(
        AnalyzerConfig(
            run_semantic=False,
            run_unused=False,
            filter_tiny_traditional=filter_tiny_traditional,
        )
    ).analyze(project)

    assert len(result.traditional_duplicates) == int(not filter_tiny_traditional)


_PYTHON_TWO_METHOD_CLASSES = dedent(
    """
    class FirstProcessor:
        def prepare(self, value):
            adjusted = value + 1
            doubled = adjusted * 2
            return doubled

        def finish(self, value):
            adjusted = value - 1
            doubled = adjusted * 2
            return doubled

    class SecondProcessor:
        def prepare(self, value):
            adjusted = value + 1
            doubled = adjusted * 2
            return doubled

        def finish(self, value):
            adjusted = value - 1
            doubled = adjusted * 2
            return doubled
    """
).strip()

_JAVASCRIPT_TWO_METHOD_CLASSES = dedent(
    """
    class FirstProcessor {
      prepare(value) {
        const adjusted = value + 1;
        const doubled = adjusted * 2;
        return doubled;
      }
      finish(value) {
        const adjusted = value - 1;
        const doubled = adjusted * 2;
        return doubled;
      }
    }
    class SecondProcessor {
      prepare(value) {
        const adjusted = value + 1;
        const doubled = adjusted * 2;
        return doubled;
      }
      finish(value) {
        const adjusted = value - 1;
        const doubled = adjusted * 2;
        return doubled;
      }
    }
    """
).strip()


@pytest.mark.parametrize(
    ("module", "source"),
    [
        ("large_classes.py", _PYTHON_TWO_METHOD_CLASSES),
        ("large_classes.js", _JAVASCRIPT_TWO_METHOD_CLASSES),
    ],
    ids=["python", "tree-sitter"],
)
def test_large_class_duplicates_are_not_filtered_by_member_count(
    tmp_path: Path, module: str, source: str
) -> None:
    """Two members with substantial bodies are not a tiny class, whichever extractor counts them."""
    project = create_project(tmp_path, source, module=module)

    result = CodeAnalyzer(
        AnalyzerConfig(run_semantic=False, run_unused=False, filter_tiny_traditional=True)
    ).analyze(project)

    classes = [unit for unit in result.units if unit.unit_type == CodeUnitType.CLASS]
    assert {unit.statement_count for unit in classes} == {2}
    assert any(
        duplicate.unit_a.unit_type == CodeUnitType.CLASS
        and duplicate.unit_b.unit_type == CodeUnitType.CLASS
        for duplicate in result.traditional_duplicates
    )


def test_large_class_duplicates_survive_private_member_filter(tmp_path: Path) -> None:
    """An incomplete visible-member list must not make substantial classes tiny."""
    source = dedent(
        """
        class FirstProcessor:
            def _prepare(self, value):
                adjusted = value + 1
                doubled = adjusted * 2
                return doubled

        class SecondProcessor:
            def _prepare(self, value):
                adjusted = value + 1
                doubled = adjusted * 2
                return doubled
        """
    ).strip()
    project = create_project(tmp_path, source, module="private_class_members.py")

    result = CodeAnalyzer(
        AnalyzerConfig(
            run_semantic=False,
            run_unused=False,
            include_private=False,
            filter_tiny_traditional=True,
        )
    ).analyze(project)

    assert {unit.name for unit in result.units} == {"FirstProcessor", "SecondProcessor"}
    assert any(
        duplicate.unit_a.unit_type == CodeUnitType.CLASS
        and duplicate.unit_b.unit_type == CodeUnitType.CLASS
        for duplicate in result.traditional_duplicates
    )


@pytest.mark.parametrize(
    ("initializer", "expected_duplicate"),
    [
        ("", False),
        ("this.ready = true;", False),
        ("if (enabled) { this.first = 1; this.second = 2; }", True),
    ],
)
def test_class_static_initializers_follow_tiny_filter(
    tmp_path: Path, initializer: str, expected_duplicate: bool
) -> None:
    """Prove the tiny filter consumes the static-block statement count.

    Per-suffix (js/jsx/ts/tsx) statement counting of static-block bodies is
    the authority of
    test_polyglot_ecmascript.py::test_class_member_count_includes_static_initializer_bodies;
    one suffix here is enough to prove the analyzer's tiny filter reacts to
    that count.
    """
    source = (
        f"class First {{ static {{ {initializer} }} }}\n"
        f"class Second {{ static {{ {initializer} }} }}\n"
    )
    project = create_project(tmp_path, source, module="initializers.js")
    unfiltered = CodeAnalyzer(
        AnalyzerConfig(run_semantic=False, run_unused=False, filter_tiny_traditional=False)
    ).analyze(project)
    assert len(unfiltered.traditional_duplicates) == 1

    filtered = CodeAnalyzer(AnalyzerConfig(run_semantic=False, run_unused=False)).analyze(project)
    assert len(filtered.units) == 2
    assert all(unit.unit_type == CodeUnitType.CLASS for unit in filtered.units)
    assert len(filtered.traditional_duplicates) == int(expected_duplicate)


@pytest.mark.parametrize("filter_tiny_traditional", [True, False])
def test_tiny_near_duplicates_follow_tiny_filter(
    tmp_path: Path, monkeypatch, filter_tiny_traditional: bool
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

    def fake_traditional(units, jaccard_threshold=0.85):
        return (
            [],
            [DuplicatePair(unit_a=units[0], unit_b=units[1], similarity=1.0, method="jaccard")],
        )

    monkeypatch.setattr(analyzer_module, "run_traditional_analysis", fake_traditional)

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=False,
            run_unused=False,
            filter_tiny_traditional=filter_tiny_traditional,
        )
    )
    result = analyzer.analyze(project)

    assert len(result.traditional_duplicates) == int(not filter_tiny_traditional)
