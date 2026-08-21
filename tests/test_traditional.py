from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from codedupes import traditional as traditional_module
from codedupes.traditional import (
    build_reference_graph,
    extract_identifiers,
    find_potentially_unused,
    run_traditional_analysis,
)
from tests.conftest import extract_units


def test_exact_duplicates_via_ast_hash(tmp_path: Path) -> None:
    source = dedent(
        """
        def foo(a, b):
            return a + b

        def bar(x, y):
            return x + y
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)

    exact, near, _ = run_traditional_analysis(units, jaccard_threshold=0.85)

    assert len(exact) == 1
    assert len(near) == 0
    methods = {pair.method for pair in exact}
    assert methods == {"ast_hash"}


def test_exact_duplicates_across_function_and_method(tmp_path: Path) -> None:
    source = dedent(
        """
        def render_summary(rows, limit, header):
            lines = [header]
            for row in rows[:limit]:
                lines.append(str(row))
            return "\\n".join(lines)

        class Report:
            @staticmethod
            def render_summary(rows, limit, header):
                lines = [header]
                for row in rows[:limit]:
                    lines.append(str(row))
                return "\\n".join(lines)
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)

    exact, _near, _ = run_traditional_analysis(units, jaccard_threshold=0.85)

    # A function copied verbatim into a class body must stay visible to exact
    # detection: functions and methods share a blocking kind, matching semantic
    # pairing.
    pairs = {
        tuple(sorted((pair.unit_a.qualified_name, pair.unit_b.qualified_name))) for pair in exact
    }
    assert ("sample.Report.render_summary", "sample.render_summary") in pairs


def test_near_duplicates_across_function_and_method(tmp_path: Path) -> None:
    source = dedent(
        """
        def collect_totals(entries, bucket, scale):
            totals = {}
            for entry in entries:
                totals[entry.bucket] = totals.get(entry.bucket, 0) + entry.value * scale
            return totals

        class Aggregator:
            def collect_totals(self, entries, bucket, scale):
                totals = {}
                for entry in entries:
                    totals[entry.bucket] = totals.get(entry.bucket, 0) + entry.value * scale
                return totals
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)

    _exact, near, _ = run_traditional_analysis(units, jaccard_threshold=0.8)

    pairs = {
        tuple(sorted((pair.unit_a.qualified_name, pair.unit_b.qualified_name))) for pair in near
    }
    assert ("sample.Aggregator.collect_totals", "sample.collect_totals") in pairs


def test_near_duplicates_threshold_boundary(tmp_path: Path) -> None:
    source = dedent(
        """
        def first(a, b):
            return a + b + a

        def second(a, c):
            return a + c + c

        def third(a, b):
            return b + 2
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)

    exact_low, near_low, _ = run_traditional_analysis(units, jaccard_threshold=0.3)
    _exact_high, near_high, _ = run_traditional_analysis(units, jaccard_threshold=0.95)

    assert len(near_low) >= 1
    assert len(near_high) == 0
    assert len(exact_low) == 0


def test_alias_aware_reference_graph(tmp_path: Path) -> None:
    source = dedent(
        """
        def helper(value):
            return value

        alias = helper

        def caller(value):
            return alias(value)

        def dead():
            return 0
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=False)
    build_reference_graph(units)

    unused = find_potentially_unused(units, strict_unused=True)
    names = {unit.name for unit in unused}

    assert "helper" not in names
    assert "caller" in names
    assert "dead" in names


def test_public_function_is_skipped_by_default(tmp_path: Path) -> None:
    source = dedent(
        """
        def public_function():
            return 1

        def _private_function():
            return 2

        def _unused_private():
            return _private_function() + public_function()
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    unused = find_potentially_unused(units, strict_unused=False)

    names = {unit.name for unit in unused}
    assert "public_function" not in names
    assert "_private_function" in names


def test_noqa_and_main_block_mark_as_used(tmp_path: Path) -> None:
    source = dedent(
        """
        def ignored_unused():  # noqa: codedupes
            return 42

        def used_by_main():
            return 7

        if __name__ == "__main__":
            used_by_main()
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    build_reference_graph(units, project_root=tmp_path)
    unused = find_potentially_unused(units, strict_unused=True)
    names = {unit.name for unit in unused}

    assert "ignored_unused" not in names
    assert "used_by_main" not in names


def test_pyproject_entry_points_mark_as_used(tmp_path: Path) -> None:
    source = dedent(
        """
        def cli_entry():
            return 1

        def helper():
            return 2
        """
    ).strip()
    (tmp_path / "pyproject.toml").write_text(
        dedent(
            """
            [project]
            name = "sample"
            scripts = { sample-cli = "sample_module:cli_entry" }
            """
        ).strip()
    )
    project = tmp_path / "src"
    project.mkdir()
    (project / "__init__.py").write_text("")
    (project / "sample_module.py").write_text(source)
    extractor_file = project / "sample_module.py"

    from codedupes.extractor import CodeExtractor

    units = list(CodeExtractor(project).extract_from_file(extractor_file))
    assert len(units) == 2
    build_reference_graph(units, project_root=tmp_path)
    unused = find_potentially_unused(units, strict_unused=True)
    names = {unit.name for unit in unused}
    assert "cli_entry" not in names
    assert "helper" in names


def test_main_block_calls_are_parsed_once_per_file(tmp_path: Path, monkeypatch) -> None:
    source = dedent(
        """
        def first():
            return 1

        def second():
            return 2

        if __name__ == "__main__":
            first()
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    calls: list[Path] = []

    def fake_extract_main_block_calls(path: Path) -> set[str]:
        calls.append(path)
        return {"first"}

    monkeypatch.setattr(
        traditional_module,
        "_extract_main_block_calls",
        fake_extract_main_block_calls,
    )

    build_reference_graph(units)

    assert len(calls) == 1


def test_extract_identifiers_filters_builtin_names() -> None:
    source = dedent(
        """
        def helper(items):
            total = len(items)
            print(total)
            return sorted(items)
        """
    ).strip()

    identifiers = extract_identifiers(source)

    assert {"helper", "items", "total"} <= identifiers
    assert not {"len", "print", "sorted"} & identifiers
