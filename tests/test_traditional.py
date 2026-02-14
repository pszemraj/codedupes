from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from codedupes.extractor import CodeExtractor, compute_token_hash
from codedupes.traditional import run_traditional_analysis
from codedupes.traditional import find_potentially_unused
from codedupes.traditional import build_reference_graph


def _extract_units(tmp_path: Path, source: str) -> list:
    file_path = tmp_path / "sample.py"
    file_path.write_text(source)
    extractor = CodeExtractor(tmp_path, exclude_patterns=[], include_private=False)
    return list(extractor.extract_from_file(file_path))


def test_exact_duplicates_via_ast_hash(tmp_path: Path) -> None:
    source = dedent(
        """
        def foo(a, b):
            return a + b

        def bar(x, y):
            return x + y
        """
    ).strip()
    units = _extract_units(tmp_path, source)

    exact, near, _ = run_traditional_analysis(units, jaccard_threshold=0.85)

    assert len(exact) == 1
    assert len(near) == 0
    methods = {pair.method for pair in exact}
    assert methods == {"ast_hash"}


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
    units = _extract_units(tmp_path, source)

    exact_low, near_low, _ = run_traditional_analysis(units, jaccard_threshold=0.3)
    exact_high, near_high, _ = run_traditional_analysis(units, jaccard_threshold=0.95)

    assert len(near_low) >= 1
    assert len(near_high) == 0
    assert len(exact_low) == 0


def test_compute_token_hash_ignores_formatting() -> None:
    assert compute_token_hash("def f(x):\n    return x + 1") == compute_token_hash(
        "def f( x ):\n\treturn x+1"
    )


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
    units = _extract_units(tmp_path, source)
    build_reference_graph(units)

    unused = find_potentially_unused(units)
    names = {unit.name for unit in unused}

    assert "helper" not in names
    assert "caller" in names
    assert "dead" in names
