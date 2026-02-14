from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from codedupes.analyzer import CodeAnalyzer, AnalyzerConfig


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
    assert "caller" in names
    assert "_private_entry" not in names
