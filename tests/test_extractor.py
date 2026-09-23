from __future__ import annotations

import codecs
import fnmatch
import os
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Any
from unittest.mock import Mock

import pytest

from codedupes.extractor import CodeExtractor
from codedupes.models import CodeUnitType
from tests.conftest import extract_units


def test_nested_scope_extraction_and_private_filtering(tmp_path: Path) -> None:
    code = dedent(
        """
        def top_level(value):
            def nested(value):
                return value * 2

            return nested(value)

        class Container:
            def method(self, value):
                return value

            class Inner:
                def inner_method(self):
                    return 1

            def _private(self):
                return 2

        class _PrivateClass:
            pass
        """
    ).strip()

    units = extract_units(tmp_path, code, include_private=False)
    names = {unit.qualified_name: unit.unit_type for unit in units}

    assert names["sample.top_level"] == CodeUnitType.FUNCTION
    assert names["sample.top_level.nested"] == CodeUnitType.FUNCTION
    assert names["sample.Container"] == CodeUnitType.CLASS
    assert names["sample.Container.method"] == CodeUnitType.METHOD
    assert names["sample.Container.Inner"] == CodeUnitType.CLASS
    assert names["sample.Container.Inner.inner_method"] == CodeUnitType.METHOD
    assert "sample.Container._private" not in names
    assert "sample._PrivateClass" not in names


def test_python_syntax_error_skips_the_broken_unit_and_reports_it(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    root.joinpath("__init__.py").write_text("")
    bad = root / "bad.py"
    bad.write_text("def broken(:\n    pass\n")
    extractor = CodeExtractor(root, include_private=False)

    assert list(extractor.extract_from_file(bad)) == []
    assert [diagnostic.code for diagnostic in extractor.diagnostics] == [
        "partial-parse",
        "unit-parse-error",
    ]
    assert all(diagnostic.language == "python" for diagnostic in extractor.diagnostics)


def test_extract_all_deduplicates_symlinked_paths(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("")

    source = dedent(
        """
        def sample():
            return 1
        """
    ).strip()
    real = package / "real.py"
    real.write_text(source)
    alias = package / "alias.py"
    alias.symlink_to(real)

    extractor = CodeExtractor(package, include_private=False)
    units = extractor.extract_all()
    assert len(units) == 1


def test_extract_all_records_every_visited_file_even_without_units(tmp_path: Path) -> None:
    """A re-export module yields nothing, yet ``extracted_files`` lists it by language."""
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "impl.py").write_text("def real():\n    return 1\n")
    (package / "api.py").write_text("from .impl import real\n\n__all__ = ['real']\n")
    (package / "helper.js").write_text("export const run = () => 1;\n")

    extractor = CodeExtractor(package, include_private=True)
    units = extractor.extract_all()

    assert [unit.qualified_name for unit in units] == ["helper.run", "impl.real"]
    assert {
        language: sorted(path.name for path in paths)
        for language, paths in extractor.extracted_files.items()
    } == {
        "javascript": ["helper.js"],
        "python": ["__init__.py", "api.py", "impl.py"],
    }


def test_extract_all_survives_symlink_to_file_outside_root(tmp_path: Path) -> None:
    outside = tmp_path / "ext"
    outside.mkdir()
    target = outside / "shared.py"
    target.write_text("def alpha(x):\n    y = x + 1\n    z = y * 2\n    return z\n")

    root = tmp_path / "proj"
    root.mkdir()
    (root / "normal.py").write_text("def beta(x):\n    y = x - 1\n    z = y * 3\n    return z\n")
    (root / "linked.py").symlink_to(target)

    extractor = CodeExtractor(root, include_private=False)
    units = extractor.extract_all()

    # The symlink is the file's in-tree identity: extraction must not abort,
    # and the module name comes from the link, not the resolved target.
    assert sorted(unit.qualified_name for unit in units) == ["linked.alpha", "normal.beta"]


def test_non_header_extraction_does_not_resolve_c_header_policy(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "package"
    root.mkdir()
    module = root / "module.py"
    module.write_text("def sample():\n    return 1\n")
    extractor = CodeExtractor(root, include_private=False)

    def fail_on_header_probe() -> bool:
        raise AssertionError("non-header extraction must not resolve the C-header policy")

    monkeypatch.setattr(extractor, "_allow_c_headers", fail_on_header_probe)

    assert [unit.name for unit in extractor.extract_from_file(module)] == ["sample"]
    assert [unit.name for unit in extractor.extract_all()] == ["sample"]


def test_header_only_tree_reports_c_header_policy_diagnostic(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    root.mkdir()
    (root / "test_ignored.h").write_text("", encoding="utf-8")
    (root / "clamp.h").write_text(
        "static inline int clamp_value(int v, int lo, int hi) {\n"
        "    if (v < lo) return lo;\n"
        "    if (v > hi) return hi;\n"
        "    return v;\n"
        "}\n"
    )

    extractor = CodeExtractor(root, include_private=True)
    units = extractor.extract_all()

    assert units == []
    codes = [diagnostic.code for diagnostic in extractor.diagnostics]
    assert codes == ["c-header-policy"]
    assert "--language c" in extractor.diagnostics[0].message
    assert extractor.diagnostics[0].message.startswith("1 .h file(s)")
    assert extractor.diagnostics[0].file_path == root / "clamp.h"


def test_cpp_presence_reports_skipped_headers(tmp_path: Path) -> None:
    root = tmp_path / "mixed"
    (root / "third_party").mkdir(parents=True)
    (root / "main.c").write_text("int main(void) {\n    return 0;\n}\n")
    (root / "util.h").write_text("static int helper(int v) {\n    return v + 1;\n}\n")
    (root / "third_party" / "x.cpp").write_text("int cpp_fn() {\n    return 2;\n}\n")

    extractor = CodeExtractor(root, include_private=True)
    units = extractor.extract_all()

    assert [unit.qualified_name for unit in units] == ["main.main"]
    codes = [diagnostic.code for diagnostic in extractor.diagnostics]
    assert "c-header-policy" in codes


@pytest.mark.parametrize(
    ("filename", "source", "code"),
    [
        ("mytool", "#!/usr/bin/env python\ndef alpha():\n    return 1\n", "unsupported-file"),
        ("types.d.ts", "export declare function alpha(v: number): number;\n", "declaration-file"),
    ],
    ids=["unsupported", "declaration"],
)
def test_explicit_skipped_file_reports_diagnostic(
    tmp_path: Path, filename: str, source: str, code: str
) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    target = root / filename
    target.write_text(source)

    extractor = CodeExtractor(root, include_private=True)
    units = list(extractor.extract_from_file(target))

    assert units == []
    assert [diagnostic.code for diagnostic in extractor.diagnostics] == [code]


def test_explicit_language_filtered_file_reports_diagnostic(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    module = root / "mod.py"
    module.write_text("def alpha():\n    return 1\n")

    extractor = CodeExtractor(root, include_private=True, languages=("rust",))
    units = list(extractor.extract_from_file(module))

    assert units == []
    diagnostic = extractor.diagnostics[0]
    assert diagnostic.code == "language-filter"
    assert diagnostic.language == "python"


def test_stub_module_name_drops_the_pyi_suffix(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("")
    stub = package / "typed_mod.pyi"
    stub.write_text("def entry() -> int: ...\n")

    extractor = CodeExtractor(package, include_private=True, include_stubs=True)
    units = list(extractor.extract_from_file(stub))

    assert len(units) == 1
    assert units[0].qualified_name == "typed_mod.entry"


def test_extract_from_file_honors_include_stubs_false(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("")
    stub = package / "typed_mod.pyi"
    stub.write_text("def entry() -> int: ...\n")

    extractor = CodeExtractor(package, include_private=True)
    units = list(extractor.extract_from_file(stub))

    assert units == []
    assert [diagnostic.code for diagnostic in extractor.diagnostics] == ["stub-policy"]


def test_extract_all_skips_common_artifact_directories(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(
        dedent(
            """
            def keep():
                return 1
            """
        ).strip()
        + "\n"
    )

    target_dir = root / "target"
    target_dir.mkdir()
    (target_dir / "generated.py").write_text(
        dedent(
            """
            def ignore_me():
                return 2
            """
        ).strip()
        + "\n"
    )

    node_modules_dir = root / "node_modules"
    node_modules_dir.mkdir()
    (node_modules_dir / "lib.py").write_text(
        dedent(
            """
            def ignore_me_too():
                return 3
            """
        ).strip()
        + "\n"
    )

    extractor = CodeExtractor(root, include_private=True)
    units = extractor.extract_all()
    qualified_names = {unit.qualified_name for unit in units}

    assert "pkg.main.keep" in qualified_names
    assert all("ignore_me" not in name for name in qualified_names)


def test_extract_all_skips_suffix_test_files_by_default(tmp_path: Path, caplog) -> None:
    source = "def entry():\n    return 1\n"
    (tmp_path / "inject_test.py").write_text(source)
    (tmp_path / "inject_tests.py").write_text(source)
    (tmp_path / "keeper.py").write_text(source)

    with caplog.at_level("INFO", logger="codedupes.extractor"):
        units = CodeExtractor(tmp_path, include_private=True).extract_all()
    assert [unit.file_path.name for unit in units] == ["keeper.py"]
    assert (
        "Skipped 2 files and 0 directories matching default test exclusions; "
        "their Python files still count as references for unused-code analysis. "
        "Use --no-default-excludes to include them in duplicate detection too."
    ) in caplog.text


@pytest.mark.parametrize("include_tests", [False, True])
def test_default_exclusion_hint_counts_pruned_directories(tmp_path: Path, caplog, include_tests):
    from codedupes.extractor import DEFAULT_EXCLUDE_PATTERNS

    for relative in ["test_one.py", "test_helpers/deep.py", "node_modules/test_dep.py", "skip.py"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def entry():\n    return 1\n", encoding="utf-8")
    patterns = ([] if include_tests else DEFAULT_EXCLUDE_PATTERNS) + ["skip.py"]

    extractor = CodeExtractor(
        tmp_path, exclude_patterns=patterns, implicit_default_excludes=not include_tests
    )
    with caplog.at_level("INFO", logger="codedupes.extractor"):
        extractor.extract_all()

    if include_tests:
        assert "matching default test exclusions" not in caplog.text
        # Nothing is skipped by a default shape when defaults are disabled, so
        # nothing needs the reference-only path either.
        assert extractor.reference_only_files == []
    else:
        assert (
            "Skipped 1 files and 1 directories matching default test exclusions; "
            "their Python files still count as references for unused-code analysis."
        ) in caplog.text
        # Both default-excluded shapes feed the reference-only list; the
        # artifact directory and the user's own "skip.py" exclusion do not.
        names = {path.name for path in extractor.reference_only_files}
        assert names == {"test_one.py", "deep.py"}


def test_extract_from_file_respects_exclude_patterns(tmp_path: Path) -> None:
    source = "def entry():\n    return 1\n"
    file_path = tmp_path / "sample.py"
    file_path.write_text(source)

    extractor = CodeExtractor(tmp_path, exclude_patterns=["sample.py"], include_private=True)
    units = list(extractor.extract_from_file(file_path))
    assert units == []


def test_extract_all_double_star_pattern_matches_root_level_files(tmp_path: Path) -> None:
    source = "def entry():\n    return 1\n"
    file_path = tmp_path / "sample.py"
    file_path.write_text(source)

    extractor = CodeExtractor(tmp_path, exclude_patterns=["**/sample.py"], include_private=True)
    units = extractor.extract_all()
    assert units == []


@pytest.mark.parametrize(
    "pattern", ["examples", "examples/", "**/examples", "**/examples/**", "exam*"]
)
def test_exclude_directory_at_any_depth(tmp_path: Path, pattern: str) -> None:
    paths = ["examples/a.py", "pkg/examples/deep/b.py", "pkg/keep.py", "myexamples/c.py"]
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def entry():\n    return 1\n", encoding="utf-8")
    extractor = CodeExtractor(tmp_path, exclude_patterns=[pattern])

    assert {
        unit.file_path.relative_to(tmp_path).as_posix() for unit in extractor.extract_all()
    } == {"pkg/keep.py", "myexamples/c.py"}
    assert list(extractor.extract_from_file(tmp_path / "pkg/examples/deep/b.py")) == []


@pytest.mark.parametrize(
    "pattern", ["./examples/", "examples/deep", "examples/deep/", "examples/deep/**"]
)
def test_exclude_root_relative_paths(tmp_path: Path, pattern: str) -> None:
    paths = ["examples/deep/a.py", "pkg/examples/deep/b.py"]
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def entry():\n    return 1\n", encoding="utf-8")
    units = CodeExtractor(tmp_path, exclude_patterns=[pattern]).extract_all()
    assert [unit.file_path.relative_to(tmp_path).as_posix() for unit in units] == [paths[1]]


def test_explicit_empty_excludes_include_tests(tmp_path: Path) -> None:
    path = tmp_path / "test_entry.py"
    path.write_text("def entry():\n    return 1\n", encoding="utf-8")
    assert len(CodeExtractor(tmp_path, exclude_patterns=[]).extract_all()) == 1


@pytest.mark.parametrize("cpp_path", ["examples/foreign.cpp", "test_foreign.cpp"])
def test_header_detection_ignores_excluded_cpp(tmp_path: Path, cpp_path: str) -> None:
    (tmp_path / "main.c").write_text("int run(void) { return 1; }", encoding="utf-8")
    foreign = tmp_path / cpp_path
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_text("", encoding="utf-8")
    from codedupes.extractor import DEFAULT_EXCLUDE_PATTERNS

    extractor = CodeExtractor(tmp_path, exclude_patterns=[*DEFAULT_EXCLUDE_PATTERNS, "examples"])
    assert extractor._allow_c_headers()


def test_header_detection_requires_included_c_source(tmp_path: Path) -> None:
    (tmp_path / "ignored.c").write_text("", encoding="utf-8")
    assert not CodeExtractor(tmp_path, exclude_patterns=["ignored.c"])._allow_c_headers()


@pytest.mark.parametrize("suffix", [".c", ".cpp"])
@pytest.mark.parametrize("exclude", ["target", "alias"])
def test_header_detection_respects_symlink_exclusions(
    tmp_path: Path, suffix: str, exclude: str
) -> None:
    if suffix == ".cpp":
        (tmp_path / "main.c").write_text("", encoding="utf-8")
    target = tmp_path / f"target{suffix}"
    target.write_text("", encoding="utf-8")
    (tmp_path / f"alias{suffix}").symlink_to(target)
    patterns = [f"{exclude}{suffix}"]
    if exclude == "alias":
        # Excluding an alias must not exclude the actual included target.
        expected = suffix == ".c"
    else:
        expected = suffix == ".cpp"
    assert CodeExtractor(tmp_path, exclude_patterns=patterns)._allow_c_headers() is expected


def test_excluded_alias_does_not_hide_included_target(tmp_path: Path) -> None:
    target = tmp_path / "z_target.py"
    target.write_text("def entry():\n    return 1\n", encoding="utf-8")
    alias = tmp_path / "a_alias.py"
    alias.symlink_to(target)
    units = CodeExtractor(tmp_path, exclude_patterns=[alias.name]).extract_all()
    assert [unit.file_path for unit in units] == [target]


def test_direct_symlink_respects_its_excluded_name(tmp_path: Path) -> None:
    target = tmp_path / "target.py"
    target.write_text("def entry():\n    return 1\n", encoding="utf-8")
    alias = tmp_path / "excluded.py"
    alias.symlink_to(target)
    extractor = CodeExtractor(tmp_path, exclude_patterns=[alias.name])
    assert list(extractor.extract_from_file(alias)) == []


@pytest.mark.parametrize("pattern", ["examples", "examples/", "examples/**", "**/examples/**"])
def test_excluded_directories_are_not_walked(tmp_path: Path, monkeypatch, pattern: str) -> None:
    (tmp_path / "examples" / "deep").mkdir(parents=True)
    visited = []
    original_walk = os.walk

    def recording_walk(*args, **kwargs):
        for entry in original_walk(*args, **kwargs):
            visited.append(Path(entry[0]))
            yield entry

    monkeypatch.setattr(os, "walk", recording_walk)
    CodeExtractor(tmp_path, exclude_patterns=[pattern]).extract_all()
    assert visited == [tmp_path]


@pytest.mark.parametrize("depth", [1, 6])
def test_walk_exclusion_matching_cost(tmp_path: Path, monkeypatch: Any, depth: int) -> None:
    directory = tmp_path.joinpath(*[f"level{i}" for i in range(depth)])
    directory.mkdir(parents=True)
    source = directory / "entry.py"
    source.touch()
    for i in range(40):
        (directory / f"notes{i}.txt").touch()
    extractor = CodeExtractor(tmp_path)
    matchers = [
        (use_path, directory_only, Mock(wraps=matcher), Mock(wraps=zero_depth))
        for use_path, directory_only, matcher, zero_depth in extractor._exclude_matchers
    ]
    extractor._exclude_matchers = matchers
    translate = Mock(side_effect=AssertionError("Patterns must be compiled before walking"))
    monkeypatch.setattr(fnmatch, "translate", translate)
    extract = Mock(return_value=iter(()))
    monkeypatch.setattr(extractor, "extract_from_file", extract)

    extractor.extract_all()

    extract.assert_called_once_with(source)
    # Each directory gets at most four matches per pattern, and the source two.
    # Unsupported files and previously visited ancestors add no matching work.
    calls = sum(m.match.call_count + z.match.call_count for _, _, m, z in matchers)
    assert calls <= len(matchers) * (4 * depth + 2)
    translate.assert_not_called()


def test_walk_symlink_checks_excluded_target_ancestors(tmp_path: Path) -> None:
    target = tmp_path / "examples" / "deep" / "target.py"
    target.parent.mkdir(parents=True)
    target.write_text("def entry():\n    return 1\n", encoding="utf-8")
    alias = tmp_path / "alias.py"
    alias.symlink_to(target)
    extractor = CodeExtractor(tmp_path, exclude_patterns=["examples/"])

    assert extractor.extract_all() == []
    assert list(extractor.extract_from_file(alias)) == []


def test_python_byte_range_matches_emitted_source_with_unicode(tmp_path: Path) -> None:
    source = (
        "# café before the unit\n"
        "\n"
        "def greet(name):\n"
        '    message = "héllo " + name\n'
        "    return message\n"
    )
    file_path = tmp_path / "unicode_sample.py"
    file_path.write_text(source, encoding="utf-8")

    unit = next(CodeExtractor(tmp_path, include_private=True).extract_from_file(file_path))
    encoded = source.encode("utf-8")

    assert encoded[unit.start_byte : unit.end_byte] == unit.source.encode("utf-8")
    assert unit.start_column == 0
    assert unit.end_column == len(b"    return message")


def test_python_source_lines_survive_form_feed_separator(tmp_path: Path) -> None:
    # PEP 8 allows form feeds as section separators, and Python line numbers do
    # not advance on ``\f``/``\v``; the emitted span must follow the same rule.
    source = (
        "def before():\n"
        '    return "\v"\n'
        "\f\n"
        "def after(name):\n"
        '    message = "hi " + name\n'
        "    return message\n"
    )
    file_path = tmp_path / "form_feed_sample.py"
    file_path.write_text(source, encoding="utf-8")

    units = list(CodeExtractor(tmp_path, include_private=True).extract_from_file(file_path))
    unit = next(unit for unit in units if unit.name == "after")
    encoded = source.encode("utf-8")

    assert unit.source == 'def after(name):\n    message = "hi " + name\n    return message'
    assert encoded[unit.start_byte : unit.end_byte] == unit.source.encode("utf-8")
    assert (unit.lineno, unit.end_lineno) == (4, 6)
    assert unit.end_column == len(b"    return message")


def test_python_crlf_source_stays_byte_exact(tmp_path: Path) -> None:
    # ``read_text`` would translate the line endings away, so the byte range would
    # describe LF text that is not what the file stores.
    file_path = tmp_path / "crlf_sample.py"
    file_path.write_bytes(
        b"# leading comment\r\ndef greet(name):\r\n"
        b'    message = "hi " + name\r\n'
        b"    return message\r\n"
    )

    units = list(CodeExtractor(tmp_path, include_private=True).extract_from_file(file_path))
    raw = file_path.read_bytes()

    assert [unit.name for unit in units] == ["greet"]
    unit = units[0]
    assert "\r\n" in unit.source
    assert raw[unit.start_byte : unit.end_byte].decode("utf-8") == unit.source
    assert (unit.lineno, unit.end_lineno) == (2, 4)


def test_python_bom_file_extracts_with_on_disk_byte_offsets(tmp_path: Path) -> None:
    # The lexer skips the BOM, so the first unit starts at byte 3 and the byte range
    # still slices the file as stored.
    file_path = tmp_path / "bom_sample.py"
    body = 'def greet(name):\n    message = "héllo " + name\n    return message\n'
    file_path.write_bytes(codecs.BOM_UTF8 + body.encode("utf-8"))

    extractor = CodeExtractor(tmp_path, include_private=True)
    units = list(extractor.extract_from_file(file_path))
    raw = file_path.read_bytes()

    assert [unit.name for unit in units] == ["greet"]
    assert extractor.diagnostics == []
    unit = units[0]
    assert unit.start_byte == len(codecs.BOM_UTF8)
    assert raw[unit.start_byte : unit.end_byte].decode("utf-8") == unit.source
    assert not unit.source.startswith("﻿")


def test_python_file_with_nul_byte_reports_a_diagnostic(tmp_path: Path) -> None:
    file_path = tmp_path / "nul_sample.py"
    file_path.write_bytes(b"def greet():\n    return 1\x00\n")

    extractor = CodeExtractor(tmp_path, include_private=True)
    units = list(extractor.extract_from_file(file_path))

    assert units == []
    assert [diagnostic.code for diagnostic in extractor.diagnostics] == [
        "partial-parse",
        "unit-parse-error",
    ]


def test_unreadable_files_do_not_abort_extraction(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "keeper.py").write_text("def alpha():\n    return 1\n")
    missing = tmp_path / "gone"
    (root / "dangling.py").symlink_to(missing / "absent.py")
    (root / "dangling.js").symlink_to(missing / "absent.js")
    (root / "loop.py").symlink_to("loop.py")

    extractor = CodeExtractor(root, include_private=True)
    units = extractor.extract_all()

    assert [unit.qualified_name for unit in units] == ["keeper.alpha"]
    read_errors = [
        diagnostic for diagnostic in extractor.diagnostics if diagnostic.code == "read-error"
    ]
    assert {diagnostic.language for diagnostic in read_errors} == {"python", "javascript"}
    assert {diagnostic.file_path.name for diagnostic in read_errors} == {
        "dangling.py",
        "dangling.js",
        "loop.py",
    }


def test_extract_all_order_is_independent_of_walk_order(tmp_path: Path, monkeypatch: Any) -> None:
    names = ["gamma.py", "alpha.py", "beta.py"]
    for name in names:
        (tmp_path / name).write_text(f"def {Path(name).stem}():\n    return 1\n")

    def walk_yielding(order: list[str]) -> Any:
        def fake_walk(top: str, followlinks: bool = True, onerror: Any = None) -> Any:
            yield str(tmp_path), [], list(order)

        return fake_walk

    def extracted(order: list[str]) -> list[str]:
        monkeypatch.setattr(os, "walk", walk_yielding(order))
        extractor = CodeExtractor(tmp_path, include_private=True)
        return [unit.qualified_name for unit in extractor.extract_all()]

    forward = extracted(names)
    reverse = extracted(list(reversed(names)))

    assert forward == reverse == ["alpha.alpha", "beta.beta", "gamma.gamma"]


requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _write_source_tree(root: Path, relative_paths: list[str]) -> None:
    """Write one three-statement function per path, named after the file stem."""
    for relative in relative_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"def {path.stem}_fn():\n    a = 1\n    b = 2\n    return a + b\n")


def _git_work_tree(tmp_path: Path) -> Path:
    """Create a repository whose root and nested ``.gitignore`` files ignore three paths."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".gitignore").write_text("ignored_dir/\nignored_module.py\n")
    (root / "sub").mkdir()
    (root / "sub" / ".gitignore").write_text("local.py\n")
    _write_source_tree(
        root,
        ["kept.py", "ignored_module.py", "ignored_dir/inner.py", "sub/local.py", "sub/kept2.py"],
    )
    return root


@requires_git
def test_extract_all_skips_gitignored_paths_and_logs_a_hint(tmp_path: Path, caplog) -> None:
    root = _git_work_tree(tmp_path)

    with caplog.at_level("INFO", logger="codedupes.extractor"):
        units = CodeExtractor(root, include_private=True).extract_all()
    assert sorted(unit.name for unit in units) == ["kept2_fn", "kept_fn"]
    assert (
        "Skipped 2 files and 1 directories ignored by git; use --no-gitignore to include them."
    ) in caplog.text
    assert "default test exclusions" not in caplog.text

    everything = CodeExtractor(root, include_private=True, respect_gitignore=False).extract_all()
    assert sorted(unit.name for unit in everything) == [
        "ignored_module_fn",
        "inner_fn",
        "kept2_fn",
        "kept_fn",
        "local_fn",
    ]


@requires_git
def test_gitignored_scan_root_and_named_files_are_analyzed(tmp_path: Path) -> None:
    """Pointing at an ignored directory or file is an explicit request for it."""
    root = _git_work_tree(tmp_path)

    inside = CodeExtractor(root / "ignored_dir", include_private=True).extract_all()
    assert [unit.name for unit in inside] == ["inner_fn"]

    extractor = CodeExtractor(root, include_private=True)
    named = list(extractor.extract_from_file(root / "sub" / "local.py"))
    assert [unit.name for unit in named] == ["local_fn"]


@requires_git
def test_gitignore_prunes_the_c_header_policy_scan_too(tmp_path: Path) -> None:
    """C++ git ignores must not flip ``.h`` handling for files the walk never visits."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".gitignore").write_text("vendor/\n")
    (root / "vendor").mkdir()
    (root / "vendor" / "addon.cpp").write_text("int addon() {\n    return 2;\n}\n")
    (root / "main.c").write_text("int main(void) {\n    return 0;\n}\n")
    (root / "util.h").write_text("static int helper(int v) {\n    return v + 1;\n}\n")

    extractor = CodeExtractor(root, include_private=True)
    assert sorted(unit.qualified_name for unit in extractor.extract_all()) == [
        "main.main",
        "util.helper",
    ]
    assert extractor.diagnostics == []

    ignoring_nothing = CodeExtractor(root, include_private=True, respect_gitignore=False)
    assert [unit.qualified_name for unit in ignoring_nothing.extract_all()] == ["main.main"]
    assert [diagnostic.code for diagnostic in ignoring_nothing.diagnostics] == ["c-header-policy"]


def test_gitignore_files_outside_a_work_tree_are_plain_files(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored_module.py\n")
    _write_source_tree(tmp_path, ["kept.py", "ignored_module.py"])

    units = CodeExtractor(tmp_path, include_private=True).extract_all()
    assert sorted(unit.name for unit in units) == ["ignored_module_fn", "kept_fn"]


@requires_git
def test_reference_only_files_are_the_default_test_exclusions(tmp_path: Path) -> None:
    """Files skipped only by a default test shape still surface for reference parsing."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".gitignore").write_text("tests/generated/\n")
    for relative in [
        "tests/test_impl.py",
        "tests/conftest.py",
        "tests/generated/test_gen.py",
        "pkg/legacy_test.py",
        "pkg/keeper.py",
        "node_modules/test_dep.py",
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def entry():\n    return 1\n")

    def reference_only_names(
        exclude_patterns: list[str] | None, *, implicit_default_excludes: bool = False
    ) -> set[str]:
        extractor = CodeExtractor(
            root,
            exclude_patterns=exclude_patterns,
            include_private=True,
            implicit_default_excludes=implicit_default_excludes,
        )
        extractor.extract_all()
        return {file.relative_to(root).as_posix() for file in extractor.reference_only_files}

    # Default configuration: both default-excluded shapes feed the reference-only
    # list; the gitignored file and the artifact directory do not.
    assert reference_only_names(None) == {
        "tests/test_impl.py",
        "tests/conftest.py",
        "pkg/legacy_test.py",
    }

    # Explicitly supplying the same shapes makes them user exclusions, not
    # implicit defaults. The CLI marks its own prefix when appending a user rule.
    from codedupes.extractor import DEFAULT_EXCLUDE_PATTERNS

    assert reference_only_names(DEFAULT_EXCLUDE_PATTERNS.copy()) == set()
    assert reference_only_names(
        [*DEFAULT_EXCLUDE_PATTERNS, "pkg/legacy_test.py"], implicit_default_excludes=True
    ) == {"tests/test_impl.py", "tests/conftest.py"}

    # Disabling default test exclusions extracts test files directly, so nothing
    # needs the reference-only path.
    assert reference_only_names([]) == set()

    # A real user exclusion for "tests" drops the whole directory from both
    # duplicate detection and unused-code references; the unrelated default
    # shape match under "pkg" is unaffected.
    assert reference_only_names(
        [*DEFAULT_EXCLUDE_PATTERNS, "tests"], implicit_default_excludes=True
    ) == {"pkg/legacy_test.py"}

    # Repeating a built-in shape explicitly is a real user exclusion for the
    # reference walk, whether or not the built-ins are otherwise active.
    assert reference_only_names(
        [*DEFAULT_EXCLUDE_PATTERNS, "**/tests/**"], implicit_default_excludes=True
    ) == {"pkg/legacy_test.py"}
    assert reference_only_names(["**/tests/**"]) == set()
