from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent

from codedupes.extractor import CodeExtractor, compute_ast_hash, compute_token_hash
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


def test_compute_ast_hash_normalizes_variable_names() -> None:
    first = ast.parse("def add(a, b):\n    return a + b").body[0]
    second = ast.parse("def total(x, y):\n    return x + y").body[0]

    assert compute_ast_hash(first) == compute_ast_hash(second)


def test_compute_token_hash_ignores_formatting() -> None:
    assert compute_token_hash("def f(x):\n    return x + 1") == compute_token_hash(
        "def f( x ):\n\treturn x+1"
    )


def test_parse_error_is_skipped(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    root.joinpath("__init__.py").write_text("")
    bad = root / "bad.py"
    bad.write_text("def broken(:\n    pass\n")
    extractor = CodeExtractor(root, include_private=False)

    assert list(extractor.extract_from_file(bad)) == []


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


def test_header_only_tree_reports_c_header_policy_diagnostic(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    root.mkdir()
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


def test_explicit_unsupported_file_reports_diagnostic(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    script = root / "mytool"
    script.write_text("#!/usr/bin/env python\ndef alpha():\n    return 1\n")

    extractor = CodeExtractor(root, include_private=True)
    units = list(extractor.extract_from_file(script))

    assert units == []
    assert [diagnostic.code for diagnostic in extractor.diagnostics] == ["unsupported-file"]


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


def test_explicit_declaration_file_reports_diagnostic(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    decl = root / "types.d.ts"
    decl.write_text("export declare function alpha(v: number): number;\n")

    extractor = CodeExtractor(root, include_private=True)
    units = list(extractor.extract_from_file(decl))

    assert units == []
    assert [diagnostic.code for diagnostic in extractor.diagnostics] == ["declaration-file"]


def test_get_module_name_handles_stub_suffix(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("")
    stub = package / "typed_mod.pyi"
    stub.write_text("def entry() -> int: ...\n")

    extractor = CodeExtractor(package, include_private=True, include_stubs=True)
    units = list(extractor.extract_from_file(stub))

    assert len(units) == 1
    assert units[0].qualified_name == "typed_mod.entry"


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


def test_extract_all_skips_suffix_test_files_by_default(tmp_path: Path) -> None:
    source = "def entry():\n    return 1\n"
    (tmp_path / "inject_test.py").write_text(source)
    (tmp_path / "inject_tests.py").write_text(source)
    (tmp_path / "keeper.py").write_text(source)

    units = CodeExtractor(tmp_path, include_private=True).extract_all()
    assert [unit.file_path.name for unit in units] == ["keeper.py"]


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
