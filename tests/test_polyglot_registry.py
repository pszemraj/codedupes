"""Language selection and grammar-package contract tests."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

import pytest

from codedupes.constants import DEFAULT_EXCLUDE_DIR_NAMES, is_default_excluded_dir
from codedupes.extractor import CodeExtractor
from codedupes.languages import registry
from codedupes.languages.registry import (
    GRAMMAR_PACKAGES,
    TREE_SITTER_PACKAGE,
    get_grammar_statuses,
    language_for_path,
    normalize_languages,
    repository_allows_c_headers,
)


def test_normalize_languages_accepts_aliases_and_deduplicates() -> None:
    assert normalize_languages(("py", "Rust", "rs", "tsx", "ts")) == (
        "python",
        "rust",
        "typescript",
    )
    assert normalize_languages(()) is None
    assert normalize_languages(None) is None


def test_normalize_languages_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unsupported language selection: java"):
        normalize_languages(("java",))


@pytest.mark.parametrize(
    ("filename", "language", "dialect"),
    [
        ("sample.py", "python", "python"),
        ("sample.c", "c", "c"),
        ("sample.rs", "rust", "rust"),
        ("sample.js", "javascript", "javascript"),
        ("sample.jsx", "javascript", "jsx"),
        ("sample.mjs", "javascript", "javascript"),
        ("sample.cjs", "javascript", "javascript"),
        ("sample.ts", "typescript", "typescript"),
        ("sample.mts", "typescript", "typescript"),
        ("sample.cts", "typescript", "typescript"),
        ("sample.tsx", "typescript", "tsx"),
    ],
)
def test_language_for_path_selects_expected_dialect(
    filename: str,
    language: str,
    dialect: str,
) -> None:
    selection = language_for_path(
        Path(filename),
        include_stubs=True,
        selected_languages=None,
        allow_c_header=False,
    )

    assert selection is not None
    assert (selection.language, selection.dialect) == (language, dialect)


@pytest.mark.parametrize("filename", ["types.d.ts", "types.d.mts", "types.d.cts"])
def test_typescript_declaration_files_are_not_executable_units(filename: str) -> None:
    assert (
        language_for_path(
            Path(filename),
            include_stubs=True,
            selected_languages=None,
            allow_c_header=False,
        )
        is None
    )


def test_python_stubs_require_explicit_inclusion() -> None:
    assert (
        language_for_path(
            Path("sample.pyi"),
            include_stubs=False,
            selected_languages=None,
            allow_c_header=False,
        )
        is None
    )
    selection = language_for_path(
        Path("sample.pyi"),
        include_stubs=True,
        selected_languages=None,
        allow_c_header=False,
    )
    assert selection is not None and selection.language == "python"


def test_language_filter_is_applied_after_extension_detection() -> None:
    assert (
        language_for_path(
            Path("sample.rs"),
            include_stubs=False,
            selected_languages=("python",),
            allow_c_header=False,
        )
        is None
    )


@pytest.mark.parametrize("filename", ["sample.C", "sample.H"])
def test_case_sensitive_cpp_suffixes_are_never_selected_as_c(filename: str) -> None:
    assert (
        language_for_path(
            Path(filename),
            include_stubs=False,
            selected_languages=("c",),
            allow_c_header=True,
        )
        is None
    )


def test_c_header_auto_detection_requires_c_without_cpp(tmp_path: Path) -> None:
    (tmp_path / "module.c").write_text("int run(void) { return 1; }\n")
    (tmp_path / "module.h").write_text("int run(void);\n")
    assert repository_allows_c_headers(tmp_path, None)

    (tmp_path / "other.cpp").write_text("int other() { return 2; }\n")
    assert not repository_allows_c_headers(tmp_path, None)


@pytest.mark.parametrize("cpp_filename", ["other.C", "other.H"])
def test_case_sensitive_cpp_suffixes_disable_c_header_detection(
    tmp_path: Path, cpp_filename: str
) -> None:
    (tmp_path / "module.c").write_text("int run(void) { return 1; }\n")
    (tmp_path / cpp_filename).write_text("int other() { return 2; }\n")

    assert not repository_allows_c_headers(tmp_path, None)


def test_c_header_detection_ignores_dependency_trees(tmp_path: Path) -> None:
    (tmp_path / "module.c").write_text("int run(void) { return 1; }\n")
    dependency = tmp_path / "node_modules" / "native"
    dependency.mkdir(parents=True)
    (dependency / "addon.cpp").write_text("int addon() { return 2; }\n")

    assert repository_allows_c_headers(tmp_path, None)


def test_c_header_detection_walks_vendor_because_extraction_does(tmp_path: Path) -> None:
    """``vendor`` is not a default exclusion, so its C++ must disable ``.h`` parsing."""
    (tmp_path / "module.c").write_text("int run(void) { return 1; }\n")
    vendored = tmp_path / "vendor" / "lib"
    vendored.mkdir(parents=True)
    (vendored / "addon.cpp").write_text("int addon() { return 2; }\n")

    assert not repository_allows_c_headers(tmp_path, None)


def test_c_header_detection_skips_egg_info_because_extraction_does(tmp_path: Path) -> None:
    """Extraction never reads ``*.egg-info``, so its C++ must not disable ``.h`` parsing."""
    (tmp_path / "module.c").write_text("int run(void) { return 1; }\n")
    packaged = tmp_path / "something.egg-info" / "native"
    packaged.mkdir(parents=True)
    (packaged / "addon.cpp").write_text("int addon() { return 2; }\n")

    assert repository_allows_c_headers(tmp_path, None)


def test_header_scan_and_extraction_prune_identical_directories() -> None:
    """A divergent second list lets one vendored file flip ``.h`` handling repo-wide."""
    for name in (*DEFAULT_EXCLUDE_DIR_NAMES, "foo.egg-info"):
        assert is_default_excluded_dir(name)
        assert CodeExtractor._is_excluded_dir_name(name)

    assert not is_default_excluded_dir("vendor")
    assert not CodeExtractor._is_excluded_dir_name("vendor")


@pytest.mark.parametrize("directory", ["venv", ".nox", ".eggs", ".mypy_cache", ".next", ".gradle"])
def test_c_header_detection_prunes_excluded_directories(tmp_path: Path, directory: str) -> None:
    (tmp_path / "module.c").write_text("int run(void) { return 1; }\n")
    vendored = tmp_path / directory / "native"
    vendored.mkdir(parents=True)
    (vendored / "addon.cpp").write_text("int addon() { return 2; }\n")

    assert repository_allows_c_headers(tmp_path, None)


def test_explicit_c_selection_resolves_header_ambiguity(tmp_path: Path) -> None:
    (tmp_path / "only.h").write_text("static inline int run(void) { return 1; }\n")

    assert repository_allows_c_headers(tmp_path, ("c",))
    assert not repository_allows_c_headers(tmp_path, ("rust",))


def test_grammar_status_requires_exact_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    installed = {
        TREE_SITTER_PACKAGE[0]: TREE_SITTER_PACKAGE[1],
        **{package: version for package, version in GRAMMAR_PACKAGES.values()},
    }

    monkeypatch.setattr(metadata, "version", installed.__getitem__)
    monkeypatch.setattr(registry, "_probe_dialect", lambda dialect: None)
    statuses = get_grammar_statuses()
    assert statuses
    assert all(status.available and status.error is None for status in statuses)

    installed["tree-sitter-rust"] = "99.0.0"
    statuses = get_grammar_statuses()
    rust = next(status for status in statuses if status.dialect == "rust")
    assert not rust.available
    assert "tree-sitter-rust==0.24.2 is required" in (rust.error or "")


def test_grammar_status_reports_wheels_that_fail_parser_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A right-version wheel that cannot build a parser must not report ready."""
    installed = {
        TREE_SITTER_PACKAGE[0]: TREE_SITTER_PACKAGE[1],
        **{package: version for package, version in GRAMMAR_PACKAGES.values()},
    }
    monkeypatch.setattr(metadata, "version", installed.__getitem__)
    monkeypatch.setattr(
        registry,
        "_probe_dialect",
        lambda dialect: "parser construction failed: broken wheel" if dialect == "c" else None,
    )

    statuses = {status.dialect: status for status in get_grammar_statuses()}

    assert not statuses["c"].available
    assert "broken wheel" in (statuses["c"].error or "")
    assert statuses["rust"].available
