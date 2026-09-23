"""Explicit file targets: stubs, symlinks, and exclusions through the analyzer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codedupes import analyzer as analyzer_module
from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from codedupes.extractor import CodeExtractor
from tests.analyzer_helpers import embedding_identity_from_kwargs


def test_analyze_explicit_stub_target_ignores_include_stubs_default(tmp_path: Path) -> None:
    stub = tmp_path / "typed_mod.pyi"
    stub.write_text("def entry() -> int: ...\n")

    config = AnalyzerConfig(run_semantic=False, run_unused=False)
    result = CodeAnalyzer(config).analyze(stub)

    assert [unit.qualified_name for unit in result.units] == ["typed_mod.entry"]
    assert result.run.include_stubs is True


def test_explicit_stub_symlink_target_ignores_include_stubs_default(
    tmp_path: Path, monkeypatch
) -> None:
    stub = tmp_path / "typed_mod.pyi"
    stub.write_text("def entry() -> int: ...\n", encoding="utf-8")
    alias = tmp_path / "typed_mod.py"
    alias.symlink_to(stub)

    check_result = CodeAnalyzer(AnalyzerConfig(run_semantic=False, run_unused=False)).analyze(alias)
    assert [unit.qualified_name for unit in check_result.units] == ["typed_mod.entry"]
    assert check_result.run.include_stubs is True

    def fake_compute_embeddings(units, **kwargs):
        return (
            np.zeros((len(units), 2), dtype=np.float32),
            embedding_identity_from_kwargs(kwargs),
        )

    monkeypatch.setattr(analyzer_module, "compute_embeddings", fake_compute_embeddings)
    search_analyzer = CodeAnalyzer(
        AnalyzerConfig(
            mode="search",
            run_traditional=False,
            run_unused=False,
            min_semantic_statements=0,
        )
    )
    assert search_analyzer.index(alias) == 1


@pytest.mark.grammar
def test_explicit_c_header_probe_honors_default_test_exclusions(tmp_path: Path) -> None:
    (tmp_path / "main.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")
    header = tmp_path / "test_util.h"
    header.write_text(
        "static inline int helper(int value) { return value + 1; }\n",
        encoding="utf-8",
    )
    (tmp_path / "test_foreign.cpp").write_text(
        "int ignored(void) { return 2; }\n",
        encoding="utf-8",
    )

    result = CodeAnalyzer(AnalyzerConfig(run_semantic=False, run_unused=False)).analyze(header)

    assert [unit.qualified_name for unit in result.units] == ["test_util.helper"]
    assert result.extraction_diagnostics == []


def test_file_target_reads_references_from_the_project_tree(tmp_path: Path) -> None:
    """A single-file target's unused analysis still sees the whole project's references."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "a.py").write_text("def helper():\n    return 1\n")
    (pkg / "b.py").write_text("from pkg.a import helper\n\n\ndef caller():\n    return helper()\n")
    config = AnalyzerConfig(
        run_traditional=False, run_semantic=False, run_unused=True, strict_unused=True
    )

    # Without a pyproject.toml and outside a git work tree, root falls back to
    # the target's own directory, which still contains the sibling that
    # references it.
    fallback_result = CodeAnalyzer(config).analyze(pkg / "a.py")
    assert "helper" not in {unit.name for unit in fallback_result.potentially_unused}

    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n')

    project_result = CodeAnalyzer(config).analyze(pkg / "a.py")
    assert [unit.qualified_name for unit in project_result.units] == ["a.helper"]
    assert "helper" not in {unit.name for unit in project_result.potentially_unused}


def test_file_target_anchors_reference_exclusions_to_its_parent(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n')
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    target = pkg / "target.py"
    target.write_text("def _helper():\n    return 1\n")
    (pkg / "excluded.py").write_text("from pkg.target import _helper\n_helper()\n")

    config = {"run_traditional": False, "run_semantic": False, "run_unused": True}
    with_reference = CodeAnalyzer(AnalyzerConfig(exclude_patterns=[], **config)).analyze(target)
    excluded = CodeAnalyzer(AnalyzerConfig(exclude_patterns=["./excluded.py"], **config)).analyze(
        target
    )

    assert with_reference.potentially_unused == []
    assert with_reference.run.unused.files == 2
    assert [unit.name for unit in excluded.potentially_unused] == ["_helper"]
    assert excluded.run.unused.files == 1


def test_file_target_reports_incomplete_reference_walk(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n')
    source = tmp_path / "entry.py"
    source.write_text("def _helper():\n    return 1\n")

    def failed_reference_walk(extractor: CodeExtractor, directory: Path) -> list[Path]:
        extractor._report_walk_error(
            PermissionError(13, "Permission denied", str(directory / "blocked"))
        )
        return []

    monkeypatch.setattr(CodeExtractor, "_collect_reference_files", failed_reference_walk)

    result = CodeAnalyzer(AnalyzerConfig(run_semantic=False, run_traditional=False)).analyze(source)

    assert [unit.name for unit in result.potentially_unused] == ["_helper"]
    assert [diagnostic.code for diagnostic in result.extraction_diagnostics] == ["walk-error"]
    assert result.analysis_status == "partial"


@pytest.mark.parametrize("target_is_file", [False, True])
def test_no_unused_skips_reference_file_discovery(
    tmp_path: Path, monkeypatch, target_is_file: bool
) -> None:
    """Duplicate-only scans avoid traversing files used solely for unused references."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n')
    source = tmp_path / "entry.py"
    source.write_text("def entry():\n    return 1\n")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_entry.py").write_text("def test_entry():\n    assert True\n")

    def unexpected_reference_walk(*_args, **_kwargs):
        pytest.fail("unused reference files were walked with run_unused=False")

    monkeypatch.setattr(CodeExtractor, "reference_files", unexpected_reference_walk)
    monkeypatch.setattr(CodeExtractor, "_collect_reference_files", unexpected_reference_walk)
    target = source if target_is_file else tmp_path

    result = CodeAnalyzer(AnalyzerConfig(run_semantic=False, run_unused=False)).analyze(target)

    assert [unit.name for unit in result.units] == ["entry"]
    assert result.run.unused is None


def test_explicit_test_file_bypasses_defaults_but_honors_configured_excludes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "test_entry.py"
    source.write_text("def entry():\n    return 1\n", encoding="utf-8")

    config = AnalyzerConfig(run_semantic=False, run_unused=False)
    result = CodeAnalyzer(config).analyze(source)

    assert [unit.qualified_name for unit in result.units] == ["test_entry.entry"]

    excluded = CodeAnalyzer(
        AnalyzerConfig(
            exclude_patterns=[source.name],
            run_semantic=False,
            run_unused=False,
        )
    ).analyze(source)

    assert excluded.units == []


def test_index_explicit_test_file_bypasses_defaults_but_honors_configured_excludes(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "test_entry.py"
    source.write_text("def entry():\n    return 1\n", encoding="utf-8")

    def fake_compute_embeddings(units, **kwargs):
        return (
            np.zeros((len(units), 2), dtype=np.float32),
            embedding_identity_from_kwargs(kwargs),
        )

    monkeypatch.setattr(analyzer_module, "compute_embeddings", fake_compute_embeddings)

    config = AnalyzerConfig(
        mode="search",
        run_traditional=False,
        run_unused=False,
        min_semantic_statements=0,
    )
    assert CodeAnalyzer(config).index(source) == 1

    excluded = AnalyzerConfig(
        mode="search",
        exclude_patterns=[source.name],
        run_traditional=False,
        run_unused=False,
        min_semantic_statements=0,
    )
    assert CodeAnalyzer(excluded).index(source) == 0


def test_run_record_file_target_has_no_default_excludes(tmp_path: Path) -> None:
    """A file target's run record shows no effective excludes; a directory target does."""
    source = tmp_path / "entry.py"
    source.write_text("def entry():\n    return 1\n", encoding="utf-8")

    file_result = CodeAnalyzer(AnalyzerConfig(run_semantic=False, run_unused=False)).analyze(source)
    assert file_result.run.exclude_patterns == ()
    assert file_result.run.include_stubs is True
    assert file_result.run.target.name == "entry.py"

    directory_result = CodeAnalyzer(AnalyzerConfig(run_semantic=False, run_unused=False)).analyze(
        tmp_path
    )
    assert directory_result.run.exclude_patterns != ()


def test_analyze_directory_still_gates_stubs_on_include_stubs(tmp_path: Path) -> None:
    (tmp_path / "typed_mod.pyi").write_text("def entry() -> int: ...\n")
    (tmp_path / "real_mod.py").write_text("def keep():\n    return 1\n")

    config = AnalyzerConfig(run_semantic=False, run_unused=False)
    result = CodeAnalyzer(config).analyze(tmp_path)

    assert [unit.qualified_name for unit in result.units] == ["real_mod.keep"]
