"""End-to-end embedding-cache tests across real filesystem transitions."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from codedupes import analyzer as analyzer_module
from codedupes import semantic as semantic_module
from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from codedupes.embedding_cache import INDEX_FILENAME, EmbeddingCache
from codedupes.models import AnalysisResult, CodeUnit
from tests.test_embedding_cache import REVISION_1, CountingModel, _patch_get_model


def _write_repo(tmp_path: Path, files: Mapping[str, str]) -> Path:
    """Write a small Python repository and return its root."""
    repo = tmp_path / "repo"
    for relative, source in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source.strip() + "\n", encoding="utf-8")
    return repo


def _analyze(
    repo: Path,
    *,
    embedding_cache: bool = True,
    **overrides: Any,
) -> AnalysisResult:
    """Analyze one test repository with deterministic semantic settings."""
    config_values: dict[str, Any] = {
        "model_name": "test-model",
        "model_revision": REVISION_1,
        "semantic_threshold": 0.0,
        "run_traditional": False,
        "run_semantic": True,
        "run_unused": True,
        "min_semantic_statements": 0,
        "embedding_cache": embedding_cache,
        "progress": "never",
    }
    config_values.update(overrides)
    config = AnalyzerConfig(
        **config_values,
    )
    return CodeAnalyzer(config).analyze(repo)


def _index(repo: Path, *, search_document: str) -> CodeAnalyzer:
    """Build one deterministic semantic search index and return its analyzer."""
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            mode="search",
            model_name="test-model",
            model_revision=REVISION_1,
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            progress="never",
            search_document=search_document,
        )
    )
    analyzer.index(repo)
    return analyzer


def _unit_identity(unit: CodeUnit) -> tuple[str, str, int, int]:
    """Return stable finding fields used for cached/uncached comparison."""
    return (str(unit.file_path), unit.qualified_name, unit.lineno, unit.end_lineno)


def _normalized_findings(result: AnalysisResult) -> dict[str, Any]:
    """Normalize finding order and float precision while omitting telemetry."""

    def raw_pair(duplicate: Any) -> tuple[Any, ...]:
        return (
            _unit_identity(duplicate.unit_a),
            _unit_identity(duplicate.unit_b),
            round(duplicate.similarity, 6),
            duplicate.method,
        )

    def hybrid_pair(duplicate: Any) -> tuple[Any, ...]:
        return (
            _unit_identity(duplicate.unit_a),
            _unit_identity(duplicate.unit_b),
            duplicate.tier,
            round(duplicate.confidence, 6),
        )

    return {
        "units": sorted(_unit_identity(unit) for unit in result.units),
        "traditional": sorted(raw_pair(duplicate) for duplicate in result.traditional_duplicates),
        "semantic": sorted(raw_pair(duplicate) for duplicate in result.semantic_duplicates),
        "hybrid": sorted(hybrid_pair(duplicate) for duplicate in result.hybrid_duplicates),
        "unused": sorted(_unit_identity(unit) for unit in result.potentially_unused),
    }


def _assert_matches_uncached(repo: Path, cached: AnalysisResult) -> None:
    """Assert cache use changes telemetry only, never findings."""
    assert _normalized_findings(cached) == _normalized_findings(
        _analyze(repo, embedding_cache=False)
    )


def test_cold_scan_encodes_every_unique_body(tmp_path, monkeypatch) -> None:
    repo = _write_repo(
        tmp_path,
        {
            "src/alpha.py": "def alpha(value):\n    return value + 1",
            "src/beta.py": "def beta(value):\n    return value + 2",
            "src/gamma.py": "def gamma(value):\n    return value + 3",
        },
    )
    _patch_get_model(monkeypatch, CountingModel())

    result = _analyze(repo)

    assert result.embedding_stats is not None
    assert result.embedding_stats.encoded_inputs == 3
    assert result.embedding_stats.unique_inputs == 3
    assert result.embedding_stats.model_loaded is True
    assert result.embedding_stats.manifest_generation == 1
    _assert_matches_uncached(repo, result)


@pytest.mark.parametrize("edit_bodies", [False, True], ids=["docstring", "body-edits"])
def test_in_file_edits_reencode_only_changed_bodies(tmp_path, monkeypatch, edit_bodies) -> None:
    source = (
        "def alpha(value):\n    return value + 1\n\n"
        "def beta(value):\n    return value + 2\n\n"
        "def gamma(value):\n    return value + 3\n"
    )
    repo = _write_repo(
        tmp_path,
        {"src/math.py": source},
    )
    _patch_get_model(monkeypatch, CountingModel())
    _analyze(repo)
    updated = (
        source.replace("+ 1", "+ 99").replace("+ 2", "+ 100")
        if edit_bodies
        else '"""Module documentation shifts every function."""\n\n' + source
    )
    (repo / "src/math.py").write_text(updated, encoding="utf-8")

    result = _analyze(repo)

    assert result.embedding_stats is not None
    expected_encoded = 2 if edit_bodies else 0
    assert result.embedding_stats.encoded_inputs == expected_encoded
    assert result.embedding_stats.cache_hit_rows == 3 - expected_encoded
    assert result.embedding_stats.moved_units_reused == 0
    assert result.embedding_stats.deleted_units == 0
    assert result.embedding_stats.orphan_rows_retained == expected_encoded
    _assert_matches_uncached(repo, result)


def test_single_file_edit_preserves_old_key_for_orphan_collection(tmp_path, monkeypatch) -> None:
    repo = _write_repo(
        tmp_path,
        {
            "a.py": "def alpha(value):\n    return value + 1",
            "b.py": "def beta(value):\n    return value + 2",
        },
    )
    _patch_get_model(monkeypatch, CountingModel())
    _analyze(repo)
    (repo / "a.py").write_text(
        "def alpha(value):\n    return value + 99\n",
        encoding="utf-8",
    )

    narrow = _analyze(repo / "a.py")

    assert narrow.embedding_stats is not None
    assert narrow.embedding_stats.encoded_inputs == 1
    assert narrow.embedding_stats.deleted_units == 0
    assert narrow.embedding_stats.orphan_rows_retained == 1

    complete = _analyze(repo)
    assert complete.embedding_stats is not None
    assert complete.embedding_stats.encoded_inputs == 0
    assert complete.embedding_stats.orphan_rows_retained == 1

    _analyze(repo)
    collected = _analyze(repo)
    assert collected.embedding_stats is not None
    assert collected.embedding_stats.orphan_rows_collected == 1
    assert EmbeddingCache().stats()["entries"] == 2


def test_rename_file_reuses_embeddings_and_reports_only_new_path(tmp_path, monkeypatch) -> None:
    repo = _write_repo(
        tmp_path,
        {
            "src/old/helpers.py": "def normalize(value):\n    return value.strip().lower()",
            "src/app.py": "def run(value):\n    return value.upper()",
        },
    )
    _patch_get_model(monkeypatch, CountingModel())
    _analyze(repo)
    new_path = repo / "src/text/normalize.py"
    new_path.parent.mkdir()
    (repo / "src/old/helpers.py").rename(new_path)

    result = _analyze(repo)

    assert result.embedding_stats is not None
    assert result.embedding_stats.encoded_inputs == 0
    assert result.embedding_stats.model_loaded is False
    assert result.embedding_stats.moved_units_reused == 1
    assert result.embedding_stats.deleted_units == 0
    reported = {unit.file_path for unit in result.units}
    assert new_path.resolve() in reported
    assert (repo / "src/old/helpers.py").resolve() not in reported
    _assert_matches_uncached(repo, result)


def test_move_file_across_directories_reuses_embeddings(tmp_path, monkeypatch) -> None:
    repo = _write_repo(
        tmp_path,
        {
            "src/first/worker.py": "def work(value):\n    return value * 2",
            "src/app.py": "def run(value):\n    return value + 1",
        },
    )
    _patch_get_model(monkeypatch, CountingModel())
    _analyze(repo)
    destination = repo / "lib/jobs/worker.py"
    destination.parent.mkdir(parents=True)
    (repo / "src/first/worker.py").rename(destination)

    result = _analyze(repo)

    assert result.embedding_stats is not None
    assert result.embedding_stats.encoded_inputs == 0
    assert result.embedding_stats.moved_units_reused == 1
    assert destination.resolve() in {unit.file_path for unit in result.units}
    _assert_matches_uncached(repo, result)


@pytest.mark.parametrize("shared_body", [False, True], ids=["unique-body", "shared-body"])
@pytest.mark.parametrize("excludes", [None, ["*_test.py"]], ids=["default", "excludes"])
def test_delete_file_removes_its_units_and_findings_without_encoding(
    tmp_path, monkeypatch, shared_body, excludes
) -> None:
    kept_source = "def kept(value):\n    return value + 1"
    repo = _write_repo(
        tmp_path,
        {
            "src/remove.py": kept_source
            if shared_body
            else "def removed(value):\n    return value - 1",
            "src/keep.py": kept_source,
        },
    )
    if excludes:
        (repo / "src/ignored_test.py").write_text("def ignored():\n    return 42\n")
    _patch_get_model(monkeypatch, CountingModel())
    initial = _analyze(repo, exclude_patterns=excludes)
    assert len(initial.units) == 2
    deleted = (repo / "src/remove.py").resolve()
    (repo / "src/remove.py").unlink()

    result = _analyze(repo, exclude_patterns=excludes)

    assert result.embedding_stats is not None
    assert result.embedding_stats.encoded_inputs == 0
    assert result.embedding_stats.moved_units_reused == 0
    assert result.embedding_stats.deleted_units == 1
    expected_orphans = int(not shared_body)
    assert result.embedding_stats.orphan_rows_retained == expected_orphans
    repo_stats = EmbeddingCache().stats()["repos"][0]
    assert repo_stats["orphan_rows"] == expected_orphans
    assert repo_stats["last_complete_generation"] == 2
    assert all(unit.file_path != deleted for unit in result.units)
    assert all(
        duplicate.unit_a.file_path != deleted and duplicate.unit_b.file_path != deleted
        for duplicate in result.hybrid_duplicates
    )
    assert all(unit.file_path != deleted for unit in result.potentially_unused)
    assert _normalized_findings(result) == _normalized_findings(
        _analyze(repo, embedding_cache=False, exclude_patterns=excludes)
    )

    for _ in range(2):
        retained = _analyze(repo, exclude_patterns=excludes)
        assert retained.embedding_stats is not None
        assert retained.embedding_stats.orphan_rows_retained == expected_orphans
        assert retained.embedding_stats.orphan_rows_collected == 0
    collected = _analyze(repo, exclude_patterns=excludes)
    assert collected.embedding_stats is not None
    assert collected.embedding_stats.orphan_rows_collected == expected_orphans
    assert collected.embedding_stats.orphan_rows_retained == 0
    collected_stats = EmbeddingCache().stats()
    assert collected_stats["entries"] == 1
    assert collected_stats["repos"][0]["orphan_rows"] == 0


def test_delete_last_unit_publishes_empty_manifest(tmp_path, monkeypatch) -> None:
    repo = _write_repo(
        tmp_path,
        {
            "only.py": "def only(value):\n    return value + 1",
            "api.h": "int value(void);",
        },
    )
    _patch_get_model(monkeypatch, CountingModel())
    _analyze(repo)
    (repo / "only.py").unlink()

    empty = _analyze(repo)

    assert empty.units == []
    # Intentional C-header selection is not an extraction coverage failure.
    assert [diagnostic.code for diagnostic in empty.extraction_diagnostics] == ["c-header-policy"]
    assert empty.embedding_stats is not None
    assert empty.embedding_stats.encoded_inputs == 0
    assert empty.embedding_stats.model_loaded is False
    assert empty.embedding_stats.deleted_units == 1
    assert empty.embedding_stats.orphan_rows_retained == 1
    assert empty.embedding_stats.manifest_generation == 2


def test_last_ineligible_candidate_publishes_empty_manifest(tmp_path, monkeypatch) -> None:
    repo = _write_repo(
        tmp_path,
        {"only.py": "def only(value):\n    adjusted = value + 1\n    return adjusted"},
    )
    _patch_get_model(monkeypatch, CountingModel())
    _analyze(repo, min_semantic_statements=2)
    (repo / "only.py").write_text(
        "def only(value):\n    return value + 1\n",
        encoding="utf-8",
    )

    filtered = _analyze(repo, min_semantic_statements=2)

    assert len(filtered.units) == 1
    assert filtered.embedding_stats is not None
    assert filtered.embedding_stats.encoded_inputs == 0
    assert filtered.embedding_stats.model_loaded is False
    assert filtered.embedding_stats.deleted_units == 1
    assert filtered.embedding_stats.orphan_rows_retained == 1


def test_single_file_scan_orphans_candidate_that_became_ineligible(tmp_path, monkeypatch) -> None:
    repo = _write_repo(
        tmp_path,
        {"only.py": "def only(value):\n    adjusted = value + 1\n    return adjusted"},
    )
    _patch_get_model(monkeypatch, CountingModel())
    _analyze(repo, min_semantic_statements=2)
    (repo / "only.py").write_text(
        "def only(value):\n    return value + 1\n",
        encoding="utf-8",
    )

    filtered = _analyze(repo / "only.py", min_semantic_statements=2)

    assert len(filtered.units) == 1
    assert filtered.embedding_stats is not None
    assert filtered.embedding_stats.requested_rows == 0
    assert filtered.embedding_stats.deleted_units == 1
    assert filtered.embedding_stats.orphan_rows_retained == 1


@pytest.mark.parametrize("operation", ["analyze", "index"])
@pytest.mark.parametrize("outside_root", [False, True])
def test_single_symlink_scan_updates_observed_target(
    tmp_path, monkeypatch, operation, outside_root
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = (tmp_path if outside_root else repo) / "target.py"
    target.write_text(
        "def only(value):\n    adjusted = value + 1\n    return adjusted\n", encoding="utf-8"
    )
    alias = repo / "alias.py"
    alias.symlink_to(target)
    _patch_get_model(monkeypatch, CountingModel())
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            model_name="test-model",
            model_revision=REVISION_1,
            semantic_threshold=0.0,
            device="cpu",
            run_traditional=False,
            run_unused=False,
            min_semantic_statements=2,
            progress="never",
        )
    )
    scan = getattr(analyzer, operation)
    scan(repo)
    target.write_text("def only(value):\n    return value + 1\n", encoding="utf-8")

    scan(alias)

    assert analyzer.embedding_stats is not None
    assert analyzer.embedding_stats.requested_rows == 0
    assert analyzer.embedding_stats.deleted_units == 1
    assert analyzer.embedding_stats.orphan_rows_retained == 1


def test_move_and_edit_reencodes_changed_function_but_hits_sibling(tmp_path, monkeypatch) -> None:
    repo = _write_repo(
        tmp_path,
        {
            "src/old/math.py": (
                "def alpha(value):\n    return value + 1\n\ndef beta(value):\n    return value + 2"
            )
        },
    )
    _patch_get_model(monkeypatch, CountingModel())
    _analyze(repo)
    destination = repo / "src/new/math.py"
    destination.parent.mkdir()
    (repo / "src/old/math.py").rename(destination)
    destination.write_text(
        "def alpha(value):\n    return value + 99\n\ndef beta(value):\n    return value + 2\n",
        encoding="utf-8",
    )

    result = _analyze(repo)

    assert result.embedding_stats is not None
    assert result.embedding_stats.encoded_inputs == 1
    assert result.embedding_stats.cache_hit_rows == 1
    _assert_matches_uncached(repo, result)


def test_identical_bodies_in_two_files_encode_once(tmp_path, monkeypatch) -> None:
    source = "def normalize(value):\n    return value.strip().lower()"
    repo = _write_repo(tmp_path, {"src/a.py": source, "src/b.py": source})
    _patch_get_model(monkeypatch, CountingModel())

    result = _analyze(repo)

    assert result.embedding_stats is not None
    assert result.embedding_stats.encoded_inputs == 1
    assert result.embedding_stats.duplicate_rows_reused == 1
    assert len({unit.file_path for unit in result.units}) == 2
    _assert_matches_uncached(repo, result)


def test_adding_caller_updates_unused_findings_without_reencoding_callee(
    tmp_path, monkeypatch
) -> None:
    repo = _write_repo(
        tmp_path,
        {"src/helpers.py": "def _normalize(value):\n    return value.strip().lower()"},
    )
    _patch_get_model(monkeypatch, CountingModel())
    initial = _analyze(repo)
    assert {unit.name for unit in initial.potentially_unused} == {"_normalize"}
    (repo / "src/app.py").write_text(
        "from helpers import _normalize\n\ndef run(value):\n    return _normalize(value)\n",
        encoding="utf-8",
    )

    result = _analyze(repo)

    assert result.embedding_stats is not None
    assert result.embedding_stats.cache_hit_rows == 1
    assert result.embedding_stats.encoded_inputs == 1
    assert "_normalize" not in {unit.name for unit in result.potentially_unused}
    _assert_matches_uncached(repo, result)


def test_narrow_rerun_does_not_change_full_scan_shard_rows(tmp_path, monkeypatch) -> None:
    repo = _write_repo(
        tmp_path,
        {
            "src/a.py": "def alpha(value):\n    return value + 1",
            "src/b.py": "def beta(value):\n    return value + 2",
        },
    )
    _patch_get_model(monkeypatch, CountingModel())
    _analyze(repo)
    cache = EmbeddingCache()
    full_shard_index = cache.shard_dir(repo, "test-model", REVISION_1) / INDEX_FILENAME
    entries_before = len(json.loads(full_shard_index.read_text(encoding="utf-8"))["keys"])

    result = _analyze(repo / "src/a.py")

    assert result.embedding_stats is not None
    assert result.embedding_stats.encoded_inputs == 1
    assert len(json.loads(full_shard_index.read_text(encoding="utf-8"))["keys"]) == entries_before
    narrow_manifest = cache.load_manifest(repo / "src", "test-model", REVISION_1)
    assert narrow_manifest is not None
    narrow_selection = next(iter(narrow_manifest.selections.values()))
    assert narrow_selection.complete_scan is False
    assert narrow_selection.orphans == {}
    _assert_matches_uncached(repo / "src/a.py", result)


@pytest.mark.parametrize(
    ("files", "overrides"),
    [
        (
            {
                "src/public.py": "def public(value):\n    return value + 1",
                "src/ignored_test.py": "def ignored(value):\n    return value + 2",
            },
            {"exclude_patterns": ["*_test.py"]},
        ),
        (
            {
                "src/public.py": "def public(value):\n    return value + 1",
                "src/private.py": "def _private(value):\n    return value + 2",
            },
            {"include_private": False},
        ),
        (
            {
                "src/python_unit.py": "def python_unit(value):\n    return value + 1",
                "src/javascript_unit.js": (
                    "function javascriptUnit(value) {\n  return value + 2;\n}"
                ),
            },
            {"languages": ("python",)},
        ),
        (
            {
                "src/small.py": "def small(value):\n    return value + 1",
                "src/larger.py": (
                    "def larger(value):\n    adjusted = value + 1\n    return adjusted * 2"
                ),
            },
            {"min_semantic_statements": 2},
        ),
    ],
    ids=["exclude-filter", "private-filter", "language-filter", "minimum-statements-filter"],
)
def test_scope_filter_does_not_age_or_collect_other_selection_rows(
    tmp_path, monkeypatch, files, overrides
) -> None:
    repo = _write_repo(tmp_path, files)
    _patch_get_model(monkeypatch, CountingModel())
    initial = _analyze(repo)
    assert initial.embedding_stats is not None
    initial_count = initial.embedding_stats.requested_rows

    for _ in range(4):
        filtered = _analyze(repo, **overrides)
        assert filtered.embedding_stats is not None
        assert filtered.embedding_stats.deleted_units == 0
        assert filtered.embedding_stats.orphan_rows_retained == 0
        assert filtered.embedding_stats.orphan_rows_collected == 0

    default_again = _analyze(repo)
    assert default_again.embedding_stats is not None
    assert default_again.embedding_stats.requested_rows == initial_count
    assert default_again.embedding_stats.encoded_inputs == 0
    assert default_again.embedding_stats.model_loaded is False


@pytest.mark.parametrize(
    "search_document",
    [None, "source", "contextual"],
    ids=["check", "search-source", "search-contextual"],
)
def test_runtime_variant_switch_preserves_warm_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, search_document: str | None
) -> None:
    """Switching precision must not orphan another variant's unchanged vectors."""
    repo = _write_repo(
        tmp_path,
        {
            "a.py": "def alpha(value):\n    return value + 1",
            "b.py": "def beta(value):\n    return value + 2",
        },
    )
    model = CountingModel()
    model_loads = _patch_get_model(monkeypatch, model)
    dtype_variant = ""
    monkeypatch.setattr(
        semantic_module, "_dtype_variant_for", lambda *_args, **_kwargs: dtype_variant
    )

    def run() -> AnalysisResult | CodeAnalyzer:
        """Run the same corpus selection under the current runtime variant."""
        if search_document is None:
            return _analyze(repo)
        return _index(repo, search_document=search_document)

    initial = run()
    assert initial.embedding_stats is not None
    assert initial.embedding_stats.encoded_inputs == 2

    dtype_variant = "dtype=torch.bfloat16"
    for _ in range(4):
        changed = run()
        assert changed.embedding_stats is not None
        assert changed.embedding_stats.orphan_rows_retained == 0
        assert changed.embedding_stats.orphan_rows_collected == 0

    dtype_variant = ""
    restored = run()
    assert restored.embedding_stats is not None
    assert restored.embedding_stats.cache_hit_rows == 2
    assert restored.embedding_stats.encoded_inputs == 0
    assert restored.embedding_stats.model_loaded is False
    assert model_loads["count"] == 2
    assert len(model.encode_calls) == 2


def test_incomplete_scans_do_not_refresh_unseen_sibling_pins(tmp_path, monkeypatch) -> None:
    repo = _write_repo(
        tmp_path,
        {
            "a.py": "def alpha(value):\n    return value + 1",
            "b.py": "def beta(value):\n    return value + 2",
        },
    )
    _patch_get_model(monkeypatch, CountingModel())
    _analyze(repo)
    _analyze(repo, languages=("python",))
    (repo / "b.py").unlink()
    deleted = _analyze(repo, languages=("python",))
    assert deleted.embedding_stats is not None
    assert deleted.embedding_stats.orphan_rows_retained == 1

    collected = None
    for _ in range(3):
        narrow = _analyze(repo / "a.py")
        assert narrow.embedding_stats is not None
        assert narrow.embedding_stats.orphan_rows_collected == 0
        collected = _analyze(repo, languages=("python",))

    assert collected is not None
    assert collected.embedding_stats is not None
    assert collected.embedding_stats.orphan_rows_retained == 0
    assert collected.embedding_stats.orphan_rows_collected == 1
    assert EmbeddingCache().stats()["entries"] == 1


def test_stale_scope_selection_stops_pinning_deleted_rows(tmp_path, monkeypatch) -> None:
    repo = _write_repo(
        tmp_path,
        {
            "src/a.py": "def alpha(value):\n    return value + 1",
            "src/b.py": "def beta(value):\n    return value + 2",
            "src/private.py": "def _private(value):\n    return value + 3",
        },
    )
    _patch_get_model(monkeypatch, CountingModel())
    _analyze(repo)
    _analyze(repo, include_private=False)
    (repo / "src/b.py").unlink()

    deleted = _analyze(repo)

    assert deleted.embedding_stats is not None
    assert deleted.embedding_stats.deleted_units == 1
    assert deleted.embedding_stats.orphan_rows_retained == 1
    assert deleted.embedding_stats.orphan_rows_collected == 0
    assert EmbeddingCache().stats()["repos"][0]["orphan_rows"] == 1

    retained = _analyze(repo)
    expired_pin = _analyze(repo)
    collected = _analyze(repo)

    assert retained.embedding_stats is not None
    assert retained.embedding_stats.orphan_rows_retained == 1
    assert retained.embedding_stats.orphan_rows_collected == 0
    assert expired_pin.embedding_stats is not None
    assert expired_pin.embedding_stats.orphan_rows_retained == 1
    assert expired_pin.embedding_stats.orphan_rows_collected == 0
    assert collected.embedding_stats is not None
    assert collected.embedding_stats.orphan_rows_retained == 0
    assert collected.embedding_stats.orphan_rows_collected == 1
    stats = EmbeddingCache().stats()
    assert stats["entries"] == 2
    assert stats["repos"][0]["orphan_rows"] == 0


def test_expired_selection_retains_deletion_baseline(tmp_path, monkeypatch) -> None:
    repo = _write_repo(
        tmp_path,
        {
            "src/a.py": "def alpha(value):\n    return value + 1",
            "src/b.py": "def beta(value):\n    return value + 2",
        },
    )
    _patch_get_model(monkeypatch, CountingModel())
    _analyze(repo)
    for _ in range(3):
        _index(repo, search_document="source")
    (repo / "src/b.py").unlink()

    deleted = _analyze(repo)

    assert deleted.embedding_stats is not None
    assert deleted.embedding_stats.deleted_units == 1
    assert deleted.embedding_stats.orphan_rows_retained == 1
    assert deleted.embedding_stats.orphan_rows_collected == 0

    _analyze(repo)
    _analyze(repo)
    collected = _analyze(repo)

    assert collected.embedding_stats is not None
    assert collected.embedding_stats.orphan_rows_retained == 0
    assert collected.embedding_stats.orphan_rows_collected == 1
    stats = EmbeddingCache().stats()
    assert stats["entries"] == 1
    assert stats["repos"][0]["orphan_rows"] == 0


def test_orphan_aging_survives_search_selection_between_checks(tmp_path, monkeypatch) -> None:
    repo = _write_repo(
        tmp_path,
        {
            "src/a.py": "def alpha(value):\n    return value + 1",
            "src/b.py": "def beta(value):\n    return value + 2",
        },
    )
    _patch_get_model(monkeypatch, CountingModel())
    _analyze(repo)
    (repo / "src/b.py").unlink()
    orphaned = _analyze(repo)
    assert orphaned.embedding_stats is not None
    assert orphaned.embedding_stats.manifest_generation == 2
    assert orphaned.embedding_stats.orphan_rows_retained == 1

    _index(repo, search_document="source")
    resumed = _analyze(repo)

    assert resumed.embedding_stats is not None
    assert resumed.embedding_stats.manifest_generation == 4
    assert resumed.embedding_stats.orphan_rows_retained == 1
    manifest = EmbeddingCache().load_manifest(repo, "test-model", REVISION_1)
    assert manifest is not None
    assert len(manifest.selections) == 2

    collected = _analyze(repo)
    assert collected.embedding_stats is not None
    assert collected.embedding_stats.manifest_generation == 5
    assert collected.embedding_stats.orphan_rows_collected == 1


def test_single_file_scan_preserves_shared_complete_baseline(tmp_path, monkeypatch) -> None:
    repo = _write_repo(
        tmp_path,
        {
            "a.py": "def alpha(value):\n    return value + 1",
            "b.py": "def beta(value):\n    return value + 2",
        },
    )
    _patch_get_model(monkeypatch, CountingModel())
    _analyze(repo)
    cache = EmbeddingCache()

    narrow = _analyze(repo / "a.py")

    assert narrow.embedding_stats is not None
    assert narrow.embedding_stats.encoded_inputs == 0
    manifest = cache.load_manifest(repo, "test-model", REVISION_1)
    assert manifest is not None
    selection = next(iter(manifest.selections.values()))
    assert selection.complete_scan is False
    assert len(selection.units) == 2

    (repo / "b.py").unlink()
    rescanned = _analyze(repo)

    assert rescanned.embedding_stats is not None
    assert rescanned.embedding_stats.deleted_units == 1
    assert rescanned.embedding_stats.orphan_rows_retained == 1


def test_failed_analysis_keeps_previous_manifest_authoritative(tmp_path, monkeypatch) -> None:
    repo = _write_repo(
        tmp_path,
        {"src/value.py": "def value(number):\n    return number + 1"},
    )
    _patch_get_model(monkeypatch, CountingModel())
    initial = _analyze(repo)
    assert initial.embedding_stats is not None
    cache = EmbeddingCache()
    previous = cache.load_manifest(repo, "test-model", REVISION_1)
    assert previous is not None
    previous_generation = previous.generation
    (repo / "src/value.py").write_text(
        "def value(number):\n    return number + 99\n",
        encoding="utf-8",
    )

    def fail_unused(*_args, **_kwargs):
        raise RuntimeError("unused failed")

    with monkeypatch.context() as crash_patch:
        crash_patch.setattr(analyzer_module, "find_potentially_unused", fail_unused)
        with pytest.raises(RuntimeError, match="unused failed"):
            _analyze(repo)

    after_failure = cache.load_manifest(repo, "test-model", REVISION_1)
    assert after_failure == previous

    recovered = _analyze(repo)
    assert recovered.embedding_stats is not None
    assert recovered.embedding_stats.encoded_inputs == 0
    assert recovered.embedding_stats.orphan_rows_retained == 1
    assert recovered.embedding_stats.manifest_generation == previous_generation + 1


@pytest.mark.parametrize("operation", ["analyze", "index"])
@pytest.mark.parametrize(
    ("failure", "single_file"),
    [
        ("parse-error", False),
        ("read-error", False),
        ("walk-error", False),
        ("parse-error", True),
        ("read-error", True),
    ],
)
def test_incomplete_extraction_keeps_previous_manifest_authoritative(
    tmp_path, monkeypatch, operation, failure, single_file
) -> None:
    affected_name = "nested/b.py" if failure == "walk-error" else "b.py"
    repo = _write_repo(
        tmp_path,
        {
            "a.py": "def alpha(value):\n    return value + 1",
            affected_name: "def beta(value):\n    return value + 2",
        },
    )
    affected = repo / affected_name
    original_source = affected.read_bytes()
    _patch_get_model(monkeypatch, CountingModel())

    def scan(target):
        analyzer = CodeAnalyzer(
            AnalyzerConfig(
                mode="check" if operation == "analyze" else "search",
                model_name="test-model",
                model_revision=REVISION_1,
                semantic_threshold=0.0,
                run_traditional=False,
                run_unused=False,
                min_semantic_statements=0,
                progress="never",
            )
        )
        getattr(analyzer, operation)(target)
        return analyzer

    scan(repo)
    cache = EmbeddingCache()
    previous = cache.load_manifest(repo, "test-model", REVISION_1)
    assert previous is not None
    assert previous.generation == 1
    if not single_file:
        # Valid work performed during an incomplete scan must still be reusable.
        (repo / "a.py").write_text("def alpha(value):\n    return value + 99\n")

    with monkeypatch.context() as failed_scan:
        if failure == "parse-error":
            affected.write_bytes(original_source + b"\ndef unfinished(\n")
        elif failure == "read-error":
            read_bytes = Path.read_bytes

            def unreadable(path):
                if path == affected:
                    raise PermissionError(13, "Permission denied", str(path))
                return read_bytes(path)

            failed_scan.setattr(Path, "read_bytes", unreadable)
        else:
            scandir = os.scandir

            def inaccessible(path):
                if Path(path) == affected.parent:
                    raise PermissionError(13, "Permission denied", str(path))
                return scandir(path)

            failed_scan.setattr(os, "scandir", inaccessible)

        # More scans than the orphan grace period must not advance its clock.
        for iteration in range(5):
            incomplete = scan(affected if single_file else repo)
            diagnostics = incomplete._extraction_diagnostics
            assert [diagnostic.code for diagnostic in diagnostics] == [failure]
            assert diagnostics[0].file_path == (
                affected.parent if failure == "walk-error" else affected
            )
            assert {unit.name for unit in incomplete._units} == (
                set() if single_file else {"alpha"}
            )
            stats = incomplete.embedding_stats
            assert stats is not None
            assert stats.encoded_inputs == int(not single_file and iteration == 0)
            assert stats.deleted_units == 0
            assert stats.orphan_rows_collected == 0
            assert stats.orphan_rows_retained == 0
            assert stats.manifest_generation is None
            assert cache.load_manifest(repo, "test-model", REVISION_1) == previous
            assert cache.stats()["entries"] == (2 if single_file else 3)

    affected.write_bytes(original_source)
    recovered = scan(repo)
    stats = recovered.embedding_stats
    assert stats is not None
    assert stats.encoded_inputs == 0
    assert stats.cache_hit_rows == 2
    assert stats.model_loaded is False
    assert stats.deleted_units == 0
    assert stats.orphan_rows_collected == 0
    assert stats.manifest_generation == previous.generation + 1


@pytest.mark.parametrize(
    ("invalid_source", "expected_codes"),
    [
        (b"function beta(value) { return value + 2; }\n// \xff\n", {"invalid-utf8"}),
        (b"function beta(value) { return value + 2; }\nfunction broken(\n", {"partial-parse"}),
        (b"function beta(value) { return value + ; }\n", {"partial-parse", "unit-parse-error"}),
    ],
    ids=["invalid-utf8", "partial-parse", "unit-parse-error"],
)
def test_tree_sitter_coverage_failures_preserve_manifest(
    tmp_path, monkeypatch, invalid_source, expected_codes
) -> None:
    repo = _write_repo(tmp_path, {"b.js": "function beta(value) { return value + 2; }"})
    source_file = repo / "b.js"
    original_source = source_file.read_bytes()
    _patch_get_model(monkeypatch, CountingModel())
    _analyze(repo, run_unused=False)
    cache = EmbeddingCache()
    previous = cache.load_manifest(repo, "test-model", REVISION_1)
    assert previous is not None

    source_file.write_bytes(invalid_source)
    for _ in range(5):
        incomplete = _analyze(repo, run_unused=False)
        assert {
            diagnostic.code for diagnostic in incomplete.extraction_diagnostics
        } == expected_codes
        assert cache.load_manifest(repo, "test-model", REVISION_1) == previous

    source_file.write_bytes(original_source)
    recovered = _analyze(repo, run_unused=False)
    assert recovered.embedding_stats is not None
    assert recovered.embedding_stats.encoded_inputs == 0
    assert recovered.embedding_stats.model_loaded is False
    assert recovered.embedding_stats.manifest_generation == previous.generation + 1


def test_contextual_search_embeds_envelope_while_analysis_embeds_source(
    tmp_path, monkeypatch
) -> None:
    repo = _write_repo(
        tmp_path,
        {"billing/refunds.py": "def validate(value):\n    return value > 0"},
    )
    model = CountingModel()
    _patch_get_model(monkeypatch, model)

    contextual = _index(repo, search_document="contextual")
    contextual_text = model.encode_calls[-1][0]
    assert contextual.embedding_stats is not None
    assert contextual_text.startswith("language: python\n")
    assert "path: billing/refunds.py\n" in contextual_text
    assert "symbol: billing.refunds.validate\n" in contextual_text
    assert "code:\ndef validate(value):" in contextual_text

    config = AnalyzerConfig(
        model_name="test-model",
        model_revision=REVISION_1,
        semantic_threshold=0.0,
        run_traditional=False,
        run_semantic=True,
        run_unused=False,
        min_semantic_statements=0,
        embedding_cache=False,
        progress="never",
        search_document="contextual",
    )
    CodeAnalyzer(config).analyze(repo)
    assert model.encode_calls[-1] == ["def validate(value):\n    return value > 0"]


def test_contextual_rename_reembeds_search_but_source_check_stays_warm(
    tmp_path, monkeypatch
) -> None:
    repo = _write_repo(
        tmp_path,
        {"src/old.py": "def normalize(value):\n    return value.strip().lower()"},
    )
    _patch_get_model(monkeypatch, CountingModel())
    _analyze(repo)
    _index(repo, search_document="contextual")
    destination = repo / "src/text/normalize.py"
    destination.parent.mkdir(parents=True)
    (repo / "src/old.py").rename(destination)

    contextual = _index(repo, search_document="contextual")
    source_check = _analyze(repo)

    assert contextual.embedding_stats is not None
    assert contextual.embedding_stats.encoded_inputs == 1
    assert source_check.embedding_stats is not None
    assert source_check.embedding_stats.encoded_inputs == 0
    assert source_check.embedding_stats.model_loaded is False
