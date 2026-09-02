"""End-to-end embedding-cache tests across real filesystem transitions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from codedupes import analyzer as analyzer_module
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


def test_edit_one_body_reencodes_only_that_body(tmp_path, monkeypatch) -> None:
    repo = _write_repo(
        tmp_path,
        {
            "src/math.py": (
                "def alpha(value):\n    return value + 1\n\ndef beta(value):\n    return value + 2"
            )
        },
    )
    _patch_get_model(monkeypatch, CountingModel())
    _analyze(repo)
    (repo / "src/math.py").write_text(
        "def alpha(value):\n    return value + 99\n\ndef beta(value):\n    return value + 2\n",
        encoding="utf-8",
    )

    result = _analyze(repo)

    assert result.embedding_stats is not None
    assert result.embedding_stats.encoded_inputs == 1
    assert result.embedding_stats.cache_hit_rows == 1
    _assert_matches_uncached(repo, result)


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


def test_delete_file_removes_its_units_and_findings_without_encoding(tmp_path, monkeypatch) -> None:
    repo = _write_repo(
        tmp_path,
        {
            "src/remove.py": "def removed(value):\n    return value - 1",
            "src/keep.py": "def kept(value):\n    return value + 1",
        },
    )
    _patch_get_model(monkeypatch, CountingModel())
    _analyze(repo)
    deleted = (repo / "src/remove.py").resolve()
    (repo / "src/remove.py").unlink()

    result = _analyze(repo)

    assert result.embedding_stats is not None
    assert result.embedding_stats.encoded_inputs == 0
    assert result.embedding_stats.deleted_units == 1
    assert result.embedding_stats.orphan_rows_retained == 1
    repo_stats = EmbeddingCache().stats()["repos"][0]
    assert repo_stats["orphan_rows"] == 1
    assert repo_stats["last_complete_generation"] == 2
    assert all(unit.file_path != deleted for unit in result.units)
    assert all(
        duplicate.unit_a.file_path != deleted and duplicate.unit_b.file_path != deleted
        for duplicate in result.hybrid_duplicates
    )
    assert all(unit.file_path != deleted for unit in result.potentially_unused)
    _assert_matches_uncached(repo, result)

    for _ in range(2):
        retained = _analyze(repo)
        assert retained.embedding_stats is not None
        assert retained.embedding_stats.orphan_rows_retained == 1
        assert retained.embedding_stats.orphan_rows_collected == 0
    collected = _analyze(repo)
    assert collected.embedding_stats is not None
    assert collected.embedding_stats.orphan_rows_collected == 1
    assert collected.embedding_stats.orphan_rows_retained == 0
    collected_stats = EmbeddingCache().stats()
    assert collected_stats["entries"] == 1
    assert collected_stats["repos"][0]["orphan_rows"] == 0


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
    assert narrow_manifest.complete_scan is False
    assert narrow_manifest.orphans == {}
    _assert_matches_uncached(repo / "src/a.py", result)


def test_selection_change_does_not_classify_excluded_units_as_deleted(
    tmp_path, monkeypatch
) -> None:
    repo = _write_repo(
        tmp_path,
        {
            "src/small.py": "def small(value):\n    return value + 1",
            "src/larger.py": (
                "def larger(value):\n    adjusted = value + 1\n    return adjusted * 2"
            ),
        },
    )
    _patch_get_model(monkeypatch, CountingModel())
    _analyze(repo)

    result = _analyze(repo, min_semantic_statements=2)

    assert result.embedding_stats is not None
    assert result.embedding_stats.deleted_units == 0
    assert result.embedding_stats.orphan_rows_retained == 0


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
