"""Embedding-cache flags, telemetry, and the ``cache info`` / ``cache clear`` commands."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import codedupes.embedding_cache as embedding_cache_module
from codedupes import cli
from codedupes.embedding_cache import CacheClearResult, EmbeddingCache
from tests.cli_helpers import build_result, build_unit
from tests.conftest import patch_cli_analyzer
from tests.test_embedding_cache import REVISION_1, CountingModel, _patch_get_model


def test_cli_json_surfaces_cache_write_failure(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry(value):\n    return value + 1\n")
    _patch_get_model(monkeypatch, CountingModel())
    monkeypatch.setattr(embedding_cache_module, "_warned_cache_error", False)

    def fail_cache_write(*_args, **_kwargs):
        raise PermissionError("cache directory is read-only")

    monkeypatch.setattr(embedding_cache_module, "_atomic_write_shard", fail_cache_write)

    result = CliRunner().invoke(
        cli.cli,
        [
            "check",
            str(path),
            "--semantic-only",
            "--no-unused",
            "--min-statements",
            "0",
            "--model",
            "test-model",
            "--model-revision",
            REVISION_1,
            "--device",
            "cpu",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert result.stderr == ""
    warnings = json.loads(result.output)["summary"]["embeddings"]["cache_warnings"]
    assert len(warnings) == 1
    assert "Embedding cache write shard failed" in warnings[0]
    assert "PermissionError: cache directory is read-only" in warnings[0]


@pytest.mark.parametrize("command", ["check", "search"])
@pytest.mark.parametrize("as_json", [False, True])
def test_cli_embedding_telemetry_tracks_filesystem_transitions(
    monkeypatch, tmp_path, command, as_json
):
    repo = tmp_path / "repo"
    repo.mkdir()
    for name in ("first.py", "second.py"):
        (repo / name).write_text("def entry(value):\n    return value + 1\n")

    class ProgressModel(CountingModel):
        def encode(self, texts, **kwargs):
            if as_json:
                assert kwargs["show_progress_bar"] is False
            return super().encode(texts, **kwargs)

    model = ProgressModel()
    _patch_get_model(monkeypatch, model)
    args = [command, str(repo)]
    if command == "check":
        args += ["--semantic-only", "--no-unused", "--fail-on", "none"]
    else:
        args += ["find entry"]
    args += [
        "--model",
        "test-model",
        "--model-revision",
        REVISION_1,
        "--device",
        "cpu",
        "--min-statements",
        "0",
        "--semantic-threshold",
        "0",
    ]
    args += ["--json"] if as_json else ["--output-width", "240"]
    if command == "search" and not as_json:
        args.append("-v")
    runner = CliRunner()
    cached_payload = None

    for phase, hits, encoded, reused, moved, deleted in (
        ("cold", 0, 1, 1, 0, 0),
        ("warm", 2, 0, 0, 0, 0),
        ("rename", 2, 0, 0, 1, 0),
        ("uncached", 0, 2, 0, 0, 0),
        ("delete", 1, 0, 0, 0, 1),
    ):
        if phase == "rename":
            (repo / "moved").mkdir()
            (repo / "first.py").rename(repo / "moved/renamed.py")
        elif phase == "delete":
            (repo / "second.py").unlink()
        result = runner.invoke(cli.cli, args + (["--no-cache"] if phase == "uncached" else []))
        assert result.exit_code == 0, result.output
        if as_json:
            assert result.stderr == ""
            payload = json.loads(result.stdout)
            stats = payload["summary"].pop("embeddings")
            assert stats["cache_hit_rows"] == hits
            assert stats["encoded_inputs"] == encoded
            assert stats["unique_inputs"] == 1
            assert stats["duplicate_rows_reused"] == reused
            assert stats["moved_units_reused"] == moved
            assert stats["deleted_units"] == deleted
            assert stats["model_loaded"] is bool(encoded)
            assert stats["cache_enabled"] is (phase != "uncached")
            assert stats["requested_rows"] == hits + encoded + reused
            assert all(Path(unit["file"]).is_file() for unit in payload["units"].values())
            if phase == "rename":
                assert payload["duplicates" if command == "check" else "results"]
                cached_payload = payload
            elif phase == "uncached":
                assert payload == cached_payload
        else:
            if command == "search":
                assert result.output.count("Effective search threshold: 0.0") == 1
                assert result.output.count("Search threshold: 0.0") == 1
            assert "Embeddings" in result.stdout
            assert f"{hits} rows from cache" in result.stdout
            assert f"{encoded} inputs encoded" in result.stdout
            assert f"{reused} duplicate rows reused" in result.stdout
            if moved:
                assert f"{moved} moved units remapped" in result.stdout
            if deleted:
                assert f"{deleted} units deleted" in result.stdout
            if not encoded:
                assert "model not loaded" in result.stdout


@pytest.mark.parametrize(
    ("command", "tail_args", "expected_exit_code"),
    [("check", [], 1), ("search", ["entry"], 0)],
)
@pytest.mark.parametrize(
    ("flag", "config_field", "expected_value"),
    [
        ("--no-cache", "embedding_cache", False),
        ("--strict-revision-cache", "strict_revision_cache", True),
        ("--loose-revision-cache", "strict_revision_cache", False),
    ],
)
def test_cli_cache_flags_plumb_to_config(
    monkeypatch,
    tmp_path,
    command,
    tail_args,
    expected_exit_code,
    flag,
    config_field,
    expected_value,
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result(tmp_path),
        search_results=[(build_unit(tmp_path), 0.9)],
        captured_configs=captured,
    )
    result = CliRunner().invoke(cli.cli, [command, str(path), *tail_args, flag])

    assert result.exit_code == expected_exit_code
    assert getattr(captured[0], config_field) is expected_value


def test_cli_traditional_only_accepts_no_cache_as_noop(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result(tmp_path),
        captured_configs=captured,
    )

    result = CliRunner().invoke(
        cli.cli,
        ["check", str(path), "--traditional-only", "--no-cache"],
    )

    assert result.exit_code == 1
    assert captured[0].run_semantic is False
    assert captured[0].embedding_cache is False


def test_cli_check_defaults_to_embedding_cache_enabled(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path)])

    assert result.exit_code == 1
    assert captured[0].embedding_cache is True


def test_cli_defaults_to_strict_revision_cache(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path)])

    assert result.exit_code == 1
    assert captured[0].strict_revision_cache is True


@pytest.mark.parametrize("command", ["check", "search"])
def test_cli_help_documents_strict_revision_cache_flag(command):
    result = CliRunner().invoke(cli.cli, [command, "--help"])

    assert result.exit_code == 0
    assert "--strict-revision-cache" in result.output
    assert "loose" in result.output and "stale warm hits" in result.output


def test_cli_cache_info_reports_empty_cache():
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cache", "info"])

    assert result.exit_code == 0
    assert "Cache path" in result.output
    assert "╭" in result.output and "│" in result.output
    assert any("Entries 0" in " ".join(line.split()) for line in result.output.splitlines())


def test_cli_cache_info_reports_populated_cache(tmp_path):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    cache.put_many(scope, "some/model", "rev1", [("k1", np.array([1.0, 2.0], dtype=np.float32))])

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cache", "info"])

    assert result.exit_code == 0
    assert any("Entries 1" in " ".join(line.split()) for line in result.output.splitlines())
    assert any("some/model 1" in " ".join(line.split()) for line in result.output.splitlines())
    assert "Per-repo breakdown" in result.output
    assert "1 shard(s), 1 entries" in result.output


def test_cli_cache_info_errors_when_cache_construction_fails(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("no home directory")

    monkeypatch.setattr(cli, "EmbeddingCache", _raise)

    result = CliRunner().invoke(cli.cli, ["cache", "info"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Cache unavailable: no home directory" in result.stderr


def test_cli_cache_clear_removes_all_entries(tmp_path):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    cache.put_many(scope, "some/model", "rev1", [("k1", np.array([1.0, 2.0], dtype=np.float32))])

    runner = CliRunner(env={"CODEDUPES_CACHE_CLEAR_MODEL": "unrelated/model"})
    result = runner.invoke(cli.cli, ["cache", "clear"])

    assert result.exit_code == 0
    assert "Cleared 1 cached embedding" in result.output
    assert cache.stats()["entries"] == 0


@pytest.mark.parametrize("model", ["", " ", "\t"])
def test_cli_cache_clear_rejects_empty_model_without_deleting(tmp_path, model):
    cache = EmbeddingCache()
    cache.put_many(tmp_path, "some/model", "rev1", [("k1", np.array([1.0, 2.0]))])

    result = CliRunner().invoke(cli.cli, ["cache", "clear", "--model", model])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "must not be empty" in result.stderr
    assert cache.stats()["entries"] == 1


def test_cli_cache_clear_scoped_to_model(tmp_path):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    cache.put_many(
        scope,
        "Alibaba-NLP/gte-modernbert-base",
        "rev1",
        [("k1", np.array([1.0, 2.0], dtype=np.float32))],
    )
    cache.put_many(scope, "other/model", "rev1", [("k2", np.array([3.0, 4.0], dtype=np.float32))])

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cache", "clear", "--model", "gte-modernbert-base"])

    assert result.exit_code == 0
    assert "Cleared 1 cached embedding" in result.output
    remaining = cache.stats()
    assert remaining["entries"] == 1
    assert remaining["models"] == {"other/model": 1}


def test_cli_cache_clear_warns_for_missing_local_model_directory(tmp_path):
    missing = tmp_path / "gone-model"

    result = CliRunner().invoke(cli.cli, ["cache", "clear", "--model", str(missing)])

    assert result.exit_code == 0
    # Rich wraps the long temporary path, so compare with line breaks collapsed.
    message = " ".join(result.stderr.split())
    assert "does not exist" in message
    assert "without --model" in message


def test_cli_cache_clear_reports_failure(monkeypatch):
    def fail_clear(_self, model=None):
        raise PermissionError("cache is read-only")

    monkeypatch.setattr(cli.EmbeddingCache, "clear", fail_clear)

    result = CliRunner().invoke(cli.cli, ["cache", "clear"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Cache clear failed: cache is read-only" in result.stderr


def test_cli_cache_clear_reports_best_effort_deletion_failures(monkeypatch):
    monkeypatch.setattr(
        cli.EmbeddingCache,
        "clear",
        lambda _self, model=None: CacheClearResult(
            removed_entries=2,
            failed_deletions=1,
        ),
    )

    result = CliRunner().invoke(cli.cli, ["cache", "clear"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "removed 2 cached embedding(s)" in result.stderr
    assert "1 deletion operation(s) failed" in result.stderr


def test_cli_cache_clear_wraps_literal_status(monkeypatch):
    model = "org/[red]" + "long-model-name-" * 10
    monkeypatch.setattr(
        cli.EmbeddingCache,
        "clear",
        lambda _self, model=None: CacheClearResult(removed_entries=1, failed_deletions=0),
    )
    result = CliRunner().invoke(
        cli.cli, ["cache", "clear", "--model", model, "--output-width", "80"]
    )
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert max(map(len, result.stdout.splitlines())) <= 80
    assert model in "".join(line.strip() for line in result.stdout.splitlines())


@pytest.mark.parametrize("command", [["info", "--verbose"], ["cache", "info"], ["cache", "clear"]])
def test_cli_cache_warnings_use_rich_stderr(monkeypatch, command):
    original_stats = EmbeddingCache.stats

    def warn():
        logging.getLogger("codedupes.embedding_cache").warning(
            "Cache operation failed at " + "deeply/nested/" * 12 + "cache.json"
        )

    def noisy_stats(cache):
        warn()
        return original_stats(cache)

    def noisy_clear(_cache, model=None):
        warn()
        return CacheClearResult(removed_entries=0, failed_deletions=1)

    monkeypatch.setattr(EmbeddingCache, "stats", noisy_stats)
    monkeypatch.setattr(EmbeddingCache, "clear", noisy_clear)
    result = CliRunner().invoke(cli.cli, [*command, "--output-width", "80"])

    assert result.exit_code == (1 if command[-1] == "clear" else 0), result.output
    assert "WARNING" in result.stderr
    assert "Cache operation failed" in result.stderr
    assert max(map(len, result.stderr.splitlines())) <= 80
    assert "Cache operation failed" not in result.stdout
