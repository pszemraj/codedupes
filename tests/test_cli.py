from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import codedupes.embedding_cache as embedding_cache_module
from codedupes import cli
from codedupes.devices import DeviceDiagnostics
from codedupes.embedding_cache import CacheClearResult, EmbeddingCache
from codedupes.languages import GrammarUnavailableError
from codedupes.logging_utils import NOISY_EXTERNAL_LOGGERS
from codedupes.models import (
    AnalysisResult,
    CodeUnit,
    DuplicatePair,
    ExtractionDiagnostic,
    HybridDuplicate,
)
from codedupes.semantic import EmbeddingRunStats, SemanticBackendError
from tests.conftest import make_code_unit, patch_cli_analyzer
from tests.test_embedding_cache import REVISION_1, CountingModel, _patch_get_model


def _build_unit(tmp_path: Path) -> CodeUnit:
    return make_code_unit(tmp_path, name="entry", source="def entry():\n    return 1")


def _build_result(tmp_path: Path) -> AnalysisResult:
    unit = _build_unit(tmp_path)
    duplicate = DuplicatePair(
        unit_a=unit,
        unit_b=unit,
        similarity=1.0,
        method="ast_hash",
    )
    hybrid = HybridDuplicate(
        unit_a=unit,
        unit_b=unit,
        tier="exact",
        confidence=1.0,
        has_exact=True,
    )

    return AnalysisResult(
        units=[unit],
        traditional_duplicates=[duplicate],
        semantic_duplicates=[],
        hybrid_duplicates=[hybrid],
        potentially_unused=[unit],
        analysis_mode="combined",
        embedding_stats=EmbeddingRunStats(
            requested_rows=1,
            unique_inputs=1,
            cache_hit_rows=1,
            model_loaded=False,
            cache_enabled=True,
            cache_revision="1" * 40,
        ),
    )


def _raise_semantic_backend_error(*_args, **_kwargs):
    raise SemanticBackendError("semantic backend mismatch")


def _build_result_with_semantic_duplicate(tmp_path: Path) -> AnalysisResult:
    result = _build_result(tmp_path)
    unit = _build_unit(tmp_path)
    result.semantic_duplicates = [
        DuplicatePair(unit_a=unit, unit_b=unit, similarity=0.95, method="semantic")
    ]
    return result


def test_cli_json_output_hybrid_default(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        search_results=[(_build_unit(tmp_path), 0.99)],
        captured_configs=captured,
    )
    runner = CliRunner()

    result = runner.invoke(cli.cli, ["check", str(path), "--json"])
    assert result.exit_code == 1
    output = json.loads(result.output)

    assert "summary" in output
    assert output["summary"]["hybrid_duplicates"] == 1
    assert output["summary"]["potentially_unused"] == 1
    assert output["summary"]["embeddings"]["cache_hit_rows"] == 1
    assert output["summary"]["embeddings"]["model_loaded"] is False
    assert output["summary"]["embeddings"]["cache_warnings"] == []
    assert output["summary"]["fail_on"] == "actionable"
    assert output["summary"]["exit_code"] == 1
    assert output["schema_version"] == 2
    assert "duplicates" in output
    assert output["duplicates"][0]["unit_a"] in output["units"]
    assert output["duplicates"][0]["unit_b"] in output["units"]
    assert output["potentially_unused"][0] in output["units"]
    assert len(output["units"]) <= 2 * len(output["duplicates"]) + len(output["potentially_unused"])
    assert "hybrid_duplicates" not in output
    assert "traditional_duplicates" not in output
    assert "semantic_duplicates" not in output
    assert captured[0].include_private is True
    assert captured[0].progress == "never"

    result = runner.invoke(cli.cli, ["search", str(path), "entry", "--json", "--top-k", "1"])
    assert result.exit_code == 0
    search_output = json.loads(result.output)
    assert search_output["query"] == "entry"
    result_uid = search_output["results"][0]["unit"]
    assert search_output["units"][result_uid]["name"] == "entry"
    assert captured[1].progress == "never"


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


def test_cli_table_output_uses_auto_progress(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )

    result = CliRunner().invoke(cli.cli, ["check", str(path)])

    assert captured[0].progress == "auto"
    assert "Embeddings" in result.output
    assert "model not loaded" in result.output


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


@pytest.mark.parametrize("initially_disabled", [False, True])
def test_cli_json_restores_huggingface_progress_state(monkeypatch, tmp_path, initially_disabled):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    patch_cli_analyzer(monkeypatch, cli, analyze_result=lambda: _build_result(tmp_path))
    state = {"disabled": initially_disabled}
    calls: list[str] = []

    def disable_progress_bars():
        calls.append("disable")
        state["disabled"] = True

    def enable_progress_bars():
        calls.append("enable")
        state["disabled"] = False

    monkeypatch.setattr(
        "huggingface_hub.utils.are_progress_bars_disabled",
        lambda: state["disabled"],
    )
    monkeypatch.setattr("huggingface_hub.utils.disable_progress_bars", disable_progress_bars)
    monkeypatch.setattr("huggingface_hub.utils.enable_progress_bars", enable_progress_bars)

    CliRunner().invoke(cli.cli, ["check", str(path), "--json"])

    assert calls == (["disable"] if initially_disabled else ["disable", "enable"])
    assert state["disabled"] is initially_disabled


def test_cli_json_discards_direct_backend_stderr_on_success(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    class NoisyAnalyzer:
        def __init__(self, _config):
            pass

        def analyze(self, _path):
            print("backend progress", file=sys.stderr)
            return _build_result(tmp_path)

    monkeypatch.setattr(cli, "CodeAnalyzer", NoisyAnalyzer)

    result = CliRunner().invoke(cli.cli, ["check", str(path), "--json"])

    assert result.exit_code == 1
    assert result.stderr == ""
    assert json.loads(result.output)["schema_version"] == 2


def _run_merged_cli(args: list[str], setup: str = "") -> subprocess.CompletedProcess[str]:
    """Run the real CLI in a subprocess with stderr merged into stdout."""
    command = ["codedupes", *args]
    if setup:
        # Fault injection needs an interpreter; ordinary runs test the installed command.
        script = (
            "from codedupes import cli\n"
            + textwrap.dedent(setup)
            + "\nraise SystemExit(cli.main())"
        )
        command = [sys.executable, "-c", script, *args]
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        check=False,
    )


@pytest.mark.parametrize("command", ["check", "search"])
def test_cli_json_isolates_custom_family_warning_before_config(tmp_path, command):
    args = [command, str(tmp_path)]
    if command == "search":
        args.append("entry")
    result = _run_merged_cli([*args, "--model", "review/gte-modernbert-base", "--json"])

    assert result.returncode == 0, result.stdout
    assert json.loads(result.stdout)["schema_version"] == 2


@pytest.mark.parametrize(
    ("command", "fail_on", "exit_code"),
    [("check", "none", 0), ("check", "actionable", 1), ("search", None, 0)],
)
def test_cli_json_isolates_native_stderr_in_completed_report(tmp_path, command, fail_on, exit_code):
    args = [command, str(tmp_path), "--json"]
    if command == "check":
        (tmp_path / "sample.py").write_text(
            "def first(value):\n    return value + 1\n\ndef second(value):\n    return value + 1\n"
        )
        args.extend(["--traditional-only", "--no-unused", "--no-tiny-filter", "--fail-on", fail_on])
    else:
        args.append("entry")
    result = _run_merged_cli(
        args,
        """
        import os
        import sys

        class NoisyAnalyzer(cli.CodeAnalyzer):
            def __init__(self, config):
                os.write(2, b"native initialization diagnostic\\n")
                super().__init__(config)

            def analyze(self, path):
                print("Python analysis diagnostic", file=sys.stderr)
                os.write(2, b"native analysis diagnostic\\n")
                return super().analyze(path)

            def index(self, path):
                print("Python indexing diagnostic", file=sys.stderr)
                os.write(2, b"native indexing diagnostic\\n")
                return super().index(path)

            def search(self, *args, **kwargs):
                os.write(2, b"native query diagnostic\\n")
                return super().search(*args, **kwargs)

        cli.CodeAnalyzer = NoisyAnalyzer
        """,
    )

    assert result.returncode == exit_code, result.stdout
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 2
    if command == "check":
        assert payload["duplicates"]
        assert payload["summary"]["exit_code"] == exit_code


@pytest.mark.parametrize("command", ["check", "search"])
def test_cli_json_replays_python_and_native_stderr_on_failure(tmp_path, command):
    args = [command, str(tmp_path), "--json"]
    if command == "search":
        args.append("entry")
    result = _run_merged_cli(
        args,
        """
        import os
        import sys

        class FailingAnalyzer(cli.CodeAnalyzer):
            def analyze(self, path):
                print("Python backend diagnostic", file=sys.stderr)
                os.write(2, b"native backend diagnostic\\n")
                raise RuntimeError("backend exploded")

            index = analyze

        cli.CodeAnalyzer = FailingAnalyzer
        """,
    )

    assert result.returncode == 1
    assert "Python backend diagnostic" in result.stdout
    assert "native backend diagnostic" in result.stdout
    error_label = "analysis" if command == "check" else "search"
    assert f"Error during {error_label}: backend exploded" in result.stdout
    assert "schema_version" not in result.stdout


def test_cli_reports_semantic_context_diagnostics(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = _build_unit(tmp_path)
    result_obj = AnalysisResult(
        units=[unit],
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[],
        potentially_unused=[],
        analysis_mode="combined",
        semantic_diagnostics=[
            ExtractionDiagnostic(
                file_path=unit.file_path,
                language="python",
                code="semantic-context-overflow",
                message="sample.entry is 4096 tokens including the encode prompt",
                lineno=1,
                end_lineno=2,
            )
        ],
    )
    patch_cli_analyzer(monkeypatch, cli, analyze_result=result_obj)
    runner = CliRunner()

    table_result = runner.invoke(cli.cli, ["check", str(path)])
    assert "Semantic diagnostics" in table_result.output
    assert "4096 tokens" in table_result.output

    json_result = runner.invoke(cli.cli, ["check", str(path), "--json"])
    payload = json.loads(json_result.output)
    assert payload["summary"]["semantic_diagnostics"] == 1
    assert payload["semantic_diagnostics"][0]["code"] == "semantic-context-overflow"


def test_cli_search_json_surfaces_semantic_diagnostics(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = _build_unit(tmp_path)
    result_obj = AnalysisResult(
        units=[unit],
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[],
        potentially_unused=[],
        analysis_mode="semantic",
    )
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=result_obj,
        search_results=[(unit, 0.91)],
        semantic_diagnostics=[
            ExtractionDiagnostic(
                file_path=unit.file_path,
                language="python",
                code="semantic-context-overflow",
                message="sample.entry is 4096 tokens including the encode prompt",
                lineno=1,
                end_lineno=2,
            )
        ],
    )
    runner = CliRunner()

    result = runner.invoke(cli.cli, ["search", str(path), "entry", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    result_uid = payload["results"][0]["unit"]
    assert payload["units"][result_uid]["name"] == "entry"
    assert payload["semantic_diagnostics"][0]["code"] == "semantic-context-overflow"


def test_cli_search_indexes_without_running_full_analysis(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    class IndexOnlyAnalyzer:
        def __init__(self, config):
            del config
            self.semantic_diagnostics = []
            self.embedding_stats = None

        def analyze(self, _path):
            raise AssertionError("search must build its corpus via index(), not analyze()")

        def index(self, _path):
            return 1

        def search(self, query, top_k=10):
            del query, top_k
            return [(_build_unit(tmp_path), 0.99)]

    monkeypatch.setattr(cli, "CodeAnalyzer", IndexOnlyAnalyzer)
    runner = CliRunner()

    result = runner.invoke(cli.cli, ["search", str(path), "entry", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    result_uid = payload["results"][0]["unit"]
    assert payload["units"][result_uid]["name"] == "entry"


def _patch_search_analyzer(
    monkeypatch,
    *,
    indexed_units: int = 0,
    extracted_unit_count: int = 1,
    results: list | None = None,
    index_error: Exception | None = None,
    semantic_diagnostics: list[ExtractionDiagnostic] | None = None,
) -> None:
    """Patch the CLI analyzer with a search double that controls the index size."""

    class StubSearchAnalyzer:
        def __init__(self, config):
            del config
            self.extracted_unit_count = extracted_unit_count
            self.semantic_diagnostics = list(semantic_diagnostics or [])
            self.embedding_stats = None

        def index(self, _path):
            if index_error is not None:
                raise index_error
            return indexed_units

        def search(self, query, top_k=10):
            del query, top_k
            return list(results or [])

    monkeypatch.setattr(cli, "CodeAnalyzer", StubSearchAnalyzer)


def test_cli_search_warns_when_candidate_filters_emptied_the_index(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    _patch_search_analyzer(monkeypatch, indexed_units=0)

    result = CliRunner().invoke(cli.cli, ["search", str(path), "entry", "--min-statements", "3"])

    assert result.exit_code == 0
    assert "search index is empty" in result.stderr
    assert "--min-statements" in result.stderr
    # The zero-hit table still renders, but no longer alone.
    assert "No matches found" in result.stdout


def test_cli_search_empty_extraction_warning_does_not_blame_candidate_filters(
    monkeypatch, tmp_path
):
    path = tmp_path / "empty"
    path.mkdir()
    _patch_search_analyzer(monkeypatch, indexed_units=0, extracted_unit_count=0)

    result = CliRunner().invoke(cli.cli, ["search", str(path), "entry"])

    assert result.exit_code == 0
    assert "extraction produced no code units" in result.stderr
    assert "--min-statements" not in result.stderr
    assert "--semantic-unit-type" not in result.stderr


def test_cli_search_empty_index_reports_semantic_context_diagnostics(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    diagnostic = ExtractionDiagnostic(
        file_path=path,
        language="python",
        code="semantic-context-overflow",
        message="sample.entry exceeds the model context window",
        lineno=1,
        end_lineno=2,
    )
    _patch_search_analyzer(
        monkeypatch,
        indexed_units=0,
        extracted_unit_count=1,
        semantic_diagnostics=[diagnostic],
    )

    # A wide console keeps the diagnostic on one line: the default width wraps
    # it at a point that depends on the pytest tmp-path length, splitting the
    # asserted message on some machines.
    result = CliRunner().invoke(cli.cli, ["search", str(path), "entry", "--output-width", "400"])

    assert result.exit_code == 0
    assert "no semantic candidates survived indexing" in result.stderr
    assert "Semantic diagnostics" in result.stdout
    assert "exceeds the model context window" in result.stdout


def test_cli_search_does_not_warn_when_the_index_has_units(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    _patch_search_analyzer(monkeypatch, indexed_units=4)

    result = CliRunner().invoke(cli.cli, ["search", str(path), "entry"])

    assert result.exit_code == 0
    assert "search index is empty" not in result.output
    assert "No matches found" in result.stdout


@pytest.mark.parametrize("indexed_units", [0, 7])
def test_cli_search_json_reports_indexed_unit_count(monkeypatch, tmp_path, indexed_units):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    _patch_search_analyzer(monkeypatch, indexed_units=indexed_units)

    result = CliRunner().invoke(cli.cli, ["search", str(path), "entry", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 2
    assert payload["summary"]["indexed_units"] == indexed_units
    assert payload["results"] == []
    assert result.stderr == ""


def test_cli_search_reports_path_deleted_after_validation(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    _patch_search_analyzer(
        monkeypatch,
        index_error=FileNotFoundError("Path does not exist"),
    )

    result = CliRunner().invoke(cli.cli, ["search", str(path), "entry"])

    assert result.exit_code == 1
    assert not isinstance(result.exception, FileNotFoundError)
    assert "Error: Path does not exist" in result.stderr


@pytest.mark.parametrize("phase", ["construction", "query"])
@pytest.mark.parametrize("as_json", [False, True])
def test_cli_search_reports_runtime_failures(monkeypatch, tmp_path, phase, as_json):
    def fail(*args, **kwargs):
        raise FileNotFoundError("model asset disappeared")

    if phase == "construction":
        monkeypatch.setattr(cli, "CodeAnalyzer", fail)
    else:
        patch_cli_analyzer(
            monkeypatch, cli, analyze_result=_build_result(tmp_path), search_results=fail
        )
    args = ["search", str(tmp_path), "entry"] + (["--json"] if as_json else [])
    result = CliRunner().invoke(cli.cli, args)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "model asset disappeared" in result.stderr
    assert "Traceback" not in result.stderr
    assert not isinstance(result.exception, FileNotFoundError)


@pytest.mark.parametrize("command", ["check", "search"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-0.1", "1.1"])
def test_cli_rejects_invalid_active_threshold_before_analysis(
    monkeypatch, tmp_path, command, value
):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid threshold reached analysis")

    monkeypatch.setattr(cli, "CodeAnalyzer", unexpected)
    args = [command, str(tmp_path)] + (["entry"] if command == "search" else [])
    result = CliRunner().invoke(cli.cli, [*args, "--threshold", value, "--json"])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "[0.0, 1.0]" in result.stderr


def test_cli_json_show_all_includes_raw_sections(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result_with_semantic_duplicate(tmp_path),
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--json", "--show-all"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert "traditional_duplicates" in payload
    assert "semantic_duplicates" in payload
    for collection in ("duplicates", "traditional_duplicates", "semantic_duplicates"):
        for edge in payload[collection]:
            assert edge["unit_a"] in payload["units"]
            assert edge["unit_b"] in payload["units"]


def test_cli_json_v2_raw_mode_uses_edge_list(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = _build_unit(tmp_path)
    duplicate = DuplicatePair(unit_a=unit, unit_b=unit, similarity=0.95, method="semantic")
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=AnalysisResult(
            units=[unit],
            traditional_duplicates=[],
            semantic_duplicates=[duplicate],
            hybrid_duplicates=[],
            potentially_unused=[],
            analysis_mode="semantic",
        ),
    )

    result = CliRunner().invoke(
        cli.cli,
        ["check", str(path), "--semantic-only", "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["schema_version"] == 2
    assert payload["duplicates"] == [
        {
            "method": "semantic",
            "similarity": 0.95,
            "unit_a": unit.uid,
            "unit_b": unit.uid,
        }
    ]
    assert payload["units"][unit.uid]["name"] == "entry"
    assert "traditional_duplicates" not in payload
    assert "semantic_duplicates" not in payload


def test_cli_json_v2_emits_each_unit_once(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    result_obj = _build_result(tmp_path)
    result_obj.hybrid_duplicates *= 4
    patch_cli_analyzer(monkeypatch, cli, analyze_result=result_obj)

    result = CliRunner().invoke(cli.cli, ["check", str(path), "--json"])

    payload = json.loads(result.output)
    assert len(payload["duplicates"]) == 4
    assert len(payload["units"]) == 1


def test_cli_no_private_option_check(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--no-private"])
    assert result.exit_code == 1
    assert captured[0].include_private is False


def test_cli_model_semantic_flags_pass_through(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            "check",
            str(path),
            "--semantic-threshold",
            "0.9",
            "--instruction-prefix",
            "Represent this code: ",
            "--model-revision",
            "test-rev",
            "--semantic-task",
            "classification",
            "--no-trust-remote-code",
            "--suppress-test-semantic",
            "--semantic-unit-type",
            "class",
            "--no-tiny-filter",
            "--tiny-cutoff",
            "4",
            "--show-all",
        ],
    )

    assert result.exit_code == 1
    assert captured[0].instruction_prefix == "Represent this code: "
    assert captured[0].model_revision == "test-rev"
    assert captured[0].trust_remote_code is False
    assert captured[0].suppress_test_semantic_matches is True
    assert captured[0].semantic_task == "classification"
    assert captured[0].semantic_unit_types == ("class",)
    assert captured[0].filter_tiny_traditional is False
    assert captured[0].tiny_unit_statement_cutoff == 4


def test_cli_check_rejects_uncalibrated_context_as_usage_error(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    patch_cli_analyzer(monkeypatch, cli, analyze_result=lambda: _build_result(tmp_path))
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--instruction-prefix", "Represent this code: "],
    )

    assert result.exit_code == 2
    assert "provide semantic_threshold explicitly" in result.output


def test_cli_search_builds_search_mode_config(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        search_results=[],
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["search", str(path), "find entry", "--instruction-prefix", "custom: "],
    )

    assert result.exit_code == 0
    assert captured[0].mode == "search"
    assert captured[0].instruction_prefix == "custom: "


def test_cli_allow_semantic_fallback_pass_through(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--allow-semantic-fallback"])

    assert result.exit_code == 1
    assert captured[0].allow_semantic_fallback is True


def test_cli_model_revision_defaults_to_auto_none(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            "check",
            str(path),
            "--model",
            "sentence-transformers/all-MiniLM-L6-v2",
        ],
    )

    assert result.exit_code == 1
    assert captured[0].model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert captured[0].model_revision is None


@pytest.mark.parametrize(
    ("command", "tail_args", "expected_exit"),
    [
        ("check", [], 1),
        ("search", ["entry"], 0),
    ],
)
def test_cli_local_model_path_pass_through(
    monkeypatch,
    tmp_path,
    command,
    tail_args,
    expected_exit,
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    model_dir = tmp_path / "saved-model"
    model_dir.mkdir()
    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        search_results=[(_build_unit(tmp_path), 0.99)],
        captured_configs=captured,
    )

    result = CliRunner().invoke(
        cli.cli,
        [command, str(path), *tail_args, "--model", str(model_dir)],
    )

    assert result.exit_code == expected_exit
    assert captured[0].model_name == str(model_dir)


def test_cli_threshold_precedence(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()

    result_default = runner.invoke(cli.cli, ["check", str(path)])
    assert result_default.exit_code == 1
    # No override: the analyzer applies the profile's per-language gates.
    assert captured[-1].semantic_threshold is None
    assert captured[-1].jaccard_threshold == cli.DEFAULT_TRADITIONAL_THRESHOLD
    assert captured[-1].semantic_unit_types == ("function", "method")
    assert captured[-1].filter_tiny_traditional is True
    # The CLI defaults must be the library defaults, not a second hardcoded copy.
    assert captured[-1].tiny_unit_statement_cutoff == cli.DEFAULT_TINY_UNIT_STATEMENT_CUTOFF
    assert captured[-1].tiny_unit_statement_cutoff == 3

    result_shared = runner.invoke(cli.cli, ["check", str(path), "--threshold", "0.67"])
    assert result_shared.exit_code == 1
    assert captured[-1].semantic_threshold == 0.67
    assert captured[-1].jaccard_threshold == 0.67

    result_override = runner.invoke(
        cli.cli,
        [
            "check",
            str(path),
            "--threshold",
            "0.67",
            "--semantic-threshold",
            "0.91",
            "--traditional-threshold",
            "0.44",
        ],
    )
    assert result_override.exit_code == 1
    assert captured[-1].semantic_threshold == 0.91
    assert captured[-1].jaccard_threshold == 0.44


def test_cli_semantic_only_shared_threshold_does_not_set_traditional_threshold(
    monkeypatch, tmp_path
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--semantic-only", "--threshold", "0.7"],
    )
    assert result.exit_code == 1
    assert captured[-1].semantic_threshold == 0.7
    assert captured[-1].jaccard_threshold == cli.DEFAULT_TRADITIONAL_THRESHOLD


def test_cli_traditional_only_omits_semantic_defaults(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--traditional-only"])
    assert result.exit_code == 1
    assert captured[-1].run_semantic is False
    assert captured[-1].semantic_threshold is None
    assert captured[-1].semantic_task is None


def test_cli_traditional_only_shared_threshold_sets_only_traditional_threshold(
    monkeypatch, tmp_path
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--traditional-only", "--threshold", "0.9"],
    )
    assert result.exit_code == 1
    assert captured[-1].jaccard_threshold == 0.9
    assert captured[-1].semantic_threshold is None
    assert captured[-1].semantic_task is None


def test_cli_cross_language_flag_passes_through(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()

    result_default = runner.invoke(cli.cli, ["check", str(path)])
    assert result_default.exit_code == 1
    assert captured[-1].cross_language is False

    result_flag = runner.invoke(cli.cli, ["check", str(path), "--cross-language"])
    assert result_flag.exit_code == 1
    assert captured[-1].cross_language is True


def test_cli_search_defaults_to_code_retrieval_task(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        search_results=[(_build_unit(tmp_path), 0.99)],
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["search", str(path), "entry"])
    assert result.exit_code == 0
    assert captured[0].semantic_task == "code-retrieval"
    assert captured[0].semantic_unit_types == ("function", "method")
    assert captured[0].search_document == "source"


def test_cli_contextual_search_requires_explicit_threshold(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    model = CountingModel()
    _patch_get_model(monkeypatch, model)
    args = [
        "search",
        str(path),
        "entry",
        "--search-document",
        "contextual",
        "--min-statements",
        "0",
        "--device",
        "cpu",
        "--json",
    ]
    runner = CliRunner()
    missing = runner.invoke(cli.cli, args)

    assert missing.exit_code == 2
    assert "contextual" in missing.output
    assert "explicit threshold" in missing.output
    assert "--semantic-threshold" in missing.output
    assert model.encode_calls == []

    for option in ("--semantic-threshold", "--threshold"):
        result = runner.invoke(cli.cli, [*args, option, "0.0"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["summary"]["results"] == 1
    assert len(model.encode_calls) == 2
    assert model.encode_calls[-1] == ["entry"]


@pytest.mark.parametrize("command_tail", [[], ["entry"]], ids=["check", "search"])
@pytest.mark.parametrize("explicit_options", [False, True])
def test_cli_options_ignore_automatic_environment_variables(
    monkeypatch, tmp_path, command_tail, explicit_options
) -> None:
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        search_results=[(_build_unit(tmp_path), 0.99)],
        captured_configs=captured,
    )
    command = "search" if command_tail else "check"
    runner = CliRunner(
        env={
            "CODEDUPES_DEVICE": "cpu",
            "CODEDUPES_MODEL": "test-model",
            "CODEDUPES_NO_CACHE": "1",
            "CODEDUPES_LANGUAGES": "invalid-language",
            "CODEDUPES_THRESHOLD": "not-a-number",
            "CODEDUPES_NO_PRIVATE": "1",
            "CODEDUPES_EXCLUDE": "*.py",
            f"CODEDUPES_{command.upper()}_DEVICE": "cpu",
            f"CODEDUPES_{command.upper()}_MODEL": "test-model",
            f"CODEDUPES_{command.upper()}_THRESHOLD": "not-a-number",
        }
    )

    args = [command, str(path), *command_tail]
    if explicit_options:
        args.extend(["--device", "cpu", "--model", "explicit-model", "--no-cache"])
    result = runner.invoke(cli.cli, args)

    assert result.exit_code == (0 if command_tail else 1), result.output
    assert captured[0].device == ("cpu" if explicit_options else cli.DEFAULT_SEMANTIC_DEVICE)
    assert captured[0].model_name == ("explicit-model" if explicit_options else cli.DEFAULT_MODEL)
    # The cache library's explicit process control is independent of CLI parsing.
    assert captured[0].embedding_cache is (not explicit_options)
    assert captured[0].languages is None
    assert captured[0].include_private is True
    assert captured[0].exclude_patterns is None

    help_result = runner.invoke(cli.cli, [command, "--help"])
    assert help_result.exit_code == 0
    assert "CODEDUPES_" not in help_result.output


def test_cli_search_semantic_unit_type_pass_through(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        search_results=[(_build_unit(tmp_path), 0.99)],
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["search", str(path), "entry", "--semantic-unit-type", "class"],
    )
    assert result.exit_code == 0
    assert captured[0].semantic_unit_types == ("class",)


def test_cli_search_threshold_precedence(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        search_results=[(_build_unit(tmp_path), 0.99)],
        captured_configs=captured,
    )
    runner = CliRunner()

    result_default = runner.invoke(cli.cli, ["search", str(path), "entry"])
    assert result_default.exit_code == 0
    assert captured[-1].semantic_threshold is None

    result_shared = runner.invoke(cli.cli, ["search", str(path), "entry", "--threshold", "0.4"])
    assert result_shared.exit_code == 0
    assert captured[-1].semantic_threshold == 0.4

    result_override = runner.invoke(
        cli.cli,
        ["search", str(path), "entry", "--threshold", "0.4", "--semantic-threshold", "0.6"],
    )
    assert result_override.exit_code == 0
    assert captured[-1].semantic_threshold == 0.6


def test_cli_requires_explicit_command(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(cli.cli, [str(path), "--no-private"])
    assert result.exit_code == 2


@pytest.mark.parametrize(
    "token",
    [".", "./src", "/tmp", "~/whatever", "srcish"],
    ids=["dot", "dot-relative", "absolute", "tilde", "bare-name"],
)
def test_cli_no_subcommand_token_exits_usage_error(token):
    """A sole path-like or bare-name token with no subcommand must be a usage error.

    Click's ``resolve_command`` re-parses unmatched command tokens whose first
    character is non-alphanumeric (``.``, ``/``, ``~``, ...); older click releases
    used to re-run this with an emptied ``ctx.args``, which spuriously hit the
    group's no-args help path and exited 0 instead of raising a usage error.
    """
    runner = CliRunner()
    result = runner.invoke(cli.cli, [token])
    assert result.exit_code == 2
    assert f"No such command {token!r}." in result.output
    assert "Commands:" not in result.output


def test_cli_no_args_prints_help_and_exits_usage_error():
    result = _run_merged_cli([])
    assert result.returncode == 2
    assert "Commands" in result.stdout
    assert "check" in result.stdout
    assert "search" in result.stdout


@pytest.mark.parametrize(
    ("command", "tail_args"),
    [("check", []), ("search", ["entry"])],
)
def test_cli_rejects_missing_path(tmp_path, command, tail_args):
    missing = tmp_path / "missing.py"
    runner = CliRunner()
    result = runner.invoke(cli.cli, [command, str(missing), *tail_args])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_cli_invalid_threshold(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--threshold", "1.2"])
    assert result.exit_code == 2
    assert "must be in [0.0, 1.0]" in result.output


def test_cli_rejects_conflicting_single_method_flags(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--semantic-only", "--traditional-only"],
    )
    assert result.exit_code == 2


@pytest.mark.parametrize(
    ("flag", "expected_message"),
    [
        ("--show-all", "--show-all is only valid in default combined mode."),
        (
            "--allow-semantic-fallback",
            "--allow-semantic-fallback is only valid in default combined mode.",
        ),
    ],
)
def test_cli_rejects_combined_only_flags_in_single_method_modes(tmp_path, flag, expected_message):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    semantic_result = runner.invoke(
        cli.cli,
        ["check", str(path), "--semantic-only", flag],
    )
    assert semantic_result.exit_code == 2
    assert expected_message in semantic_result.output

    traditional_result = runner.invoke(
        cli.cli,
        ["check", str(path), "--traditional-only", flag],
    )
    assert traditional_result.exit_code == 2
    assert expected_message in traditional_result.output


@pytest.mark.parametrize(
    ("command", "tail_args", "rich_args", "expected_option"),
    [
        ("check", [], ["--show-source"], "--show-source"),
        ("search", ["entry"], ["--verbose"], "--verbose"),
        ("check", [], ["--output-width", "160"], "--output-width"),
    ],
)
def test_cli_rejects_json_with_rich_only_flags(
    tmp_path,
    command,
    tail_args,
    rich_args,
    expected_option,
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [command, str(path), *tail_args, "--json", *rich_args],
    )
    assert result.exit_code == 2
    assert f"Cannot use {expected_option} with --json." in result.output


@pytest.mark.parametrize(
    ("command", "tail_args", "enabled_flag", "disabled_flag"),
    [
        ("check", [], "--trust-remote-code", "--no-trust-remote-code"),
        ("search", ["entry"], "--trust-remote-code", "--no-trust-remote-code"),
        ("check", [], "--mps-fallback", "--no-mps-fallback"),
        ("search", ["entry"], "--mps-fallback", "--no-mps-fallback"),
    ],
)
def test_cli_rejects_conflicting_paired_flags(
    tmp_path,
    command,
    tail_args,
    enabled_flag,
    disabled_flag,
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [command, str(path), *tail_args, enabled_flag, disabled_flag],
    )
    assert result.exit_code == 2
    assert f"Cannot combine {enabled_flag} and {disabled_flag}." in result.output


@pytest.mark.parametrize(
    ("extra_args", "expected_option"),
    [
        (["--semantic-threshold", "0.9"], "--semantic-threshold"),
        (["--cross-language"], "--cross-language"),
        (["--semantic-task", "classification"], "--semantic-task"),
        (["--instruction-prefix", "prefix"], "--instruction-prefix"),
        (["--model", "sentence-transformers/all-MiniLM-L6-v2"], "--model"),
        (["--model-revision", "rev1"], "--model-revision"),
        (["--trust-remote-code"], "--trust-remote-code"),
        (["--no-trust-remote-code"], "--no-trust-remote-code"),
        (["--batch-size", "4"], "--batch-size"),
        (["--min-statements", "1"], "--min-statements"),
        (["--semantic-unit-type", "class"], "--semantic-unit-type"),
        (["--suppress-test-semantic"], "--suppress-test-semantic"),
    ],
)
def test_cli_rejects_all_semantic_mode_flags_with_traditional_only(
    tmp_path, extra_args, expected_option
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--traditional-only", *extra_args],
    )

    assert result.exit_code == 2
    assert f"Cannot use {expected_option}" in result.output


@pytest.mark.parametrize(
    ("extra_args", "expected_option"),
    [
        (["--traditional-threshold", "0.8"], "--traditional-threshold"),
        (["--no-tiny-filter"], "--no-tiny-filter"),
        (["--tiny-cutoff", "4"], "--tiny-cutoff"),
    ],
)
def test_cli_rejects_all_traditional_mode_flags_with_semantic_only(
    tmp_path, extra_args, expected_option
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--semantic-only", *extra_args],
    )

    assert result.exit_code == 2
    assert f"Cannot use {expected_option}" in result.output


def test_cli_rejects_strict_unused_with_no_unused(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--no-unused", "--strict-unused"],
    )

    assert result.exit_code == 2
    assert "Cannot combine --no-unused and --strict-unused" in result.output


def test_cli_info_exit_zero():
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["info"])
    assert result.exit_code == 0
    assert "codedupes" in result.output.lower()
    assert "pytorch:" in result.output.lower()
    assert "mps built/available:" in result.output.lower()
    assert "mlx loaded in process:" in result.output.lower()
    assert "built-in semantic model aliases" in result.output.lower()
    assert "family=gte-modernbert search_threshold=0.5" in result.output
    assert (
        "semantic duplicate gates: python=0.8, c=0.82, rust=0.74, "
        "javascript=0.7, typescript=0.68 (fallback=0.82)" in result.output
    )
    default_revision = cli.resolve_model_profile(cli.DEFAULT_MODEL).default_revision
    assert f"Default model revision: {default_revision}" in result.output


def test_cli_info_configures_mps_environment_before_diagnostics(monkeypatch):
    order: list[str] = []

    def _record_configure(requested_device, *, fallback):
        order.append(f"configure:{requested_device}:{fallback}")

    def _record_diagnostics(requested_device):
        order.append(f"diagnostics:{requested_device}")
        return DeviceDiagnostics(
            requested=requested_device,
            resolved="cpu",
            torch_available=True,
            cuda_available=False,
            mps_built=False,
            mps_available=False,
            mps_fallback_env="1",
            mlx_loaded=False,
            cpu_name="Test CPU",
            cpu_architecture="arm64",
            cpu_bf16_isa=False,
            cpu_mkldnn_available=False,
            cpu_bf16_native=False,
        )

    monkeypatch.setattr(cli, "configure_mps_environment", _record_configure)
    monkeypatch.setattr(cli, "get_device_diagnostics", _record_diagnostics)

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["info"])

    assert result.exit_code == 0
    assert order == [
        f"configure:{cli.DEFAULT_SEMANTIC_DEVICE}:None",
        f"diagnostics:{cli.DEFAULT_SEMANTIC_DEVICE}",
    ]


@pytest.mark.parametrize(
    ("command", "tail_args"),
    [("check", []), ("search", ["entry"])],
)
def test_cli_surfaces_analyzer_config_validation_error(monkeypatch, tmp_path, command, tail_args):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    def _raise_config_error(**_kwargs):
        raise ValueError("invalid config")

    monkeypatch.setattr(cli, "AnalyzerConfig", _raise_config_error)

    runner = CliRunner()
    result = runner.invoke(cli.cli, [command, str(path), *tail_args])
    assert result.exit_code == 2
    assert "invalid config" in result.output


def test_cli_help_and_version():
    help_result = _run_merged_cli(["--help"])
    assert help_result.returncode == 0
    assert "Commands" in help_result.stdout
    assert "check" in help_result.stdout
    assert "search" in help_result.stdout

    version_result = _run_merged_cli(["--version"])
    assert version_result.returncode == 0
    assert version_result.stdout.lower().startswith("codedupes")


def test_cli_search_help_is_search_specific() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["search", "--help"])

    assert result.exit_code == 0
    assert "also narrows traditional duplicate scope in combined mode" not in result.output
    assert "Built-in" in result.output
    assert "still apply." in result.output


@pytest.mark.parametrize("command", ["check", "search"])
def test_cli_help_advertises_local_model_directories(command: str) -> None:
    result = CliRunner(env={"COLUMNS": "200"}).invoke(cli.cli, [command, "--help"])

    assert result.exit_code == 0
    assert "complete local model directory" in result.output


def test_cli_output_width_option(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result_with_semantic_duplicate(tmp_path),
    )

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--output-width", "200"])
    assert result.exit_code == 1
    assert cli.console.width == 200


def test_cli_table_locations_disambiguate_same_named_files(monkeypatch, tmp_path):
    unit_a = make_code_unit(
        tmp_path, name="helper", source="def helper():\n    return 1", lineno=12
    )
    unit_b = make_code_unit(
        tmp_path, name="helper", source="def helper():\n    return 1", lineno=12
    )
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    unit_a.file_path = tmp_path / "alpha" / "utils.py"
    unit_b.file_path = tmp_path / "beta" / "utils.py"
    monkeypatch.chdir(tmp_path)

    assert cli.format_location(unit_a) == os.path.join("alpha", "utils.py") + ":12"
    assert cli.format_location(unit_b) == os.path.join("beta", "utils.py") + ":12"
    assert cli.format_location(unit_a) != cli.format_location(unit_b)


def test_cli_table_locations_preserve_bracketed_path_segments(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = _build_unit(tmp_path)
    unit.file_path = tmp_path / "corpus" / "pages" / "[id].ts"
    monkeypatch.chdir(tmp_path)
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        search_results=[(unit, 0.99)],
    )

    result = CliRunner().invoke(cli.cli, ["search", str(path), "entry"])

    assert result.exit_code == 0
    assert os.path.join("corpus", "pages", "[id].ts") + ":1" in result.stdout


def test_cli_diagnostics_preserve_bracketed_fields(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    result_obj = _build_result(tmp_path)
    result_obj.extraction_diagnostics = [
        ExtractionDiagnostic(
            file_path=tmp_path / "pages" / "[id].ts",
            language="typescript",
            code="partial-parse",
            message="unexpected [token]",
            lineno=1,
            end_lineno=1,
        )
    ]
    patch_cli_analyzer(monkeypatch, cli, analyze_result=result_obj)

    # A wide console keeps the diagnostic on one line: the default width wraps
    # it at a point that depends on the pytest tmp-path length, splitting the
    # asserted message on some machines.
    result = CliRunner().invoke(cli.cli, ["check", str(path), "--output-width", "400"])

    assert result.exit_code == 1
    assert "[typescript]" in result.stdout
    assert "[id].ts" in result.stdout
    assert "unexpected [token]" in result.stdout


def test_cli_table_location_uses_absolute_path_when_relative_path_is_longer(monkeypatch, tmp_path):
    deep_cwd = tmp_path.joinpath(*(f"level-{index}" for index in range(60)))
    deep_cwd.mkdir(parents=True)
    unit = _build_unit(tmp_path)
    unit.file_path = tmp_path / "corpus" / "algorithm.py"
    monkeypatch.chdir(deep_cwd)

    assert cli.format_location(unit) == f"{unit.file_path}:1"


def test_cli_show_all_prints_raw_sections(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result_with_semantic_duplicate(tmp_path),
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--show-all"])
    assert result.exit_code == 1
    assert "Traditional Duplicates (Raw" in result.output
    assert "Semantic Duplicates (Raw" in result.output


def test_cli_never_reports_a_filtered_raw_duplicate_count(monkeypatch, tmp_path):
    # Every candidate pair now reaches a tier, so the old always-zero counter
    # and its surfaces are gone.
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result_with_semantic_duplicate(tmp_path),
    )
    runner = CliRunner()

    table_result = runner.invoke(cli.cli, ["check", str(path), "--show-all"])
    assert "Filtered raw duplicates" not in table_result.output
    assert "raw duplicate pairs" not in table_result.output

    json_result = runner.invoke(cli.cli, ["check", str(path), "--json"])
    assert "filtered_raw_duplicates" not in json.loads(json_result.output)["summary"]


def test_cli_traditional_panel_label_is_language_neutral(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = _build_unit(tmp_path)
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=AnalysisResult(
            units=[unit],
            traditional_duplicates=[
                DuplicatePair(unit_a=unit, unit_b=unit, similarity=1.0, method="token_hash")
            ],
            semantic_duplicates=[],
            hybrid_duplicates=[],
            potentially_unused=[],
            analysis_mode="traditional",
        ),
    )

    result = CliRunner().invoke(cli.cli, ["check", str(path), "--traditional-only"])
    # Only Python parses to an AST here; the other backends fingerprint tokens.
    assert "Traditional Duplicates (Structural/Token/Jaccard)" in result.output
    assert "AST" not in result.output


def test_cli_full_table_disables_truncation(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = _build_unit(tmp_path)
    hybrid = HybridDuplicate(
        unit_a=unit,
        unit_b=unit,
        tier="exact",
        confidence=1.0,
        has_exact=True,
    )
    result_obj = AnalysisResult(
        units=[unit],
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[hybrid for _ in range(25)],
        potentially_unused=[],
        analysis_mode="combined",
    )
    patch_cli_analyzer(monkeypatch, cli, analyze_result=result_obj)

    runner = CliRunner()
    default_result = runner.invoke(cli.cli, ["check", str(path)])
    assert default_result.exit_code == 1
    assert "... and 5 more" in default_result.output

    full_result = runner.invoke(cli.cli, ["check", str(path), "--full-table"])
    assert full_result.exit_code == 1
    assert "... and 5 more" not in full_result.output


def test_cli_invalid_output_width(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--output-width", "60"])
    assert result.exit_code == 2
    assert "must be >= 80" in result.output


def test_cli_check_fails_on_semantic_backend_error_without_fallback(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def _dead():\n    return 1\n\ndef keep(y):\n    return y + 1\n")

    from codedupes import analyzer as analyzer_module

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", _raise_semantic_backend_error)

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--min-statements", "0"])
    assert result.exit_code == 1
    assert "Error during analysis" in result.output
    assert "--allow-semantic-fallback" in result.output
    # The wrapper must carry the root cause: --verbose is the only other route
    # to it and it is rejected with --json.
    assert "semantic backend mismatch" in result.output


def test_cli_check_degrades_on_semantic_backend_error_with_fallback(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def _dead():\n    return 1\n\ndef keep(y):\n    return y + 1\n")

    from codedupes import analyzer as analyzer_module

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", _raise_semantic_backend_error)

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--min-statements", "0", "--allow-semantic-fallback"],
    )
    assert result.exit_code == 0
    assert "Semantic analysis unavailable" in result.output


def test_cli_check_degrades_on_semantic_backend_error_in_json(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def _dead():\n    return 1\n\ndef keep(y):\n    return y + 1\n")

    from codedupes import analyzer as analyzer_module

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", _raise_semantic_backend_error)

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--min-statements", "0", "--allow-semantic-fallback", "--json"],
    )
    assert result.exit_code == 0

    assert result.output.lstrip().startswith("{"), (
        f"Expected pure JSON output, got: {result.output!r}"
    )
    payload = json.loads(result.output)
    assert payload["summary"]["exit_code"] == 0
    assert payload["summary"]["semantic_fallback"] is True
    assert payload["summary"]["semantic_fallback_reason"] is not None
    assert "Semantic analysis unavailable" in payload["summary"]["semantic_fallback_reason"]


@pytest.mark.parametrize(
    ("error", "expected_text"),
    [
        (RuntimeError("analysis exploded"), "Error during analysis: analysis exploded"),
        (GrammarUnavailableError("grammar missing"), "Parser unavailable: grammar missing"),
        (FileNotFoundError("Path does not exist"), "Error: Path does not exist"),
    ],
    ids=["runtime", "grammar", "missing-path"],
)
def test_cli_check_json_keeps_errors_off_stdout(monkeypatch, tmp_path, error, expected_text):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    def _raise() -> AnalysisResult:
        raise error

    patch_cli_analyzer(monkeypatch, cli, analyze_result=_raise)

    result = CliRunner().invoke(cli.cli, ["check", str(path), "--json"])

    assert result.exit_code == 1
    # --json promises machine-parseable JSON only on stdout.
    assert result.stdout == ""
    assert expected_text in result.stderr


def test_cli_check_log_output_goes_to_stderr(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    result = CliRunner().invoke(cli.cli, ["check", str(path), "--traditional-only"])

    assert "Extracting code units" in result.stderr
    assert "Extracting code units" not in result.stdout


def test_cli_check_verbose_traceback_goes_to_stderr(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    def _raise() -> AnalysisResult:
        raise RuntimeError("analysis exploded")

    patch_cli_analyzer(monkeypatch, cli, analyze_result=_raise)

    result = CliRunner().invoke(cli.cli, ["check", str(path), "--verbose"])

    assert result.exit_code == 1
    assert "Traceback" in result.stderr
    assert "Traceback" not in result.stdout


@pytest.mark.parametrize(
    ("args", "expected_message"),
    [
        (["check", "--semantic-only", "--min-statements", "0"], "Error during analysis"),
        (["search", "entry"], "Error during search"),
    ],
)
def test_cli_semantic_required_modes_fail_on_semantic_backend_error(
    monkeypatch, tmp_path, args, expected_message
):
    path = tmp_path / "sample.py"
    path.write_text("def entry(x):\n    return x + 1\n")

    from codedupes import analyzer as analyzer_module

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", _raise_semantic_backend_error)
    # `search` builds its corpus through index()/compute_embeddings, not the
    # duplicate-mining entry point.
    monkeypatch.setattr(analyzer_module, "compute_embeddings", _raise_semantic_backend_error)

    runner = CliRunner()
    result = runner.invoke(cli.cli, [args[0], str(path), *args[1:]])
    assert result.exit_code == 1
    assert expected_message in result.output


def test_cli_combined_exit_code_ignores_raw_filtered_findings(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = _build_unit(tmp_path)
    duplicate = DuplicatePair(unit_a=unit, unit_b=unit, similarity=1.0, method="jaccard")
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=AnalysisResult(
            units=[unit],
            traditional_duplicates=[duplicate],
            semantic_duplicates=[],
            hybrid_duplicates=[],
            potentially_unused=[],
            analysis_mode="traditional",
        ),
    )

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path)])
    assert result.exit_code == 0


@pytest.mark.parametrize(
    (
        "policy",
        "combined_mode",
        "strict_unused",
        "tier",
        "raw_duplicate",
        "include_unused",
        "expected",
    ),
    [
        ("actionable", True, False, "semantic_review", False, True, False),
        ("all", True, False, "semantic_review", False, True, True),
        ("none", True, True, "hybrid_confirmed", False, True, False),
        ("actionable", True, True, None, False, True, True),
        ("actionable", True, False, "hybrid_confirmed", False, False, True),
        ("actionable", True, False, "semantic_high_confidence", False, False, False),
        ("actionable", False, False, None, True, False, True),
    ],
)
def test_run_should_fail_policy(
    tmp_path,
    policy,
    combined_mode,
    strict_unused,
    tier,
    raw_duplicate,
    include_unused,
    expected,
):
    unit = _build_unit(tmp_path)
    hybrid = (
        [
            HybridDuplicate(
                unit_a=unit,
                unit_b=unit,
                tier=tier,
                confidence=0.9,
            )
        ]
        if tier is not None
        else []
    )
    raw = (
        [DuplicatePair(unit_a=unit, unit_b=unit, similarity=0.9, method="semantic")]
        if raw_duplicate
        else []
    )
    result = AnalysisResult(
        units=[unit],
        traditional_duplicates=raw,
        semantic_duplicates=[],
        hybrid_duplicates=hybrid,
        potentially_unused=[unit] if include_unused else [],
        analysis_mode="combined" if combined_mode else "traditional",
    )

    assert (
        cli.run_should_fail(
            result,
            policy=policy,
            combined_mode=combined_mode,
            strict_unused=strict_unused,
        )
        is expected
    )


def test_cli_fail_on_all_and_none(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = _build_unit(tmp_path)
    result_obj = AnalysisResult(
        units=[unit],
        traditional_duplicates=[],
        semantic_duplicates=[],
        hybrid_duplicates=[
            HybridDuplicate(
                unit_a=unit,
                unit_b=unit,
                tier="semantic_review",
                confidence=0.8,
            )
        ],
        potentially_unused=[unit],
        analysis_mode="combined",
    )
    patch_cli_analyzer(monkeypatch, cli, analyze_result=result_obj)
    runner = CliRunner()

    default_result = runner.invoke(cli.cli, ["check", str(path)])
    all_result = runner.invoke(cli.cli, ["check", str(path), "--fail-on", "all", "--json"])
    none_result = runner.invoke(cli.cli, ["check", str(path), "--fail-on", "none"])

    assert default_result.exit_code == 0
    assert all_result.exit_code == 1
    assert none_result.exit_code == 0
    assert "Failure policy" in default_result.output
    assert "actionable" in default_result.output
    assert "Finding status" in default_result.output
    assert "pass (exit 0)" in default_result.output
    summary = json.loads(all_result.output)["summary"]
    assert summary["fail_on"] == "all"
    assert summary["exit_code"] == 1


def test_cli_semantic_only_uses_raw_findings_for_exit(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = _build_unit(tmp_path)
    duplicate = DuplicatePair(unit_a=unit, unit_b=unit, similarity=0.95, method="semantic")
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=AnalysisResult(
            units=[unit],
            traditional_duplicates=[],
            semantic_duplicates=[duplicate],
            hybrid_duplicates=[],
            potentially_unused=[],
            analysis_mode="semantic",
        ),
    )

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path), "--semantic-only"])
    assert result.exit_code == 1
    assert "Semantic Duplicates (Embedding)" in result.output


def test_setup_logging_quiets_external_loggers() -> None:
    cli.setup_logging(verbose=False)
    for logger_name in NOISY_EXTERNAL_LOGGERS:
        assert logging.getLogger(logger_name).level == logging.WARNING


def test_main_propagates_check_exit_code(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    patch_cli_analyzer(monkeypatch, cli, analyze_result=lambda: _build_result(tmp_path))
    monkeypatch.setattr(sys, "argv", ["codedupes", "check", str(path), "--json"])

    assert cli.main() == 1


@pytest.mark.parametrize(
    ("command", "tail_args", "mps_fallback_flag", "expected_mps_fallback"),
    [
        ("check", [], "--no-mps-fallback", False),
        ("search", ["entry"], "--no-mps-fallback", False),
        ("check", [], "--mps-fallback", True),
        ("search", ["entry"], "--mps-fallback", True),
    ],
)
def test_cli_device_controls_pass_through(
    monkeypatch,
    tmp_path,
    command,
    tail_args,
    mps_fallback_flag,
    expected_mps_fallback,
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        search_results=[(_build_unit(tmp_path), 0.99)],
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            command,
            str(path),
            *tail_args,
            "--device",
            "mps",
            mps_fallback_flag,
            "--mps-memory-fraction",
            "0.8",
        ],
    )

    expected_exit = 1 if command == "check" else 0
    assert result.exit_code == expected_exit, result.output
    assert captured[0].device == "mps"
    assert captured[0].mps_fallback is expected_mps_fallback
    assert captured[0].mps_memory_fraction == 0.8


def test_cli_rejects_unsafe_mps_memory_fraction(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--mps-memory-fraction", "0"],
    )

    assert result.exit_code == 2
    assert "must be finite and in the interval (0.0, 2.0]" in result.output


def test_cli_rejects_mps_memory_fraction_with_cpu_device(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            "check",
            str(path),
            "--device",
            "cpu",
            "--mps-memory-fraction",
            "0.8",
        ],
    )

    assert result.exit_code == 2
    assert "mps_memory_fraction requires device='mps' or device='auto'" in result.output


@pytest.mark.parametrize(
    ("extra_args", "expected_option"),
    [
        (["--device", "mps"], "--device"),
        (["--mps-fallback"], "--mps-fallback"),
        (["--no-mps-fallback"], "--no-mps-fallback"),
        (["--mps-memory-fraction", "0.8"], "--mps-memory-fraction"),
        (["--strict-revision-cache"], "--strict-revision-cache"),
    ],
)
def test_cli_rejects_device_controls_with_traditional_only(
    tmp_path,
    extra_args,
    expected_option,
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["check", str(path), "--traditional-only", *extra_args],
    )

    assert result.exit_code == 2
    assert f"Cannot use {expected_option}" in result.output


@pytest.mark.parametrize(
    ("command", "tail_args", "expected_exit_code"),
    [("check", [], 1), ("search", ["entry"], 0)],
)
def test_cli_no_cache_flag_disables_embedding_cache(
    monkeypatch,
    tmp_path,
    command,
    tail_args,
    expected_exit_code,
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        search_results=[(_build_unit(tmp_path), 0.9)],
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, [command, str(path), *tail_args, "--no-cache"])

    assert result.exit_code == expected_exit_code
    assert captured[0].embedding_cache is False


def test_cli_traditional_only_accepts_no_cache_as_noop(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
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
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path)])

    assert result.exit_code == 1
    assert captured[0].embedding_cache is True


@pytest.mark.parametrize(
    ("command", "tail_args", "expected_exit_code"),
    [("check", [], 1), ("search", ["entry"], 0)],
)
def test_cli_strict_revision_cache_flag_plumbs_to_config(
    monkeypatch,
    tmp_path,
    command,
    tail_args,
    expected_exit_code,
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        search_results=[(_build_unit(tmp_path), 0.9)],
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, [command, str(path), *tail_args, "--strict-revision-cache"])

    assert result.exit_code == expected_exit_code
    assert captured[0].strict_revision_cache is True


def test_cli_defaults_to_loose_revision_cache(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: _build_result(tmp_path),
        captured_configs=captured,
    )
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["check", str(path)])

    assert result.exit_code == 1
    assert captured[0].strict_revision_cache is False


@pytest.mark.parametrize("command", ["check", "search"])
def test_cli_help_documents_strict_revision_cache_flag(command):
    result = CliRunner().invoke(cli.cli, [command, "--help"])

    assert result.exit_code == 0
    assert "--strict-revision-cache" in result.output


def test_cli_cache_info_reports_empty_cache():
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cache", "info"])

    assert result.exit_code == 0
    assert "Cache path:" in result.output
    assert "Entries: 0" in result.output


def test_cli_cache_info_reports_populated_cache(tmp_path):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    cache.put_many(scope, "some/model", "rev1", [("k1", np.array([1.0, 2.0], dtype=np.float32))])

    runner = CliRunner()
    result = runner.invoke(cli.cli, ["cache", "info"])

    assert result.exit_code == 0
    assert "Entries: 1" in result.output
    assert "some/model: 1" in result.output


def test_cli_cache_info_errors_when_cache_construction_fails(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("no home directory")

    monkeypatch.setattr(cli, "EmbeddingCache", _raise)

    result = CliRunner().invoke(cli.cli, ["cache", "info"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Cache unavailable: no home directory" in result.stderr


def test_cli_info_survives_cache_construction_failure(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("no home directory")

    monkeypatch.setattr(cli, "EmbeddingCache", _raise)

    result = CliRunner().invoke(cli.cli, ["info"])

    assert result.exit_code == 0
    assert "unavailable: no home directory" in result.output
    assert "Run with --help for CLI usage" in result.output


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


@pytest.mark.parametrize(("command", "expected_exit_code"), [("check", 1), ("search", 0)])
@pytest.mark.parametrize("include_tests", [False, True])
def test_cli_exclusions_extend_defaults(
    monkeypatch, tmp_path, command, expected_exit_code, include_tests
):
    from codedupes.extractor import CodeExtractor

    for relative in ["keep.py", "test_entry.py", "pkg/examples/deep.py", "node_modules/mod.py"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def entry():\n    return 1\n", encoding="utf-8")
    captured = []
    patch_cli_analyzer(
        monkeypatch, cli, analyze_result=_build_result(tmp_path), captured_configs=captured
    )
    args = [command, str(tmp_path)] + (["entry"] if command == "search" else ["--traditional-only"])
    args += ["--exclude", "examples", "--json"]
    if include_tests:
        args.append("--no-default-excludes")
    result = CliRunner().invoke(cli.cli, args)
    assert result.exit_code == expected_exit_code, result.output
    units = CodeExtractor(tmp_path, exclude_patterns=captured[0].exclude_patterns).extract_all()
    assert {unit.file_path.name for unit in units} == (
        {"keep.py", "test_entry.py"} if include_tests else {"keep.py"}
    )


@pytest.mark.parametrize(("command", "expected_exit_code"), [("check", 1), ("search", 0)])
def test_cli_implicit_default_exclusions_preserve_analyzer_default(
    monkeypatch, tmp_path, command, expected_exit_code
):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n", encoding="utf-8")
    captured = []
    patch_cli_analyzer(
        monkeypatch, cli, analyze_result=_build_result(tmp_path), captured_configs=captured
    )

    args = [command, str(path)]
    if command == "check":
        args.append("--traditional-only")
    else:
        args.append("entry")
    result = CliRunner().invoke(cli.cli, args)

    assert result.exit_code == expected_exit_code, result.output
    assert captured[0].exclude_patterns is None


@pytest.mark.parametrize("command", ["check", "search"])
@pytest.mark.parametrize("outside_root", [False, True])
@pytest.mark.parametrize("symlinked_parent", [False, True])
def test_cli_explicit_symlink_exclusions(
    monkeypatch, tmp_path, command, outside_root, symlinked_parent
):
    root = tmp_path / "project"
    root.mkdir()
    target = (tmp_path if outside_root else root) / "target.py"
    target.write_text("def entry():\n    return 1\n", encoding="utf-8")
    alias = root / "test_alias.py"
    alias.symlink_to(target)
    if symlinked_parent:
        parent_alias = tmp_path / "project_link"
        parent_alias.symlink_to(root, target_is_directory=True)
        alias = parent_alias / alias.name
    model = CountingModel()
    _patch_get_model(monkeypatch, model)

    args = [command, str(alias), "--json"]
    if command == "check":
        args += ["--traditional-only", "--no-unused"]
        count_key = "total_units"
    else:
        args += ["entry", "--device", "cpu", "--min-statements", "0", "--threshold", "0"]
        count_key = "indexed_units"

    for options, expected_count in (
        ([], 0),
        (["--no-default-excludes"], 1),
        (["--no-default-excludes", "--exclude", alias.name], 0),
    ):
        result = CliRunner().invoke(cli.cli, [*args, *options])
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["summary"][count_key] == expected_count
        assert result.stderr == ""


@pytest.mark.grammar
@pytest.mark.parametrize("exclude", ["ignored.cpp", "examples"])
def test_cli_header_detection_ignores_excluded_symlink_targets(tmp_path, exclude):
    (tmp_path / "main.c").write_text("int main(void) { return 1; }", encoding="utf-8")
    (tmp_path / "header.h").write_text(
        "static inline int header(void) { return 2; }", encoding="utf-8"
    )
    target = tmp_path / "examples" / "ignored.cpp"
    target.parent.mkdir()
    target.write_text("", encoding="utf-8")
    (tmp_path / "alias.cpp").symlink_to(target)

    result = CliRunner().invoke(
        cli.cli,
        [
            "check",
            str(tmp_path),
            "--exclude",
            exclude,
            "--traditional-only",
            "--no-unused",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["summary"]["units_by_language"] == {"c": 2}
    assert payload["extraction_diagnostics"] == []


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
    assert "does not exist" in result.stderr
    assert "without --model" in result.stderr


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
