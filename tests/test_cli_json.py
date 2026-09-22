"""``--json`` output: schema payloads and keeping stdout clean of backend noise."""

from __future__ import annotations

import json
import sys

import pytest
from click.testing import CliRunner

from codedupes import cli
from codedupes.models import (
    AnalysisResult,
    DuplicatePair,
    HybridDuplicate,
)
from tests.cli_helpers import build_copy, build_result, build_unit, run_cli_subprocess
from tests.conftest import patch_cli_analyzer


def test_cli_json_output_hybrid_default(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    captured = []
    patch_cli_analyzer(
        monkeypatch,
        cli,
        analyze_result=lambda: build_result(tmp_path),
        search_results=[(build_unit(tmp_path), 0.99)],
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
    assert output["summary"]["strict_unused"] is False
    assert output["summary"]["exit_code"] == 1
    assert output["summary"]["hidden_only_failure"] == []
    assert output["schema_version"] == 4
    assert output["summary"]["max_duplicates"] == 20
    assert output["summary"]["reported_duplicates"] == 1
    assert output["summary"]["omitted_review_duplicates"] == 0
    assert output["summary"]["actionable_duplicates"] == 1
    assert output["summary"]["reported_actionable_duplicates"] == 1
    assert output["summary"]["duplicates_by_tier"]["exact"] == 1
    assert output["summary"]["exact_family_members"] == 2
    assert output["summary"]["max_unused"] == 20
    assert output["summary"]["reported_unused"] == 1
    assert output["summary"]["truncated_unused"] == 0
    # The exact pair is one family record; the pairwise list holds no exact edge.
    assert output["exact_families"] == [
        {"method": "structural_hash", "members": ["u0", "u1"], "lines": 2, "redundant_lines": 2}
    ]
    assert output["duplicates"] == []
    assert output["potentially_unused"][0] in output["units"]
    assert output["units"]["u0"]["uid"] == build_unit(tmp_path).uid
    assert output["units"]["u1"]["uid"] == build_copy(tmp_path).uid
    assert len(output["units"]) == 2
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


@pytest.mark.parametrize("initially_disabled", [False, True])
def test_cli_json_restores_huggingface_progress_state(monkeypatch, tmp_path, initially_disabled):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    patch_cli_analyzer(monkeypatch, cli, analyze_result=lambda: build_result(tmp_path))
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


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_cli_json_discards_direct_backend_output_on_success(monkeypatch, tmp_path, stream):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")

    class NoisyAnalyzer:
        def __init__(self, _config):
            pass

        def analyze(self, _path):
            print("backend progress", file=getattr(sys, stream))
            return build_result(tmp_path)

    monkeypatch.setattr(cli, "CodeAnalyzer", NoisyAnalyzer)

    result = CliRunner().invoke(cli.cli, ["check", str(path), "--json"])

    assert result.exit_code == 1
    assert result.stderr == ""
    assert json.loads(result.output)["schema_version"] == 4


@pytest.mark.parametrize("command", ["check", "search"])
def test_cli_json_isolates_custom_family_warning_before_config(tmp_path, command):
    args = [command, str(tmp_path)]
    if command == "search":
        args.append("entry")
    result = run_cli_subprocess(
        [
            *args,
            "--model",
            "review/gte-modernbert-base",
            "--loose-revision-cache",
            "--json",
        ]
    )

    assert result.returncode == 0, result.stdout
    assert json.loads(result.stdout)["schema_version"] == 4


@pytest.mark.parametrize(
    ("command", "fail_on", "exit_code"),
    # Findings exit 1 must not replay backend noise as an operation failure.
    [("check", "actionable", 1), ("search", None, 0)],
)
@pytest.mark.parametrize("stream_fd", [1, 2])
def test_cli_json_isolates_backend_output_in_completed_report(
    tmp_path, command, fail_on, exit_code, stream_fd
):
    args = [command, str(tmp_path), "--json"]
    if command == "check":
        (tmp_path / "sample.py").write_text(
            "def first(value):\n    return value + 1\n\ndef second(value):\n    return value + 1\n"
        )
        args.extend(["--traditional-only", "--no-unused", "--no-tiny-filter", "--fail-on", fail_on])
    else:
        args.append("entry")
    result = run_cli_subprocess(
        args,
        f"""
        import ctypes
        import os
        import sys

        native_printf = ctypes.CDLL("ucrtbase" if os.name == "nt" else None).printf
        native_printf.argtypes = [ctypes.c_char_p]
        native_printf.restype = ctypes.c_int

        class NoisyAnalyzer(cli.CodeAnalyzer):
            def __init__(self, config):
                os.write({stream_fd}, b"native initialization diagnostic\\n")
                native_printf(b"buffered native initialization diagnostic")
                super().__init__(config)

            def analyze(self, path):
                print("Python analysis diagnostic", file=sys.{"stdout" if stream_fd == 1 else "stderr"})
                os.write({stream_fd}, b"native analysis diagnostic\\n")
                return super().analyze(path)

            def index(self, path):
                print("Python indexing diagnostic", file=sys.{"stdout" if stream_fd == 1 else "stderr"})
                os.write({stream_fd}, b"native indexing diagnostic\\n")
                return super().index(path)

            def search(self, *args, **kwargs):
                os.write({stream_fd}, b"native query diagnostic\\n")
                return super().search(*args, **kwargs)

        cli.CodeAnalyzer = NoisyAnalyzer
        """,
    )

    assert result.returncode == exit_code, result.stdout
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 4
    if command == "check":
        assert payload["exact_families"]
        assert payload["summary"]["exit_code"] == exit_code


@pytest.mark.parametrize("command", ["check", "search"])
@pytest.mark.parametrize("stream_fd", [1, 2])
def test_cli_json_replays_python_and_native_output_on_failure(tmp_path, command, stream_fd):
    args = [command, str(tmp_path), "--json"]
    if command == "search":
        args.append("entry")
    result = run_cli_subprocess(
        args,
        f"""
        import ctypes
        import os
        import sys

        native_printf = ctypes.CDLL("ucrtbase" if os.name == "nt" else None).printf
        native_printf.argtypes = [ctypes.c_char_p]
        native_printf.restype = ctypes.c_int

        class FailingAnalyzer(cli.CodeAnalyzer):
            def analyze(self, path):
                print("Python backend diagnostic", file=sys.{"stdout" if stream_fd == 1 else "stderr"})
                os.write({stream_fd}, b"native backend diagnostic\\n")
                native_printf(b"buffered native backend diagnostic")
                raise RuntimeError("backend exploded")

            index = analyze

        cli.CodeAnalyzer = FailingAnalyzer
        """,
        merge_stderr=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "Python backend diagnostic" in result.stderr
    assert "native backend diagnostic" in result.stderr
    error_label = "analysis" if command == "check" else "search"
    assert f"Error during {error_label}: backend exploded" in result.stderr
    assert "buffered native backend diagnostic" in result.stderr
    assert "schema_version" not in result.stderr


def test_cli_json_v4_raw_mode_uses_edge_list(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    unit = build_unit(tmp_path)
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
    assert payload["schema_version"] == 4
    assert payload["duplicates"] == [
        {
            "method": "semantic",
            "similarity": 0.95,
            "unit_a": "u0",
            "unit_b": "u0",
        }
    ]
    assert payload["units"]["u0"]["name"] == "entry"
    assert payload["units"]["u0"]["uid"] == unit.uid
    assert payload["exact_families"] == []
    assert set(payload["summary"]["duplicates_by_tier"].values()) == {0}
    assert "traditional_duplicates" not in payload
    assert "semantic_duplicates" not in payload


def test_cli_json_v4_emits_each_unit_once(monkeypatch, tmp_path):
    path = tmp_path / "sample.py"
    path.write_text("def entry():\n    return 1\n")
    result_obj = build_result(tmp_path)
    unit, copy = result_obj.units
    near = HybridDuplicate(
        unit, copy, "hybrid_confirmed", 0.9, jaccard_similarity=0.9, semantic_similarity=0.9
    )
    result_obj.hybrid_duplicates = [near] * 4
    patch_cli_analyzer(monkeypatch, cli, analyze_result=result_obj)

    result = CliRunner().invoke(cli.cli, ["check", str(path), "--json"])

    payload = json.loads(result.output)
    assert len(payload["duplicates"]) == 4
    assert len(payload["units"]) == 2
