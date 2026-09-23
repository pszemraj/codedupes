"""Shared builders and runners for the ``codedupes`` CLI tests."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from codedupes.models import (
    AnalysisResult,
    CodeUnit,
    DuplicatePair,
    HybridDuplicate,
    UnitCounts,
)
from codedupes.semantic import EmbeddingRunStats
from tests.conftest import make_code_unit, make_run_record


def build_unit(tmp_path: Path) -> CodeUnit:
    return make_code_unit(tmp_path, name="entry", source="def entry():\n    return 1")


def build_copy(tmp_path: Path) -> CodeUnit:
    """Return a second unit that is an exact copy of :func:`build_unit`."""
    return make_code_unit(
        tmp_path, name="entry_copy", source="def entry_copy():\n    return 1", lineno=5
    )


def build_result(tmp_path: Path) -> AnalysisResult:
    """Combined result with one two-member exact family and one unused unit."""
    unit = build_unit(tmp_path)
    copy = build_copy(tmp_path)
    duplicate = DuplicatePair(
        unit_a=unit,
        unit_b=copy,
        similarity=1.0,
        method="structural_hash",
    )
    hybrid = HybridDuplicate(
        unit_a=unit,
        unit_b=copy,
        tier="exact",
        score=1.0,
        exact_method="structural_hash",
    )

    units = [unit, copy]
    return AnalysisResult(
        units=units,
        traditional_duplicates=[duplicate],
        semantic_duplicates=[],
        hybrid_duplicates=[hybrid],
        potentially_unused=[unit],
        run=make_run_record(
            tmp_path,
            mode="combined",
            units=UnitCounts.from_units(units, semantic_eligible=len(units)),
        ),
        embedding_stats=EmbeddingRunStats(
            requested_rows=1,
            unique_inputs=1,
            cache_hit_rows=1,
            model_loaded=False,
            cache_enabled=True,
            cache_revision="1" * 40,
        ),
    )


def build_result_with_semantic_duplicate(tmp_path: Path) -> AnalysisResult:
    result = build_result(tmp_path)
    unit = build_unit(tmp_path)
    result.semantic_duplicates = [
        DuplicatePair(unit_a=unit, unit_b=unit, similarity=0.95, method="semantic")
    ]
    return result


def run_cli_subprocess(
    args: list[str], setup: str = "", *, merge_stderr: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run the real CLI in a subprocess, optionally merging stderr into stdout."""
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
        stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        check=False,
    )
