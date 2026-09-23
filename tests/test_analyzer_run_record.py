"""Resolved run-record provenance and derived per-check status."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from codedupes import analyzer as analyzer_module
from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from codedupes.extractor import DEFAULT_EXCLUDE_PATTERNS
from codedupes.models import AnalysisMode, CodeUnitType, UnitCounts
from tests.analyzer_helpers import embedding_identity_from_kwargs, make_semantic_runner
from tests.conftest import create_project, make_code_unit


@pytest.mark.parametrize("mode", ["combined", "traditional", "semantic", "unused"])
def test_run_record_captures_resolved_settings(
    tmp_path: Path, monkeypatch, mode: AnalysisMode
) -> None:
    source = "def entry(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)
    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", make_semantic_runner())

    run_traditional = mode in {"combined", "traditional"}
    run_semantic = mode in {"combined", "semantic"}
    config = AnalyzerConfig(
        run_traditional=run_traditional,
        run_semantic=run_semantic,
        run_unused=True,
        **({"min_semantic_statements": 0} if run_semantic else {}),
    )
    analyzer = CodeAnalyzer(config)
    result = analyzer.analyze(project)

    run = result.run
    assert run.analysis_mode == mode
    assert result.analysis_mode == mode
    assert analyzer.run_record is run
    assert run.target == project.resolve()
    assert run.root == project.resolve()
    assert run.units.extracted == len(result.units)
    assert run.exclude_patterns == tuple(DEFAULT_EXCLUDE_PATTERNS)
    assert (run.traditional is not None) == run_traditional
    assert (run.semantic is not None) == run_semantic
    assert run.unused is not None

    if run_traditional:
        assert run.traditional.jaccard_threshold == config.jaccard_threshold
        assert run.traditional.tiny_filter == config.filter_tiny_traditional
    else:
        assert run.traditional is None

    if run_semantic:
        assert run.semantic.requested_model == config.model_name
        assert run.semantic.task == "semantic-similarity"
        assert run.semantic.unit_types == config.semantic_unit_types
    else:
        assert run.semantic is None

    assert run.unused.strict == config.strict_unused
    assert run.unused.files >= 1


def test_index_run_record_tracks_semantic_work_with_check_config(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "entry.py"
    source.write_text("def entry():\n    first = 1\n    second = first + 1\n    return second\n")

    def fake_compute_embeddings(units, **kwargs):
        return np.zeros((len(units), 2), dtype=np.float32), embedding_identity_from_kwargs(kwargs)

    monkeypatch.setattr(analyzer_module, "compute_embeddings", fake_compute_embeddings)
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_semantic=False,
            run_traditional=True,
            run_unused=False,
        )
    )

    assert analyzer.index(source) == 1
    assert analyzer.run_record.analysis_mode == "semantic"
    assert analyzer.run_record.semantic is not None
    assert analyzer.run_record.semantic.task == "code-retrieval"


def test_unit_counts_break_down_by_language_and_every_unit_type(tmp_path: Path) -> None:
    function = make_code_unit(tmp_path, name="helper", source="def helper():\n    pass\n")
    method = make_code_unit(
        tmp_path, name="run", source="def run(self):\n    pass\n", unit_type=CodeUnitType.METHOD
    )
    rust = replace(function, language="rust", name="parse")

    counts = UnitCounts.from_units([function, method, rust], semantic_eligible=2)

    assert counts.extracted == 3
    assert counts.semantic_eligible == 2
    # Languages are open-ended, so only those present appear; the unit-type enum is
    # closed, so every type appears and an absent one counts zero.
    assert counts.by_language == {"python": 2, "rust": 1}
    assert counts.by_type == {"class": 0, "function": 2, "method": 1}
    assert list(counts.by_language) == sorted(counts.by_language)
    assert list(counts.by_type) == sorted(counts.by_type)


def test_run_record_unit_counts_agree_with_the_extracted_units(tmp_path: Path) -> None:
    source = (
        "class Service:\n"
        "    def handle(self, x):\n"
        "        return x + 1\n"
        "\n"
        "\n"
        "def entry(x):\n"
        "    return Service().handle(x)\n"
    )
    project = create_project(tmp_path, source)
    result = CodeAnalyzer(
        AnalyzerConfig(run_traditional=True, run_semantic=False, run_unused=True)
    ).analyze(project)

    units = result.run.units
    assert units.extracted == len(result.units) == 3
    assert units.by_type == {"class": 1, "function": 1, "method": 1}
    assert units.by_language == {"python": 3}
    assert sum(units.by_type.values()) == sum(units.by_language.values()) == units.extracted


def test_run_record_marks_semantic_fallback_as_a_partial_check(tmp_path: Path, monkeypatch) -> None:
    project = create_project(tmp_path, "def entry(x):\n    return x + 1\n")
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(error=RuntimeError("backend unavailable")),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            allow_semantic_fallback=True,
            run_unused=False,
            min_semantic_statements=0,
            filter_tiny_traditional=False,
        )
    )
    result = analyzer.analyze(project)

    assert result.semantic_fallback is True
    assert result.checks.semantic.status == "fallback"
    assert result.checks.extraction.status == "completed"
    assert result.analysis_status == "partial"
    assert "semantic analysis fell back to traditional results" in result.checks.incomplete_reasons


def test_checks_extraction_is_partial_for_parse_errors_but_not_for_advisory_notices(
    tmp_path: Path,
) -> None:
    broken_root = tmp_path / "broken"
    broken_root.mkdir()
    (broken_root / "good.py").write_text("def entry():\n    return 1\n")
    (broken_root / "bad.py").write_text("def broken(\n")

    broken_result = CodeAnalyzer(AnalyzerConfig(run_semantic=False, run_unused=False)).analyze(
        broken_root
    )

    assert any(d.code == "partial-parse" for d in broken_result.extraction_diagnostics)
    assert broken_result.checks.extraction.status == "partial"
    assert broken_result.analysis_status == "partial"
    assert broken_result.checks.extraction.files_failed == 1

    advisory_root = tmp_path / "advisory"
    advisory_root.mkdir()
    (advisory_root / "good.py").write_text("def entry():\n    return 1\n")
    (advisory_root / "legacy.h").write_text("int legacy(void);\n")

    advisory_result = CodeAnalyzer(AnalyzerConfig(run_semantic=False, run_unused=False)).analyze(
        advisory_root
    )

    assert any(d.code == "c-header-policy" for d in advisory_result.extraction_diagnostics)
    assert advisory_result.checks.extraction.status == "completed"
    assert advisory_result.analysis_status == "complete"
