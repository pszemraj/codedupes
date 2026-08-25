"""Calibration-script contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from scripts.validate_calibration_corpus import _rejected_extraction_diagnostics, main

pytestmark = pytest.mark.grammar


def _write_corpus(corpus_path: Path) -> None:
    """Write a two-file Python corpus holding one exact clone pair and one near pair.

    :param Path corpus_path: Directory to populate.
    :return None: ``None``.
    """
    corpus_path.mkdir(parents=True, exist_ok=True)
    clone = (
        "def accumulate(values):\n"
        "    total = 0\n"
        "    for value in values:\n"
        "        total += value\n"
        "    return total\n"
    )
    (corpus_path / "alpha.py").write_text(clone)
    (corpus_path / "beta.py").write_text(clone)
    (corpus_path / "gamma.py").write_text(
        "def summarize(rows):\n"
        "    seen = []\n"
        "    for row in rows:\n"
        "        seen.append(row)\n"
        "    return len(seen)\n"
    )
    (corpus_path / "delta.py").write_text(
        "def tally(entries):\n"
        "    count = 0\n"
        "    while entries:\n"
        "        entries.pop()\n"
        "        count += 1\n"
        "    return count\n"
    )


def _run_validator(monkeypatch: pytest.MonkeyPatch, corpus_path: Path, labels_path: Path) -> int:
    """Invoke the validator entry point against one corpus/labels pair.

    :param pytest.MonkeyPatch monkeypatch: Fixture used to set ``sys.argv``.
    :param Path corpus_path: Corpus root.
    :param Path labels_path: Labels JSON path.
    :return int: Validator exit code.
    """
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_calibration_corpus.py",
            "--corpus-path",
            str(corpus_path),
            "--labels-path",
            str(labels_path),
            "--language",
            "python",
        ],
    )
    return main()


def test_validator_reports_every_failure_when_a_label_spec_cannot_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One renamed corpus symbol must not replace the failure report with a traceback."""
    corpus_path = tmp_path / "corpus"
    _write_corpus(corpus_path)
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "positive_groups": [
                    ["alpha.py::accumulate", "beta.py::accumulate"],
                    ["gamma.py::summarize", "delta.py::renamed_away"],
                ],
                "categories": {
                    "exact": [["alpha.py::accumulate", "beta.py::accumulate"]],
                    "near_rename": [["gamma.py::summarize", "delta.py::renamed_away"]],
                },
            }
        )
    )

    exit_code = _run_validator(monkeypatch, corpus_path, labels_path)

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL (" in output
    assert "matched 0 units" in output
    # The pre-existing per-category summary still prints alongside the failures.
    assert "exact: 1 labeled pairs" in output
    # The unresolved group IS in positive_groups; the resolution failure above is
    # the whole story, and a bogus partition complaint would point authors at the
    # labels file's categories map instead of the missing corpus symbol.
    assert "missing from positive_groups" not in output


def test_validator_flags_deterministic_pairs_missing_from_the_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An exact clone pair no label claims is an unmeasured decision, not a pass."""
    corpus_path = tmp_path / "corpus"
    _write_corpus(corpus_path)
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "positive_groups": [["gamma.py::summarize", "delta.py::tally"]],
                "categories": {
                    "near_rename": [["gamma.py::summarize", "delta.py::tally"]],
                },
            }
        )
    )

    exit_code = _run_validator(monkeypatch, corpus_path, labels_path)

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "unlabeled deterministic pair" in output
    assert "alpha.py::accumulate" in output
    assert "beta.py::accumulate" in output


def test_validator_counts_negative_controls_as_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The negative-control line reports groups and their pair expansion, not groups as pairs."""
    corpus_path = tmp_path / "corpus"
    _write_corpus(corpus_path)
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "positive_groups": [["alpha.py::accumulate", "beta.py::accumulate"]],
                "categories": {"exact": [["alpha.py::accumulate", "beta.py::accumulate"]]},
                "negative_controls": [
                    ["gamma.py::summarize", "delta.py::tally", "alpha.py::accumulate"]
                ],
            }
        )
    )

    _run_validator(monkeypatch, corpus_path, labels_path)

    assert "negative_controls: 1 groups (3 pairs)" in capsys.readouterr().out


def test_validator_flags_a_category_with_no_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty category list is a labeling mistake, not a vacuously covered category."""
    corpus_path = tmp_path / "corpus"
    _write_corpus(corpus_path)
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "positive_groups": [["alpha.py::accumulate", "beta.py::accumulate"]],
                "categories": {
                    "exact": [["alpha.py::accumulate", "beta.py::accumulate"]],
                    "near_rename": [],
                },
            }
        )
    )

    exit_code = _run_validator(monkeypatch, corpus_path, labels_path)

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "category 'near_rename' lists no positive groups" in output


def test_validator_ignores_filesystem_debris(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finder droppings and bytecode caches are not corpus files that must produce units.

    The zero-unit-file walk shares the sweep digest's exclusions: before that,
    one ``.DS_Store`` from browsing a language directory failed the whole corpus.
    """
    corpus_path = tmp_path / "corpus"
    _write_corpus(corpus_path)
    (corpus_path / ".DS_Store").write_bytes(b"\x00\x01Bud1")
    pycache_path = corpus_path / "__pycache__"
    pycache_path.mkdir()
    (pycache_path / "alpha.cpython-312.pyc").write_bytes(b"\x00")
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "positive_groups": [
                    ["alpha.py::accumulate", "beta.py::accumulate"],
                    ["gamma.py::summarize", "delta.py::tally"],
                ],
                "categories": {
                    "exact": [["alpha.py::accumulate", "beta.py::accumulate"]],
                    "near_rename": [["gamma.py::summarize", "delta.py::tally"]],
                },
            }
        )
    )

    exit_code = _run_validator(monkeypatch, corpus_path, labels_path)

    output = capsys.readouterr().out
    assert "produced zero units" not in output
    assert exit_code == 0


def test_validator_rejects_partial_parse_with_usable_units(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.rs"
    source_path.write_text(
        "pub fn valid() -> i32 { 1 }\nlet = ;\n",
        encoding="utf-8",
    )
    result = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=False,
            run_unused=False,
            languages=("rust",),
        )
    ).analyze(tmp_path)

    assert [unit.name for unit in result.units] == ["valid"]
    rejected = _rejected_extraction_diagnostics(result.extraction_diagnostics)
    assert [diagnostic.code for diagnostic in rejected] == ["partial-parse"]
