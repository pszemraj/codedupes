from __future__ import annotations

import logging
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from textwrap import dedent

import pytest

from codedupes import traditional as traditional_module
from codedupes.models import CodeUnit, CodeUnitType
from codedupes.traditional import (
    _block_kind,
    build_reference_graph,
    extract_identifiers,
    find_near_duplicates_jaccard,
    find_potentially_unused,
    jaccard_similarity,
    run_traditional_analysis,
    unit_identifier_set,
)
from tests.conftest import extract_units


def test_skipped_unused_analysis_does_not_log_a_zero_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Do not report an unused result for a phase that did not compute one."""
    units = extract_units(tmp_path, "def example():\n    return 1", include_private=True)

    with caplog.at_level(logging.INFO, logger="codedupes.traditional"):
        run_traditional_analysis(units, compute_unused=False)

    assert "potentially unused" not in caplog.text


def test_exact_duplicates_via_ast_hash(tmp_path: Path) -> None:
    source = dedent(
        """
        def foo(a, b):
            return a + b

        def bar(x, y):
            return x + y
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)

    exact, near, _ = run_traditional_analysis(units, jaccard_threshold=0.85)

    assert len(exact) == 1
    assert len(near) == 0
    methods = {pair.method for pair in exact}
    assert methods == {"ast_hash"}


def test_exact_duplicates_across_function_and_method(tmp_path: Path) -> None:
    source = dedent(
        """
        def render_summary(rows, limit, header):
            lines = [header]
            for row in rows[:limit]:
                lines.append(str(row))
            return "\\n".join(lines)

        class Report:
            @staticmethod
            def render_summary(rows, limit, header):
                lines = [header]
                for row in rows[:limit]:
                    lines.append(str(row))
                return "\\n".join(lines)
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)

    exact, _near, _ = run_traditional_analysis(units, jaccard_threshold=0.85)

    # A function copied verbatim into a class body must stay visible to exact
    # detection: functions and methods share a blocking kind, matching semantic
    # pairing.
    pairs = {
        tuple(sorted((pair.unit_a.qualified_name, pair.unit_b.qualified_name))) for pair in exact
    }
    assert ("sample.Report.render_summary", "sample.render_summary") in pairs


def test_near_duplicates_across_function_and_method(tmp_path: Path) -> None:
    source = dedent(
        """
        def collect_totals(entries, bucket, scale):
            totals = {}
            for entry in entries:
                totals[entry.bucket] = totals.get(entry.bucket, 0) + entry.value * scale
            return totals

        class Aggregator:
            def collect_totals(self, entries, bucket, scale):
                totals = {}
                for entry in entries:
                    totals[entry.bucket] = totals.get(entry.bucket, 0) + entry.value * scale
                return totals
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)

    _exact, near, _ = run_traditional_analysis(units, jaccard_threshold=0.8)

    pairs = {
        tuple(sorted((pair.unit_a.qualified_name, pair.unit_b.qualified_name))) for pair in near
    }
    assert ("sample.Aggregator.collect_totals", "sample.collect_totals") in pairs


def test_near_duplicates_threshold_boundary(tmp_path: Path) -> None:
    source = dedent(
        """
        def first(a, b):
            return a + b + a

        def second(a, c):
            return a + c + c

        def third(a, b):
            return b + 2
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)

    exact_low, near_low, _ = run_traditional_analysis(units, jaccard_threshold=0.3)
    _exact_high, near_high, _ = run_traditional_analysis(units, jaccard_threshold=0.95)

    assert len(near_low) >= 1
    assert len(near_high) == 0
    assert len(exact_low) == 0


_COMMON_TOKENS = ("self", "value", "result", "config")
_RARE_TOKENS = tuple(f"sym_{index}" for index in range(200))


def _fake_unit(
    file_path: Path,
    index: int,
    tokens: set[str],
    language: str,
    unit_type: CodeUnitType,
    start_byte: int,
) -> CodeUnit:
    """Build one non-Python unit whose identifier set is taken verbatim.

    :param file_path: File all corpus units share, so range overlap can trigger.
    :param index: Corpus position, used to keep names and uids unique.
    :param tokens: Identifier set for the unit.
    :param language: Unit language.
    :param unit_type: Unit type.
    :param start_byte: Start of the unit's byte range.
    :return: Constructed code unit.
    """
    return CodeUnit(
        name=f"unit_{index}",
        qualified_name=f"sample.unit_{index}",
        unit_type=unit_type,
        file_path=file_path,
        lineno=index + 1,
        end_lineno=index + 1,
        source="",
        # A non-Python language keeps unit_identifier_set from reparsing source,
        # so empty sets stay empty.
        language=language,
        start_byte=start_byte,
        end_byte=start_byte + 50,
        identifiers=frozenset(tokens),
    )


def _random_corpus(rng: random.Random, file_path: Path, size: int) -> list[CodeUnit]:
    """Generate a seeded corpus mixing empty, skewed, and near-copied identifier sets.

    :param rng: Seeded random source.
    :param file_path: File all units share.
    :param size: Number of units to generate.
    :return: Synthetic code units.
    """
    languages = ("go", "rust")
    unit_types = (CodeUnitType.FUNCTION, CodeUnitType.METHOD, CodeUnitType.CLASS)

    units: list[CodeUnit] = []
    token_sets: list[set[str]] = []
    for index in range(size):
        if token_sets and rng.random() < 0.3:
            # Near-copies keep the high thresholds from testing empty results only.
            tokens = set(rng.choice(token_sets))
            for _ in range(rng.randint(0, 2)):
                if tokens and rng.random() < 0.5:
                    tokens.discard(rng.choice(sorted(tokens)))
                else:
                    tokens.add(rng.choice(_RARE_TOKENS))
        else:
            tokens = set(rng.sample(_RARE_TOKENS, rng.randint(0, 40)))
            if tokens:
                tokens.update(token for token in _COMMON_TOKENS if rng.random() < 0.8)
        token_sets.append(tokens)

        start_byte = index * 100
        if index and rng.random() < 0.1:
            # Straddle the previous unit so the source-overlap skip fires.
            start_byte = (index - 1) * 100 + 10
        units.append(
            _fake_unit(
                file_path,
                index,
                tokens,
                rng.choice(languages),
                rng.choice(unit_types),
                start_byte,
            )
        )
    return units


def _brute_force_near_duplicates(
    units: list[CodeUnit], threshold: float
) -> list[tuple[str, str, float]]:
    """Score every in-block pair directly, as the reference for the prefix-filtered join.

    :param units: Candidate units.
    :param threshold: Jaccard cutoff.
    :return: ``(uid_a, uid_b, similarity)`` triples in report order.
    """
    identifier_sets = {unit.uid: unit_identifier_set(unit) for unit in units}
    groups: dict[tuple[str, str], list[CodeUnit]] = defaultdict(list)
    for unit in units:
        groups[(unit.language, _block_kind(unit.unit_type))].append(unit)

    duplicates: list[tuple[str, str, float]] = []
    for group in groups.values():
        for a, b in combinations(group, 2):
            if a.overlaps(b):
                continue
            set_a = identifier_sets[a.uid]
            set_b = identifier_sets[b.uid]
            if not set_a or not set_b:
                continue
            size_ratio = min(len(set_a), len(set_b)) / max(len(set_a), len(set_b), 1)
            if size_ratio < threshold / 2:
                continue
            sim = jaccard_similarity(set_a, set_b)
            if sim >= threshold:
                duplicates.append((a.uid, b.uid, sim))
    return duplicates


@pytest.mark.parametrize("threshold", [0.0, 0.5, 0.8, 0.85, 0.93, 1.0])
@pytest.mark.parametrize("seed", [1, 7, 13])
def test_jaccard_join_matches_brute_force(tmp_path: Path, seed: int, threshold: float) -> None:
    units = _random_corpus(random.Random(seed), tmp_path / "corpus.go", 300)

    expected = _brute_force_near_duplicates(units, threshold)
    actual = [
        (pair.unit_a.uid, pair.unit_b.uid, pair.similarity)
        for pair in find_near_duplicates_jaccard(units, threshold=threshold)
    ]

    # Pairs, scores, and report order must all survive candidate pruning.
    assert actual == expected
    assert expected, "corpus produced no duplicates; the comparison would be vacuous"


def test_alias_aware_reference_graph(tmp_path: Path) -> None:
    source = dedent(
        """
        def helper(value):
            return value

        alias = helper

        def caller(value):
            return alias(value)

        def dead():
            return 0
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=False)
    build_reference_graph(units)

    unused = find_potentially_unused(units, strict_unused=True)
    names = {unit.name for unit in unused}

    assert "helper" not in names
    assert "caller" in names
    assert "dead" in names


def test_public_function_is_skipped_by_default(tmp_path: Path) -> None:
    source = dedent(
        """
        def public_function():
            return 1

        def _private_function():
            return 2

        def _unused_private():
            return _private_function() + public_function()
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    unused = find_potentially_unused(units, strict_unused=False)

    names = {unit.name for unit in unused}
    assert "public_function" not in names
    assert "_private_function" in names


def test_noqa_and_main_block_mark_as_used(tmp_path: Path) -> None:
    source = dedent(
        """
        def ignored_unused():  # noqa: codedupes
            return 42

        def used_by_main():
            return 7

        if __name__ == "__main__":
            used_by_main()
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    build_reference_graph(units, project_root=tmp_path)
    unused = find_potentially_unused(units, strict_unused=True)
    names = {unit.name for unit in unused}

    assert "ignored_unused" not in names
    assert "used_by_main" not in names


def test_main_block_references_survive_a_bom(tmp_path: Path) -> None:
    from codedupes.extractor import CodeExtractor

    source = dedent(
        """
        def used_by_main():
            return 7

        if __name__ == "__main__":
            used_by_main()
        """
    ).strip()
    path = tmp_path / "bom_sample.py"
    path.write_bytes(b"\xef\xbb\xbf" + source.encode("utf-8"))

    units = list(CodeExtractor(tmp_path, include_private=True).extract_from_file(path))
    build_reference_graph(units, project_root=tmp_path)
    unused = find_potentially_unused(units, strict_unused=True)

    assert "used_by_main" not in {unit.name for unit in unused}


def test_pyproject_entry_points_mark_as_used(tmp_path: Path) -> None:
    source = dedent(
        """
        def cli_entry():
            return 1

        def helper():
            return 2
        """
    ).strip()
    (tmp_path / "pyproject.toml").write_text(
        dedent(
            """
            [project]
            name = "sample"
            scripts = { sample-cli = "sample_module:cli_entry" }
            """
        ).strip()
    )
    project = tmp_path / "src"
    project.mkdir()
    (project / "__init__.py").write_text("")
    (project / "sample_module.py").write_text(source)
    extractor_file = project / "sample_module.py"

    from codedupes.extractor import CodeExtractor

    units = list(CodeExtractor(project).extract_from_file(extractor_file))
    assert len(units) == 2
    build_reference_graph(units, project_root=tmp_path)
    unused = find_potentially_unused(units, strict_unused=True)
    names = {unit.name for unit in unused}
    assert "cli_entry" not in names
    assert "helper" in names


def test_main_block_calls_are_parsed_once_per_file(tmp_path: Path, monkeypatch) -> None:
    source = dedent(
        """
        def first():
            return 1

        def second():
            return 2

        if __name__ == "__main__":
            first()
        """
    ).strip()
    units = extract_units(tmp_path, source, include_private=True)
    calls: list[Path] = []

    def fake_extract_main_block_calls(path: Path) -> set[str]:
        calls.append(path)
        return {"first"}

    monkeypatch.setattr(
        traditional_module,
        "_extract_main_block_calls",
        fake_extract_main_block_calls,
    )

    build_reference_graph(units)

    assert len(calls) == 1


def test_extract_identifiers_filters_builtin_names() -> None:
    source = dedent(
        """
        def helper(items):
            total = len(items)
            print(total)
            return sorted(items)
        """
    ).strip()

    identifiers = extract_identifiers(source)

    assert {"helper", "items", "total"} <= identifiers
    assert not {"len", "print", "sorted"} & identifiers
