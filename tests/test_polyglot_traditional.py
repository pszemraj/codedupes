"""Language-neutral duplicate-engine contract tests."""

from __future__ import annotations

from pathlib import Path

from codedupes.models import CodeUnit, CodeUnitType
from codedupes.traditional import (
    find_near_duplicates_jaccard,
    find_potentially_unused,
    run_traditional_analysis,
)


def _unit(
    tmp_path: Path,
    *,
    name: str,
    language: str,
    start_byte: int,
    end_byte: int,
    structural_hash: str | None = None,
    token_hash: str | None = None,
    identifiers: frozenset[str] = frozenset(),
    unit_type: CodeUnitType = CodeUnitType.FUNCTION,
) -> CodeUnit:
    suffix = {
        "python": ".py",
        "c": ".c",
        "rust": ".rs",
        "javascript": ".js",
        "typescript": ".ts",
    }[language]
    return CodeUnit(
        name=name,
        qualified_name=f"sample.{name}",
        unit_type=unit_type,
        file_path=tmp_path / f"sample{suffix}",
        lineno=start_byte + 1,
        end_lineno=end_byte + 1,
        source=f"{name} body",
        language=language,
        dialect=language,
        start_byte=start_byte,
        end_byte=end_byte,
        structural_hash=structural_hash,
        token_hash=token_hash,
        identifiers=identifiers,
    )


def test_exact_hashes_never_cross_language_boundaries(tmp_path: Path) -> None:
    units = [
        _unit(
            tmp_path,
            name="c_impl",
            language="c",
            start_byte=0,
            end_byte=10,
            structural_hash="same",
            token_hash="same-token",
        ),
        _unit(
            tmp_path,
            name="rust_impl",
            language="rust",
            start_byte=20,
            end_byte=30,
            structural_hash="same",
            token_hash="same-token",
        ),
    ]

    exact, near, _unused = run_traditional_analysis(
        units,
        jaccard_threshold=0.8,
        compute_unused=False,
    )

    assert exact == []
    assert near == []


def test_exact_hashes_still_match_within_one_language(tmp_path: Path) -> None:
    units = [
        _unit(
            tmp_path,
            name="first",
            language="rust",
            start_byte=0,
            end_byte=10,
            structural_hash="same",
        ),
        _unit(
            tmp_path,
            name="second",
            language="rust",
            start_byte=20,
            end_byte=30,
            structural_hash="same",
        ),
    ]

    exact, _near, _unused = run_traditional_analysis(
        units,
        jaccard_threshold=0.8,
        compute_unused=False,
    )

    assert len(exact) == 1
    assert exact[0].method == "ast_hash"


def test_overlapping_nested_units_are_not_reported_as_exact(tmp_path: Path) -> None:
    units = [
        _unit(
            tmp_path,
            name="outer",
            language="javascript",
            start_byte=0,
            end_byte=100,
            structural_hash="same",
        ),
        _unit(
            tmp_path,
            name="inner",
            language="javascript",
            start_byte=20,
            end_byte=40,
            structural_hash="same",
        ),
    ]

    exact, _near, _unused = run_traditional_analysis(
        units,
        jaccard_threshold=0.8,
        compute_unused=False,
    )

    assert exact == []


def test_identifier_jaccard_is_blocked_by_language_and_unit_type(tmp_path: Path) -> None:
    identifiers = frozenset({"request", "retry", "delay"})
    units = [
        _unit(
            tmp_path,
            name="js_fn",
            language="javascript",
            start_byte=0,
            end_byte=10,
            identifiers=identifiers,
        ),
        _unit(
            tmp_path,
            name="ts_fn",
            language="typescript",
            start_byte=20,
            end_byte=30,
            identifiers=identifiers,
        ),
        _unit(
            tmp_path,
            name="js_class",
            language="javascript",
            start_byte=40,
            end_byte=50,
            identifiers=identifiers,
            unit_type=CodeUnitType.CLASS,
        ),
    ]

    assert find_near_duplicates_jaccard(units, threshold=1.0) == []


def test_unused_analysis_is_explicitly_python_only(tmp_path: Path) -> None:
    rust = _unit(
        tmp_path,
        name="dead_rust",
        language="rust",
        start_byte=0,
        end_byte=10,
    )
    python = _unit(
        tmp_path,
        name="dead_python",
        language="python",
        start_byte=20,
        end_byte=30,
    )

    unused = find_potentially_unused([rust, python], strict_unused=True)

    assert [unit.name for unit in unused] == ["dead_python"]
