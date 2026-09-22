"""Fixture-corpus regression check: equal token hashes imply equal structural hashes."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from codedupes.extractor import CodeExtractor
from codedupes.models import CodeUnitType

pytestmark = pytest.mark.grammar


_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "test_fixtures"


@pytest.mark.parametrize(
    "relative_root",
    ["calibration/ledger", "exact_family", "search_probes"],
)
def test_python_fixture_roots_compile(relative_root: str) -> None:
    """tree-sitter accepts input CPython rejects, so fixtures need their own ``ast.parse`` check.

    A tree-sitter-python parse can come back with no ``ERROR`` node for source
    that is not actually valid Python (see ``docs/polyglot-languages.md``), so
    extraction cannot catch an invalid fixture on its own. This guards every
    Python fixture root against silently regressing into invalid syntax.
    """
    for path in sorted((_FIXTURE_ROOT / relative_root).rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize(
    ("relative_root", "language", "expects_token_group"),
    [
        ("calibration/ledger", "python", False),
        ("exact_family", "python", True),
        ("cowsay_wasm/src", "rust", True),
        ("calibration/c_metering", "c", False),
        ("calibration/javascript", "javascript", False),
        ("calibration/harbor-ts", "typescript", False),
    ],
)
def test_equal_token_hashes_imply_equal_structural_hashes_on_fixtures(
    relative_root: str, language: str, expects_token_group: bool
) -> None:
    """Report families label a token-identical group ``token_hash``; that label is only meaningful while token equality never crosses a structural boundary.

    This is a corpus regression check, not a backend theorem: Python
    indentation changes the tree without changing the token stream, so a
    ``return`` inside versus after an ``if`` block is token-equal and
    structurally different. Grouping is therefore done per fingerprint, and
    this test guards the fixtures the family tests rely on.
    """
    extractor = CodeExtractor(
        _FIXTURE_ROOT / relative_root,
        exclude_patterns=[],
        include_private=True,
        languages=(language,),
    )
    units = extractor.extract_all()

    assert units
    assert [d for d in extractor.diagnostics if d.severity == "error"] == []
    members: dict[tuple[str, str, str], int] = {}
    structural: dict[tuple[str, str, str], set[str]] = {}
    for unit in units:
        assert unit.token_hash and unit.structural_hash
        kind = (
            "callable"
            if unit.unit_type in (CodeUnitType.FUNCTION, CodeUnitType.METHOD)
            else "class"
        )
        key = (unit.language, kind, unit.token_hash)
        members[key] = members.get(key, 0) + 1
        structural.setdefault(key, set()).add(unit.structural_hash)

    assert all(len(hashes) == 1 for hashes in structural.values())
    # The check is only non-vacuous where a token group has two or more members.
    assert any(count >= 2 for count in members.values()) is expects_token_group
