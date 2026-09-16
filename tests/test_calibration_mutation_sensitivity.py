"""Prove the ledger behavior suite catches the pilot's critical contract mutations."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE = Path("test_fixtures/calibration/ledger")

MUTATIONS = {
    "validation_after_void_filter": (
        "src/ledger/settlement.py",
        "if not isinstance(row.invoice_id, str) or not row.invoice_id.strip():",
        (
            "if not row.voided and (not isinstance(row.invoice_id, str) "
            "or not row.invoice_id.strip()):"
        ),
    ),
    "deferral_includes_boundary": (
        "src/ledger/planning.py",
        "if 0 < item.total_minor < minimum_payout_minor:",
        "if 0 < item.total_minor <= minimum_payout_minor:",
    ),
    "reserve_external_id_before_acceptance": (
        "src/ledger/imports.py",
        "for position, row in enumerate(rows):\n        try:",
        (
            "for position, row in enumerate(rows):\n"
            '        seen.add(str(row.get("external_id", "")).strip())\n'
            "        try:"
        ),
    ),
    "shared_normalizer_keeps_invoice_whitespace": (
        "src/ledger/imports.py",
        "external_id.strip(), invoice_id.strip(), int(amount), currency, posted_on, voided",
        "external_id.strip(), invoice_id, int(amount), currency, posted_on, voided",
    ),
}


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_behavior_suite_rejects_temporary_mutations(tmp_path: Path, mutation: str):
    copy = tmp_path / "ledger"
    shutil.copytree(FIXTURE, copy, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    relative, before, after = MUTATIONS[mutation]
    target = copy / relative
    source = target.read_text()
    assert source.count(before) >= 1
    target.write_text(source.replace(before, after, 1))
    environment = os.environ | {"PYTHONPATH": str(copy / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=copy,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0, (
        f"mutation survived: {mutation}\n{result.stdout}\n{result.stderr}"
    )
