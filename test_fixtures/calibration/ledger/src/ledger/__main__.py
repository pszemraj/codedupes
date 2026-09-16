"""Demonstrate the offline import, settlement, and audit application."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from .adapters import import_csv_file, ingest_api_batch
from .audit import (
    assess_credit_risk,
    audit_chronology,
    audit_currency_limits,
    audit_reference_usage,
)
from .imports import ingest_rows, prepare_import
from .models import Transaction
from .planning import plan_batch_with_deferrals, plan_settlements
from .settlement import build_invoice_report, summarize_import


def main() -> None:
    """Exercise supported import and reporting paths with a small local batch."""
    rows = [
        {"external_id": "r1", "invoice_id": " A ", "amount_minor": "75",
         "currency": "USD", "posted_on": "2024-02-29", "voided": "false"},
        {"external_id": "r2", "invoice_id": "A", "amount_minor": "-25",
         "currency": "USD", "posted_on": "2024-02-28", "voided": "false"},
    ]
    inline = ingest_rows(rows)
    prepared = prepare_import(rows)
    assert inline == prepared
    with TemporaryDirectory() as directory:
        path = Path(directory) / "transactions.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        file_report = import_csv_file(path)
    api_report = ingest_api_batch({"batch_id": "demo", "items": rows}, set())
    transactions = [Transaction(row.invoice_id, row.amount_minor, row.voided) for row in inline.accepted]
    summary = summarize_import(transactions)
    assert summary == build_invoice_report(transactions)
    print("imports", len(file_report["accepted"]), len(api_report["results"]))
    print("settlements", plan_settlements(summary), plan_batch_with_deferrals(summary, 100))
    print("audits", assess_credit_risk(summary, 40), audit_reference_usage(inline.accepted),
          audit_chronology(inline.accepted), audit_currency_limits(inline.accepted, {"USD": 40}))


if __name__ == "__main__":
    main()
