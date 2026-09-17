"""Explicit oracles for settlement policies and import side effects."""

import copy
import csv
import io
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from ledger.adapters import import_csv_file, ingest_api_batch
from ledger.audit import (
    assess_credit_risk,
    audit_chronology,
    audit_currency_limits,
    audit_reference_usage,
)
from ledger.imports import ingest_rows, normalize_transaction, prepare_import
from ledger.models import InvoiceSummary
from ledger.planning import (
    plan_batch_with_deferrals,
    plan_settlements,
    plan_settlements_indexed,
)


def record(**changes):
    row = {
        "external_id": "r1",
        "invoice_id": " A ",
        "amount_minor": "-25",
        "currency": "USD",
        "posted_on": "2024-02-29",
        "voided": "false",
    }
    return row | changes


class PlanningTests(unittest.TestCase):
    def test_baseline_and_active_extension(self):
        rows = [
            InvoiceSummary("D", 0, 1),
            InvoiceSummary("C", -30, 2),
            InvoiceSummary("B", 100, 1),
            InvoiceSummary("A", 75, 1),
        ]
        before = tuple(rows)
        base = plan_settlements(rows)
        self.assertEqual(base, plan_settlements_indexed(rows))
        self.assertEqual(base, plan_batch_with_deferrals(rows, 0))
        self.assertEqual(base.transfers, (("A", 75), ("B", 100)))
        extended = plan_batch_with_deferrals(rows, 100)
        self.assertEqual(extended.transfers, (("B", 100),))
        self.assertEqual(
            extended.lines,
            (
                ("A", "deferred", 75),
                ("B", "ready", 100),
                ("C", "refund_due", -30),
                ("D", "balanced", 0),
            ),
        )
        self.assertEqual(extended.events, (("A", "below_minimum", 100),))
        self.assertEqual(extended.deferred_total, 75)
        self.assertEqual(tuple(rows), before)

    def test_validation_empty_and_thresholds(self):
        self.assertEqual(plan_settlements([]), plan_batch_with_deferrals([], 10))
        for minimum in [-1, True, 2.5]:
            with self.assertRaisesRegex(ValueError, "minimum_payout_minor"):
                plan_batch_with_deferrals([], minimum)
        invalid = [
            None,
            InvoiceSummary(" ", 1, 1),
            InvoiceSummary("A", True, 1),
            InvoiceSummary("A", 1, 0),
            InvoiceSummary("A", 1, True),
        ]
        for implementation in [
            plan_settlements,
            plan_settlements_indexed,
            plan_batch_with_deferrals,
        ]:
            for item in invalid:
                with self.assertRaises((ValueError, TypeError)):
                    implementation([item])
            with self.assertRaisesRegex(ValueError, "duplicate invoice_id"):
                implementation([InvoiceSummary("A", 1, 1)] * 2)


class ImportTests(unittest.TestCase):
    def test_duplicate_acceptance_and_date_bounds(self):
        rows = [
            record(amount_minor="bad"),
            record(),
            record(),
            record(external_id="r2", posted_on="2024-03-01", amount_minor="50"),
        ]
        before = copy.deepcopy(rows)
        for implementation in [ingest_rows, prepare_import]:
            result = implementation(rows)
            self.assertEqual([r.external_id for r in result.accepted], ["r1", "r2"])
            self.assertEqual(result.rejected, ((0, "amount_minor"), (2, "duplicate_id")))
            self.assertEqual(result.error_counts, (("amount_minor", 1), ("duplicate_id", 1)))
            self.assertEqual(result.date_bounds, (date(2024, 2, 29), date(2024, 3, 1)))
        self.assertEqual(rows, before)
        self.assertEqual(ingest_rows(rows), prepare_import(rows))

    def test_field_boundaries_and_first_error(self):
        cases = [
            ("external_id", " "),
            ("invoice_id", ""),
            ("amount_minor", True),
            ("amount_minor", "+1"),
            ("amount_minor", "1e2"),
            ("amount_minor", " 2"),
            ("amount_minor", "２"),
            ("amount_minor", "1.5"),
            ("currency", "usd"),
            ("posted_on", "2023-02-29"),
            ("posted_on", "2024-2-29"),
            ("voided", "yes"),
        ]
        for field, value in cases:
            row = record(**{field: value})
            for implementation in [ingest_rows, prepare_import]:
                self.assertEqual(implementation([row]).rejected, ((0, field),))
            with self.assertRaisesRegex(ValueError, field):
                normalize_transaction(row)
        self.assertEqual(
            ingest_rows([record(external_id="", amount_minor="bad")]).rejected,
            ((0, "external_id"),),
        )
        self.assertEqual(ingest_rows([]), prepare_import([]))
        for value in [True, False, "true", "false"]:
            self.assertEqual(
                ingest_rows([record(voided=value)]), prepare_import([record(voided=value)])
            )

    def test_csv_api_shared_region_and_distinct_reports(self):
        rows = [
            record(invoice_id=" A, 東京 "),
            record(external_id="r2", amount_minor="1.2"),
            record(external_id="r3", currency="XXX"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            report = import_csv_file(path)
        completed = set()
        api = ingest_api_batch({"batch_id": " b1 ", "items": rows}, completed)
        self.assertEqual(report["accepted"][0], api["results"][0][2])
        self.assertEqual(report["accepted"][0].invoice_id, "A, 東京")
        self.assertEqual(report["accepted"][0].amount_minor, -25)
        self.assertEqual(report["line_errors"], ((3, "amount_minor"), (4, "currency")))
        self.assertEqual(
            api["results"][1:], ((1, "rejected", "amount_minor"), (2, "rejected", "currency"))
        )
        self.assertEqual(completed, {"b1"})
        self.assertEqual(
            ingest_api_batch({"batch_id": "b1", "items": rows}, completed),
            {"batch_id": "b1", "replayed": True, "results": ()},
        )

    def test_file_closure_and_envelope_failure(self):
        for contents in [
            "wrong,headers\nx,y\n",
            "external_id,invoice_id,amount_minor,currency,posted_on,voided\n",
        ]:
            stream = io.StringIO(contents)
            with patch.object(Path, "open", return_value=stream):
                if contents.startswith("wrong"):
                    with self.assertRaisesRegex(ValueError, "headers"):
                        import_csv_file(Path("input.csv"))
                else:
                    self.assertEqual(import_csv_file(Path("input.csv"))["accepted"], ())
            self.assertTrue(stream.closed)
        completed = {"old"}
        for envelope in [None, {}, {"batch_id": "b", "items": ()}, {"batch_id": " ", "items": []}]:
            with self.assertRaises(ValueError):
                ingest_api_batch(envelope, completed)
            self.assertEqual(completed, {"old"})

    def test_adapters_enforce_acceptance_before_id_reservation(self):
        rows = [record(amount_minor="bad"), record(), record()]
        report = ingest_api_batch({"batch_id": "batch", "items": rows}, set())
        self.assertEqual(
            [item[1] for item in report["results"]], ["rejected", "accepted", "rejected"]
        )
        self.assertEqual(report["results"][2][2], "duplicate_id")

    def test_audit_witnesses(self):
        rows = [
            normalize_transaction(record(amount_minor="100")),
            normalize_transaction(record(external_id="r2", posted_on="2024-02-28")),
        ]
        self.assertEqual(audit_reference_usage(rows), (("A", ("r1", "r2")),))
        self.assertEqual(audit_chronology(rows), ((1, "r2", "2024-02-29"),))
        self.assertEqual(audit_currency_limits(rows, {"USD": 80}), ((0, "USD", "exhausted"),))
        self.assertEqual(
            assess_credit_risk([InvoiceSummary("A", 75, 2)], 50), (("A", "over_limit"),)
        )


if __name__ == "__main__":
    unittest.main()
