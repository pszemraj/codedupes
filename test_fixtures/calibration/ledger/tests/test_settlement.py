"""Behavioral contract tests for settlement implementations."""

import random
import unittest

from ledger.settlement import (
    InvoiceSummary,
    Transaction,
    build_invoice_report,
    summarize_import,
)

IMPLEMENTATIONS = (summarize_import, build_invoice_report)


class SettlementTests(unittest.TestCase):
    def test_documented_result(self):
        rows = [
            Transaction(" A ", 100),
            Transaction("B", 70),
            Transaction("A", -25),
            Transaction("A", 500, True),
            Transaction("東京", 0),
        ]
        expected = (InvoiceSummary("A", 75, 2), InvoiceSummary("B", 70, 1),
                    InvoiceSummary("東京", 0, 1))
        for implementation in IMPLEMENTATIONS:
            with self.subTest(implementation=implementation.__name__):
                self.assertEqual(implementation(rows), expected)

    def test_empty_and_all_voided(self):
        for implementation in IMPLEMENTATIONS:
            self.assertEqual(implementation([]), ())
            self.assertEqual(implementation([Transaction("A", 5, True)]), ())

    def test_arbitrary_precision_and_zero_total(self):
        rows = [Transaction("A", 10**100), Transaction("A", -(10**100))]
        for implementation in IMPLEMENTATIONS:
            self.assertEqual(implementation(rows), (InvoiceSummary("A", 0, 2),))

    def test_case_is_preserved(self):
        for implementation in IMPLEMENTATIONS:
            self.assertEqual(implementation([Transaction("a", 1), Transaction("A", 2)]),
                             (InvoiceSummary("A", 2, 1), InvoiceSummary("a", 1, 1)))

    def test_invalid_voided_row_is_still_rejected(self):
        rows = [Transaction("OK", 1), Transaction(" ", 9, True)]
        for implementation in IMPLEMENTATIONS:
            with self.assertRaisesRegex(ValueError, "row 1: invoice_id"):
                implementation(rows)

    def test_invalid_fields_and_first_error_order(self):
        cases = [
            ([None], TypeError, "row 0: expected Transaction"),
            ([Transaction(None, 3)], ValueError, "row 0: invoice_id must be nonblank text"),
            ([Transaction("A", True)], ValueError, "row 0: amount_minor must be an integer"),
            ([Transaction("A", 3, 1)], ValueError, "row 0: voided must be a boolean"),
            ([Transaction("A", 2.5, True)], ValueError,
             "row 0: amount_minor must be an integer"),
            ([Transaction(" ", False, 1)], ValueError,
             "row 0: invoice_id must be nonblank text"),
        ]
        for rows, exception_type, message in cases:
            for implementation in IMPLEMENTATIONS:
                with self.subTest(rows=rows, implementation=implementation.__name__):
                    with self.assertRaises(exception_type) as raised:
                        implementation(rows)
                    self.assertEqual(str(raised.exception), message)

    def test_seeded_differential_and_aggregate_invariants(self):
        rng = random.Random(20)
        for case in range(1000):
            rows = [Transaction(rng.choice(["A", " B ", "東京", "é", "a"]),
                                rng.randrange(-10000, 10001), bool(rng.randrange(4) == 0))
                    for _ in range(rng.randrange(45))]
            before = tuple(rows)
            left = summarize_import(rows)
            right = build_invoice_report(rows)
            with self.subTest(case=case):
                self.assertEqual(left, right)
                self.assertEqual(tuple(rows), before)
                self.assertEqual(sum(item.total_minor for item in left),
                                 sum(row.amount_minor for row in rows if not row.voided))
                self.assertEqual(sum(item.transaction_count for item in left),
                                 sum(not row.voided for row in rows))
                self.assertEqual([item.invoice_id for item in left],
                                 sorted({row.invoice_id.strip() for row in rows if not row.voided}))


if __name__ == "__main__":
    unittest.main()
