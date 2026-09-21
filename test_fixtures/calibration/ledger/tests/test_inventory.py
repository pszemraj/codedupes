"""Behavioral evidence for pure and stateful inventory allocation."""

import copy
import unittest

from ledger.inventory import InventoryBook, allocate_orders, audit_stock_levels


class InventoryTests(unittest.TestCase):
    def test_pure_and_stateful_paths_share_the_explicit_state_mapping(self):
        stock = {"A": 5, "B": 2}
        orders = (("A", 3), ("A", 4), ("B", 1))
        before = copy.deepcopy(stock)
        new_stock, allocations, shortages = allocate_orders(stock, orders, True)
        book = InventoryBook(stock, True)
        self.assertEqual(book.reserve(orders), (allocations, shortages))
        self.assertEqual(book.stock, new_stock)
        self.assertEqual(stock, before)
        self.assertEqual(allocations, (("A", 3, 3), ("A", 4, 2), ("B", 1, 1)))
        self.assertEqual(shortages, (("A", 4, 2),))

    def test_non_partial_policy_leaves_unfilled_quantity_available(self):
        orders = (("A", 3), ("A", 4))
        pure_stock, pure_allocations, pure_shortages = allocate_orders({"A": 5}, orders, False)
        book = InventoryBook({"A": 5}, False)
        self.assertEqual(book.reserve(orders), (pure_allocations, pure_shortages))
        self.assertEqual(book.stock, pure_stock)
        self.assertEqual(pure_stock, {"A": 2})
        self.assertEqual(pure_allocations, (("A", 3, 3), ("A", 4, 0)))
        self.assertEqual(pure_shortages, (("A", 4, 4),))

    def test_missing_stock_and_repeated_skus_follow_fifo_order(self):
        orders = (("missing", 2), ("A", 1), ("A", 2))
        new_stock, allocations, shortages = allocate_orders({"A": 2}, orders, True)
        self.assertEqual(new_stock, {"A": 0, "missing": 0})
        self.assertEqual(allocations, (("missing", 2, 0), ("A", 1, 1), ("A", 2, 1)))
        self.assertEqual(shortages, (("missing", 2, 2), ("A", 2, 1)))

    def test_invalid_later_order_rolls_back_the_stateful_book(self):
        book = InventoryBook({"A": 5}, True)
        with self.assertRaisesRegex(ValueError, "order_quantity"):
            book.reserve((("A", 2), ("A", -1)))
        self.assertEqual(book.stock, {"A": 5})
        with self.assertRaisesRegex(ValueError, "order_quantity"):
            allocate_orders({"A": 5}, (("A", 2), ("A", 0)), True)

    def test_multiple_calls_commit_only_valid_reservations(self):
        book = InventoryBook({"A": 4}, True)
        self.assertEqual(book.reserve((("A", 3),)), ((("A", 3, 3),), ()))
        self.assertEqual(book.restock((("A", 2), ("B", 1))), {"A": 3, "B": 1})
        self.assertEqual(book.reserve((("A", 5),)), ((("A", 5, 3),), (("A", 5, 2),)))
        self.assertEqual(book.stock, {"A": 0, "B": 1})

    def test_inventory_audit_is_a_non_allocating_neighbor(self):
        stock = {"A": 1, "B": 4}
        self.assertEqual(
            audit_stock_levels(stock, {"A": 3, "B": 2, "C": 1}), (("A", 3, 2), ("C", 1, 1))
        )
        self.assertEqual(stock, {"A": 1, "B": 4})
        with self.assertRaisesRegex(ValueError, "minimum_quantity"):
            audit_stock_levels(stock, {"A": -1})
