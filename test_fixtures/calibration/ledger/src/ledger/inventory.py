"""Allocate stock through a pure workflow and a stateful inventory book."""

from collections.abc import Mapping, Sequence

Allocation = tuple[str, int, int]
Shortage = tuple[str, int, int]


def allocate_orders(
    stock: Mapping[str, int], orders: Sequence[tuple[str, int]], allow_partial: bool
) -> tuple[dict[str, int], tuple[Allocation, ...], tuple[Shortage, ...]]:
    """Allocate an entire order batch without changing the caller's stock mapping."""
    if type(allow_partial) is not bool:
        raise ValueError("allow_partial")
    remaining = {}
    for sku, available in stock.items():
        if not isinstance(sku, str) or not sku.strip():
            raise ValueError("stock_sku")
        if type(available) is not int or available < 0:
            raise ValueError("stock_quantity")
        remaining[sku] = available
    checked_orders = []
    for order in orders:
        if not isinstance(order, tuple) or len(order) != 2:
            raise ValueError("order")
        sku, requested = order
        if not isinstance(sku, str) or not sku.strip():
            raise ValueError("order_sku")
        if type(requested) is not int or requested <= 0:
            raise ValueError("order_quantity")
        checked_orders.append((sku, requested))
    allocations = []
    shortages = []
    for sku, requested in checked_orders:
        available = remaining.get(sku, 0)
        filled = requested if available >= requested else available if allow_partial else 0
        remaining[sku] = available - filled
        allocations.append((sku, requested, filled))
        if filled != requested:
            shortages.append((sku, requested, requested - filled))
    return remaining, tuple(allocations), tuple(shortages)


class InventoryBook:
    """Store stock and commit validated reservations as application state."""

    def __init__(self, stock: Mapping[str, int], allow_partial: bool) -> None:
        """Create an inventory book with a private mutable stock snapshot."""
        if type(allow_partial) is not bool:
            raise ValueError("allow_partial")
        self.allow_partial = allow_partial
        self._stock = {}
        for sku, available in stock.items():
            if not isinstance(sku, str) or not sku.strip():
                raise ValueError("stock_sku")
            if type(available) is not int or available < 0:
                raise ValueError("stock_quantity")
            self._stock[sku] = available

    @property
    def stock(self) -> dict[str, int]:
        """Return a caller-safe snapshot of the committed inventory state."""
        return dict(self._stock)

    def _pending_orders(self, orders: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
        """Decode a complete reservation request before touching stored stock."""
        pending = []
        for entry in orders:
            try:
                item, demand = entry
            except (TypeError, ValueError):
                raise ValueError("order") from None
            if not isinstance(item, str) or item.strip() == "":
                raise ValueError("order_sku")
            if isinstance(demand, bool) or not isinstance(demand, int) or demand < 1:
                raise ValueError("order_quantity")
            pending.append((item, demand))
        return pending

    def reserve(
        self, orders: Sequence[tuple[str, int]]
    ) -> tuple[tuple[Allocation, ...], tuple[Shortage, ...]]:
        """Validate then commit one FIFO reservation batch against book state."""
        pending = self._pending_orders(orders)
        candidate = dict(self._stock)
        fulfilled = []
        deficits = []
        position = 0
        while position < len(pending):
            item, demand = pending[position]
            on_hand = candidate.get(item, 0)
            granted = min(on_hand, demand)
            if granted < demand and not self.allow_partial:
                granted = 0
            candidate[item] = on_hand - granted
            fulfilled.append((item, demand, granted))
            if granted < demand:
                deficits.append((item, demand, demand - granted))
            position += 1
        self._stock = candidate
        return tuple(fulfilled), tuple(deficits)

    def restock(self, deliveries: Sequence[tuple[str, int]]) -> dict[str, int]:
        """Commit received quantities without performing order allocation."""
        checked_deliveries = []
        for delivery in deliveries:
            if not isinstance(delivery, tuple) or len(delivery) != 2:
                raise ValueError("delivery")
            sku, received = delivery
            if not isinstance(sku, str) or not sku.strip():
                raise ValueError("delivery_sku")
            if type(received) is not int or received <= 0:
                raise ValueError("delivery_quantity")
            checked_deliveries.append((sku, received))
        for sku, received in checked_deliveries:
            self._stock[sku] = self._stock.get(sku, 0) + received
        return self.stock


def audit_stock_levels(
    stock: Mapping[str, int], minimums: Mapping[str, int]
) -> tuple[Shortage, ...]:
    """Report stock-policy breaches without allocating or changing inventory."""
    findings = []
    for sku, minimum in minimums.items():
        if not isinstance(sku, str) or not sku.strip():
            raise ValueError("minimum_sku")
        if type(minimum) is not int or minimum < 0:
            raise ValueError("minimum_quantity")
        available = stock.get(sku, 0)
        if type(available) is not int or available < 0:
            raise ValueError("stock_quantity")
        if available < minimum:
            findings.append((sku, minimum, minimum - available))
    return tuple(sorted(findings))
