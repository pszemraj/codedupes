"""Prepare transfer instructions and audit-visible settlement decisions."""

from collections.abc import Sequence

from .models import InvoiceSummary, SettlementPlan


def plan_settlements(summaries: Sequence[InvoiceSummary]) -> SettlementPlan:
    """Validate invoices and prepare an ordered settlement batch."""
    seen = set()
    for item in summaries:
        if not isinstance(item, InvoiceSummary):
            raise TypeError("expected InvoiceSummary")
        if not isinstance(item.invoice_id, str) or not item.invoice_id.strip():
            raise ValueError("invoice_id")
        if item.invoice_id in seen:
            raise ValueError("duplicate invoice_id")
        if type(item.total_minor) is not int:
            raise ValueError("total_minor")
        if type(item.transaction_count) is not int or item.transaction_count <= 0:
            raise ValueError("transaction_count")
        seen.add(item.invoice_id)
    lines = []
    transfers = []
    for item in sorted(summaries, key=lambda entry: entry.invoice_id):
        if item.total_minor > 0:
            lines.append((item.invoice_id, "ready", item.total_minor))
            transfers.append((item.invoice_id, item.total_minor))
        elif item.total_minor < 0:
            lines.append((item.invoice_id, "refund_due", item.total_minor))
        else:
            lines.append((item.invoice_id, "balanced", 0))
    return SettlementPlan(tuple(lines), tuple(transfers), 0, ())


def plan_batch_with_deferrals(
    summaries: Sequence[InvoiceSummary], minimum_payout_minor: int = 0
) -> SettlementPlan:
    """Prepare transfers while retaining small positive balances for a later batch."""
    if type(minimum_payout_minor) is not int or minimum_payout_minor < 0:
        raise ValueError("minimum_payout_minor")
    seen = set()
    for item in summaries:
        if not isinstance(item, InvoiceSummary):
            raise TypeError("expected InvoiceSummary")
        if not isinstance(item.invoice_id, str) or not item.invoice_id.strip():
            raise ValueError("invoice_id")
        if item.invoice_id in seen:
            raise ValueError("duplicate invoice_id")
        if type(item.total_minor) is not int:
            raise ValueError("total_minor")
        if type(item.transaction_count) is not int or item.transaction_count <= 0:
            raise ValueError("transaction_count")
        seen.add(item.invoice_id)
    lines = []
    transfers = []
    events = []
    deferred_total = 0
    for item in sorted(summaries, key=lambda entry: entry.invoice_id):
        if 0 < item.total_minor < minimum_payout_minor:
            lines.append((item.invoice_id, "deferred", item.total_minor))
            deferred_total += item.total_minor
            events.append((item.invoice_id, "below_minimum", minimum_payout_minor))
        elif item.total_minor > 0:
            lines.append((item.invoice_id, "ready", item.total_minor))
            transfers.append((item.invoice_id, item.total_minor))
        elif item.total_minor < 0:
            lines.append((item.invoice_id, "refund_due", item.total_minor))
        else:
            lines.append((item.invoice_id, "balanced", 0))
    return SettlementPlan(tuple(lines), tuple(transfers), deferred_total, tuple(events))


def plan_settlements_indexed(summaries: Sequence[InvoiceSummary]) -> SettlementPlan:
    """Prepare the baseline settlement plan with explicit index traversal."""
    seen = set()
    for position in range(len(summaries)):
        summary = summaries[position]
        if not isinstance(summary, InvoiceSummary):
            raise TypeError("expected InvoiceSummary")
        if not isinstance(summary.invoice_id, str) or not summary.invoice_id.strip():
            raise ValueError("invoice_id")
        if summary.invoice_id in seen:
            raise ValueError("duplicate invoice_id")
        if type(summary.total_minor) is not int:
            raise ValueError("total_minor")
        if type(summary.transaction_count) is not int or summary.transaction_count <= 0:
            raise ValueError("transaction_count")
        seen.add(summary.invoice_id)
    lines = []
    transfers = []
    ordered = sorted(summaries, key=lambda entry: entry.invoice_id)
    for position in range(len(ordered)):
        summary = ordered[position]
        if summary.total_minor > 0:
            lines.append((summary.invoice_id, "ready", summary.total_minor))
            transfers.append((summary.invoice_id, summary.total_minor))
        elif summary.total_minor < 0:
            lines.append((summary.invoice_id, "refund_due", summary.total_minor))
        else:
            lines.append((summary.invoice_id, "balanced", 0))
    return SettlementPlan(tuple(lines), tuple(transfers), 0, ())
