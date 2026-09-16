"""Invoice settlement for offline transaction imports."""

from collections.abc import Sequence
from itertools import groupby
from operator import itemgetter

from .models import InvoiceSummary, Transaction


def summarize_import(rows: Sequence[Transaction]) -> tuple[InvoiceSummary, ...]:
    """Validate an import and return invoice totals in identifier order."""
    totals: dict[str, int] = {}
    counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Transaction):
            raise TypeError(f"row {index}: expected Transaction")
        if not isinstance(row.invoice_id, str) or not row.invoice_id.strip():
            raise ValueError(f"row {index}: invoice_id must be nonblank text")
        if type(row.amount_minor) is not int:
            raise ValueError(f"row {index}: amount_minor must be an integer")
        if type(row.voided) is not bool:
            raise ValueError(f"row {index}: voided must be a boolean")
        invoice = row.invoice_id.strip()
        if row.voided:
            continue
        if invoice not in totals:
            totals[invoice] = 0
            counts[invoice] = 0
        totals[invoice] += row.amount_minor
        counts[invoice] += 1
    result = []
    for invoice in sorted(totals):
        result.append(InvoiceSummary(invoice, totals[invoice], counts[invoice]))
    return tuple(result)


def build_invoice_report(rows: Sequence[Transaction]) -> tuple[InvoiceSummary, ...]:
    """Prepare invoice groups and produce a stable settlement report."""
    entries: list[tuple[str, int]] = []
    for position, transaction in enumerate(rows):
        if not isinstance(transaction, Transaction):
            raise TypeError(f"row {position}: expected Transaction")
        name = transaction.invoice_id
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"row {position}: invoice_id must be nonblank text")
        amount = transaction.amount_minor
        if type(amount) is not int:
            raise ValueError(f"row {position}: amount_minor must be an integer")
        is_void = transaction.voided
        if type(is_void) is not bool:
            raise ValueError(f"row {position}: voided must be a boolean")
        if not is_void:
            entries.append((name.strip(), amount))
    ordered = sorted(entries, key=itemgetter(0))
    summaries = []
    for name, members in groupby(ordered, key=itemgetter(0)):
        amounts = [entry[1] for entry in members]
        summaries.append(InvoiceSummary(name, sum(amounts), len(amounts)))
    return tuple(summaries)
