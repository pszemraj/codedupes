"""Independent risk and consistency reports for ledger operators."""

from collections.abc import Mapping, Sequence

from .models import ImportedTransaction, InvoiceSummary


def assess_credit_risk(summaries: Sequence[InvoiceSummary], credit_limit: int) -> tuple:
    """Classify exposure without creating or deferring payment instructions."""
    if type(credit_limit) is not int or credit_limit <= 0:
        raise ValueError("credit_limit")
    flagged = []
    for item in summaries:
        if item.total_minor < 0:
            reason = "credit_balance"
        elif item.total_minor > credit_limit:
            reason = "over_limit"
        elif item.transaction_count > 10:
            reason = "high_activity"
        else:
            continue
        flagged.append((item.invoice_id, reason))
    return tuple(sorted(flagged))


def audit_reference_usage(rows: Sequence[ImportedTransaction]) -> tuple:
    """Report invoice references associated with multiple source identifiers."""
    references = {}
    for row in rows:
        if row.voided:
            continue
        references.setdefault(row.invoice_id, set()).add(row.external_id)
    findings = []
    for invoice, identifiers in sorted(references.items()):
        if len(identifiers) > 1:
            findings.append((invoice, tuple(sorted(identifiers))))
    return tuple(findings)


def audit_chronology(rows: Sequence[ImportedTransaction]) -> tuple:
    """Find date regressions in the accepted event stream for each invoice."""
    latest = {}
    violations = []
    for position, row in enumerate(rows):
        previous = latest.get(row.invoice_id)
        if previous is not None and row.posted_on < previous:
            violations.append((position, row.external_id, previous.isoformat()))
        if previous is None or row.posted_on > previous:
            latest[row.invoice_id] = row.posted_on
    return tuple(violations)


def audit_currency_limits(rows: Sequence[ImportedTransaction], limits: Mapping[str, int]) -> tuple:
    """Check net currency exposure against explicit operator budgets."""
    remaining = dict(limits)
    violations = []
    for position, row in enumerate(rows):
        if row.voided:
            continue
        if row.currency not in remaining:
            violations.append((position, row.currency, "unbudgeted"))
            continue
        remaining[row.currency] -= row.amount_minor
        if remaining[row.currency] < 0:
            violations.append((position, row.currency, "exhausted"))
    return tuple(violations)
