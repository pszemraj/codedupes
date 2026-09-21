"""Immutable records used by imports, settlements, and audit reports."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class Transaction:
    invoice_id: str
    amount_minor: int
    voided: bool = False


@dataclass(frozen=True, slots=True)
class InvoiceSummary:
    invoice_id: str
    total_minor: int
    transaction_count: int


@dataclass(frozen=True, slots=True)
class ImportedTransaction:
    external_id: str
    invoice_id: str
    amount_minor: int
    currency: str
    posted_on: date
    voided: bool


@dataclass(frozen=True, slots=True)
class ImportReport:
    accepted: tuple[ImportedTransaction, ...]
    rejected: tuple[tuple[int, str], ...]
    error_counts: tuple[tuple[str, int], ...]
    date_bounds: tuple[date, date] | None


@dataclass(frozen=True, slots=True)
class SettlementPlan:
    lines: tuple[tuple[str, str, int], ...]
    transfers: tuple[tuple[str, int], ...]
    deferred_total: int
    events: tuple[tuple[str, str, int], ...]
