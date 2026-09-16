"""Normalize external records and build ordered import reports."""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date

from .models import ImportedTransaction, ImportReport


def normalize_transaction(row: Mapping[str, object]) -> ImportedTransaction:
    """Decode one record, reporting the first invalid domain field."""
    if not isinstance(row, Mapping):
        raise ValueError("external_id")
    external_id = row.get("external_id")
    if not isinstance(external_id, str) or not external_id.strip():
        raise ValueError("external_id")
    invoice_id = row.get("invoice_id")
    if not isinstance(invoice_id, str) or not invoice_id.strip():
        raise ValueError("invoice_id")
    amount = row.get("amount_minor")
    if not isinstance(amount, str) or re.fullmatch(r"-?[0-9]+", amount) is None:
        raise ValueError("amount_minor")
    currency = row.get("currency")
    if currency not in ("USD", "EUR", "GBP"):
        raise ValueError("currency")
    day = row.get("posted_on")
    if not isinstance(day, str) or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", day) is None:
        raise ValueError("posted_on")
    try:
        posted_on = date.fromisoformat(day)
    except ValueError as exc:
        raise ValueError("posted_on") from exc
    voided = row.get("voided")
    if type(voided) is not bool:
        if voided not in ("true", "false"):
            raise ValueError("voided")
        voided = voided == "true"
    return ImportedTransaction(
        external_id.strip(), invoice_id.strip(), int(amount), currency, posted_on, voided
    )


def ingest_rows(rows: Sequence[Mapping[str, object]]) -> ImportReport:
    """Import rows in source order while retaining row-local rejections."""
    accepted = []
    rejected = []
    errors = Counter()
    seen = set()
    for position, row in enumerate(rows):
        try:
            if not isinstance(row, Mapping):
                raise ValueError("external_id")
            external_id = row.get("external_id")
            if not isinstance(external_id, str) or not external_id.strip():
                raise ValueError("external_id")
            invoice_id = row.get("invoice_id")
            if not isinstance(invoice_id, str) or not invoice_id.strip():
                raise ValueError("invoice_id")
            amount = row.get("amount_minor")
            if not isinstance(amount, str) or re.fullmatch(r"-?[0-9]+", amount) is None:
                raise ValueError("amount_minor")
            currency = row.get("currency")
            if currency not in ("USD", "EUR", "GBP"):
                raise ValueError("currency")
            day = row.get("posted_on")
            if not isinstance(day, str) or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", day) is None:
                raise ValueError("posted_on")
            try:
                posted_on = date.fromisoformat(day)
            except ValueError as exc:
                raise ValueError("posted_on") from exc
            voided = row.get("voided")
            if type(voided) is not bool:
                if voided not in ("true", "false"):
                    raise ValueError("voided")
                voided = voided == "true"
            transaction = ImportedTransaction(
                external_id.strip(), invoice_id.strip(), int(amount), currency, posted_on, voided
            )
            if transaction.external_id in seen:
                raise ValueError("duplicate_id")
            seen.add(transaction.external_id)
            accepted.append(transaction)
        except ValueError as exc:
            code = str(exc)
            rejected.append((position, code))
            errors[code] += 1
    days = [item.posted_on for item in accepted]
    bounds = (min(days), max(days)) if days else None
    return ImportReport(tuple(accepted), tuple(rejected), tuple(sorted(errors.items())), bounds)


def prepare_import(rows: Sequence[Mapping[str, object]]) -> ImportReport:
    """Prepare a batch using domain decoding and ordered acceptance accounting."""
    accepted = []
    rejected = []
    errors = Counter()
    seen = set()
    earliest = None
    latest = None
    for position, row in enumerate(rows):
        try:
            transaction = normalize_transaction(row)
            if transaction.external_id in seen:
                raise ValueError("duplicate_id")
        except ValueError as exc:
            code = str(exc)
            rejected.append((position, code))
            errors[code] += 1
            continue
        seen.add(transaction.external_id)
        accepted.append(transaction)
        if earliest is None or transaction.posted_on < earliest:
            earliest = transaction.posted_on
        if latest is None or transaction.posted_on > latest:
            latest = transaction.posted_on
    bounds = (earliest, latest) if earliest is not None else None
    return ImportReport(tuple(accepted), tuple(rejected), tuple(sorted(errors.items())), bounds)
