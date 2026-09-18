"""Read file and API imports with adapter-specific ownership and reporting."""

import csv
import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path

from .models import ImportedTransaction


def import_csv_file(path: Path) -> dict:
    """Read a UTF-8 transaction file and report source-line rejections."""
    accepted = []
    rejected = []
    seen = set()
    required = {"external_id", "invoice_id", "amount_minor", "currency", "posted_on", "voided"}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("headers")
        for row in reader:
            try:
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
                rejected.append((reader.line_num, str(exc)))
    return {"source": path.name, "accepted": tuple(accepted), "line_errors": tuple(rejected)}


def ingest_api_batch(envelope: Mapping[str, object], completed_batches: set[str]) -> dict:
    """Ingest an identified API batch and remember completed batches for replay."""
    if not isinstance(envelope, Mapping):
        raise ValueError("envelope")
    batch_id = envelope.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ValueError("batch_id")
    items = envelope.get("items")
    if not isinstance(items, list):
        raise ValueError("items")
    batch_id = batch_id.strip()
    if batch_id in completed_batches:
        return {"batch_id": batch_id, "replayed": True, "results": ()}
    results = []
    seen = set()
    for position, row in enumerate(items):
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
            results.append((position, "accepted", transaction))
        except ValueError as exc:
            results.append((position, "rejected", str(exc)))
    completed_batches.add(batch_id)
    return {"batch_id": batch_id, "replayed": False, "results": tuple(results)}
