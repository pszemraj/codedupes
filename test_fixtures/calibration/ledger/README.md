# Offline ledger pilot

This development application imports transaction records, summarizes invoices,
plans settlements, and produces independent audit reports. Multiple supported
implementations are intentional. Their maintenance judgments live in the corpus
annotations, outside analyzed source.

Run `PYTHONPATH=src python -m unittest discover -s tests` and
`PYTHONPATH=src python -m ledger` from this directory (using the repository's
`inf` environment). The entry point exercises both summary and planning paths,
both row import paths, the file/API adapters, and the audit reports.

## Contracts

Aggregation validates records in input order before ignoring voided records.
Invoice identifiers are stripped, case-sensitive Unicode text; empty identifiers
are invalid. Amounts are arbitrary-size Python integers, excluding booleans.
Voided flags must be booleans. Results are immutable and sorted by invoice;
active zero-total invoices remain present. Inputs are never modified.

Settlement summaries must have unique nonblank identifiers, integer totals, and
positive integer counts. Validate the entire input before constructing output.
Positive totals create transfers, negative totals are `refund_due`, and zero is
`balanced`. Optional deferral applies only to positive totals strictly below a
nonnegative integer minimum. Deferred invoices create no transfer and do create
an audit event. Minimum zero gives the complete baseline result, including empty
audit events and zero deferred total. Output is sorted by invoice.

Raw imports require `external_id`, `invoice_id`, `amount_minor`, `currency`,
`posted_on`, and `voided`. Identifiers are stripped nonblank strings. Amount text
is an optional minus followed by ASCII digits; plus signs, whitespace, decimals,
exponents, and booleans are rejected. Currency is exactly `USD`, `EUR`, or `GBP`.
Dates have strict `YYYY-MM-DD` syntax and must be real calendar dates. Voided
accepts a boolean or the exact strings `true` and `false`. Validation precedence
is external ID, invoice ID, amount, currency, date, voided. Invalid rows produce
ordered field-code rejections rather than aborting the batch. Only an accepted
row reserves its external ID. Reports preserve source order, include counters
and accepted-date bounds, and leave inputs unchanged.

CSV uses UTF-8, a context manager, and all six required headers (additional
columns are allowed). Rejection positions are physical CSV ending-line numbers.
Malformed headers abort and close the file. API envelopes require a nonblank
batch ID and a list of items. A previously completed batch returns a replay
result without processing items again. The supplied completed-batch set is
updated only after processing the entire valid envelope. Item positions are
zero-based. CSV and API reports deliberately have different shapes.

Audit responsibilities are separate: credit risk classifies exposure using a
credit limit; reference auditing identifies invoice identifiers occurring under
multiple external IDs; chronological auditing finds out-of-order accepted
records; currency auditing validates per-currency net exposure against budgets.
Different output alone is not a negative label: the annotations explain why
these operations lack a substantial shared maintenance region with their peers.
