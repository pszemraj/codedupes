# Offline fulfillment pilot

This small Node.js application accepts invoice rows from a webhook, prepares
payout instructions, and delivers receipt messages. It deliberately retains
several independently written implementations of the same maintenance work:
invoice aggregation, payout planning, webhook normalization, asynchronous
receipt delivery, and cursor pagination. Labels and pair judgments follow the
shared [calibration corpus contract](../README.md).

Run `npm test` and `npm start` from this directory. Both commands use only the
Node.js standard library. The entry point invokes every implementation family.

## Fixture contracts

Invoice aggregation validates every row before skipping voided invoices,
preserves its input, and returns invoice-sorted totals and counts. Payout plans
validate the complete summary set before producing sorted transfer instructions.
Positive balances strictly below an enabled minimum are deferred; a zero
minimum is the baseline behavior.

Webhook records require nonblank IDs, a normalizable email address, a safe
integer amount, and a boolean `voided` flag. Invoice aggregation and deferred
payout totals also reject additions that leave JavaScript's safe-integer range.
Batch processors only reserve IDs after acceptance, so a malformed first record
never prevents a corrected later record from being accepted. Batch and
JSON-lines adapters deliberately report failures differently.

Receipt adapters write one normalized delivery record to the supplied mailbox.
The callback, queued outbox Promise, and async transport forms have intentionally
different ownership and calling conventions while preserving the same
successful-delivery state.

Pagination uses an injected asynchronous transport. The loop/Set and recursive
Promise/Map implementations both validate page and row shapes, retain the first
record for each ID, reject cursor cycles before page-budget exhaustion, and
discard responses observed after cancellation. Transport errors retain their
original identity.
