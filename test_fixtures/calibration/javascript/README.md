# Offline fulfillment pilot

This small Node.js application accepts invoice rows from a webhook, prepares
payout instructions, and delivers receipt messages. It deliberately retains
several independently written implementations of the same maintenance work:
invoice aggregation, payout planning, webhook normalization, and asynchronous
receipt delivery. The annotations explain which regions are candidates for
consolidation and which similar-looking operations are not.

Run `npm test` and `npm start` from this directory. Both commands use only the
Node.js standard library. The entry point invokes every implementation family.

## Fixture contracts

Invoice aggregation validates every row before skipping voided invoices,
preserves its input, and returns invoice-sorted totals and counts. Payout plans
validate the complete summary set before producing sorted transfer instructions.
Positive balances strictly below an enabled minimum are deferred; a zero
minimum is the baseline behavior.

Webhook records require nonblank IDs, a normalizable email address, a finite
integer amount, and a boolean `voided` flag. Batch processors only reserve IDs
after acceptance, so a malformed first record never prevents a corrected later
record from being accepted. Batch and JSON-lines adapters deliberately report
failures differently.

Receipt adapters write one normalized delivery record to the supplied mailbox.
The callback, Promise, and async transport forms have intentionally different
calling conventions while preserving the same successful-delivery state.
