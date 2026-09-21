"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const {
  aggregateInvoices,
  buildInvoiceDigest,
  summarizeInvoiceAccounts,
} = require("../src/aggregation");
const { importWebhookBatch, importWebhookLines, normalizeWebhookRecord } = require("../src/ingestion");
const { dispatchReceipt, sendReceiptCallback } = require("../src/notifications");
const { publishConfirmation } = require("../src/outbox");
const {
  planTransfers,
  planTransfersWithMinimum,
  schedulePayouts,
} = require("../src/settlement");

const records = [
  { invoiceId: " b", amountMinor: 50, voided: false },
  { invoiceId: "a", amountMinor: 70, voided: false },
  { invoiceId: "b", amountMinor: -20, voided: false },
  { invoiceId: "bad", amountMinor: 1, voided: true },
];

test("translated aggregation agrees and validates before void filtering", () => {
  const original = structuredClone(records);
  const expected = [
    { invoiceId: "a", totalMinor: 70, count: 1 },
    { invoiceId: "b", totalMinor: 30, count: 2 },
  ];
  assert.deepEqual(aggregateInvoices(records), expected);
  assert.deepEqual(buildInvoiceDigest(records), expected);
  assert.deepEqual(summarizeInvoiceAccounts(records), expected);
  assert.deepEqual(records, original);
  const invalidVoided = [{ invoiceId: "discard", amountMinor: 1.5, voided: true }];
  assert.throws(() => aggregateInvoices(invalidVoided), /safe integer/);
  assert.throws(() => buildInvoiceDigest(invalidVoided), /safe integer/);
  assert.throws(() => summarizeInvoiceAccounts(invalidVoided), /safe integer/);
});

test("aggregation variants retain exact Unicode keys across input permutations", () => {
  const unicodeRecords = [
    { invoiceId: "\u00e9", amountMinor: 10, voided: false },
    { invoiceId: "e\u0301", amountMinor: 20, voided: false },
    { invoiceId: "\u00e9", amountMinor: 30, voided: false },
  ];
  const expected = [
    { invoiceId: "e\u0301", totalMinor: 20, count: 1 },
    { invoiceId: "\u00e9", totalMinor: 40, count: 2 },
  ];
  const permutations = [
    [0, 1, 2], [0, 2, 1], [1, 0, 2],
    [1, 2, 0], [2, 0, 1], [2, 1, 0],
  ];
  for (const permutation of permutations) {
    const source = permutation.map((index) => unicodeRecords[index]);
    const original = structuredClone(source);
    for (const aggregate of [
      aggregateInvoices,
      buildInvoiceDigest,
      summarizeInvoiceAccounts,
    ]) {
      assert.deepEqual(aggregate(source), expected);
      assert.deepEqual(source, original);
    }
  }
});

test("aggregation variants reject totals outside the safe-integer range", () => {
  const overflowing = [
    { invoiceId: "a", amountMinor: Number.MAX_SAFE_INTEGER, voided: false },
    { invoiceId: "a", amountMinor: 1, voided: false },
  ];
  for (const aggregate of [
    aggregateInvoices,
    buildInvoiceDigest,
    summarizeInvoiceAccounts,
  ]) {
    assert.throws(() => aggregate(overflowing), /invoice total must be a safe integer/);
  }
});

test("payout extension preserves zero minimum and records deferral boundary", () => {
  const summaries = aggregateInvoices(records);
  const baseline = planTransfers(summaries);
  assert.deepEqual(planTransfersWithMinimum(summaries), baseline);
  assert.deepEqual(schedulePayouts(summaries), baseline);
  const deferred = planTransfersWithMinimum(summaries, 100);
  assert.deepEqual(deferred.transfers, []);
  assert.deepEqual(deferred.events, [["a", "below_minimum", 100], ["b", "below_minimum", 100]]);
  assert.equal(deferred.deferredTotal, 100);
  assert.deepEqual(schedulePayouts(summaries, { minimumPayoutMinor: 100 }), deferred);
});

test("payout variants reject deferred totals outside the safe-integer range", () => {
  const summaries = [
    { invoiceId: "a", totalMinor: Number.MAX_SAFE_INTEGER - 1, count: 1 },
    { invoiceId: "b", totalMinor: 1, count: 1 },
    { invoiceId: "c", totalMinor: 1, count: 1 },
  ];
  assert.throws(
    () => planTransfersWithMinimum(summaries, Number.MAX_SAFE_INTEGER),
    /deferredTotal must be a safe integer/,
  );
  assert.throws(
    () => schedulePayouts(summaries, { minimumPayoutMinor: Number.MAX_SAFE_INTEGER }),
    /deferredTotal must be a safe integer/,
  );
});

test("webhook normalization keeps acceptance and adapter failures distinct", () => {
  const source = { id: " x1 ", email: "USER@Example.Test ", amountMinor: 45, voided: false };
  assert.deepEqual(normalizeWebhookRecord(source), { id: "x1", email: "user@example.test", amountMinor: 45, voided: false });
  const completed = new Set();
  const batch = importWebhookBatch({ batchId: "receipt-1", items: [{ id: "x1", email: "bad", amountMinor: 2, voided: false }, source, source] }, completed);
  assert.deepEqual(batch.accepted, [{ id: "x1", email: "user@example.test", amountMinor: 45, voided: false }]);
  assert.deepEqual(batch.rejected.map((entry) => entry.index), [0, 2]);
  assert.equal(importWebhookBatch({ batchId: "receipt-1", items: [] }, completed).replayed, true);
  const lines = importWebhookLines(["{bad json", JSON.stringify(source), JSON.stringify(source)]);
  assert.deepEqual(lines.rejected.map((entry) => entry.line), [1, 3]);
  assert.equal(lines.accepted[0].id, "x1");
});

test("receipt adapters preserve delivery state across callback and promise styles", async () => {
  const callbackMailbox = { sent: [] };
  const message = { invoiceId: " invoice-9 ", email: "TO@Example.Test", totalMinor: 150 };
  const callbackDelivery = await new Promise((resolve, reject) => {
    sendReceiptCallback(callbackMailbox, message, (error, delivery) => (error ? reject(error) : resolve(delivery)));
  });
  const promiseMailbox = { sent: [] };
  const promised = await publishConfirmation(promiseMailbox, message);
  const transportMailbox = { sent: [] };
  const dispatched = await dispatchReceipt(transportMailbox, message);
  assert.deepEqual(callbackDelivery, promised);
  assert.deepEqual(promised, dispatched);
  assert.deepEqual(callbackMailbox.sent, [callbackDelivery]);
  await assert.rejects(() => dispatchReceipt(transportMailbox, message, "sms"), /unsupported transport/);
});
