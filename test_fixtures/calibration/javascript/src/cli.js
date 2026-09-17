"use strict";

const {
  aggregateInvoices,
  buildInvoiceDigest,
  reduceInvoiceTotals,
  summarizeInvoiceAccounts,
} = require("./aggregation");
const { importWebhookBatch, importWebhookLines } = require("./ingestion");
const { dispatchReceipt } = require("./notifications");
const { publishConfirmation } = require("./outbox");
const {
  planTransfers,
  planTransfersWithMinimum,
  schedulePayouts,
} = require("./settlement");

async function main() {
  const { fetchRecords, readSnapshot } = await import("./pagination.mjs");
  const records = [
    { invoiceId: "  east-1", amountMinor: 75, voided: false },
    { invoiceId: "east-1", amountMinor: 25, voided: false },
    { invoiceId: "west-2", amountMinor: -10, voided: false },
  ];
  const summaries = aggregateInvoices(records);
  const completed = new Set();
  const batch = importWebhookBatch({ batchId: "demo", items: [{ id: "e1", email: "a@example.test", amountMinor: 100, voided: false }] }, completed);
  const lines = importWebhookLines([JSON.stringify({ id: "w2", email: "b@example.test", amountMinor: 20, voided: false })]);
  const mailbox = { sent: [] };
  await publishConfirmation(mailbox, { invoiceId: "east-1", email: "a@example.test", totalMinor: 100 });
  await dispatchReceipt(mailbox, { invoiceId: "west-2", email: "b@example.test", totalMinor: -10 });
  const pages = new Map([
    [null, { rows: [{ id: "first", value: 1 }], next: "more" }],
    ["more", { rows: [{ id: "second", value: 2 }], next: null }],
  ]);
  const fetchPage = async (cursor) => pages.get(cursor);
  return {
    aggregation: [
      summaries,
      buildInvoiceDigest(records),
      summarizeInvoiceAccounts(records),
      reduceInvoiceTotals(records),
    ],
    plans: [
      planTransfers(summaries),
      planTransfersWithMinimum(summaries, 100),
      schedulePayouts(summaries),
    ],
    ingestion: [batch, lines],
    deliveries: mailbox.sent,
    pagination: [await fetchRecords(fetchPage), await readSnapshot(fetchPage)],
  };
}

if (require.main === module) {
  main().then((result) => process.stdout.write(`${JSON.stringify(result)}\n`));
}

module.exports = { main };
