"use strict";

const { aggregateInvoices, buildInvoiceDigest, summarizeInvoiceAccounts } = require("./aggregation");
const { importWebhookBatch, importWebhookLines } = require("./ingestion");
const { dispatchReceipt, sendReceiptPromise } = require("./notifications");
const { planTransfers, planTransfersWithMinimum, schedulePayouts } = require("./settlement");

async function main() {
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
  await sendReceiptPromise(mailbox, { invoiceId: "east-1", email: "a@example.test", totalMinor: 100 });
  await dispatchReceipt(mailbox, { invoiceId: "west-2", email: "b@example.test", totalMinor: -10 });
  return {
    aggregation: [summaries, buildInvoiceDigest(records), summarizeInvoiceAccounts(records)],
    plans: [planTransfers(summaries), planTransfersWithMinimum(summaries, 100), schedulePayouts(summaries)],
    ingestion: [batch, lines],
    deliveries: mailbox.sent,
  };
}

if (require.main === module) {
  main().then((result) => process.stdout.write(`${JSON.stringify(result)}\n`));
}

module.exports = { main };
