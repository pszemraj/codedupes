"use strict";

function assertInvoiceRecord(record, index) {
  if (record === null || typeof record !== "object") {
    throw new TypeError(`row ${index}: expected invoice record`);
  }
  if (typeof record.invoiceId !== "string" || record.invoiceId.trim() === "") {
    throw new TypeError(`row ${index}: invoiceId must be nonblank text`);
  }
  if (!Number.isSafeInteger(record.amountMinor)) {
    throw new TypeError(`row ${index}: amountMinor must be a safe integer`);
  }
  if (typeof record.voided !== "boolean") {
    throw new TypeError(`row ${index}: voided must be boolean`);
  }
}

function aggregateInvoices(records) {
  const byInvoice = new Map();
  for (const [index, record] of records.entries()) {
    assertInvoiceRecord(record, index);
    if (record.voided) continue;
    const invoiceId = record.invoiceId.trim();
    const previous = byInvoice.get(invoiceId) ?? { totalMinor: 0, count: 0 };
    byInvoice.set(invoiceId, {
      totalMinor: previous.totalMinor + record.amountMinor,
      count: previous.count + 1,
    });
  }
  return [...byInvoice.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([invoiceId, value]) => ({ invoiceId, ...value }));
}

function buildInvoiceDigest(records) {
  const accepted = [];
  records.forEach((record, index) => {
    assertInvoiceRecord(record, index);
    if (!record.voided) accepted.push([record.invoiceId.trim(), record.amountMinor]);
  });
  accepted.sort(([left], [right]) => left.localeCompare(right));
  const digest = [];
  for (const [invoiceId, amountMinor] of accepted) {
    const tail = digest.at(-1);
    if (tail?.invoiceId === invoiceId) {
      tail.totalMinor += amountMinor;
      tail.count += 1;
    } else {
      digest.push({ invoiceId, totalMinor: amountMinor, count: 1 });
    }
  }
  return digest;
}

function summarizeInvoiceAccounts(records) {
  const accounts = Object.create(null);
  for (let index = 0; index < records.length; index += 1) {
    const record = records[index];
    assertInvoiceRecord(record, index);
    if (record.voided) continue;
    const key = record.invoiceId.trim();
    accounts[key] ??= { totalMinor: 0, count: 0 };
    accounts[key].totalMinor += record.amountMinor;
    accounts[key].count += 1;
  }
  return Object.keys(accounts)
    .sort((left, right) => left.localeCompare(right))
    .map((invoiceId) => ({ invoiceId, ...accounts[invoiceId] }));
}

module.exports = { aggregateInvoices, buildInvoiceDigest, summarizeInvoiceAccounts };
