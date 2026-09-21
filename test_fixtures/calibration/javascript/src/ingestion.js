"use strict";

function normalizeWebhookRecord(record) {
  if (record === null || typeof record !== "object") throw new TypeError("record must be an object");
  if (typeof record.id !== "string" || record.id.trim() === "") throw new TypeError("id must be nonblank text");
  if (typeof record.email !== "string" || !record.email.includes("@")) throw new TypeError("email must contain @");
  if (!Number.isSafeInteger(record.amountMinor)) throw new TypeError("amountMinor must be a safe integer");
  if (typeof record.voided !== "boolean") throw new TypeError("voided must be boolean");
  return {
    id: record.id.trim(),
    email: record.email.trim().toLowerCase(),
    amountMinor: record.amountMinor,
    voided: record.voided,
  };
}

function importWebhookBatch(envelope, completedBatches = new Set()) {
  if (envelope === null || typeof envelope !== "object") throw new TypeError("envelope must be an object");
  if (typeof envelope.batchId !== "string" || envelope.batchId.trim() === "") throw new TypeError("batchId must be nonblank text");
  if (!Array.isArray(envelope.items)) throw new TypeError("items must be an array");
  const batchId = envelope.batchId.trim();
  if (completedBatches.has(batchId)) return { batchId, accepted: [], rejected: [], replayed: true };
  const accepted = [];
  const rejected = [];
  const seenIds = new Set();
  envelope.items.forEach((record, index) => {
    try {
      if (record === null || typeof record !== "object") throw new TypeError("record must be an object");
      if (typeof record.id !== "string" || record.id.trim() === "") throw new TypeError("id must be nonblank text");
      if (typeof record.email !== "string" || !record.email.includes("@")) throw new TypeError("email must contain @");
      if (!Number.isSafeInteger(record.amountMinor)) throw new TypeError("amountMinor must be a safe integer");
      if (typeof record.voided !== "boolean") throw new TypeError("voided must be boolean");
      const normalized = { id: record.id.trim(), email: record.email.trim().toLowerCase(), amountMinor: record.amountMinor, voided: record.voided };
      if (seenIds.has(normalized.id)) throw new Error("duplicate id");
      seenIds.add(normalized.id);
      accepted.push(normalized);
    } catch (error) {
      rejected.push({ index, reason: error.message });
    }
  });
  completedBatches.add(batchId);
  return { batchId, accepted, rejected, replayed: false };
}

function importWebhookLines(lines, reservedIds = new Set()) {
  const accepted = [];
  const rejected = [];
  lines.forEach((line, lineNumber) => {
    try {
      const record = JSON.parse(line);
      if (record === null || typeof record !== "object") throw new TypeError("record must be an object");
      if (typeof record.id !== "string" || record.id.trim() === "") throw new TypeError("id must be nonblank text");
      if (typeof record.email !== "string" || !record.email.includes("@")) throw new TypeError("email must contain @");
      if (!Number.isSafeInteger(record.amountMinor)) throw new TypeError("amountMinor must be a safe integer");
      if (typeof record.voided !== "boolean") throw new TypeError("voided must be boolean");
      const normalized = { id: record.id.trim(), email: record.email.trim().toLowerCase(), amountMinor: record.amountMinor, voided: record.voided };
      if (reservedIds.has(normalized.id)) throw new Error("duplicate id");
      reservedIds.add(normalized.id);
      accepted.push(normalized);
    } catch (error) {
      rejected.push({ line: lineNumber + 1, reason: error.message });
    }
  });
  return { accepted, rejected };
}

module.exports = { importWebhookBatch, importWebhookLines, normalizeWebhookRecord };
