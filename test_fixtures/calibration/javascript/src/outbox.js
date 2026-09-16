"use strict";

const waiting = [];

function createFuture() {
  let finish;
  const wait = new Promise((resolve, reject) => {
    finish = { resolve, reject };
  });
  return { finish, wait };
}

function publishConfirmation(archive, draft) {
  const future = createFuture();
  waiting.push({ archive, draft, future });
  if (waiting.length === 1) setImmediate(releaseNext);
  return future.wait;
}

function releaseNext() {
  const envelope = waiting.shift();
  if (envelope === undefined) return;
  try {
    const entry = prepareEntry(envelope.draft);
    envelope.archive.sent.push(entry);
    envelope.future.finish.resolve(entry);
  } catch (error) {
    envelope.future.finish.reject(error);
  }
  if (waiting.length > 0) setImmediate(releaseNext);
}

function prepareEntry(draft) {
  if (draft === null || typeof draft !== "object") throw new TypeError("draft must be an object");
  const reference = requiredText(draft.invoiceId, "invoiceId");
  const recipient = requiredText(draft.email, "email").toLowerCase();
  if (!recipient.includes("@")) throw new TypeError("email must contain @");
  if (!Number.isSafeInteger(draft.totalMinor)) throw new TypeError("totalMinor must be a safe integer");
  return { invoiceId: reference, email: recipient, totalMinor: draft.totalMinor, status: "sent", channel: "email" };
}

function requiredText(value, field) {
  if (typeof value !== "string" || value.trim() === "") throw new TypeError(`${field} must be nonblank text`);
  return value.trim();
}

module.exports = { publishConfirmation };
