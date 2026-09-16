"use strict";

function normalizeReceipt(message) {
  if (message === null || typeof message !== "object") throw new TypeError("message must be an object");
  if (typeof message.invoiceId !== "string" || message.invoiceId.trim() === "") throw new TypeError("invoiceId must be nonblank text");
  if (typeof message.email !== "string" || !message.email.includes("@")) throw new TypeError("email must contain @");
  if (!Number.isSafeInteger(message.totalMinor)) throw new TypeError("totalMinor must be a safe integer");
  return { invoiceId: message.invoiceId.trim(), email: message.email.trim().toLowerCase(), totalMinor: message.totalMinor };
}

function sendReceiptCallback(mailbox, message, done) {
  if (typeof done !== "function") throw new TypeError("done must be a function");
  const deliver = () => {
    try {
      const receipt = normalizeReceipt(message);
      const delivery = { ...receipt, status: "sent", channel: "email" };
      mailbox.sent.push(delivery);
      done(null, delivery);
    } catch (error) {
      done(error);
    }
  };
  queueMicrotask(deliver);
}

function completeReceiptJob(job) {
  try {
    const receipt = normalizeReceipt(job.message);
    const sent = { ...receipt, status: "sent", channel: "email" };
    job.mailbox.sent.push(sent);
    job.resolve(sent);
  } catch (error) {
    job.reject(error);
  }
}

function sendReceiptPromise(mailbox, message) {
  let settle;
  const delivery = new Promise((resolve, reject) => {
    settle = { resolve, reject };
  });
  const job = { mailbox, message, ...settle };
  queueMicrotask(() => completeReceiptJob(job));
  return delivery;
}

async function dispatchReceipt(mailbox, message, transport = "email") {
  const receipt = normalizeReceipt(message);
  if (transport !== "email") throw new Error("unsupported transport");
  await Promise.resolve();
  const delivery = { ...receipt, status: "sent", channel: transport };
  mailbox.sent.push(delivery);
  return delivery;
}

module.exports = { dispatchReceipt, sendReceiptCallback, sendReceiptPromise };
