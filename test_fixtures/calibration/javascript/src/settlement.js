"use strict";

function checkedSummaries(summaries) {
  const seen = new Set();
  for (const summary of summaries) {
    if (summary === null || typeof summary !== "object") throw new TypeError("expected summary");
    if (typeof summary.invoiceId !== "string" || summary.invoiceId.trim() === "") {
      throw new TypeError("invoiceId must be nonblank text");
    }
    if (!Number.isSafeInteger(summary.totalMinor) || !Number.isSafeInteger(summary.count) || summary.count < 1) {
      throw new TypeError("summary totals and counts must be integers");
    }
    if (seen.has(summary.invoiceId)) throw new Error("duplicate invoiceId");
    seen.add(summary.invoiceId);
  }
  return [...summaries].sort((left, right) => left.invoiceId.localeCompare(right.invoiceId));
}

function planTransfers(summaries) {
  const lines = [];
  const transfers = [];
  for (const summary of checkedSummaries(summaries)) {
    if (summary.totalMinor > 0) {
      lines.push([summary.invoiceId, "ready", summary.totalMinor]);
      transfers.push([summary.invoiceId, summary.totalMinor]);
    } else if (summary.totalMinor < 0) {
      lines.push([summary.invoiceId, "refund_due", summary.totalMinor]);
    } else {
      lines.push([summary.invoiceId, "balanced", 0]);
    }
  }
  return { lines, transfers, deferredTotal: 0, events: [] };
}

function planTransfersWithMinimum(summaries, minimumPayoutMinor = 0) {
  if (!Number.isSafeInteger(minimumPayoutMinor) || minimumPayoutMinor < 0) {
    throw new TypeError("minimumPayoutMinor must be a nonnegative integer");
  }
  const lines = [];
  const transfers = [];
  const events = [];
  let deferredTotal = 0;
  for (const summary of checkedSummaries(summaries)) {
    if (summary.totalMinor > 0 && summary.totalMinor < minimumPayoutMinor) {
      lines.push([summary.invoiceId, "deferred", summary.totalMinor]);
      deferredTotal += summary.totalMinor;
      events.push([summary.invoiceId, "below_minimum", minimumPayoutMinor]);
    } else if (summary.totalMinor > 0) {
      lines.push([summary.invoiceId, "ready", summary.totalMinor]);
      transfers.push([summary.invoiceId, summary.totalMinor]);
    } else if (summary.totalMinor < 0) {
      lines.push([summary.invoiceId, "refund_due", summary.totalMinor]);
    } else {
      lines.push([summary.invoiceId, "balanced", 0]);
    }
  }
  return { lines, transfers, deferredTotal, events };
}

function schedulePayouts(summaries, options = {}) {
  const minimum = options.minimumPayoutMinor ?? 0;
  if (!Number.isSafeInteger(minimum) || minimum < 0) {
    throw new TypeError("minimumPayoutMinor must be a nonnegative integer");
  }
  return checkedSummaries(summaries).reduce(
    (plan, summary) => {
      if (summary.totalMinor > 0 && summary.totalMinor < minimum) {
        plan.lines.push([summary.invoiceId, "deferred", summary.totalMinor]);
        plan.deferredTotal += summary.totalMinor;
        plan.events.push([summary.invoiceId, "below_minimum", minimum]);
      } else if (summary.totalMinor > 0) {
        plan.lines.push([summary.invoiceId, "ready", summary.totalMinor]);
        plan.transfers.push([summary.invoiceId, summary.totalMinor]);
      } else if (summary.totalMinor < 0) {
        plan.lines.push([summary.invoiceId, "refund_due", summary.totalMinor]);
      } else {
        plan.lines.push([summary.invoiceId, "balanced", 0]);
      }
      return plan;
    },
    { lines: [], transfers: [], deferredTotal: 0, events: [] },
  );
}

function compileTransfers(summaries, minimumPayoutMinor = 0) {
  if (!Number.isSafeInteger(minimumPayoutMinor) || minimumPayoutMinor < 0) {
    throw new TypeError("minimumPayoutMinor must be a nonnegative integer");
  }
  const plan = { lines: [], transfers: [], deferredTotal: 0, events: [] };
  const ordered = checkedSummaries(summaries);
  let index = 0;
  while (index < ordered.length) {
    const summary = ordered[index];
    if (summary.totalMinor > 0 && summary.totalMinor < minimumPayoutMinor) {
      plan.lines.push([summary.invoiceId, "deferred", summary.totalMinor]);
      plan.deferredTotal += summary.totalMinor;
      plan.events.push([summary.invoiceId, "below_minimum", minimumPayoutMinor]);
    } else if (summary.totalMinor > 0) {
      plan.lines.push([summary.invoiceId, "ready", summary.totalMinor]);
      plan.transfers.push([summary.invoiceId, summary.totalMinor]);
    } else if (summary.totalMinor < 0) {
      plan.lines.push([summary.invoiceId, "refund_due", summary.totalMinor]);
    } else {
      plan.lines.push([summary.invoiceId, "balanced", 0]);
    }
    index += 1;
  }
  return plan;
}

module.exports = { compileTransfers, planTransfers, planTransfersWithMinimum, schedulePayouts };
