"use strict";

function addSafeInteger(total, amount, field) {
  const nextTotal = total + amount;
  if (!Number.isSafeInteger(nextTotal)) {
    throw new RangeError(`${field} must be a safe integer`);
  }
  return nextTotal;
}

module.exports = { addSafeInteger };
