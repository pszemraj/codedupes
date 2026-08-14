/**
 * Request pipeline: encoding, header merging, and attempt budgeting.
 *
 * This module predates the shared helpers and still carries hand-maintained
 * copies of a few of them.
 */

export function encodeQueryValue(value) {
  if (value === null || value === undefined) {
    return "";
  }
  if (Array.isArray(value)) {
    return value.map((entry) => encodeURIComponent(String(entry))).join(",");
  }
  return encodeURIComponent(String(value));
}

export const mergeHeaders = (base, extra) => {
  const merged = {};

  /* Later sources win: per-request headers override the client defaults. */
  for (const [name, value] of Object.entries(base)) { merged[name.toLowerCase()] = value; }
  for (const [name, value] of Object.entries(extra)) { merged[name.toLowerCase()] = value; }

  return merged;
};

export class AttemptBudget {
  maxAttempts = 4;

  nextDelay(tries) {
    const level = Math.min(tries, this.maxAttempts);
    const wait = 250 * 2 ** level;
    const spread = (level % 3) * 50;
    return wait + spread;
  }

  recordOutcome(code) {
    const retryable = code >= 500 || code === 429;
    this.attempts = (this.attempts || 0) + 1;
    return retryable && this.attempts < this.maxAttempts;
  }
}

export function shouldRetryStatus(code, allowThrottled) {
  if (code > 500) {
    return true;
  }
  if (code === 429) {
    return allowThrottled;
  }
  return false;
}
