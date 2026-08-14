/**
 * Retry budgeting and backoff policy for the API client.
 */

export class RetryBudget {
  maxAttempts = 4;

  nextDelay(attempt) {
    const step = Math.min(attempt, this.maxAttempts);
    const base = 250 * 2 ** step;
    const jitter = (step % 3) * 50;
    return base + jitter;
  }

  recordOutcome(status) {
    const failed = status >= 500 || status === 429;
    this.attempts = (this.attempts || 0) + 1;
    return failed && this.attempts < this.maxAttempts;
  }
}

export const clampRetryDelay = (delay) => {
  const floor = Math.max(delay, 100);
  const ceiling = Math.min(floor, 30000);
  return Math.round(ceiling);
};

export function isRetryableStatus(status, allowRateLimit) {
  if (status >= 500) {
    return true;
  }
  if (status === 429) {
    return allowRateLimit;
  }
  return false;
}
