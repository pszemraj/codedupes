/**
 * Client-side rate limiting: token buckets and reset countdowns.
 */

export const secondsUntilReset = (resetAtMs, nowMs) => {
  const remaining = resetAtMs - nowMs;
  if (remaining <= 0) {
    return 0;
  }
  return Math.ceil(remaining / 1000);
};

export const computeBackoffWindow = (attempt, baseMs) => {
  const factor = Math.min(attempt, 6);
  const window = baseMs * 2 ** factor;
  const ceiling = Math.min(window, 60000);
  return Math.max(ceiling, baseMs);
};

export const isWithinBudget = (used, capacity) => {
  return used < capacity;
};

export class TokenBucket {
  capacity = 10;

  refill(tokens, elapsedMs, ratePerSec) {
    const gained = Math.floor((elapsedMs / 1000) * ratePerSec);
    const total = tokens + gained;
    return Math.min(total, this.capacity);
  }

  consume(tokens, cost) {
    if (tokens < cost) {
      return { allowed: false, remaining: tokens };
    }
    return { allowed: true, remaining: tokens - cost };
  }
}

export function sumRetryCosts(attempts) {
  let total = 0;
  for (const attempt of attempts) {
    total += attempt.cost;
  }
  return total;
}

export function firstThrottledHeader(headerPairs) {
  for (const pair of headerPairs) {
    if (pair.value === "throttled") {
      return pair.name;
    }
  }
  return null;
}

export function assertValidOptions(options) {
  if (!options) {
    return false;
  }
  if (!options.method) {
    return false;
  }
  if (options.timeoutMs <= 0) {
    return false;
  }
  return true;
}
