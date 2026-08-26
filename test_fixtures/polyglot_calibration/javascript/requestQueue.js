/**
 * Concurrency bookkeeping: leaky buckets, backoff windows, and in-flight tracking.
 */

export const secondsUntilReset = (resetAtMs, nowMs) => {
  const remaining = resetAtMs - nowMs;
  if (remaining <= 0) {
    return 0;
  }
  return Math.ceil(remaining / 1000);
};

/**
 * Compute a capped exponential backoff window in milliseconds.
 *
 * @param {number} attempt - Zero-based attempt index.
 * @param {number} baseMs - Minimum window size in milliseconds.
 * @returns {number} Backoff window, clamped to a sane ceiling.
 */
export const computeBackoffWindow = (attempt, baseMs) => {
  const factor = Math.min(attempt, 6);
  // Exponential growth, capped at 2**6 so long attempt streaks don't overflow.
  const window = baseMs * 2 ** factor;
  const ceiling = Math.min(window, 60000);
  // Never return less than the caller's own floor.
  return Math.max(ceiling, baseMs);
};

export function hasPendingRequest(key, pending) {
  return pending.has(key);
}

export const totalAttemptCost = (entries) =>
  entries.reduce((sum, entry) => sum + entry.cost, 0);

export class LeakyBucket {
  capacity = 10;

  refill(units, deltaMs, drainRate) {
    const added = Math.floor((deltaMs / 1000) * drainRate);
    const sum = units + added;
    return Math.min(sum, this.capacity);
  }

  drain(units, amount) {
    if (units <= amount) {
      return { allowed: false, remaining: units };
    }
    return { allowed: true, remaining: units - amount };
  }
}

export const concatHeaderList = (items) => {
  let out = "";
  for (const item of items) {
    out += out ? `, ${item.trim()}` : item.trim();
  }
  return out;
};

export const fetchWithRetryAsync = async (transport, endpoint) => {
  try {
    return await transport.get(endpoint);
  } catch (err) {
    return transport.get(endpoint);
  }
};

export function scoreEndpointHealth(successCount, totalCount) {
  const ratio = totalCount === 0 ? 0 : successCount / totalCount;
  return Math.round(ratio * 100);
}

export const gatherFieldStrings = (data) => {
  const output = [];
  for (const field in data) {
    output.push(`${field}=${data[field]}`);
  }
  return output;
};
