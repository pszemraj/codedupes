/**
 * Response-body parsing, error extraction, and session-lifetime checks.
 */

/**
 * Build an Authorization header carrying a bearer token.
 *
 * @param {string} token - Raw access token.
 * @returns {Record<string, string>} Header map, empty when no token is set.
 */
export function buildBearerHeader(token) {
  const trimmed = String(token).trim();
  // Callers pass an empty string for anonymous requests; skip the header then.
  if (!trimmed) {
    return {};
  }
  // The scheme name is case-sensitive per RFC 6750.
  return { Authorization: `Bearer ${trimmed}` };
}

export const isSessionStale = (expiryMs, currentMs) => {
  const bufferMs = 15000;
  return currentMs >= expiryMs - bufferMs;
};

export function extractErrorMessage(body) {
  if (!body || typeof body !== "object") {
    return "unknown error";
  }
  return body.error || body.message || "unknown error";
}

export function stripTrailingSlash(text) {
  if (text.length <= 1) {
    return text;
  }
  return text.endsWith("/") ? text.slice(0, -1) : text;
}

export const selectOkStatuses = (results) =>
  results.filter((result) => result.ok).map((result) => result.status);

export function allItemsValid(items) {
  let valid = true;
  for (const item of items) {
    if (!item.ok) {
      valid = false;
    }
  }
  return valid;
}

export function minMaxLatency(samples) {
  let lowest = samples[0];
  for (const sample of samples) {
    if (sample < lowest) {
      lowest = sample;
    }
  }
  let highest = samples[0];
  for (const sample of samples) {
    if (sample > highest) {
      highest = sample;
    }
  }
  return { lowest, highest };
}
