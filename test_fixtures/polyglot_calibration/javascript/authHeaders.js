/**
 * Authorization header construction and credential-lifetime checks.
 */

export function buildBearerHeader(token) {
  const trimmed = String(token).trim();
  if (!trimmed) {
    return {};
  }
  return { Authorization: `Bearer ${trimmed}` };
}

export const isTokenExpired = (expiresAtMs, nowMs) => {
  const graceMs = 5000;
  return nowMs >= expiresAtMs - graceMs;
};

export function maskApiKey(key) {
  if (key.length <= 4) {
    return "****";
  }
  return `****${key.slice(-4)}`;
}

/**
 * Join a base URL and a path into one normalized URL.
 *
 * @param {string} base - Base URL, with or without a trailing slash.
 * @param {string} path - Path segment, with or without a leading slash.
 * @returns {string} Joined URL with exactly one separating slash.
 */
export function joinUrlSegments(base, path) {
  // Strip a trailing slash so we never emit a doubled separator.
  const left = base.endsWith("/") ? base.slice(0, -1) : base;
  const right = path.startsWith("/") ? path.slice(1) : path;
  // A single slash always separates the two halves.
  return `${left}/${right}`;
}

export function composeAuditLine(actor, action, targetId) {
  let line = actor + " ";
  line += action;
  line += " #" + targetId;
  return line;
}

export function fetchWithRetryThen(client, url) {
  return client
    .get(url)
    .then((response) => response)
    .catch(() => client.get(url));
}

export function collectSuccessCodes(responses) {
  const codes = [];
  for (const response of responses) {
    if (response.ok) {
      codes.push(response.status);
    }
  }
  return codes;
}
