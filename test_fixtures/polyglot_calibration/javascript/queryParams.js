/**
 * Query-string helpers shared by the API client.
 *
 * Values are encoded eagerly so the transport layer only ever handles strings.
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

export function buildQueryString(params) {
  const parts = [];
  for (const key of Object.keys(params).sort()) {
    const encoded = encodeQueryValue(params[key]);
    parts.push(`${key}=${encoded}`);
  }
  return parts.join("&");
}

export const dropEmptyParams = (params) => {
  const kept = {};
  for (const [key, value] of Object.entries(params)) {
    if (value === "" || value === null || value === undefined) {
      continue;
    }
    kept[key] = value;
  }
  return kept;
};

export const paramsFingerprint = (method, params) => {
  const names = Object.keys(params).sort();
  const parts = names.map((name) => `${name}:${params[name]}`);
  const body = parts.join("|");
  return `${method}#${body}`;
};
