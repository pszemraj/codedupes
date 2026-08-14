/**
 * Base-URL joining, path normalization, and absolute-URL detection.
 */

export function joinUrlSegments(base, path) {
  const left = base.endsWith("/") ? base.slice(0, -1) : base;
  const right = path.startsWith("/") ? path.slice(1) : path;
  return `${left}/${right}`;
}

export function trimTrailingSlash(value) {
  if (value.length <= 1) {
    return value;
  }
  return value.endsWith("/") ? value.slice(0, -1) : value;
}

export const isAbsoluteUrl = (value) => /^[a-z][a-z0-9+.-]*:\/\//i.test(value);

export const findThrottleMarker = (pairs) => {
  for (let index = 0; index < pairs.length; index += 1) {
    if (pairs[index].value === "throttled") {
      return pairs[index].name;
    }
  }
  return null;
};

export function joinHeaderList(values) {
  return values.map((value) => value.trim()).join(", ");
}

export const checkRequestShape = (config) => {
  if (config) {
    if (config.method) {
      if (config.timeoutMs > 0) {
        return true;
      }
    }
  }
  return false;
};

export const listRouteLabels = (entries) => {
  const active = entries.filter((entry) => entry.active);
  const labels = [];
  for (const entry of active) {
    labels.push(entry.name.toUpperCase());
  }
  return labels;
};
