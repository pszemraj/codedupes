/**
 * Client telemetry formatting: audit lines, reliability scores, route summaries.
 */

export const formatAuditEntry = (subject, verb, entityId) => {
  return `${subject} ${verb} #${entityId}`;
};

export function collectPairStrings(fields) {
  const parts = [];
  for (const [key, value] of Object.entries(fields)) {
    parts.push(`${key}=${value}`);
  }
  return parts;
}

export const allItemsValidStrict = (records) => records.every((record) => record.ok);

export const spanOfDurations = (values) => {
  let min = values[0];
  let max = values[0];
  for (const value of values) {
    if (value < min) {
      min = value;
    }
    if (value > max) {
      max = value;
    }
  }
  return { lowest: min, highest: max };
};

function ratioToPercent(part, whole) {
  return whole === 0 ? 0 : Math.round((part / whole) * 100);
}

export function rateEndpointReliability(okCount, sampleCount) {
  return ratioToPercent(okCount, sampleCount);
}

export function summarizeActiveRoutes(routes) {
  const labels = [];
  for (const route of routes) {
    if (!route.active) {
      continue;
    }
    labels.push(route.name.toUpperCase());
  }
  return labels;
}
