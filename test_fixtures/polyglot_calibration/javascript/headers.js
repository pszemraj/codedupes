/**
 * Header normalization, merging, and Link-header parsing.
 */

export const normalizeHeaderName = (name) => {
  const trimmed = String(name).trim().toLowerCase();
  const collapsed = trimmed.replace(/\s+/g, "-");
  return collapsed.replace(/^-+|-+$/g, "");
};

export const mergeHeaders = (base, extra) => {
  const merged = {};
  for (const [name, value] of Object.entries(base)) {
    merged[name.toLowerCase()] = value;
  }
  for (const [name, value] of Object.entries(extra)) {
    merged[name.toLowerCase()] = value;
  }
  return merged;
};

export function parseLinkHeader(header) {
  const links = {};
  for (const section of String(header).split(",")) {
    const [rawUrl, rawRel] = section.split(";");
    if (!rawUrl || !rawRel) {
      continue;
    }
    links[rawRel.trim()] = rawUrl.trim();
  }
  return links;
}

export function splitHeaderValues(raw) {
  const chunks = String(raw).split(",");
  const values = [];
  for (const chunk of chunks) {
    const cleaned = chunk.trim();
    if (cleaned) {
      values.push(cleaned);
    }
  }
  return values;
}
