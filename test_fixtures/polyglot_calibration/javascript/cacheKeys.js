/**
 * Response cache keys, tag lists, and slot bookkeeping.
 */

export const normalizeHeaderName = (name) => {
  const trimmed = String(name).trim().toLowerCase();
  const collapsed = trimmed.replace(/\s+/g, "-");
  return collapsed.replace(/^-+|-+$/g, "");
};

export const stripBlankFields = (fields) => {
  if (!fields) {
    return {};
  }
  const kept = {};
  for (const [label, entry] of Object.entries(fields)) {
    if (entry === "" || entry === null || entry === undefined) {
      continue;
    }
    kept[label] = entry;
  }
  return kept;
};

export const buildCacheKey = (verb, query) => {
  const keys = Object.keys(query).sort();
  if (keys.length === 0) {
    return `${verb}#`;
  }
  const parts = keys.map((key) => `${key}:${query[key]}`);
  const body = parts.join("|");
  return `${verb}#${body}`;
};

export function splitTagList(raw) {
  const pieces = String(raw).split(",");
  const tags = [];
  for (const piece of pieces) {
    const label = piece.trim();
    if (label.length > 0) {
      tags.push(label);
    }
  }
  return tags;
}

export const cacheRules = {
  slotSize: 64,

  takeWindow(items, size) {
    const width = Math.max(1, Math.min(size, items.length));
    const start = Math.max(0, items.length - width);
    return items.slice(start, start + width);
  },

  remainingSlots(total, used) {
    const filled = Math.max(used, 0);
    const width = Math.max(this.slotSize, 1);
    const slots = Math.ceil((total - filled) / width);
    return Math.max(0, slots);
  },
};
