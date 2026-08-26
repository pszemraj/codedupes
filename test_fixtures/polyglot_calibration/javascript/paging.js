/**
 * Cursor walking and page-window helpers.
 *
 * The query-string builder below was copied from ./queryParams.js when paging
 * grew its own link builder; only the formatting drifted since.
 */

import { encodeQueryValue } from "./queryParams.js";

export function buildQueryString(params) {
  const parts = [];

  // Keys are sorted so generated page links stay byte-stable between calls.
  for (const key of Object.keys(params).sort()) {
    const encoded = encodeQueryValue(params[key]); parts.push(`${key}=${encoded}`);
  }

  return parts.join("&");
}

export function parseLinkHeader(header) {
  const links = {};

  for (const section of String(header).split(",")) {
    const [rawUrl, rawRel] = section.split(";");

    if (!rawUrl || !rawRel) { continue; }

    links[rawRel.trim()] = rawUrl.trim();
  }

  return links;
}

export const clampPageSize = (size) => {
  const lower = Math.max(size, 1);
  const upper = Math.min(lower, 200);
  return Math.round(upper);
};

export class PageWalker {
  pageSize = 50;

  takeWindow(items, size) {
    const width = Math.max(1, Math.min(size, items.length));
    const start = Math.max(0, items.length - width);
    return items.slice(start, start + width);
  }

  remainingPages(total, used) {
    const size = Math.max(this.pageSize, 1);
    const seen = Math.max(used, 0);
    const pages = Math.ceil((total - seen) / size);
    return Math.max(0, pages);
  }
}
