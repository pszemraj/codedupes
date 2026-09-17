/** Offline pagination clients using an injected asynchronous transport. */

export async function fetchRecords(fetchPage, { maxPages = 100, signal } = {}) {
  if (!Number.isInteger(maxPages) || maxPages < 1) {
    throw new RangeError("INVALID_PAGE_LIMIT");
  }
  const visited = new Set();
  const identifiers = new Set();
  const records = [];
  let cursor = null;
  let pages = 0;
  while (true) {
    if (signal?.aborted) throw new Error("ABORTED");
    if (visited.has(cursor)) throw new Error("CURSOR_CYCLE");
    if (pages === maxPages) throw new RangeError("PAGE_LIMIT");
    visited.add(cursor);
    const page = await fetchPage(cursor);
    if (signal?.aborted) throw new Error("ABORTED");
    if (page === null || typeof page !== "object" || !Array.isArray(page.rows)) {
      throw new TypeError("INVALID_PAGE");
    }
    if (page.next !== null && (typeof page.next !== "string" || page.next.length === 0)) {
      throw new TypeError("INVALID_CURSOR");
    }
    for (const row of page.rows) {
      if (
        row === null ||
        typeof row !== "object" ||
        typeof row.id !== "string" ||
        row.id.length === 0 ||
        !Number.isSafeInteger(row.value)
      ) {
        throw new TypeError("INVALID_ROW");
      }
      if (!identifiers.has(row.id)) {
        identifiers.add(row.id);
        records.push({ id: row.id, value: row.value });
      }
    }
    pages += 1;
    if (page.next === null) return { records, pages };
    cursor = page.next;
  }
}

export function readSnapshot(fetchPage, { maxPages = 100, signal } = {}) {
  if (!Number.isInteger(maxPages) || maxPages < 1) {
    return Promise.reject(new RangeError("INVALID_PAGE_LIMIT"));
  }
  const requested = new Set();
  const collected = new Map();
  let completed = 0;
  function visit(token) {
    try {
      if (signal?.aborted) throw new Error("ABORTED");
      if (requested.has(token)) throw new Error("CURSOR_CYCLE");
      if (completed >= maxPages) throw new RangeError("PAGE_LIMIT");
      requested.add(token);
      const pending = fetchPage(token);
      return Promise.resolve(pending).then((response) => {
        if (signal?.aborted) throw new Error("ABORTED");
        if (response === null || typeof response !== "object" || !Array.isArray(response.rows)) {
          throw new TypeError("INVALID_PAGE");
        }
        const continuation = response.next;
        if (
          continuation !== null &&
          (typeof continuation !== "string" || continuation.length === 0)
        ) {
          throw new TypeError("INVALID_CURSOR");
        }
        Array.from(response.rows).forEach((item) => {
          if (
            item === null ||
            typeof item !== "object" ||
            typeof item.id !== "string" ||
            item.id.length === 0 ||
            !Number.isSafeInteger(item.value)
          ) {
            throw new TypeError("INVALID_ROW");
          }
          if (!collected.has(item.id)) {
            collected.set(item.id, { id: item.id, value: item.value });
          }
        });
        completed += 1;
        if (continuation === null) {
          return { records: Array.from(collected.values()), pages: completed };
        }
        return visit(continuation);
      });
    } catch (error) {
      return Promise.reject(error);
    }
  }
  return visit(null);
}
