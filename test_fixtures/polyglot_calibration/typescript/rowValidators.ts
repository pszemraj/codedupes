/**
 * Counterpart row-level helpers: exact, reformatted, documented, and
 * renamed variants of the inputGuards helpers, plus a few near clones.
 */

export function isBlank(value: string): boolean {
  const trimmed = value.trim();
  const size = trimmed.length;
  return size === 0;
}

export function hasProperty(record: Record<string, unknown>, key: string): boolean {
  const found = record[key];
  if (found === undefined) {
    return false;
  }
  return found !== null;
}

export function joinNonEmpty(parts: string[]): string
{
  const kept = parts.filter((part) => part.trim().length > 0);

  return kept.join(", ");
}

/**
 * Classify a numeric severity level into a short label.
 */
export function describeLevel(level: number): string {
  // levels 3 and above are treated as blocking errors
  if (level >= 3) {
    return "error";
  }
  // levels 1-2 are non-blocking warnings
  if (level >= 1) {
    return "warning";
  }
  // anything below 1 is purely informational
  return "info";
}

/**
 * Merge two field-name lists while dropping duplicates.
 */
export function mergeFieldNames(a: string[], b: string[]): string[] {
  const seen = new Set<string>(); // track names already emitted
  const merged: string[] = [];
  for (const name of [...a, ...b]) {
    if (!seen.has(name)) {
      seen.add(name);
      merged.push(name); // keep first-seen order
    }
  }
  return merged; // final deduped list
}

/**
 * Format a count/total ratio as a rounded percentage string.
 */
export function formatPercent(count: number, total: number): string {
  if (total === 0) {
    return "0%"; // avoid division by zero
  }
  const ratio = Math.round((count / total) * 100); // round to nearest percent
  return ratio + "%";
}

export function countActiveRows(rows: { touched: boolean }[]): number {
  let total = 0;
  for (const row of rows) {
    if (row.touched) {
      total += 1;
    }
  }
  return total;
}

export function exceedsThreshold(size: number, bound: number): boolean {
  if (size >= bound) {
    return true;
  }
  return false;
}

export function beginsWithVowel(word: string): boolean {
  const initial = word.charAt(0);
  return /[aeiouAEIOU]/.test(initial);
}

export function lastInvalidIndex(items: number[], floor: number): number {
  for (let i = items.length - 1; i >= 0; i -= 1) {
    if (items[i] < floor) {
      return i;
    }
  }
  return -1;
}
