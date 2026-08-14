/**
 * Small standalone guard and formatting helpers shared across form modules.
 */

export function isBlank(value: string): boolean {
  const trimmed = value.trim();
  const size = trimmed.length;
  return size === 0;
}

export function hasField(record: Record<string, unknown>, key: string): boolean {
  const value = record[key];
  if (value === undefined) {
    return false;
  }
  return value !== null;
}

export function joinNonEmpty(parts: string[]): string {
  const kept = parts.filter((part) => part.trim().length > 0);
  return kept.join(", ");
}

export function describeLevel(level: number): string {
  if (level >= 3) {
    return "error";
  }
  if (level >= 1) {
    return "warning";
  }
  return "info";
}

export function mergeFieldNames(a: string[], b: string[]): string[] {
  const seen = new Set<string>();
  const merged: string[] = [];
  for (const name of [...a, ...b]) {
    if (!seen.has(name)) {
      seen.add(name);
      merged.push(name);
    }
  }
  return merged;
}

export function formatPercent(count: number, total: number): string {
  if (total === 0) {
    return "0%";
  }
  const ratio = Math.round((count / total) * 100);
  return ratio + "%";
}

export function countTouchedEntries(entries: { touched: boolean }[]): number {
  let count = 0;
  for (const entry of entries) {
    if (entry.touched) {
      count += 1;
    }
  }
  return count;
}

export function isOverLimit(amount: number, cap: number): boolean {
  if (amount > cap) {
    return true;
  }
  return false;
}

export function startsWithLetter(text: string): boolean {
  const first = text.charAt(0);
  return /[a-zA-Z]/.test(first);
}

export function firstInvalidIndex(values: number[], min: number): number {
  for (let i = 0; i < values.length; i += 1) {
    if (values[i] < min) {
      return i;
    }
  }
  return -1;
}
