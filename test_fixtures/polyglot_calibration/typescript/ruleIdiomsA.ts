/**
 * Imperative-style implementations paired with functional-style
 * counterparts in ruleIdiomsB.ts. Same behavior, different idiom.
 */

export function totalFieldLength(fields: string[]): number {
  let sum = 0;
  for (const field of fields) {
    sum += field.length;
  }
  return sum;
}

export function allFieldsFilled(values: string[]): boolean {
  for (const value of values) {
    if (value.trim().length === 0) {
      return false;
    }
  }
  return true;
}

export function anyValueTooLarge(values: number[], cap: number): boolean {
  for (const value of values) {
    if (value > cap) {
      return true;
    }
  }
  return false;
}

export function levelLabel(score: number): string {
  let label: string;
  if (score >= 90) {
    label = "excellent";
  } else if (score >= 70) {
    label = "good";
  } else if (score >= 50) {
    label = "fair";
  } else {
    label = "poor";
  }
  return label;
}

export function countTruthyProps(record: Record<string, boolean>): number {
  let total = 0;
  for (const [, flag] of Object.entries(record)) {
    if (flag) {
      total += 1;
    }
  }
  return total;
}

export function readNestedLabel(entry: { profile?: { label?: string } }): string {
  if (entry.profile && entry.profile.label) {
    return entry.profile.label;
  }
  return "unlabeled";
}

export function prefixAllNames(names: string[], tag: string): string[] {
  const result: string[] = [];
  for (const name of names) {
    result.push(tag + ":" + name);
  }
  return result;
}

export function isRowComplete(name: string, value: string, touched: boolean): boolean {
  if (name.trim().length === 0) {
    return false;
  }
  if (value.trim().length === 0) {
    return false;
  }
  if (!touched) {
    return false;
  }
  return true;
}
