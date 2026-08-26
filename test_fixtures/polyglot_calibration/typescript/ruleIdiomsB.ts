/**
 * Functional-style counterparts to ruleIdiomsA.ts. Same observable
 * behavior, expressed with array methods, early returns, and optional
 * chaining instead of loops and explicit guards.
 */

export function combinedLabelWidth(labels: string[]): number {
  return labels.reduce((width, label) => width + label.length, 0);
}

export function everyEntryPresent(entries: string[]): boolean {
  return entries.every((entry) => entry.trim().length > 0);
}

export function someAmountExceeds(amounts: number[], limit: number): boolean {
  return amounts.some((amount) => amount > limit);
}

export function gradeTier(mark: number): string {
  if (mark >= 90) {
    return "excellent";
  }
  if (mark >= 70) {
    return "good";
  }
  if (mark >= 50) {
    return "fair";
  }
  return "poor";
}

export function countEnabledFlags(map: Record<string, boolean>): number {
  let count = 0;
  for (const name of Object.keys(map)) {
    if (map[name]) {
      count += 1;
    }
  }
  return count;
}

export function resolveDisplayName(item: { profile?: { label?: string } }): string {
  return item.profile?.label ?? "unlabeled";
}

export function tagEachLabel(labels: string[], marker: string): string[] {
  return labels.map((label) => marker + ":" + label);
}

export function isEntryFilled(label: string, content: string, active: boolean): boolean {
  return label.trim().length > 0 && content.trim().length > 0 && active;
}
