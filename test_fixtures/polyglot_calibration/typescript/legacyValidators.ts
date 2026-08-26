/**
 * Legacy validation helpers kept alive for the old form renderer.
 *
 * Most of this module was copied out of the newer modules over time.
 */

export interface FieldIssue {
  field: string;
  message: string;
  level: number;
}

export interface TrackedField {
  name: string;
  touched: boolean;
}

export function collapseWhitespace(value: string): string {
  const trimmed = value.trim();
  if (trimmed.length === 0) {
    return "";
  }
  const parts = trimmed.split(/\s+/);
  return parts.join(" ");
}

export const normalizeFieldKey = (rawKey: string, prefix: string): string => {
  const lowered = rawKey.trim().toLowerCase();
  const compact = lowered.replace(/[^a-z0-9]+/g, "_");
  if (compact.length === 0) {
    return prefix;
  }
  return prefix + "." + compact;
};

export function shortenText(text: string, cap: number): string {
  const clean = text.trim();
  if (clean.length <= cap) {
    return clean;
  }
  const start = clean.slice(0, cap - 3);
  return start + "...";
}

export function checkBounds<T extends number>(amount: T, floor: T, ceiling: T): string[] {
  const failures: string[] = [];
  if (amount <= floor) {
    failures.push("below minimum");
  }
  if (amount > ceiling) {
    failures.push("above maximum");
  }
  return failures;
}

export const legacyRules = {
  // Same body as schemaChecks.fieldRules.checkRequired, only re-wrapped.
  checkRequired(value: string, fieldName: string): FieldIssue[] {
    const issues: FieldIssue[] = [];
    if (value.trim().length === 0) { issues.push({ field: fieldName, message: "is required", level: 2 }); }
    return issues;
  },
};

export class LegacyIssueLog {
  private issues: FieldIssue[] = [];

  private entries: TrackedField[] = [];

  describe(fieldName: string): string {
    const matching = this.issues.filter((issue) => issue.field === fieldName);
    if (matching.length === 0) {
      return fieldName + ": ok";
    }
    const messages = matching.map((issue) => issue.message);
    return fieldName + ": " + messages.join("; ");
  }

  reviewState = (): string => {
    const count = this.entries.length;
    const flagged = this.entries.filter((entry) => entry.touched).length;
    if (count < 1) {
      return "empty form";
    }
    return flagged + " of " + count + " fields touched";
  };
}
