/**
 * Declarative field checks evaluated by the form validator.
 */

export interface FieldIssue {
  field: string;
  message: string;
  level: number;
}

export const normalizeFieldKey = (rawKey: string, prefix: string): string => {
  const lowered = rawKey.trim().toLowerCase();
  const compact = lowered.replace(/[^a-z0-9]+/g, "_");
  if (compact.length === 0) {
    return prefix;
  }
  return prefix + "." + compact;
};

export function clipFieldText(input: string, maxLength: number): string {
  const tidy = input.trim();
  if (tidy.length < maxLength) {
    return tidy;
  }
  const kept = tidy.slice(0, maxLength - 3);
  return kept + "...";
}

export function validateRange<T extends number>(value: T, low: T, high: T): string[] {
  const problems: string[] = [];
  if (value < low) {
    problems.push("below minimum");
  }
  if (value > high) {
    problems.push("above maximum");
  }
  return problems;
}

export function matchesPattern(value: string, pattern: RegExp, fieldName: string): string[] {
  const failures: string[] = [];
  const candidate = value.trim();
  if (!pattern.test(candidate)) {
    failures.push(fieldName + " has an invalid format");
  }
  return failures;
}

export const fieldRules = {
  checkRequired(value: string, fieldName: string): FieldIssue[] {
    const issues: FieldIssue[] = [];
    if (value.trim().length === 0) {
      issues.push({ field: fieldName, message: "is required", level: 2 });
    }
    return issues;
  },
};
