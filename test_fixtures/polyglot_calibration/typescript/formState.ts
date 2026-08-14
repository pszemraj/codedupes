/**
 * Tracks which fields the user touched and summarizes the current form state.
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

export const canonicalizeKey = (label: string, scope: string): string => {
  const folded = label.trim().toLowerCase();
  const squeezed = folded.replace(/[^a-z0-9]+/g, "_");
  if (squeezed.length === 0) {
    return scope;
  }
  return scope + "." + squeezed;
};

export class FormTracker {
  private problems: FieldIssue[] = [];

  private fields: TrackedField[] = [];

  summarize(fieldLabel: string): string {
    const relevant = this.problems.filter((problem) => problem.field === fieldLabel);
    if (relevant.length === 0) {
      return fieldLabel + ": ok";
    }
    const lines = relevant.map((problem) => problem.message);
    lines.sort();
    return fieldLabel + ": " + lines.join("; ");
  }

  summarizeState = (): string => {
    const total = this.fields.length;
    const dirty = this.fields.filter((field) => field.touched).length;
    if (total === 0) {
      return "empty form";
    }
    return dirty + " of " + total + " fields touched";
  };
}
