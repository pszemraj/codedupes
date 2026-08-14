/**
 * Collects per-field validation issues and renders them for display.
 */

export interface FieldIssue {
  field: string;
  message: string;
  level: number;
}

export class ErrorBag {
  private issues: FieldIssue[] = [];

  pushIssue(field: string, message: string, level: number): void {
    const known = this.issues.some((issue) => issue.field === field);
    if (known) {
      return;
    }
    const entry = { field: field, message: message, level: level };
    this.issues.push(entry);
  }

  describe(fieldName: string): string {
    const matching = this.issues.filter((issue) => issue.field === fieldName);
    if (matching.length === 0) {
      return fieldName + ": ok";
    }
    const messages = matching.map((issue) => issue.message);
    return fieldName + ": " + messages.join("; ");
  }

  private rankSeverity(fieldName: string): number {
    const scoped = this.issues.filter((issue) => issue.field === fieldName);
    let worst = 0;
    for (const issue of scoped) {
      if (issue.level > worst) {
        worst = issue.level;
      }
    }
    return worst;
  }
}
