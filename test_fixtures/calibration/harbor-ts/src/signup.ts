/** Typed validation rules for an offline account-registration workflow. */
export type Submission =
  | { kind: "person"; email: string; age: number; marketing: boolean }
  | { kind: "business"; email: string; company: string; country: "GB" | "US"; vatId: string };

export type Field = "email" | "age" | "marketing" | "company" | "vatId";
export type Code = "invalid_email" | "invalid_age" | "underage_marketing" |
  "invalid_company" | "invalid_vat";
export interface Issue { field: Field; code: Code }

export function validateSignup(input: Submission): Issue[] {
  const issues: Issue[] = [];
  const email = input.email.trim().toLowerCase();
  if (!/^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$/.test(email)) {
    issues.push({ field: "email", code: "invalid_email" });
  }
  if (input.kind === "person") {
    const validAge = Number.isInteger(input.age) && input.age >= 0 && input.age <= 130;
    if (!validAge) {
      issues.push({ field: "age", code: "invalid_age" });
    } else if (input.marketing && input.age < 18) {
      issues.push({ field: "marketing", code: "underage_marketing" });
    }
  } else {
    const company = input.company.trim();
    if (company.length === 0 || company.length > 80) {
      issues.push({ field: "company", code: "invalid_company" });
    }
    if (input.country === "GB") {
      const vat = input.vatId.replace(/[ -]/g, "").toUpperCase();
      if (!/^GB[0-9]{9}$/.test(vat)) {
        issues.push({ field: "vatId", code: "invalid_vat" });
      }
    }
  }
  return issues;
}

export function evaluateRegistration(input: Submission): Issue[] {
  type Rule = { field: Field; code: Code; reject: (value: Submission) => boolean };
  const rules: readonly Rule[] = [
    {
      field: "email", code: "invalid_email",
      reject: value => !/^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$/.test(value.email.trim().toLowerCase()),
    },
    {
      field: "age", code: "invalid_age",
      reject: value => value.kind === "person" &&
        (!Number.isInteger(value.age) || value.age < 0 || value.age > 130),
    },
    {
      field: "marketing", code: "underage_marketing",
      reject: value => value.kind === "person" && Number.isInteger(value.age) &&
        value.age >= 0 && value.age <= 130 && value.age < 18 && value.marketing,
    },
    {
      field: "company", code: "invalid_company",
      reject: value => value.kind === "business" &&
        (value.company.trim().length === 0 || value.company.trim().length > 80),
    },
    {
      field: "vatId", code: "invalid_vat",
      reject: value => value.kind === "business" && value.country === "GB" &&
        !/^GB[0-9]{9}$/.test(value.vatId.replace(/[ -]/g, "").toUpperCase()),
    },
  ];
  const failures: Issue[] = [];
  for (const rule of rules) {
    if (rule.reject(input)) {
      failures.push({ field: rule.field, code: rule.code });
    }
  }
  return failures;
}
