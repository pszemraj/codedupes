import { evaluateRegistration, validateSignup, type Issue, type Submission } from "../src/signup.ts";

function equal(actual: unknown, expected: unknown): void {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`Expected ${JSON.stringify(expected)}; received ${JSON.stringify(actual)}`);
  }
}

function check(input: Submission, expected: Issue[]): void {
  const snapshot = JSON.stringify(input);
  Object.freeze(input);
  equal(validateSignup(input), expected);
  equal(evaluateRegistration(input), expected);
  equal(JSON.stringify(input), snapshot);
}

check({ kind: "person", email: " A@EXAMPLE.COM ", age: 18, marketing: true }, []);
check({ kind: "person", email: "K@example.com", age: 18, marketing: false }, []);
check({ kind: "person", email: "a@example.com", age: 17, marketing: true },
      [{ field: "marketing", code: "underage_marketing" }]);
check({ kind: "person", email: "bad", age: NaN, marketing: true },
      [{ field: "email", code: "invalid_email" }, { field: "age", code: "invalid_age" }]);
check({ kind: "business", email: "a@example.com", company: " A ", country: "GB", vatId: "gb 123-456-789" }, []);
check({ kind: "business", email: "bad", company: " ", country: "GB", vatId: "123" },
      [{ field: "email", code: "invalid_email" }, { field: "company", code: "invalid_company" },
       { field: "vatId", code: "invalid_vat" }]);
check({ kind: "business", email: "a@example.com", company: "X", country: "US", vatId: "ignored" }, []);
check({ kind: "business", email: "a@example.com", company: "X", country: "GB", vatId: "GB\t123456789" },
      [{ field: "vatId", code: "invalid_vat" }]);

let personCases = 0;
for (const age of [-1, 0, 1.5, 17, 18, 130, 131, NaN, Infinity]) {
  for (const email of [" A@EXAMPLE.COM ", "bad", "é@example.com", "a@b.co"]) {
    for (const marketing of [false, true]) {
      const input: Submission = { kind: "person", age, email, marketing };
      const expected: Issue[] = [];
      if (email === "bad" || email === "é@example.com") expected.push({ field: "email", code: "invalid_email" });
      const validAge = Number.isInteger(age) && age >= 0 && age <= 130;
      if (!validAge) expected.push({ field: "age", code: "invalid_age" });
      if (validAge && marketing && age < 18) expected.push({ field: "marketing", code: "underage_marketing" });
      check(input, expected);
      personCases += 1;
    }
  }
}
let businessCases = 0;
for (const country of ["GB", "US"] as const) {
  for (const company of ["", " ", "Company", "X".repeat(80), "X".repeat(81)]) {
    for (const vatId of ["gb 123-456-789", "GB000000000", "123456789", "GB123", "GB\t123456789"]) {
      const input: Submission = { kind: "business", email: "a@example.com", country, company, vatId };
      const expected: Issue[] = [];
      if (company.trim().length === 0 || company.length > 80) expected.push({ field: "company", code: "invalid_company" });
      if (country === "GB" && vatId !== "gb 123-456-789" && vatId !== "GB000000000") {
        expected.push({ field: "vatId", code: "invalid_vat" });
      }
      check(input, expected);
      businessCases += 1;
    }
  }
}
console.log(`signup contract: passed; ${personCases} person cases, ${businessCases} business cases, 8 explicit cases`);
