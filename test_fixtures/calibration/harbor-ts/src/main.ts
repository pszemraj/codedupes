import { summarizeDockLoads } from "./aggregation.ts";
import { planDispatchesWithHoldback } from "./dispatch.ts";
import { evaluateRegistration, validateSignup } from "./signup.ts";

const summaries = summarizeDockLoads([
  { loadId: "L-1", zone: "east", weightKg: 120, status: "active" },
  { loadId: "L-2", zone: "west", weightKg: 40, status: "active" },
]);
const plan = planDispatchesWithHoldback(summaries, 100);
const signup = {
  kind: "person" as const,
  email: "crew@example.test",
  age: 17,
  marketing: true,
};
console.log(
  JSON.stringify({
    plan,
    registration: [validateSignup(signup), evaluateRegistration(signup)],
  }),
);
