import { summarizeDockLoads } from "./aggregation.ts";
import { planDispatchesWithHoldback } from "./dispatch.ts";

const summaries = summarizeDockLoads([
  { loadId: "L-1", zone: "east", weightKg: 120, status: "active" },
  { loadId: "L-2", zone: "west", weightKg: 40, status: "active" },
]);
const plan = planDispatchesWithHoldback(summaries, 100);
console.log(JSON.stringify(plan));
