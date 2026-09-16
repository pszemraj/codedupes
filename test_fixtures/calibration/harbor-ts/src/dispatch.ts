import type { DispatchPlan, ZoneSummary } from "./models.ts";

function validateSummaries(summaries: readonly ZoneSummary[]): void {
  for (const summary of summaries) {
    if (!summary.zone.trim()) throw new Error("zone is required");
    if (!Number.isInteger(summary.totalKg) || summary.totalKg <= 0) {
      throw new Error("totalKg must be a positive integer");
    }
    if (!Number.isInteger(summary.loadCount) || summary.loadCount <= 0) {
      throw new Error("loadCount must be a positive integer");
    }
  }
}

/** Build the baseline queue: every valid zone receives one dispatch instruction. */
export function planDispatches(summaries: readonly ZoneSummary[]): DispatchPlan {
  validateSummaries(summaries);
  return {
    instructions: [...summaries]
      .sort((left, right) => left.zone.localeCompare(right.zone))
      .map((summary) => ({ zone: summary.zone, totalKg: summary.totalKg, state: "queued" })),
    heldKg: 0,
    auditEvents: [],
  };
}

/** Defer zones below a global minimum while preserving baseline output at zero. */
export function planDispatchesWithHoldback(
  summaries: readonly ZoneSummary[],
  minimumDispatchKg: number,
): DispatchPlan {
  validateSummaries(summaries);
  if (!Number.isInteger(minimumDispatchKg) || minimumDispatchKg < 0) {
    throw new Error("minimumDispatchKg must be a non-negative integer");
  }
  const instructions = [] as DispatchPlan["instructions"];
  const auditEvents: string[] = [];
  let heldKg = 0;
  for (const summary of [...summaries].sort((left, right) => left.zone.localeCompare(right.zone))) {
    if (minimumDispatchKg > 0 && summary.totalKg < minimumDispatchKg) {
      heldKg += summary.totalKg;
      instructions.push({ zone: summary.zone, totalKg: summary.totalKg, state: "held" });
      auditEvents.push(`held:${summary.zone}:${summary.totalKg}`);
    } else {
      instructions.push({ zone: summary.zone, totalKg: summary.totalKg, state: "queued" });
    }
  }
  return { instructions, heldKg, auditEvents };
}

/** Apply per-zone service floors, a distinct policy extension of the queue planner. */
export function planDispatchesWithServiceFloors(
  summaries: readonly ZoneSummary[],
  floors: Readonly<Record<string, number>>,
): DispatchPlan {
  validateSummaries(summaries);
  const instructions = [] as DispatchPlan["instructions"];
  const auditEvents: string[] = [];
  let heldKg = 0;
  for (const summary of [...summaries].sort((left, right) => left.zone.localeCompare(right.zone))) {
    const floor = floors[summary.zone] ?? 0;
    if (!Number.isInteger(floor) || floor < 0) throw new Error("service floor must be non-negative");
    if (floor > 0 && summary.totalKg < floor) {
      heldKg += summary.totalKg;
      instructions.push({ zone: summary.zone, totalKg: summary.totalKg, state: "held" });
      auditEvents.push(`service-floor:${summary.zone}:${floor}`);
    } else {
      instructions.push({ zone: summary.zone, totalKg: summary.totalKg, state: "queued" });
    }
  }
  return { instructions, heldKg, auditEvents };
}
