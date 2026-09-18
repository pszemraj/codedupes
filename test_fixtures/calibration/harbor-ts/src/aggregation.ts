import type { DockLoad, ZoneSummary } from "./models.ts";

function checkedLoad(load: DockLoad): void {
  if (!load.loadId.trim()) throw new Error("loadId is required");
  if (!load.zone.trim()) throw new Error("zone is required");
  if (!Number.isSafeInteger(load.weightKg) || load.weightKg <= 0) {
    throw new Error("weightKg must be a positive integer");
  }
  if (load.status !== "active" && load.status !== "cancelled") {
    throw new Error("unknown load status");
  }
}

function addWeight(totalKg: number, weightKg: number): number {
  const nextTotal = totalKg + weightKg;
  if (!Number.isSafeInteger(nextTotal)) {
    throw new Error("totalKg must be a safe integer");
  }
  return nextTotal;
}

/** Aggregate active loads by zone with a Map-based implementation. */
export function summarizeDockLoads(loads: readonly DockLoad[]): ZoneSummary[] {
  const totals = new Map<string, { totalKg: number; loadCount: number }>();
  for (const load of loads) {
    checkedLoad(load);
    if (load.status === "cancelled") continue;
    const zone = load.zone.trim().toUpperCase();
    const current = totals.get(zone) ?? { totalKg: 0, loadCount: 0 };
    current.totalKg = addWeight(current.totalKg, load.weightKg);
    current.loadCount += 1;
    totals.set(zone, current);
  }
  return [...totals]
    .map(([zone, total]) => ({ zone, ...total }))
    .sort((left, right) => left.zone.localeCompare(right.zone));
}

/** Aggregate the same contract through sorting and contiguous zone groups. */
export function buildManifestTotals(loads: readonly DockLoad[]): ZoneSummary[] {
  const active: DockLoad[] = [];
  for (const load of loads) {
    checkedLoad(load);
    if (load.status === "active") active.push({ ...load, zone: load.zone.trim().toUpperCase() });
  }
  active.sort((left, right) => left.zone.localeCompare(right.zone));
  const summaries: ZoneSummary[] = [];
  for (const load of active) {
    const previous = summaries.at(-1);
    if (previous?.zone === load.zone) {
      previous.totalKg = addWeight(previous.totalKg, load.weightKg);
      previous.loadCount += 1;
    } else {
      summaries.push({ zone: load.zone, totalKg: load.weightKg, loadCount: 1 });
    }
  }
  return summaries;
}

/** Aggregate the same contract with an immutable object accumulator. */
export function collectZoneWeightTotals(loads: readonly DockLoad[]): ZoneSummary[] {
  const totals: Record<string, ZoneSummary> = {};
  loads.forEach((load) => {
    checkedLoad(load);
    if (load.status === "cancelled") return;
    const zone = load.zone.trim().toUpperCase();
    const known = totals[zone] ?? { zone, totalKg: 0, loadCount: 0 };
    totals[zone] = {
      zone,
      totalKg: addWeight(known.totalKg, load.weightKg),
      loadCount: known.loadCount + 1,
    };
  });
  return Object.values(totals).sort((left, right) => left.zone.localeCompare(right.zone));
}
