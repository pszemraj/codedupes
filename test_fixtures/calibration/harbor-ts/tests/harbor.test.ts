import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  buildManifestTotals,
  collectZoneWeightTotals,
  summarizeDockLoads,
} from "../src/aggregation.ts";
import { importScheduleCsv, ingestScheduleBatch, prepareWebhookSchedule } from "../src/adapters.ts";
import { auditArrivalWindow, auditZoneCapacity, auditZoneSpread } from "../src/audit.ts";
import { acceptBookingsWithNormalizer, acceptInlineBookings } from "../src/bookings.ts";
import { planDispatches, planDispatchesWithHoldback, planDispatchesWithServiceFloors } from "../src/dispatch.ts";
import { admitManifestNotices } from "../src/legacy.ts";

const loads = [
  { loadId: "L-2", zone: "west", weightKg: 40, status: "active" as const },
  { loadId: "L-1", zone: " east ", weightKg: 120, status: "active" as const },
  { loadId: "L-3", zone: "east", weightKg: 10, status: "cancelled" as const },
];

const rows = [
  { bookingId: " B-1 ", zone: " east ", arrivalDate: "2026-10-04", weightKg: "120" },
  { bookingId: "B-2", zone: "west", arrivalDate: "2026-10-05", weightKg: "40" },
  { bookingId: "B-1", zone: "north", arrivalDate: "2026-10-06", weightKg: "50" },
  { bookingId: "", zone: "east", arrivalDate: "bad", weightKg: "0" },
];

test("aggregation variants preserve validation, ordering, and caller input", () => {
  const expected = [{ zone: "EAST", totalKg: 120, loadCount: 1 }, { zone: "WEST", totalKg: 40, loadCount: 1 }];
  assert.deepEqual(summarizeDockLoads(loads), expected);
  assert.deepEqual(buildManifestTotals(loads), expected);
  assert.deepEqual(collectZoneWeightTotals(loads), expected);
  assert.equal(loads[1].zone, " east ");
  assert.throws(() => summarizeDockLoads([{ ...loads[2], weightKg: 0 }]), /positive integer/);
});

test("dispatch extensions equal the baseline when disabled and retain hold witnesses", () => {
  const summaries = summarizeDockLoads(loads);
  assert.deepEqual(planDispatchesWithHoldback(summaries, 0), planDispatches(summaries));
  assert.deepEqual(planDispatchesWithServiceFloors(summaries, {}), planDispatches(summaries));
  assert.deepEqual(planDispatchesWithHoldback(summaries, 100), {
    instructions: [{ zone: "EAST", totalKg: 120, state: "queued" }, { zone: "WEST", totalKg: 40, state: "held" }],
    heldKg: 40,
    auditEvents: ["held:WEST:40"],
  });
  assert.deepEqual(planDispatchesWithServiceFloors(summaries, { WEST: 100 }), {
    instructions: [{ zone: "EAST", totalKg: 120, state: "queued" }, { zone: "WEST", totalKg: 40, state: "held" }],
    heldKg: 40,
    auditEvents: ["service-floor:WEST:100"],
  });
});

test("inline and extracted booking decoders agree without reserving invalid IDs", () => {
  const inline = acceptInlineBookings(rows);
  const extracted = acceptBookingsWithNormalizer(rows);
  assert.deepEqual(inline, extracted);
  assert.deepEqual(extracted.reservedIds, ["B-1", "B-2"]);
  assert.match(extracted.errors[0], /duplicate bookingId/);
  assert.match(extracted.errors[1], /bookingId is required/);
});

test("legacy manifest notices preserve the current booking acceptance contract", () => {
  const notices = rows.map((row) => [
    row.bookingId,
    row.zone,
    row.arrivalDate,
    row.weightKg,
  ]);
  assert.deepEqual(admitManifestNotices(notices), acceptBookingsWithNormalizer(rows));
});

test("CSV, API, and webhook workflows share acceptance while retaining their own evidence", () => {
  const directory = mkdtempSync(join(tmpdir(), "harbor-ts-"));
  const path = join(directory, "schedule.csv");
  try {
    writeFileSync(path, "bookingId,zone,arrivalDate,weightKg\nB-1,east,2026-10-04,120\nB-1,east,2026-10-04,120\n");
    const csv = importScheduleCsv(path);
    assert.match(csv.errors[0], /^line 3:/);
    const completed = new Set<string>();
    const api = ingestScheduleBatch("batch-1", rows, completed);
    assert.equal(api.replayed, false);
    assert.match(api.report.errors[0], /^item 2:/);
    assert.equal(ingestScheduleBatch("batch-1", rows, completed).replayed, true);
    assert.equal(prepareWebhookSchedule({ eventId: "evt-1", bookings: rows }).acknowledgement, "accepted:evt-1:2");
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test("audit endpoints share harbor vocabulary but enforce independent policies", () => {
  assert.deepEqual(auditZoneCapacity([{ zone: "EAST", totalKg: 120, loadCount: 1 }], 100), ["capacity:EAST:120"]);
  const bookings = [
    { bookingId: "B-1", zone: "EAST", arrivalDate: "2026-10-02", weightKg: 40 },
    { bookingId: "B-2", zone: "EAST", arrivalDate: "2026-10-04", weightKg: 50 },
  ];
  assert.deepEqual(auditArrivalWindow(bookings, "2026-10-03"), ["arrival:B-1:2026-10-02"]);
  assert.deepEqual(auditZoneSpread(bookings, 1), ["spread:EAST:2"]);
});
