import { readFileSync } from "node:fs";

import { acceptBookingsWithNormalizer } from "./bookings.ts";
import type { RawBooking } from "./bookings.ts";
import type { ApiBatchResult, BookingReport } from "./models.ts";

function csvRows(text: string): RawBooking[] {
  const [header, ...lines] = text.trim().split(/\r?\n/);
  if (header !== "bookingId,zone,arrivalDate,weightKg") {
    throw new Error("unexpected CSV header");
  }
  return lines.filter(Boolean).map((line) => {
    const [bookingId, zone, arrivalDate, weightKg] = line.split(",");
    return { bookingId, zone, arrivalDate, weightKg };
  });
}

/** Read a schedule CSV from disk and retain physical line numbers in its errors. */
export function importScheduleCsv(filePath: string): BookingReport {
  const rows = csvRows(readFileSync(filePath, "utf8"));
  const report = acceptBookingsWithNormalizer(rows);
  return {
    ...report,
    errors: report.errors.map((message) => message.replace(/^row (\d+):/, (_, row) => `line ${Number(row) + 2}:`)),
  };
}

/** Ingest an API batch with idempotency ownership and item-oriented errors. */
export function ingestScheduleBatch(
  batchId: string,
  rows: readonly RawBooking[],
  completed: Set<string>,
): ApiBatchResult {
  if (!batchId.trim()) throw new Error("batchId is required");
  if (completed.has(batchId)) {
    return { replayed: true, report: { accepted: [], errors: [], reservedIds: [] } };
  }
  const report = acceptBookingsWithNormalizer(rows);
  completed.add(batchId);
  return {
    replayed: false,
    report: { ...report, errors: report.errors.map((message) => message.replace("row", "item")) },
  };
}

/** Decode a webhook envelope that shares booking acceptance but owns acknowledgement text. */
export function prepareWebhookSchedule(payload: {
  eventId: string;
  bookings: readonly RawBooking[];
}): { acknowledgement: string; report: BookingReport } {
  if (!payload.eventId.trim()) throw new Error("eventId is required");
  const report = acceptBookingsWithNormalizer(payload.bookings);
  return { acknowledgement: `accepted:${payload.eventId}:${report.accepted.length}`, report };
}
