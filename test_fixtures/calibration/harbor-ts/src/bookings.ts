import type { Booking, BookingReport } from "./models.ts";

export type RawBooking = Record<string, unknown>;

function strictWeight(value: unknown): number {
  if (typeof value !== "string" || !/^[1-9]\d*$/.test(value)) {
    throw new Error("weightKg must be a positive integer string");
  }
  return parseSafeWeight(value);
}

function strictDate(value: unknown): string {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    throw new Error("arrivalDate must use YYYY-MM-DD");
  }
  const parsed = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(parsed.valueOf()) || parsed.toISOString().slice(0, 10) !== value) {
    throw new Error("arrivalDate is not a calendar day");
  }
  return value;
}

/** Decode one external booking after applying the shared field precedence. */
export function normalizeBooking(raw: RawBooking): Booking {
  const bookingId = typeof raw.bookingId === "string" ? raw.bookingId.trim() : "";
  if (!bookingId) throw new Error("bookingId is required");
  const zone = typeof raw.zone === "string" ? raw.zone.trim().toUpperCase() : "";
  if (!zone) throw new Error("zone is required");
  const arrivalDate = strictDate(raw.arrivalDate);
  const weightKg = strictWeight(raw.weightKg);
  return { bookingId, zone, arrivalDate, weightKg };
}

/** Retain the former inline decoder while accepting an ordered upload. */
export function acceptInlineBookings(rows: readonly RawBooking[]): BookingReport {
  const accepted: Booking[] = [];
  const errors: string[] = [];
  const reserved = new Set<string>();
  rows.forEach((raw, index) => {
    try {
      const bookingId = typeof raw.bookingId === "string" ? raw.bookingId.trim() : "";
      if (!bookingId) throw new Error("bookingId is required");
      const zone = typeof raw.zone === "string" ? raw.zone.trim().toUpperCase() : "";
      if (!zone) throw new Error("zone is required");
      const arrivalDate = strictDate(raw.arrivalDate);
      const weightKg = strictWeight(raw.weightKg);
      if (reserved.has(bookingId)) throw new Error("duplicate bookingId");
      reserved.add(bookingId);
      accepted.push({ bookingId, zone, arrivalDate, weightKg });
    } catch (error) {
      errors.push(`row ${index}: ${(error as Error).message}`);
    }
  });
  return { accepted, errors, reservedIds: [...reserved].sort() };
}

/** Use the extracted normalizer while retaining upload ordering and duplicate accounting. */
export function acceptBookingsWithNormalizer(rows: readonly RawBooking[]): BookingReport {
  const accepted: Booking[] = [];
  const errors: string[] = [];
  const reserved = new Set<string>();
  for (const [index, raw] of rows.entries()) {
    try {
      const booking = normalizeBooking(raw);
      if (reserved.has(booking.bookingId)) throw new Error("duplicate bookingId");
      reserved.add(booking.bookingId);
      accepted.push(booking);
    } catch (error) {
      errors.push(`row ${index}: ${(error as Error).message}`);
    }
  }
  return { accepted, errors, reservedIds: [...reserved].sort() };
}

function parseSafeWeight(value: string): number {
  const weightKg = Number(value);
  if (!Number.isSafeInteger(weightKg)) {
    throw new Error("weightKg must be a positive integer string");
  }
  return weightKg;
}
