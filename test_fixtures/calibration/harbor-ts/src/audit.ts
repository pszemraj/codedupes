import type { Booking, ZoneSummary } from "./models.ts";

/** Flag zones whose active load exceeds a declared capacity. */
export function auditZoneCapacity(summaries: readonly ZoneSummary[], capacityKg: number): string[] {
  return summaries
    .filter((summary) => summary.totalKg > capacityKg)
    .map((summary) => `capacity:${summary.zone}:${summary.totalKg}`)
    .sort();
}

/** Flag booking dates that fall before a declared arrival window. */
export function auditArrivalWindow(bookings: readonly Booking[], earliestDate: string): string[] {
  return bookings
    .filter((booking) => booking.arrivalDate < earliestDate)
    .map((booking) => `arrival:${booking.bookingId}:${booking.arrivalDate}`)
    .sort();
}

/** Flag a customer-facing schedule when a zone appears on too many booking dates. */
export function auditZoneSpread(bookings: readonly Booking[], maximumDates: number): string[] {
  const datesByZone = new Map<string, Set<string>>();
  for (const booking of bookings) {
    const dates = datesByZone.get(booking.zone) ?? new Set<string>();
    dates.add(booking.arrivalDate);
    datesByZone.set(booking.zone, dates);
  }
  return [...datesByZone]
    .filter(([, dates]) => dates.size > maximumDates)
    .map(([zone, dates]) => `spread:${zone}:${dates.size}`)
    .sort();
}
