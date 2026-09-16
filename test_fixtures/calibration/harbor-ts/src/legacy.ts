type ManifestNotice = readonly string[];

type ParsedNotice =
  | { reference: string; berth: string; gateDay: string; kilograms: number }
  | { problem: string };

type ManifestLedger = {
  docked: { reference: string; berth: string; gateDay: string; kilograms: number }[];
  faults: string[];
  tickets: Set<string>;
};

function readManifestNotice(cells: ManifestNotice): ParsedNotice {
  const [reference = "", berth = "", gateDay = "", kilograms = ""] = cells;
  if (!reference.trim()) return { problem: "bookingId is required" };
  if (!berth.trim()) return { problem: "zone is required" };
  if (!/^\d{4}-\d{2}-\d{2}$/.test(gateDay)) return { problem: "arrivalDate must use YYYY-MM-DD" };
  const calendar = new Date(`${gateDay}T00:00:00Z`);
  if (Number.isNaN(calendar.valueOf()) || calendar.toISOString().slice(0, 10) !== gateDay) {
    return { problem: "arrivalDate is not a calendar day" };
  }
  if (!/^[1-9]\d*$/.test(kilograms)) return { problem: "weightKg must be a positive integer string" };
  return { reference: reference.trim(), berth: berth.trim().toUpperCase(), gateDay, kilograms: Number(kilograms) };
}

function renderBookingReport(ledger: ManifestLedger) {
  return {
    accepted: ledger.docked.map(({ reference, berth, gateDay, kilograms }) => ({
      bookingId: reference,
      zone: berth,
      arrivalDate: gateDay,
      weightKg: kilograms,
    })),
    errors: ledger.faults,
    reservedIds: [...ledger.tickets].sort(),
  };
}

/** Admit positional notices from a legacy manifest into the current booking-report shape. */
export function admitManifestNotices(notices: readonly ManifestNotice[]) {
  const ledger: ManifestLedger = { docked: [], faults: [], tickets: new Set<string>() };
  for (let position = 0; position < notices.length; position += 1) {
    const decoded = readManifestNotice(notices[position]);
    if ("problem" in decoded) {
      ledger.faults.push(`row ${position}: ${decoded.problem}`);
    } else if (ledger.tickets.has(decoded.reference)) {
      ledger.faults.push(`row ${position}: duplicate bookingId`);
    } else {
      ledger.tickets.add(decoded.reference);
      ledger.docked.push(decoded);
    }
  }
  return renderBookingReport(ledger);
}
