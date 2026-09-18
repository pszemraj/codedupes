export type LoadStatus = "active" | "cancelled";

export interface DockLoad {
  loadId: string;
  zone: string;
  weightKg: number;
  status: LoadStatus;
}

export interface ZoneSummary {
  zone: string;
  totalKg: number;
  loadCount: number;
}

export interface DispatchInstruction {
  zone: string;
  totalKg: number;
  state: "queued" | "held";
}

export interface DispatchPlan {
  instructions: DispatchInstruction[];
  heldKg: number;
  auditEvents: string[];
}

export interface Booking {
  bookingId: string;
  zone: string;
  arrivalDate: string;
  weightKg: number;
}

export interface BookingReport {
  accepted: Booking[];
  errors: string[];
  reservedIds: string[];
}

export interface ApiBatchResult {
  replayed: boolean;
  report: BookingReport;
}
