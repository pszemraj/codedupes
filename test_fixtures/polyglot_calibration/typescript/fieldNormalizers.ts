/**
 * Field-level normalization helpers used before any schema rule runs.
 */

export function collapseWhitespace(value: string): string {
  const trimmed = value.trim();
  if (trimmed.length === 0) {
    return "";
  }
  const parts = trimmed.split(/\s+/);
  return parts.join(" ");
}

export function truncateInput(value: string, limit: number): string {
  const trimmed = value.trim();
  if (trimmed.length <= limit) {
    return trimmed;
  }
  const head = trimmed.slice(0, limit - 3);
  return head + "...";
}

export function stripNonDigits(value: string): string {
  const trimmed = value.trim();
  if (trimmed.length === 0) {
    return "";
  }
  const digits = trimmed.replace(/[^0-9]/g, "");
  return digits;
}

export function padFieldCode(code: number, width: number): string;
export function padFieldCode(code: string, width: number): string;
export function padFieldCode(code: number | string, width: number): string {
  const text = String(code);
  if (text.length >= width) {
    return text;
  }
  const filler = "0".repeat(width - text.length);
  return filler + text;
}
