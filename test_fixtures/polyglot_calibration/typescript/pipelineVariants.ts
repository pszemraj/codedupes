/**
 * Near-restructure pairs: same observable behavior, different
 * decomposition (inline vs. helper, one-pass vs. two-pass, accumulator
 * object vs. parallel arrays, iterative vs. recursive).
 */

export function summarizeRow(cells: string[]): string {
  const trimmedCells = cells.map((cell) => cell.trim());
  const nonEmpty = trimmedCells.filter((cell) => cell.length > 0);
  return nonEmpty.join(" | ");
}

function cleanCell(cell: string): string {
  return cell.trim();
}

export function renderRowSummary(cells: string[]): string {
  const cleaned = cells.map(cleanCell).filter((cell) => cell.length > 0);
  return cleaned.join(" | ");
}

export function fieldLengthRange(values: string[]): { min: number; max: number } {
  const lengths = values.map((value) => value.length);
  const min = Math.min(...lengths);
  const max = Math.max(...lengths);
  return { min, max };
}

export function computeWidthBounds(items: string[]): { min: number; max: number } {
  let low = Infinity;
  let high = -Infinity;
  for (const item of items) {
    const width = item.length;
    if (width < low) {
      low = width;
    }
    if (width > high) {
      high = width;
    }
  }
  return { min: low, max: high };
}

export function tallyFieldOccurrences(fields: string[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const field of fields) {
    counts[field] = (counts[field] || 0) + 1;
  }
  return counts;
}

export function buildFieldFrequency(items: string[]): Record<string, number> {
  const names: string[] = [];
  const tallies: number[] = [];
  for (const item of items) {
    const at = names.indexOf(item);
    if (at === -1) {
      names.push(item);
      tallies.push(1);
    } else {
      tallies[at] += 1;
    }
  }
  const frequency: Record<string, number> = {};
  for (let i = 0; i < names.length; i += 1) {
    frequency[names[i]] = tallies[i];
  }
  return frequency;
}

export function highestScore(scores: number[]): number {
  let best = scores[0];
  for (let i = 1; i < scores.length; i += 1) {
    if (scores[i] > best) {
      best = scores[i];
    }
  }
  return best;
}

export function topMark(marks: number[]): number {
  if (marks.length === 1) {
    return marks[0];
  }
  const rest = topMark(marks.slice(1));
  return marks[0] > rest ? marks[0] : rest;
}
