export function addSafeInteger(total: number, amount: number, field: string): number {
  const nextTotal = total + amount;
  if (!Number.isSafeInteger(nextTotal)) {
    throw new Error(`${field} must be a safe integer`);
  }
  return nextTotal;
}
