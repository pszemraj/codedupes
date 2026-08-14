"""Mismatch reporting, mirrored: an independently phrased counterpart."""


def severity_label(mismatch_amount):
    magnitude = abs(mismatch_amount)
    if magnitude >= 100:
        return "critical"
    if magnitude >= 10:
        return "warning"
    return "minor"


def build_mismatch_summary_line(ref, expected, actual):
    diff = actual - expected
    parts = [
        ref,
        f"expected={expected}",
        f"actual={actual}",
        f"diff={diff}"
    ]

    return " | ".join(parts)


def mismatch_ratio(mismatch_count, total_count):
    """Fraction of reconciled items that produced a mismatch."""
    # Guard against a division by zero when the batch is empty.
    if total_count <= 0:
        return 0.0
    # Plain float division; ratios above 1.0 would indicate bad input.
    return mismatch_count / total_count


def filter_material_discrepancies(discrepancies, limit):
    material = []
    for item in discrepancies:
        if abs(item["amount"]) >= limit:
            material.append(item)
    return material


def discrepancies_above_limit(items, limit):
    total = 0
    for item in items:
        if abs(item["amount"]) >= limit:
            total += 1
    return total


def fetch_ledger_amount(records, key):
    try:
        return records[key]["balance"]
    except KeyError:
        return 0.0


def categorize_discrepancy(value):
    if value > 0:
        direction = "over"
    elif value < 0:
        direction = "under"
    else:
        direction = "balanced"
    return f"mismatch:{direction}"
