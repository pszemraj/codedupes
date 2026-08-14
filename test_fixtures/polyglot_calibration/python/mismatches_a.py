"""Mismatch reporting: severity, summaries, ratios, and classification."""


def severity_label(mismatch_amount):
    magnitude = abs(mismatch_amount)
    if magnitude >= 100:
        return "critical"
    if magnitude >= 10:
        return "warning"
    return "minor"


def build_mismatch_summary_line(ref, expected, actual):
    diff = actual - expected
    parts = [ref, f"expected={expected}", f"actual={actual}", f"diff={diff}"]
    return " | ".join(parts)


def mismatch_ratio(mismatch_count, total_count):
    if total_count <= 0:
        return 0.0
    return mismatch_count / total_count


def flag_large_mismatches(mismatches, cutoff):
    flagged = []
    for entry in mismatches:
        if abs(entry["amount"]) >= cutoff:
            flagged.append(entry)
    return flagged


def mismatch_count_over_threshold(mismatches, threshold):
    count = 0
    for entry in mismatches:
        if abs(entry["amount"]) > threshold:
            count += 1
    return count


def lookup_account_balance(balances, account_id):
    return balances.get(account_id, {}).get("balance", 0.0)


def _variance_direction(amount):
    if amount > 0:
        return "over"
    if amount < 0:
        return "under"
    return "balanced"


def classify_mismatch(amount):
    direction = _variance_direction(amount)
    return f"mismatch:{direction}"
