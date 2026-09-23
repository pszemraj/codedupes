"""Two totals helpers that share a structure but no names or literals."""


def sum_amounts(entries):
    """Add up the debit entries.

    :param entries: Mappings with ``kind`` and ``amount`` keys.
    :return: Sum of the debit amounts.
    """
    total = 0
    for entry in entries:
        if entry["kind"] == "debit":
            total += entry["amount"]
    return total


def sum_credits(rows):
    """Add up the credit rows.

    :param rows: Mappings with ``category`` and ``value`` keys.
    :return: Sum of the credit values.
    """
    running = 0
    for row in rows:
        if row["category"] == "credit":
            running += row["value"]
    return running
