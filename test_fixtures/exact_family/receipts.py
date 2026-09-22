"""Receipt rendering for receipts."""


def render_receipt(entries):
    """Render entries as a right-aligned two-column receipt.

    :param entries: Mappings with ``label`` and ``amount`` keys.
    :return: Receipt text with a total line.
    """
    lines = []
    total = 0
    for entry in entries:
        label = entry["label"]
        amount = entry["amount"]
        lines.append(f"{label:<20}{amount:>10.2f}")
        total += amount
    lines.append("-" * 30)
    lines.append(f"{'total':<20}{total:>10.2f}")
    return "
".join(lines)
