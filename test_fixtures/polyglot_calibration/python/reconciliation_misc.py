"""Cross-cutting reconciliation helpers not tied to one batch/window/mismatch file."""


def duplicate_reference_count(references):
    seen = set()
    duplicates = 0
    for ref in references:
        if ref in seen:
            duplicates += 1
        seen.add(ref)
    return duplicates


def recurring_tag_total(tags):
    known = set()
    hits = 0
    for tag in tags:
        if tag in known and tag != "":
            hits += 1
        known.add(tag)
    return hits


def is_batch_balanced(entries, tolerance=0.01):
    total = 0.0
    for entry in entries:
        total += entry["amount"]
    return abs(total) <= tolerance


def group_is_reconciled(items, margin=0.05):
    total = 0.0
    for item in items:
        total += item["amount"]
    return abs(total) <= margin


def count_batches_over_limit(batch_sizes, limit):
    count = 0
    index = 0
    while index < len(batch_sizes):
        if batch_sizes[index] > limit:
            count += 1
        index += 1
    return count


def tally_big_groups(measurements, cap):
    total = 0
    for size in measurements:
        if size > cap:
            total += 1
    return total
