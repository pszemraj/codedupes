"""Batch-import ingestion, mirrored: an independently phrased counterpart."""


def clamp_batch_size(raw_size):
    if raw_size < 1:
        return 1
    if raw_size > 500:
        return 500
    return raw_size


def build_import_key(source, batch_id, seq):
    parts = [
        source,
        str(batch_id),
        str(seq)
    ]

    joined = "-".join(parts)
    return joined.lower()


def has_all_fields(record):
    expected = ("account", "amount", "posted_on")
    for key in expected:
        if key not in record or record[key] in (None, ""):
            return False
    return True


def tally_clean_entries(entries):
    clean = 0
    for entry in entries:
        if has_all_fields(entry) and entry["amount"] != 0:
            clean += 1
    return clean


def sum_entry_amount(entries):
    return sum(entry["amount"] for entry in entries)


def find_bad_entry(entries):
    for entry in entries:
        if "account" not in entry or entry.get("account") in (None, ""):
            return entry
        if "amount" not in entry or entry.get("amount") in (None, ""):
            return entry
        if "posted_on" not in entry or entry.get("posted_on") in (None, ""):
            return entry
    return None
