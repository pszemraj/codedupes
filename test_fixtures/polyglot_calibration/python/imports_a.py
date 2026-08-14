"""Batch-import ingestion: row validation, key building, and totals."""


def clamp_batch_size(raw_size):
    if raw_size < 1:
        return 1
    if raw_size > 500:
        return 500
    return raw_size


def build_import_key(source, batch_id, seq):
    parts = [source, str(batch_id), str(seq)]
    joined = "-".join(parts)
    return joined.lower()


def is_row_complete(row):
    required = ("account", "amount", "posted_on")
    for field in required:
        if field not in row or row[field] in (None, ""):
            return False
    return True


def count_valid_rows(rows):
    valid = 0
    for row in rows:
        if is_row_complete(row):
            valid += 1
    return valid


def total_batch_amount(rows):
    total = 0.0
    for row in rows:
        total += row["amount"]
    return total


def first_invalid_row(rows):
    for row in rows:
        if "account" not in row or row.get("account") in (None, ""):
            return row
        else:
            if "amount" not in row or row.get("amount") in (None, ""):
                return row
            else:
                if "posted_on" not in row or row.get("posted_on") in (None, ""):
                    return row
    return None
