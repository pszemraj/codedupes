"""Deliberate search-probe corpus for semantic search calibration.

Each function implements a distinct, well-known programming task so that the
natural-language queries in queries.json have exactly one intended hit. The
opt-in smoke test in tests/test_semantic_smoke.py asserts that every relevant
query clears the default search threshold while off-topic queries return
nothing. Do not deduplicate or lint-fix this file; it is an analysis input.
"""

import time


def parse_csv_rows(text):
    """Parse comma-separated text into a list of row dicts."""
    lines = [line for line in text.splitlines() if line.strip()]
    header = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        values = line.split(",")
        rows.append(dict(zip(header, values)))
    return rows


def retry_with_backoff(operation, max_attempts=5, base_delay=0.5):
    """Retry an operation with exponential backoff between attempts."""
    for attempt in range(max_attempts):
        try:
            return operation()
        except OSError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(base_delay * (2**attempt))
    return None


def evict_least_recently_used(cache, capacity):
    """Drop the oldest entries until the cache fits its capacity."""
    while len(cache) > capacity:
        oldest_key = next(iter(cache))
        del cache[oldest_key]
    return cache


def format_bytes_human(num_bytes):
    """Format a byte count as a human readable size string."""
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PiB"


def find_insertion_index(sorted_values, target):
    """Binary search for the index where target keeps the list sorted."""
    low, high = 0, len(sorted_values)
    while low < high:
        mid = (low + high) // 2
        if sorted_values[mid] < target:
            low = mid + 1
        else:
            high = mid
    return low


def take_rate_limit_token(bucket, now):
    """Refill a token bucket by elapsed time and consume one token if available."""
    elapsed = now - bucket["last_refill"]
    bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + elapsed * bucket["rate"])
    bucket["last_refill"] = now
    if bucket["tokens"] >= 1.0:
        bucket["tokens"] -= 1.0
        return True
    return False
