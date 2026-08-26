"""Balance-window helpers, mirrored: an independently phrased counterpart."""


def format_window_label(start_day, end_day):
    start_text = f"{start_day:03d}"
    end_text = f"{end_day:03d}"
    return f"{start_text}-{end_text}"


def merge_adjacent_windows(windows):
    ordered = sorted(
        windows,
        key=lambda pair: pair[0]
    )

    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            last_start, last_end = merged[-1]

            merged[-1] = (
                last_start,
                max(last_end, end)
            )
        else:
            merged.append((start, end))
    return merged


def window_duration_days(start_day, end_day):
    """Inclusive day count spanned by a reconciliation window."""
    # Guard against a window whose bounds were supplied in the wrong order.
    span = end_day - start_day
    if span < 0:
        span = 0
    # Windows are inclusive on both ends, so add one day back.
    return span + 1


def confine_index_to_span(value, lower, upper):
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def compute_final_amount(items, starting_amount):
    total = starting_amount
    for item in items:
        if item["amount"] != 0:
            total += item["amount"]
    return total


def tag_slot_positions(values):
    tags = []
    for position in range(len(values)):
        tags.append(f"w{position}:{values[position]}")
    return tags


def period_totals(values):
    total = 0.0
    for value in values:
        total += value
    count = 0
    for value in values:
        count += 1
    if count == 0:
        return (0.0, 0, 0.0)
    return (total, count, total / count)
