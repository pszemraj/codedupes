"""Balance-window helpers: labeling, merging, bounding, and per-window stats."""


def format_window_label(start_day, end_day):
    start_text = f"{start_day:03d}"
    end_text = f"{end_day:03d}"
    return f"{start_text}-{end_text}"


def merge_adjacent_windows(windows):
    ordered = sorted(windows, key=lambda pair: pair[0])
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            last_start, last_end = merged[-1]
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def window_duration_days(start_day, end_day):
    span = end_day - start_day
    if span < 0:
        span = 0
    return span + 1


def clip_window_to_bounds(day, floor_day, ceil_day):
    if day < floor_day:
        return floor_day
    if day > ceil_day:
        return ceil_day
    return day


def running_balance_after(entries, start_balance):
    balance = start_balance
    for entry in entries:
        balance += entry["amount"]
    return balance


def label_window_indices(days):
    labels = []
    for index, day in enumerate(days):
        labels.append(f"w{index}:{day}")
    return labels


def window_stats(amounts):
    total = 0.0
    count = 0
    for amount in amounts:
        total += amount
        count += 1
    if count == 0:
        return (0.0, 0, 0.0)
    return (total, count, total / count)
