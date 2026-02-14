"""Visibility interval helpers."""


def minutes_to_hms(total_minutes: int) -> str:
    """Render minute count as HH:MM."""
    if total_minutes < 0:
        raise ValueError("minutes must be non-negative")
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


def merge_overlapping_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge sorted/unsorted minute windows."""
    if not windows:
        return []

    merged: list[tuple[int, int]] = []
    for start, end in sorted(windows):
        if not merged:
            merged.append((start, end))
            continue

        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def visible_window_minutes(windows: list[tuple[int, int]]) -> int:
    """Count total minutes in merged windows."""
    merged = merge_overlapping_windows(windows)
    return sum(max(0, end - start) for start, end in merged)
