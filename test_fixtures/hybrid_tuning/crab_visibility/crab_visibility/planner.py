"""Planning helpers for nightly observation windows."""


def compute_visibility_bands(
    timeline: list[tuple[int, float | None]],
    min_altitude_deg: float = 5.0,
) -> list[dict[str, int]]:
    """Return contiguous visibility bands with peak altitude metadata."""
    output: list[dict[str, int]] = []
    active_start: int | None = None
    active_peak = -999

    for minute, altitude in timeline:
        visible = altitude is not None and altitude >= min_altitude_deg
        if visible:
            if active_start is None:
                active_start = minute
                active_peak = int(altitude)
            else:
                active_peak = max(active_peak, int(altitude))
            continue

        if active_start is not None:
            output.append({"start": active_start, "end": minute, "peak": active_peak})
            active_start = None
            active_peak = -999

    if active_start is not None and timeline:
        output.append({"start": active_start, "end": timeline[-1][0] + 1, "peak": active_peak})

    return output


def total_dark_minutes(
    timeline: list[tuple[int, float | None]], min_altitude_deg: float = 5.0
) -> int:
    """Count minutes where target is below threshold."""
    below = 0
    for _minute, altitude in timeline:
        if altitude is None or altitude < min_altitude_deg:
            below += 1
    return below
