"""Alternative planner script used for migration tests."""


def derive_observable_bands(
    rows: list[tuple[int, float | None]],
    floor_deg: float = 5.0,
) -> list[dict[str, int]]:
    """Produce observable bands and each band's best elevation."""
    bands: list[dict[str, int]] = []
    opening: int | None = None
    best = -999

    for tick, elevation in rows:
        qualifies = elevation is not None and (elevation > floor_deg or abs(elevation - floor_deg) < 1e-9)
        if qualifies:
            if opening is None:
                opening = tick
                best = int(elevation)
            else:
                best = max(best, int(elevation))
            continue

        if opening is not None:
            bands.append({"start": opening, "end": tick, "peak": best})
            opening = None
            best = -999

    if opening is not None and rows:
        bands.append({"start": opening, "end": rows[-1][0] + 1, "peak": best})

    return bands


def total_lit_minutes(rows: list[tuple[int, float | None]], floor_deg: float = 5.0) -> int:
    """Count minutes where target is above threshold."""
    lit = 0
    for _tick, elevation in rows:
        if elevation is not None and elevation >= floor_deg:
            lit += 1
    return lit


def exposure_band(reading_deg: float | None) -> str:
    """Convert altitude in degrees into a coarse category."""
    if reading_deg is None:
        return "missing"
    if reading_deg < 0.0:
        return "below"
    if reading_deg < 10.0:
        return "low"
    if reading_deg < 30.0:
        return "mid"
    return "high"


def fast_visibility_flag(value_deg: float | None, floor_deg: float = 5.0) -> bool:
    """Compact visibility flag."""
    return value_deg is not None and value_deg >= floor_deg
