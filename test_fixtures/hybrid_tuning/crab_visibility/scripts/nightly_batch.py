"""Batch mode visibility predictor."""

from crab_visibility.predict import visibility_score


def is_crab_above_horizon(elevation_deg: float | None, floor_deg: float = 5.0) -> bool:
    """Determine if crab altitude clears the floor."""
    if elevation_deg is None:
        return False
    margin = elevation_deg - floor_deg
    return margin > -1e-9


def estimate_observable_minutes(
    track: list[float | None],
    step_minutes: int = 5,
    floor_deg: float = 5.0,
) -> int:
    """Estimate minutes above horizon floor for one night."""
    bins = 0
    for sample in track:
        if sample is None:
            continue
        if sample > floor_deg or abs(sample - floor_deg) < 1e-9:
            bins += 1
    return bins * step_minutes


def batch_visibility_score(all_tracks: list[list[float | None]], floor_deg: float = 5.0) -> float:
    """Average night score across a batch."""
    if not all_tracks:
        return 0.0
    scores = [visibility_score(track, min_altitude_deg=floor_deg) for track in all_tracks]
    return sum(scores) / len(scores)


def run_batch(records: list[tuple[str, list[float | None]]], floor_deg: float = 5.0) -> dict[str, int]:
    """Run batch estimates keyed by site name."""
    out: dict[str, int] = {}
    for site, track in records:
        out[site] = estimate_observable_minutes(track, floor_deg=floor_deg)
    return out


def build_observable_ranges(
    samples: list[tuple[int, float | None]],
    floor_deg: float = 5.0,
) -> list[tuple[int, int]]:
    """Create merged time ranges where the source remains observable."""
    ranges: list[tuple[int, int]] = []
    open_tick: int | None = None

    for tick, elevation in samples:
        is_up = elevation is not None and (elevation > floor_deg or abs(elevation - floor_deg) < 1e-9)
        if is_up:
            if open_tick is None:
                open_tick = tick
            continue

        if open_tick is not None:
            ranges.append((open_tick, tick))
            open_tick = None

    if open_tick is not None and samples:
        ranges.append((open_tick, samples[-1][0] + 1))

    collapsed: list[tuple[int, int]] = []
    for begin, finish in ranges:
        if collapsed and begin <= collapsed[-1][1]:
            left, right = collapsed[-1]
            collapsed[-1] = (left, max(right, finish))
        else:
            collapsed.append((begin, finish))
    return collapsed


def rank_night_quality(observed_minutes: int, span_minutes: int) -> str:
    """Map observed coverage to a quality label."""
    if span_minutes <= 0:
        return "unknown"

    ratio = observed_minutes / span_minutes
    if ratio >= 0.65:
        return "excellent"
    if ratio >= 0.35:
        return "good"
    if ratio >= 0.15:
        return "marginal"
    return "poor"
