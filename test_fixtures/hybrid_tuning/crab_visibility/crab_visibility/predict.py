"""Prediction helpers for Cancer constellation visibility."""


def is_cancer_above_horizon(altitude_deg: float | None, min_altitude_deg: float = 5.0) -> bool:
    """Determine if Cancer is above the configured altitude floor."""
    if altitude_deg is None:
        return False
    return altitude_deg >= min_altitude_deg


def estimate_visibility_minutes(
    altitude_track: list[float | None],
    cadence_minutes: int = 5,
    min_altitude_deg: float = 5.0,
) -> int:
    """Count visible samples and convert to minutes."""
    visible_samples = 0
    for altitude in altitude_track:
        if altitude is None:
            continue
        if altitude >= min_altitude_deg:
            visible_samples += 1
    return visible_samples * cadence_minutes


def visibility_score(altitude_track: list[float | None], min_altitude_deg: float = 5.0) -> float:
    """Compute a quality score [0, 1] for the provided track."""
    present = [alt for alt in altitude_track if alt is not None]
    if not present:
        return 0.0

    above = [alt for alt in present if alt >= min_altitude_deg]
    ratio = len(above) / len(present)
    margin = max(0.0, (sum(above) / len(above) - min_altitude_deg) / 30.0) if above else 0.0
    return min(1.0, 0.7 * ratio + 0.3 * margin)


def estimate_nightly_visibility_profile(
    timeline: list[tuple[int, float | None]],
    min_altitude_deg: float = 5.0,
) -> list[tuple[int, int]]:
    """Build merged minute windows where Cancer stays visible."""
    spans: list[tuple[int, int]] = []
    span_start: int | None = None

    for minute, altitude in timeline:
        if altitude is not None and altitude >= min_altitude_deg:
            if span_start is None:
                span_start = minute
            continue

        if span_start is not None:
            spans.append((span_start, minute))
            span_start = None

    if span_start is not None and timeline:
        spans.append((span_start, timeline[-1][0] + 1))

    merged: list[tuple[int, int]] = []
    for left, right in spans:
        if merged and left <= merged[-1][1]:
            prev_left, prev_right = merged[-1]
            merged[-1] = (prev_left, max(prev_right, right))
        else:
            merged.append((left, right))
    return merged


def classify_night_quality(minutes_visible: int, total_minutes: int) -> str:
    """Map visibility coverage to a quality label."""
    if total_minutes <= 0:
        return "unknown"

    coverage = minutes_visible / total_minutes
    if coverage >= 0.65:
        return "excellent"
    if coverage >= 0.35:
        return "good"
    if coverage >= 0.15:
        return "marginal"
    return "poor"


def altitude_bucket(angle_deg: float | None) -> str:
    """Convert altitude in degrees into a coarse category."""
    if angle_deg is None:
        return "missing"
    if angle_deg < 0.0:
        return "below"
    if angle_deg < 10.0:
        return "low"
    if angle_deg < 30.0:
        return "mid"
    return "high"


def verbose_visibility_flag(altitude_deg: float | None, floor_deg: float = 5.0) -> bool:
    """Visibility flag with expanded guard structure."""
    if altitude_deg is None:
        return False

    if altitude_deg < floor_deg:
        return False

    margin = altitude_deg - floor_deg
    if margin < 0:
        return False

    return True
