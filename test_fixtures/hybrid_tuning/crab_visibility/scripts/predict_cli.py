"""Command-line frontend for crab visibility predictions."""

from crab_visibility.predict import estimate_visibility_minutes
from crab_visibility.reporting import summarize_visibility


def parse_lat_lon(value: str) -> tuple[float, float]:
    """Parse comma separated latitude and longitude string."""
    left, right = value.split(",", 1)
    return float(left.strip()), float(right.strip())


def predict_for_track(track: list[float | None]) -> str:
    """Predict visibility for a single synthetic altitude track."""
    minutes = estimate_visibility_minutes(track)
    return summarize_visibility(minutes, never_visible=minutes == 0)


def main() -> None:
    """Print one sample prediction for local manual testing."""
    sample_track = [None, -3.0, 2.5, 6.5, 8.1, 10.0, 3.1]
    print(predict_for_track(sample_track))


if __name__ == "__main__":
    main()
