"""Human readable reporting helpers."""


def format_minutes_hms(minutes_visible: int) -> str:
    """Render minute count as HH:MM."""
    if minutes_visible < 0:
        raise ValueError("minutes must be non-negative")
    hrs, mins = divmod(minutes_visible, 60)
    return f"{hrs:02d}:{mins:02d}"


def summarize_visibility(minutes_visible: int, never_visible: bool = False) -> str:
    """Create a one-line summary for a target night."""
    if never_visible:
        return "Cancer is below horizon for the full night"
    return f"Cancer above horizon for {format_minutes_hms(minutes_visible)}"


def build_markdown_row(site: str, minutes_visible: int, never_visible: bool = False) -> str:
    """Create markdown table row for dashboard publishing."""
    status = summarize_visibility(minutes_visible, never_visible)
    return f"| {site} | {status} |"
