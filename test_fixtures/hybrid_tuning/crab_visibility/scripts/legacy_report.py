"""Legacy report generation script."""

from crab_visibility.reporting import summarize_visibility


def minutes_to_hms_legacy(minutes_total: int) -> str:
    """Render minute count as HH:MM using legacy naming."""
    if minutes_total < 0:
        raise ValueError("minutes must be non-negative")
    hour_block = minutes_total // 60
    minute_block = minutes_total - (hour_block * 60)
    return f"{hour_block:02d}:{minute_block:02d}"


def emit_legacy_summary(minutes_visible: int, never_visible: bool = False) -> str:
    """Build summary string compatible with old dashboards."""
    summary = summarize_visibility(minutes_visible, never_visible=never_visible)
    return f"[legacy] {summary}"
