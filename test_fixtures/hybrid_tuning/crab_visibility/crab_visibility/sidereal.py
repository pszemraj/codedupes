"""Sidereal time utilities."""

from datetime import datetime, timezone


def normalize_longitude_deg(lon_deg: float) -> float:
    """Wrap longitude into [0, 360)."""
    wrapped = lon_deg % 360.0
    if wrapped < 0.0:
        wrapped += 360.0
    return wrapped


def utc_to_julian_day(moment: datetime) -> float:
    """Convert UTC datetime to Julian day."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    utc = moment.astimezone(timezone.utc)

    year = utc.year
    month = utc.month
    day = utc.day + (utc.hour + utc.minute / 60.0 + utc.second / 3600.0) / 24.0

    if month <= 2:
        year -= 1
        month += 12

    a = year // 100
    b = 2 - a + (a // 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
    return jd


def local_sidereal_time_hours(moment: datetime, longitude_deg: float) -> float:
    """Approximate local sidereal time in hours."""
    jd = utc_to_julian_day(moment)
    t = (jd - 2451545.0) / 36525.0

    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t * t
        - (t * t * t) / 38710000.0
    )
    lmst_deg = normalize_longitude_deg(gmst_deg + longitude_deg)
    return lmst_deg / 15.0
