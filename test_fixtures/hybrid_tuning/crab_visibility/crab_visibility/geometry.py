"""Coordinate helpers for crab visibility calculations."""

import math


def normalize_longitude_deg(lon_deg: float) -> float:
    """Wrap longitude into [0, 360)."""
    wrapped = lon_deg % 360.0
    if wrapped < 0.0:
        wrapped += 360.0
    return wrapped


def clamp_latitude_deg(lat_deg: float) -> float:
    """Clamp latitude into the physical range."""
    return max(-90.0, min(90.0, lat_deg))


def deg_to_rad(deg: float) -> float:
    """Convert degrees to radians."""
    return deg * math.pi / 180.0


def rad_to_deg(rad: float) -> float:
    """Convert radians to degrees."""
    return rad * 180.0 / math.pi


def angular_separation_deg(ra_a: float, dec_a: float, ra_b: float, dec_b: float) -> float:
    """Compute angular separation using spherical cosine law."""
    ra_a_r = deg_to_rad(ra_a)
    dec_a_r = deg_to_rad(dec_a)
    ra_b_r = deg_to_rad(ra_b)
    dec_b_r = deg_to_rad(dec_b)

    cos_angle = (
        math.sin(dec_a_r) * math.sin(dec_b_r)
        + math.cos(dec_a_r) * math.cos(dec_b_r) * math.cos(ra_a_r - ra_b_r)
    )
    clamped = max(-1.0, min(1.0, cos_angle))
    return rad_to_deg(math.acos(clamped))
