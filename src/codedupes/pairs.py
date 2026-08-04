"""Pair-key helpers for duplicate matching."""

from __future__ import annotations

from typing import Protocol


class _HasUid(Protocol):
    """Protocol for objects with a stable uid.

    :param str uid: Unique identifier.
    """

    uid: str


def ordered_pair_key(unit_a: _HasUid, unit_b: _HasUid) -> tuple[str, str]:
    """Return a stable ordered key for two units.

    :param _HasUid unit_a: First unit.
    :param _HasUid unit_b: Second unit.
    :return tuple[str, str]: Ordered uid pair.
    """

    return (min(unit_a.uid, unit_b.uid), max(unit_a.uid, unit_b.uid))


def unordered_pair_key(unit_a: _HasUid, unit_b: _HasUid) -> frozenset[str]:
    """Return an unordered uid key for two units.

    :param _HasUid unit_a: First unit.
    :param _HasUid unit_b: Second unit.
    :return frozenset[str]: Unordered uid set.
    """

    return frozenset((unit_a.uid, unit_b.uid))
