"""Coordinates of the condition space: where in the field, and where in the pupil."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    """A field point: ``(x, y)`` in the unit of the declared fields, or relative to the largest of them.

    ``Field.normalized(hy=1.0)`` is the largest declared field along y.
    """

    x: float = 0.0
    y: float = 0.0
    relative: bool = False

    @classmethod
    def normalized(cls, hx: float = 0.0, hy: float = 0.0) -> "Field":
        return cls(float(hx), float(hy), relative=True)


@dataclass(frozen=True)
class Pupil:
    """A point of the pupil in normalized coordinates: ``|p| = 1`` is the rim the aperture fills."""

    px: float = 0.0
    py: float = 0.0


__all__ = ["Field", "Pupil"]
