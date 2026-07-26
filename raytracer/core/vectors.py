"""Deprecated: moved to :mod:`raytracer.math.vectors`."""

from __future__ import annotations

from ..math.vectors import (  # noqa: F401
    EPS,
    as_vector,
    direction_from_angle,
    normalize,
    normalize_rows,
    perpendicular_2d,
    rotation_2d,
)

__all__ = [
    "EPS",
    "as_vector",
    "normalize",
    "normalize_rows",
    "direction_from_angle",
    "perpendicular_2d",
    "rotation_2d",
]
