"""Dimension-agnostic math utilities: vectors and rigid-frame transforms.

This is the utility layer: it knows nothing about optics. Everything above
it (:mod:`raytracer.physics`, :mod:`raytracer.geometry`, the sequential and
non-sequential engines) is built out of these primitives, never the other
way around.
"""

from .frames import LocalFrame
from .vectors import (
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
    "LocalFrame",
]
