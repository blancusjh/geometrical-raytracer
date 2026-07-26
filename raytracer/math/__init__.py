"""Dimension-agnostic math utilities: vectors, rigid transforms, and the
pure numeric ray/shape intersection solvers.

This is the utility layer: it knows nothing about optics. Everything above
it (:mod:`raytracer.physics`, :mod:`raytracer.shapes`, :mod:`raytracer.optics`,
the propagation engines) is built out of these primitives, never the other
way around.
"""

from .intersections import (
    intersect_fermat_oval,
    intersect_profile,
    intersect_profile_batch,
    intersect_quadratic,
    intersect_segment,
    quadratic_normal,
)
from .transforms import RigidTransform
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
    "RigidTransform",
    "intersect_segment",
    "intersect_quadratic",
    "quadratic_normal",
    "intersect_profile",
    "intersect_profile_batch",
    "intersect_fermat_oval",
]
