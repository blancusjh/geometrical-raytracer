"""Vectors, rigid transforms, and the ray/surface intersection solvers.

A utility layer: it knows nothing about optics.
"""

from .intersections import (
    intersect_ray_with_surface,
    intersect_rays_with_profile_surface,
    solve_implicit_bracket,
    solve_implicit_newton,
    solve_implicit_quadratic,
    solve_parametric,
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
    "intersect_ray_with_surface",
    "intersect_rays_with_profile_surface",
    "solve_implicit_quadratic",
    "solve_implicit_newton",
    "solve_implicit_bracket",
    "solve_parametric",
]
