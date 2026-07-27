"""Surfaces: the shapes a ray can meet, and how each describes itself.

Every surface here states its own geometry — implicitly (``f_Sigma(x) = 0``)
or parametrically (``x = P(t)``) — plus where it sits, how far it extends,
and which media it separates. It never solves for an intersection itself:
that is done once, generically, in
:func:`raytracer.math.intersections.intersect_ray_with_surface`, which reads
the description off the surface.

This package depends on :mod:`raytracer.math` alone. It does not import
:mod:`raytracer.optics` — a surface is a shape between two media, and knows
nothing about the rays that hit it or the laws they obey when they do.
"""

from .cartesian_oval import (
    CartesianOvalProfile,
    CartesianOvalSurface,
    cartesian_oval_implicit,
    cartesian_oval_implicit_gradient,
    cartesian_oval_parametric_curve,
    cartesian_oval_sag_and_slope,
    gots_params,
    max_usable_height,
)
from .conic import (
    CircleSurface,
    ConicSurface,
    EllipseSurface,
    HyperbolaSurface,
    ParabolaSurface,
    quadratic_coeffs_from_ep,
)
from .profile import AsphereProfile, ProfileSurface
from .segment import LineSegment
from .surface import Intersection, Surface

__all__ = [
    "Surface",
    "Intersection",
    "AsphereProfile",
    "ProfileSurface",
    "LineSegment",
    "quadratic_coeffs_from_ep",
    "ConicSurface",
    "EllipseSurface",
    "CircleSurface",
    "ParabolaSurface",
    "HyperbolaSurface",
    "gots_params",
    "cartesian_oval_parametric_curve",
    "cartesian_oval_sag_and_slope",
    "cartesian_oval_implicit",
    "cartesian_oval_implicit_gradient",
    "max_usable_height",
    "CartesianOvalProfile",
    "CartesianOvalSurface",
]
