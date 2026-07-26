"""Conic sections: the pure algebra defining their quadratic form.

Conics are expressed in focal (polar) form ``r(theta) = p / (1 + e cos
theta)``. This module only defines that shape as a quadratic form; solving
a ray against it is intersection math (see
:func:`raytracer.math.intersections.intersect_quadratic`), and turning it
into a traceable surface is an engine concern (see
:mod:`raytracer.optics.conics`).
"""

from __future__ import annotations


def quadratic_coeffs_from_ep(e: float, p: float) -> tuple[float, float, float, float, float, float]:
    """Return quadratic-form coefficients ``(A, B, C, D, E, F)`` for a conic
    in polar form (semi-latus rectum *p*, eccentricity *e*)."""

    A: float = 1.0 - e**2
    B: float = 0.0
    C: float = 1.0
    D: float = 2.0 * e * p
    E: float = 0.0
    F: float = -(p**2)

    return (A, B, C, D, E, F)


__all__ = ["quadratic_coeffs_from_ep"]
