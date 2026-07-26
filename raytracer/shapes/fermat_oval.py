"""Cartesian oval: the implicit form derived from Fermat's principle.

Pure shape math (no engine dependency): given object/image points and their
media, :func:`fermat_oval_F` is the surface's defining equation and
:func:`fermat_oval_grad` its gradient (normal direction). The closed-form
GOTS parametrization in :mod:`raytracer.shapes.cartesian_oval` traces the
same family of surfaces exactly; :func:`sigma_parametric` bridges the two
for drawing. Solving a ray against this implicit form is intersection math
(see :func:`raytracer.math.intersections.intersect_fermat_oval`); turning it
into a traceable surface is an engine concern (see
:class:`raytracer.optics.cartesian_oval.CartesianOvalSurface`).
"""

from __future__ import annotations

import numpy as np

from .cartesian_oval import cartesian_oval_parametric_curve, gots_params

# ============================================================================
# FERMAT PRINCIPLE: Cartesian Oval Implicit Form
# ============================================================================
# For object at z0, image at zi, indices n0 (exterior) and ni (interior):
# The surface satisfies: n0·dist(object, surface) + ni·dist(surface, image) = constant
# ============================================================================

def fermat_oval_F(z: float, r: float, z0: float, zi: float, n0: float, ni: float) -> float:
    """
    Fermat-based implicit form: F(z,r) = 0 defines the Cartesian oval surface.

    F = n0·√((z-z0)² + r²) + ni·√((z-zi)² + r²) - (n0·|z0| + ni·zi)

    This ensures perfect stigmatism: all rays from z0 focus to zi.
    """
    dist_to_object = np.sqrt((z - z0)**2 + r**2)
    dist_to_image = np.sqrt((z - zi)**2 + r**2)
    constant = n0 * abs(z0) + ni * abs(zi)

    return n0 * dist_to_object + ni * dist_to_image - constant


def fermat_oval_grad(z: float, r: float, z0: float, zi: float, n0: float, ni: float) -> np.ndarray:
    """
    Gradient ∇F = (∂F/∂z, ∂F/∂r) for the Fermat implicit form.
    This gives the surface normal direction.
    """
    eps = 1e-12
    dist_to_object = np.sqrt((z - z0)**2 + r**2) + eps
    dist_to_image = np.sqrt((z - zi)**2 + r**2) + eps

    dF_dz = n0 * (z - z0) / dist_to_object + ni * (z - zi) / dist_to_image
    dF_dr = n0 * r / dist_to_object + ni * r / dist_to_image

    return np.array([dF_dz, dF_dr], dtype=float)


def sigma_parametric(z0: float, zi: float, rho: np.ndarray, n0: float, ni: float):
    """
    Exact parametric form ``rho -> (z, r)`` for drawing the Cartesian oval.

    Uses the closed-form GOTS parametrization (``raytracer.shapes.
    cartesian_oval``): plugging ``(z(rho), r(rho))`` back into
    ``fermat_oval_F`` gives a residual at machine precision, so the drawn
    curve matches the surface the ray tracer actually intersects.
    """
    G, O, T, S = gots_params(n0, z0, ni, zi)
    return cartesian_oval_parametric_curve(rho, G, O, T, S)


__all__ = [
    "fermat_oval_F",
    "fermat_oval_grad",
    "sigma_parametric",
]
