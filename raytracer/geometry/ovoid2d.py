"""Cartesian ovoid: stigmatic refractive surface from Fermat's principle.

Pure implicit-form math (no engine dependency): given object/image points
and their media, :func:`fermat_ovoid_F` is the surface's defining equation
and :func:`fermat_ovoid_grad` its gradient (normal direction). The
closed-form GOTS parametrization in
:mod:`raytracer.geometry.cartesian_oval` traces the same family of surfaces
exactly; :func:`sigma_parametric` bridges the two for drawing. Turning this
into a traceable ``Surface2D`` is an engine concern — see
:class:`raytracer.nonseq.ovoid2d.CartesianOvoid2D`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .cartesian_oval import cartesian_oval_parametric_curve, gots_params

# ============================================================================
# FERMAT PRINCIPLE: Cartesian Ovoid Implicit Form
# ============================================================================
# For object at z0, image at zi, indices n0 (exterior) and ni (interior):
# The surface satisfies: n0·dist(object, surface) + ni·dist(surface, image) = constant
# ============================================================================

def fermat_ovoid_F(z: float, r: float, z0: float, zi: float, n0: float, ni: float) -> float:
    """
    Fermat-based implicit form: F(z,r) = 0 defines the Cartesian ovoid surface.

    F = n0·√((z-z0)² + r²) + ni·√((z-zi)² + r²) - (n0·|z0| + ni·zi)

    This ensures perfect stigmatism: all rays from z0 focus to zi.
    """
    dist_to_object = np.sqrt((z - z0)**2 + r**2)
    dist_to_image = np.sqrt((z - zi)**2 + r**2)
    constant = n0 * abs(z0) + ni * abs(zi)

    return n0 * dist_to_object + ni * dist_to_image - constant


def fermat_ovoid_grad(z: float, r: float, z0: float, zi: float, n0: float, ni: float) -> np.ndarray:
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


def _intersect_fermat_ovoid(origin: np.ndarray, direction: np.ndarray, *,
                            z0: float, zi: float, n0: float, ni: float) -> Optional[Tuple[np.ndarray, float]]:
    """
    Find ray-surface intersection using bisection method (robust and accurate).

    Returns: (intersection_point, distance) or None
    """
    # Search for sign change in F along the ray
    t_samples = np.linspace(0, 50, 500)
    F_samples = []

    for t in t_samples:
        pt = origin + t * direction
        F_val = fermat_ovoid_F(pt[0], abs(pt[1]), z0, zi, n0, ni)
        F_samples.append(F_val)

    F_samples = np.array(F_samples)

    # Find first sign change (intersection)
    sign_changes = np.where(np.diff(np.sign(F_samples)) != 0)[0]

    if len(sign_changes) == 0:
        return None

    # Use first intersection
    idx = sign_changes[0]
    t_min = float(t_samples[idx])
    t_max = float(t_samples[idx + 1])

    # Bisection refinement
    for _ in range(100):
        t_mid = (t_min + t_max) / 2
        pt_mid = origin + t_mid * direction
        F_mid = fermat_ovoid_F(pt_mid[0], abs(pt_mid[1]), z0, zi, n0, ni)

        if abs(F_mid) < 1e-10:
            return pt_mid, float(t_mid)

        pt_min = origin + t_min * direction
        F_min = fermat_ovoid_F(pt_min[0], abs(pt_min[1]), z0, zi, n0, ni)

        if F_min * F_mid < 0:
            t_max = t_mid
        else:
            t_min = t_mid

        if abs(t_max - t_min) < 1e-12:
            break

    pt_final = origin + t_mid * direction
    F_final = fermat_ovoid_F(pt_final[0], abs(pt_final[1]), z0, zi, n0, ni)

    if abs(F_final) < 1e-6:
        return pt_final, float(t_mid)

    return None


# ============================================================================
# Parametric form for visualization
# ============================================================================

def sigma_parametric(z0: float, zi: float, rho: np.ndarray, n0: float, ni: float):
    """
    Exact parametric form ``rho -> (z, r)`` for drawing the Cartesian ovoid.

    Uses the closed-form GOTS parametrization (``raytracer.geometry.
    cartesian_oval``): plugging ``(z(rho), r(rho))`` back into
    ``fermat_ovoid_F`` gives a residual at machine precision, so the drawn
    curve matches the surface the ray tracer actually intersects.
    """
    G, O, T, S = gots_params(n0, z0, ni, zi)
    return cartesian_oval_parametric_curve(rho, G, O, T, S)


__all__ = [
    "fermat_ovoid_F",
    "fermat_ovoid_grad",
    "sigma_parametric",
]


def __getattr__(name: str):
    # Deprecated: the Surface2D adapter moved to raytracer.nonseq.ovoid2d.
    if name == "CartesianOvoid2D":
        import warnings

        from ..nonseq.ovoid2d import CartesianOvoid2D

        warnings.warn(
            "raytracer.geometry.ovoid2d.CartesianOvoid2D is deprecated; import it "
            "from raytracer.nonseq.ovoid2d instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return CartesianOvoid2D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
