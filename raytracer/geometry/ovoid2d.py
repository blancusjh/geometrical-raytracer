"""Cartesian ovoid: stigmatic refractive surface from Fermat's principle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ..core.frames import LocalFrame
from ..core.vectors import normalize
from ..nonseq.rays import Intersection2D, Ray2D
from ..nonseq.surfaces import Surface2D


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
# Parametric form for visualization (optional - can return empty for now)
# ============================================================================

def sigma_parametric(z0: float, zi: float, rho: np.ndarray, n0: float, ni: float):
    """
    Parametric form for drawing. Note: This may not match the Fermat surface exactly.
    For now, we'll use a simple approximation or return zeros.
    """
    # TODO: Implement accurate parametric form or use numerical sampling of Fermat surface
    rho = np.asarray(rho, dtype=float)
    z = np.zeros_like(rho)
    r = rho
    return z, r


# ---------------------------------------------------------------------
# Surface class (draw with σ, hit with implicit quartic)
# ---------------------------------------------------------------------
@dataclass
class CartesianOvoid2D(Surface2D):
    """
    Cartesian Ovoid defined by (z0, zi, n0, ni).
    Local axes: x ≡ z (optical axis), y ≡ r (radial).
    """
    z0: float
    zi: float
    n_exterior: float = 1.0  # n0
    n_interior: float = 1.5  # ni
    origin: np.ndarray = field(default_factory=lambda: np.zeros(2))
    angle: float = 0.0
    surface_id: str = "cartesian_ovoid"

    def __post_init__(self) -> None:
        Surface2D.__init__(self,
                           surface_id=self.surface_id,
                           n_exterior=self.n_exterior,
                           n_interior=self.n_interior)
        self.origin = np.asarray(self.origin, dtype=float)
        c, s = np.cos(self.angle), np.sin(self.angle)
        R = np.array([[c, -s], [s, c]], dtype=float)
        self.frame = LocalFrame(origin=self.origin, rotation=R)

    # ---------- drawing ----------
    def _rho_max(self) -> float:
        """
        Find the first positive ρ where r(ρ)=0 ↔ ρ^2 - z(ρ)^2 = 0
        using robust bracket + bisection. This mirrors your 'biseccion_mod'
        behavior so the drawn curve closes cleanly.
        """
        def g(rho):
            z, _ = sigma_parametric(self.z0, self.zi, np.array([rho]), self.n_exterior, self.n_interior)
            z = float(z[0])
            return rho * rho - z * z

        # bracket
        lo, hi = 0.0, 1.0
        glo = g(lo)  # <= 0
        for _ in range(64):
            ghi = g(hi)
            if ghi >= 0.0:
                break
            hi *= 2.0
        else:
            # fallback if never crosses: just take a sane radius to “see” the curve
            return max(2.0, hi)

        # bisection
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            gm = g(mid)
            if abs(gm) < 1e-10 or (hi - lo) < 1e-6:
                return mid
            if np.sign(gm) == np.sign(glo):
                lo, glo = mid, gm
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def as_points(self, samples: int = 600) -> np.ndarray:
        rho_max = self._rho_max()
        if not np.isfinite(rho_max) or rho_max <= 1e-9:
            return np.empty((0, 2), dtype=float)

        rho = np.linspace(0.0, rho_max, samples)
        z, r = sigma_parametric(self.z0, self.zi, rho, self.n_exterior, self.n_interior)

        # two branches (±r) to mirror how your circle is rendered
        upper = np.column_stack((z,  r))
        lower = np.column_stack((z[::-1], -r[::-1]))
        pts_local = np.vstack((upper, lower))

        pts_world = np.array([self.frame.to_world(p) for p in pts_local], dtype=float)
        return pts_world

    def polyline(self, samples: int = 512) -> np.ndarray:
        return self.as_points(samples=samples)

    # ---------- intersection ----------
    def intersect(self, ray: Ray2D) -> Optional[Intersection2D]:
        o_loc = self.frame.to_local(ray.origin)
        d_loc = self.frame.direction_to_local(ray.direction)

        res = _intersect_fermat_ovoid(o_loc, d_loc,
                                       z0=self.z0, zi=self.zi,
                                       n0=self.n_exterior, ni=self.n_interior)
        if res is None:
            return None
        p_loc, lam = res

        # Compute normal from gradient (handle sign for r coordinate)
        grad = fermat_ovoid_grad(p_loc[0], abs(p_loc[1]),
                                 self.z0, self.zi,
                                 self.n_exterior, self.n_interior)

        # Account for sign of r (y-coordinate)
        if p_loc[1] < 0:
            grad = np.array([grad[0], -grad[1]])

        n_loc = normalize(grad)

        p_w = self.frame.to_world(p_loc)
        n_w = normalize(self.frame.direction_to_world(n_loc))

        return Intersection2D(
            point=p_w,
            normal=n_w,
            distance=float(lam),
            surface_id=self.surface_id,
            parameters=np.asarray(p_loc, dtype=float),
            surface=self,
        )
