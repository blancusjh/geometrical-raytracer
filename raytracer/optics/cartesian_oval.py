"""Cartesian-oval surface for the 2-D engine.

Wraps the pure Fermat-principle math in :mod:`raytracer.shapes.fermat_oval`
as a ``Surface``: drawn with the exact GOTS parametrization, hit-tested via
the bisection solve in :mod:`raytracer.math.intersections`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..math.intersections import intersect_fermat_oval
from ..math.transforms import RigidTransform
from ..math.vectors import normalize
from ..shapes.fermat_oval import fermat_oval_F, fermat_oval_grad, sigma_parametric
from .ray import Ray
from .surface import Intersection, Surface


@dataclass
class CartesianOvalSurface(Surface):
    """
    Cartesian oval defined by (z0, zi, n0, ni).
    Local axes: x ≡ z (optical axis), y ≡ r (radial).
    """
    z0: float
    zi: float
    n_exterior: float = 1.0  # n0
    n_interior: float = 1.5  # ni
    origin: np.ndarray = field(default_factory=lambda: np.zeros(2))
    angle: float = 0.0
    surface_id: str = "cartesian_oval"
    semidiameter: float | None = None

    def __post_init__(self) -> None:
        Surface.__init__(self,
                         surface_id=self.surface_id,
                         n_exterior=self.n_exterior,
                         n_interior=self.n_interior)
        self.origin = np.asarray(self.origin, dtype=float)
        self.frame = RigidTransform.from_angle_2d(origin=self.origin, angle=self.angle)

    # ---------- drawing ----------
    def _rho_max(self) -> float:
        """
        Find the closure point of the oval: the first ρ > 0 (past the vertex)
        where r(ρ) = 0, i.e. ρ² - z(ρ)² = 0, using bracket + bisection.

        Near the vertex, z(ρ) is higher-order in ρ (paraxial sag), so
        g(ρ) = ρ² - z(ρ)² starts positive; the oval closes where g first
        turns negative (z catches up with ρ).
        """
        def g(rho):
            z, _ = sigma_parametric(self.z0, self.zi, np.array([rho]), self.n_exterior, self.n_interior)
            return rho * rho - float(z[0]) ** 2

        lo, hi = 1e-6, 1.0
        glo = g(lo)  # > 0 just past the vertex
        for _ in range(64):
            ghi = g(hi)
            if ghi <= 0.0:
                break
            hi *= 2.0
        else:
            # never closes within range: fall back to a sane radius to "see" the curve
            return hi

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            gm = g(mid)
            if abs(gm) < 1e-10 or (hi - lo) < 1e-9:
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
    def hit(self, ray: Ray) -> Optional[Intersection]:
        o_loc = self.frame.to_local(ray.origin)
        d_loc = self.frame.direction_to_local(ray.direction)

        res = intersect_fermat_oval(
            o_loc, d_loc,
            z0=self.z0, zi=self.zi,
            n0=self.n_exterior, ni=self.n_interior,
            implicit_form=fermat_oval_F,
        )
        if res is None:
            return None
        p_loc, lam = res

        if self.semidiameter is not None and abs(p_loc[1]) > self.semidiameter:
            return None

        # Compute normal from gradient (handle sign for r coordinate)
        grad = fermat_oval_grad(p_loc[0], abs(p_loc[1]),
                                 self.z0, self.zi,
                                 self.n_exterior, self.n_interior)

        # Account for sign of r (y-coordinate)
        if p_loc[1] < 0:
            grad = np.array([grad[0], -grad[1]])

        n_loc = normalize(grad)

        p_w = self.frame.to_world(p_loc)
        n_w = normalize(self.frame.direction_to_world(n_loc))

        return Intersection(
            point=p_w,
            normal=n_w,
            distance=float(lam),
            surface_id=self.surface_id,
            parameters=np.asarray(p_loc, dtype=float),
            surface=self,
        )


__all__ = ["CartesianOvalSurface"]
