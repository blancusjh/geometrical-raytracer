"""Conic-section surfaces: ellipse, circle, parabola, hyperbola.

A conic in focal (polar) form ``r(theta) = p / (1 + e cos theta)`` has the
implicit description ``f_Sigma(x, y) = A x^2 + B xy + C y^2 + D x + E y + F``,
which is quadratic — so the intersection solver reaches it in closed form,
and the gradient of the same polynomial gives the normal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..math.transforms import RigidTransform
from .surface import Surface


def quadratic_coeffs_from_ep(e: float, p: float) -> tuple[float, float, float, float, float, float]:
    """Return ``(A, B, C, D, E, F)`` for a conic in polar form (semi-latus
    rectum *p*, eccentricity *e*)."""

    A: float = 1.0 - e**2
    B: float = 0.0
    C: float = 1.0
    D: float = 2.0 * e * p
    E: float = 0.0
    F: float = -(p**2)

    return (A, B, C, D, E, F)


@dataclass
class ConicSurface(Surface):
    """General conic section with optional rotation and refractive indices."""

    e: float
    p: float
    focus: np.ndarray = field(default_factory=lambda: np.zeros(2))
    angle: float = 0.0
    surface_id: str = "conic"
    n_exterior: float = 1.0
    n_interior: float = 1.0
    semidiameter: float | None = None

    def __post_init__(self) -> None:
        Surface.__init__(
            self,
            surface_id=self.surface_id,
            n_exterior=self.n_exterior,
            n_interior=self.n_interior,
        )
        self.focus = np.asarray(self.focus, dtype=float)
        self.angle = float(self.angle)
        self.frame = RigidTransform.from_angle_2d(origin=self.focus, angle=self.angle)
        self.quadratic_form = quadratic_coeffs_from_ep(self.e, self.p)

    # -- implicit description ----------------------------------------------

    def implicit(self, point: np.ndarray) -> float:
        x, y = point
        A, B, C, D, E, F = self.quadratic_form
        return A * x * x + B * x * y + C * y * y + D * x + E * y + F

    def implicit_gradient(self, point: np.ndarray) -> np.ndarray:
        x, y = point
        A, B, C, D, E, _ = self.quadratic_form
        return np.array([2.0 * A * x + B * y + D, B * x + 2.0 * C * y + E], dtype=float)

    def within_aperture(self, point: np.ndarray) -> bool:
        return self.semidiameter is None or abs(point[1]) <= self.semidiameter

    # -- drawing ------------------------------------------------------------

    def r(self, theta: float) -> Optional[float]:
        denom = 1.0 + self.e * np.cos(theta)
        if abs(denom) <= 1e-12:
            return None
        return self.p / denom

    def as_points(
        self,
        samples: int = 800,
        theta_span: tuple[float, float] = (-np.pi, np.pi),
        r_clip: float = 1e3,
    ) -> np.ndarray:
        thetas = np.linspace(theta_span[0], theta_span[1], samples)
        rs = np.array([self.r(theta) for theta in thetas], dtype=float)
        rs = np.where(np.isfinite(rs), np.clip(rs, -r_clip, r_clip), np.nan)
        pts_local = np.column_stack((rs * np.cos(thetas), rs * np.sin(thetas)))
        valid = [self.frame.to_world(pt) for pt in pts_local if not np.isnan(pt).any()]
        if not valid:
            return np.empty((0, 2), dtype=float)
        return np.array(valid, dtype=float)

    def polyline(self, samples: int = 512) -> np.ndarray:
        return self.as_points(samples=samples)


class EllipseSurface(ConicSurface):
    """Ellipse defined by semi-axes (a, b) with focus at *focus*."""

    def __init__(
        self,
        semi_major: float,
        semi_minor: float,
        focus: Optional[np.ndarray] = None,
        angle: float = 0.0,
        surface_id: str = "ellipse",
        n_exterior: float = 1.0,
        n_interior: float = 1.0,
        semidiameter: float | None = None,
    ) -> None:
        if semi_major <= 0.0 or semi_minor <= 0.0:
            raise ValueError("Semi-axes must be positive.")
        if semi_minor > semi_major:
            raise ValueError("Semi-minor axis cannot exceed semi-major axis.")
        e = np.sqrt(1.0 - (semi_minor**2) / (semi_major**2))
        p = (semi_minor**2) / semi_major
        focus_vec = np.zeros(2) if focus is None else np.asarray(focus, dtype=float)
        super().__init__(
            e=e,
            p=p,
            focus=focus_vec,
            angle=angle,
            surface_id=surface_id,
            n_exterior=n_exterior,
            n_interior=n_interior,
            semidiameter=semidiameter,
        )


class CircleSurface(ConicSurface):
    """Circle (eccentricity zero) with radius *radius*."""

    def __init__(
        self,
        radius: float,
        center: Optional[np.ndarray] = None,
        surface_id: str = "circle",
        n_exterior: float = 1.0,
        n_interior: float = 1.0,
        semidiameter: float | None = None,
    ) -> None:
        if radius <= 0.0:
            raise ValueError("Radius must be positive.")
        center_vec = np.zeros(2) if center is None else np.asarray(center, dtype=float)
        super().__init__(
            e=0.0,
            p=radius,
            focus=center_vec,
            surface_id=surface_id,
            n_exterior=n_exterior,
            n_interior=n_interior,
            semidiameter=semidiameter,
        )


class ParabolaSurface(ConicSurface):
    """Parabola (eccentricity one) with semi-latus rectum *p*."""

    def __init__(
        self,
        p: float,
        focus: Optional[np.ndarray] = None,
        angle: float = 0.0,
        surface_id: str = "parabola",
        n_exterior: float = 1.0,
        n_interior: float = 1.0,
        semidiameter: float | None = None,
    ) -> None:
        if p <= 0.0:
            raise ValueError("Semi-latus rectum must be positive.")
        focus_vec = np.zeros(2) if focus is None else np.asarray(focus, dtype=float)
        super().__init__(
            e=1.0,
            p=p,
            focus=focus_vec,
            angle=angle,
            surface_id=surface_id,
            n_exterior=n_exterior,
            n_interior=n_interior,
            semidiameter=semidiameter,
        )


class HyperbolaSurface(ConicSurface):
    """Right-opening hyperbola defined by semi-axes (a, b)."""

    def __init__(
        self,
        semi_major: float,
        semi_minor: float,
        focus: Optional[np.ndarray] = None,
        angle: float = 0.0,
        surface_id: str = "hyperbola",
        n_exterior: float = 1.0,
        n_interior: float = 1.0,
        semidiameter: float | None = None,
    ) -> None:
        if semi_major <= 0.0 or semi_minor <= 0.0:
            raise ValueError("Semi-axes must be positive.")
        e = np.sqrt(1.0 + (semi_minor**2) / (semi_major**2))
        p = (semi_minor**2) / semi_major
        focus_vec = np.zeros(2) if focus is None else np.asarray(focus, dtype=float)
        super().__init__(
            e=e,
            p=p,
            focus=focus_vec,
            angle=angle,
            surface_id=surface_id,
            n_exterior=n_exterior,
            n_interior=n_interior,
            semidiameter=semidiameter,
        )


__all__ = [
    "quadratic_coeffs_from_ep",
    "ConicSurface",
    "EllipseSurface",
    "CircleSurface",
    "ParabolaSurface",
    "HyperbolaSurface",
]
