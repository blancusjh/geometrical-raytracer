"""Conic-section interfaces for the 2-D non-sequential engine.

Conics are expressed in focal (polar) form r(theta) = p / (1 + e cos theta)
and intersected through their implicit quadratic form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ..core.frames import LocalFrame
from ..core.vectors import normalize
from ..nonseq.rays import Intersection2D, Ray2D
from ..nonseq.surfaces import Surface2D


def quadratic_coeffs_from_ep(e: float, p: float) -> tuple[float, float, float, float, float, float]:
    """Return quadratic-form coefficients for a conic in polar form."""

    A: float = 1.0 - e**2
    B: float = 0.0
    C: float = 1.0
    D: float = 2.0 * e * p
    E: float = 0.0
    F: float = -(p**2)

    return (A, B, C, D, E, F)


@dataclass
class ConicProfile:
    """Reusable solver for conic sections expressed in quadratic form."""

    coeffs: Tuple[float, float, float, float, float, float]

    def intersect(
        self, origin: np.ndarray, direction: np.ndarray, eps: float = 1e-12
    ) -> Tuple[np.ndarray, float] | None:
        x0, y0 = origin
        dx, dy = direction
        A, B, C, D, E, F = self.coeffs

        a = A * dx * dx + B * dx * dy + C * dy * dy
        b = 2.0 * A * x0 * dx + B * (x0 * dy + y0 * dx) + 2.0 * C * y0 * dy + D * dx + E * dy
        c = A * x0 * x0 + B * x0 * y0 + C * y0 * y0 + D * x0 + E * y0 + F

        lam: float | None = None

        if abs(a) < eps:
            if abs(b) >= eps:
                candidate = -c / b
                if candidate >= 0.0:
                    lam = candidate
        else:
            disc = b * b - 4.0 * a * c
            if disc >= -eps:
                disc = max(0.0, disc)
                sqrt_disc = np.sqrt(disc)
                lam1 = (-b - sqrt_disc) / (2.0 * a)
                lam2 = (-b + sqrt_disc) / (2.0 * a)
                candidates = [val for val in (lam1, lam2) if val >= 0.0]
                if candidates:
                    lam = min(candidates)

        if lam is None:
            return None

        point = origin + lam * direction
        return point, float(lam)

    def normal(self, point: np.ndarray) -> np.ndarray:
        x, y = point
        A, B, C, D, E, _ = self.coeffs
        nx = 2.0 * A * x + B * y + D
        ny = B * x + 2.0 * C * y + E
        return normalize(np.array([nx, ny], dtype=float))


@dataclass
class ConicInterface2D(Surface2D):
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
        Surface2D.__init__(
            self,
            surface_id=self.surface_id,
            n_exterior=self.n_exterior,
            n_interior=self.n_interior,
        )
        self.focus = np.asarray(self.focus, dtype=float)
        self.angle = float(self.angle)
        self.frame = LocalFrame.from_angle_2d(origin=self.focus, angle=self.angle)
        self.coeffs = quadratic_coeffs_from_ep(self.e, self.p)
        self.profile = ConicProfile(self.coeffs)

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

    def intersect(self, ray: Ray2D) -> Optional[Intersection2D]:
        origin_local = self.frame.to_local(ray.origin)
        direction_local = self.frame.direction_to_local(ray.direction)
        result = self.profile.intersect(origin_local, direction_local)
        if result is None:
            return None
        point_local, lam = result
        if self.semidiameter is not None and abs(point_local[1]) > self.semidiameter:
            return None
        point_world = self.frame.to_world(point_local)
        normal_world = normalize(self.frame.direction_to_world(self.profile.normal(point_local)))
        return Intersection2D(
            point=point_world,
            normal=normal_world,
            distance=float(lam),
            surface_id=self.surface_id,
            parameters=np.asarray(point_local, dtype=float),
            surface=self,
        )


class EllipseConic(ConicInterface2D):
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


class CircleConic(ConicInterface2D):
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


class ParabolaConic(ConicInterface2D):
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


class HyperbolaConic(ConicInterface2D):
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
    "ConicProfile",
    "ConicInterface2D",
    "EllipseConic",
    "CircleConic",
    "ParabolaConic",
    "HyperbolaConic",
]
