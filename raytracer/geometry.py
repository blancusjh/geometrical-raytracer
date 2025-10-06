"""Geometric surfaces and intersection helpers for 2D tracing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .rays import Intersection2D, Ray, normalize


def quadratic_coeffs_from_ep(e: float, p: float) -> tuple[float, float, float, float, float, float]:
    """Return quadratic-form coefficients for a conic in polar form."""
    A: float = 1.0 - e ** 2
    B: float = 0.0
    C: float = 1.0
    D: float = 2.0 * e * p
    E: float = 0.0
    F: float = -(p ** 2)
    return (A, B, C, D, E, F)


class Surface2D:
    """Abstract base for planar optical elements."""

    surface_id: str

    def first_intersection(self, ray: Ray) -> Optional[Intersection2D]:
        raise NotImplementedError

    def polyline(self, samples: int = 512) -> np.ndarray:
        raise NotImplementedError


@dataclass
class ConicalDioptrique(Surface2D):
    """General conic with focus at *focus* and optical axis rotated by *angle*."""

    e: float
    p: float
    focus: np.ndarray = field(default_factory=lambda: np.zeros(2))
    angle: float = 0.0
    surface_id: str = "conic"

    def __post_init__(self) -> None:
        self.focus = np.asarray(self.focus, dtype=float)
        self.angle = float(self.angle)
        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)
        self._R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        self._RT = self._R.T
        self.coeffs = quadratic_coeffs_from_ep(self.e, self.p)

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
        rs = [self.r(theta) for theta in thetas]
        rs = np.array([np.nan if val is None else val for val in rs])
        rs = np.clip(rs, -r_clip, r_clip)
        pts_local = np.column_stack((rs * np.cos(thetas), rs * np.sin(thetas)))
        pts_world = (self._R @ pts_local.T).T + self.focus
        pts_world = pts_world[~np.isnan(pts_world).any(axis=1)]
        return pts_world

    def first_intersection(self, ray: Ray) -> Optional[Intersection2D]:
        origin_local = self._to_local_point(ray.origin)
        direction_local = self._to_local_direction(ray.direction)

        result = self._intersect_local(origin_local, direction_local)
        if result is None:
            return None

        point_local, distance = result
        point_world = self._to_world_point(point_local)
        normal_local = self._normal_local(point_local)
        normal_world = normalize(self._R @ normal_local)

        return Intersection2D(
            point=point_world,
            normal=normal_world,
            distance=float(distance),
            surface_id=self.surface_id,
            surface=self,
        )

    def polyline(self, samples: int = 512) -> np.ndarray:
        return self.as_points(samples=samples)

    def _intersect_local(self, origin: np.ndarray, direction: np.ndarray) -> Optional[tuple[np.ndarray, float]]:
        x0, y0 = origin
        dx, dy = direction
        A, B, C, D, E, F = self.coeffs

        a = A * dx * dx + B * dx * dy + C * dy * dy
        b = 2.0 * A * x0 * dx + B * (x0 * dy + y0 * dx) + 2.0 * C * y0 * dy + D * dx + E * dy
        c = A * x0 * x0 + B * x0 * y0 + C * y0 * y0 + D * x0 + E * y0 + F

        eps = 1e-12
        lam: Optional[float] = None

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

        point_local = origin + lam * direction
        return point_local, lam

    def _normal_local(self, point_local: np.ndarray) -> np.ndarray:
        x, y = point_local
        A, B, C, D, E, _ = self.coeffs
        nx = 2.0 * A * x + B * y + D
        ny = B * x + 2.0 * C * y + E
        return normalize(np.array([nx, ny]))

    def _to_local_point(self, point: np.ndarray) -> np.ndarray:
        return self._RT @ (point - self.focus)

    def _to_world_point(self, point_local: np.ndarray) -> np.ndarray:
        return self._R @ point_local + self.focus

    def _to_local_direction(self, direction: np.ndarray) -> np.ndarray:
        return self._RT @ direction


class EllipseConic(ConicalDioptrique):
    """Ellipse defined by semi-axes (a, b) with focus at *focus*."""

    def __init__(
        self,
        semi_major: float,
        semi_minor: float,
        focus: Optional[np.ndarray] = None,
        angle: float = 0.0,
        surface_id: str = "ellipse",
    ) -> None:
        if semi_major <= 0.0 or semi_minor <= 0.0:
            raise ValueError("Semi-axes must be positive.")
        if semi_minor > semi_major:
            raise ValueError("Semi-minor axis cannot exceed semi-major axis.")
        e = np.sqrt(1.0 - (semi_minor ** 2) / (semi_major ** 2))
        p = (semi_minor ** 2) / semi_major
        focus_vec = np.zeros(2) if focus is None else np.asarray(focus, dtype=float)
        super().__init__(e=e, p=p, focus=focus_vec, angle=angle, surface_id=surface_id)


class CircleConic(ConicalDioptrique):
    """Circle (eccentricity zero) with radius *radius*."""

    def __init__(
        self,
        radius: float,
        center: Optional[np.ndarray] = None,
        surface_id: str = "circle",
    ) -> None:
        if radius <= 0.0:
            raise ValueError("Radius must be positive.")
        center_vec = np.zeros(2) if center is None else np.asarray(center, dtype=float)
        super().__init__(e=0.0, p=radius, focus=center_vec, surface_id=surface_id)


class ParabolaConic(ConicalDioptrique):
    """Parabola (eccentricity one) with semi-latus rectum *p*."""

    def __init__(
        self,
        p: float,
        focus: Optional[np.ndarray] = None,
        angle: float = 0.0,
        surface_id: str = "parabola",
    ) -> None:
        if p <= 0.0:
            raise ValueError("Semi-latus rectum must be positive.")
        focus_vec = np.zeros(2) if focus is None else np.asarray(focus, dtype=float)
        super().__init__(e=1.0, p=p, focus=focus_vec, angle=angle, surface_id=surface_id)
