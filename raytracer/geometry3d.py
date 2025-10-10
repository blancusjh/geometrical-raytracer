"""3D surface primitives built atop the dimension-agnostic framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np

from .geometry import ConicProfile
from .dioptrics import SigmaCurve
from .rays import RayND, normalize
from .surfaces import SurfaceIntersection, SurfaceND

_EPS = 1e-9

if TYPE_CHECKING:  # pragma: no cover - used for annotations only
    from .geometry import ConicalDioptrique


def _sanitize_generatrix(points: np.ndarray) -> np.ndarray:
    """Return points sorted by axial coordinate with duplicate radii removed."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Generatrix must be an (N, 2) array of (x, rho) samples")
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return pts
    pts[:, 1] = np.abs(pts[:, 1])
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    dedup = [pts[0]]
    for row in pts[1:]:
        if np.linalg.norm(row - dedup[-1]) > 1e-9:
            dedup.append(row)
    curve = np.array(dedup)
    curve[0, 1] = 0.0
    return curve


def _quad_roots(a: float, b: float, c: float, eps: float = 1e-12) -> list[float]:
    """Solve ax^2 + bx + c = 0 returning the real roots."""

    if abs(a) < eps:
        if abs(b) < eps:
            return []
        return [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < -eps:
        return []
    disc = max(0.0, disc)
    sqrt_disc = float(np.sqrt(disc))
    denom = 2.0 * a
    return [(-b - sqrt_disc) / denom, (-b + sqrt_disc) / denom]


def _conic_roots_for_rho(coeffs: tuple[float, float, float, float, float, float], rho: float) -> list[float]:
    """Return axial positions for a given radial coordinate on the conic."""

    A, B, C, D, E, F = coeffs
    a = A
    b = B * rho + D
    c = C * rho * rho + E * rho + F
    return _quad_roots(a, b, c)


@dataclass
class AxisymmetricConicSurface3D(SurfaceND):
    """Revolution of a 2D conic profile around the x-axis."""

    profile: ConicProfile
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    surface_id: str = "axisymmetric_conic"
    max_distance: float = 1e4
    initial_step: float = 0.1
    tol: float = 1e-9
    profile_curve: "ConicalDioptrique | None" = None
    n_exterior: float = 1.0
    n_interior: float = 1.0

    def __post_init__(self) -> None:
        SurfaceND.__init__(self, dimension=3, parameter_dimension=3, surface_id=self.surface_id)
        self.origin = np.asarray(self.origin, dtype=float)
        if self.origin.shape != (3,):
            raise ValueError("origin must be a 3D vector")
        self.coeffs = self.profile.coeffs
        self.profile_curve = self.profile_curve

    # -- Helpers -----------------------------------------------------------------
    def _implicit(self, x: float, rho: float) -> float:
        A, B, C, D, E, F = self.coeffs
        return A * x * x + B * x * rho + C * rho * rho + D * x + E * rho + F

    def _implicit_derivative(self, x: float, rho: float, dx: float, drho: float) -> float:
        A, B, C, D, E, _ = self.coeffs
        return (2.0 * A * x + D) * dx + (B * x + 2.0 * C * rho + E) * drho + B * rho * dx

    def _gradient(self, point_local: np.ndarray) -> np.ndarray:
        x, y, z = point_local
        rho = np.hypot(y, z)
        A, B, C, D, E, _ = self.coeffs
        nx = 2.0 * A * x + B * rho + D
        dF_drho = B * x + 2.0 * C * rho + E
        if rho > _EPS:
            coeff = dF_drho / rho
            ny = coeff * y
            nz = coeff * z
        else:
            ny = 0.0
            nz = 0.0
        return normalize(np.array([nx, ny, nz], dtype=float))

    def _evaluate(self, origin: np.ndarray, direction: np.ndarray, t: float) -> tuple[float, float, float, float]:
        point = origin + t * direction
        x = point[0]
        y = point[1]
        z = point[2]
        rho = np.hypot(y, z)
        value = self._implicit(x, rho)
        dy = direction[1]
        dz = direction[2]
        if rho > _EPS:
            drho = (y * dy + z * dz) / rho
        else:
            drho = np.hypot(dy, dz)
        dx = direction[0]
        deriv = self._implicit_derivative(x, rho, dx, drho)
        return value, deriv, rho, x

    def _find_root(self, origin: np.ndarray, direction: np.ndarray) -> Optional[float]:
        t_low = 0.0
        val_low, _, _, _ = self._evaluate(origin, direction, t_low)
        if abs(val_low) <= self.tol:
            return t_low

        step = self.initial_step
        t_high = t_low + step
        val_high, _, _, _ = self._evaluate(origin, direction, t_high)
        max_iter = 200
        iterations = 0
        while val_low * val_high > 0.0 and t_high <= self.max_distance and iterations < max_iter:
            t_low, val_low = t_high, val_high
            step *= 1.5
            t_high = t_low + step
            val_high, _, _, _ = self._evaluate(origin, direction, t_high)
            iterations += 1

        if val_low * val_high > 0.0:
            return None

        for _ in range(80):
            t_mid = 0.5 * (t_low + t_high)
            val_mid, _, _, _ = self._evaluate(origin, direction, t_mid)
            if abs(val_mid) <= self.tol:
                return t_mid
            if val_low * val_mid <= 0.0:
                t_high, val_high = t_mid, val_mid
            else:
                t_low, val_low = t_mid, val_mid
        return 0.5 * (t_low + t_high)

    # -- SurfaceND hooks ---------------------------------------------------------
    def solve_intersection(self, ray: RayND) -> SurfaceIntersection | None:
        if ray.dimension != 3:
            raise ValueError("AxisymmetricConicSurface3D expects a 3D ray")
        origin = ray.origin - self.origin
        direction = ray.direction
        t = self._find_root(origin, direction)
        if t is None or t < self.tol:
            return None
        point_local = origin + t * direction
        rho = np.hypot(point_local[1], point_local[2])
        phi = np.arctan2(point_local[2], point_local[1]) if rho > _EPS else 0.0
        params = np.array([point_local[0], rho, phi], dtype=float)
        point_world = self.origin + point_local
        normal_world = self._gradient(point_local)
        return SurfaceIntersection(
            distance=float(t),
            parameters=params,
            point=point_world,
            normal=normal_world,
            meta={"rho": float(rho), "phi": float(phi)},
        )

    def point_from_parameters(self, parameters: np.ndarray) -> np.ndarray:
        x, rho, phi = parameters
        y = rho * np.cos(phi)
        z = rho * np.sin(phi)
        return self.origin + np.array([x, y, z], dtype=float)

    def normal_from_parameters(self, parameters: np.ndarray) -> np.ndarray:
        point_local = self.point_from_parameters(parameters) - self.origin
        return self._gradient(point_local)

    # -- Sampling utilities -----------------------------------------------------
    def _auto_rho_limit(self, *, rho_step: float = 0.05, rho_max: float = 5.0) -> float:
        limit = 0.0
        rho = float(rho_step)
        while rho <= rho_max:
            roots = _conic_roots_for_rho(self.coeffs, rho)
            if roots:
                limit = rho
            else:
                if limit > 0.0:
                    return limit
            rho += rho_step
        return limit if limit > 0.0 else rho_max

    def generatrix_polyline(
        self,
        samples: int = 256,
        *,
        rho_max: float | None = None,
        branch: int = -1,
    ) -> np.ndarray:
        """Return (x, rho) samples describing the generating curve."""

        if samples < 2:
            raise ValueError("samples must be >= 2")

        if self.profile_curve is not None:
            pts = self.profile_curve.polyline(samples=samples)
            curve = np.column_stack((pts[:, 0], np.abs(pts[:, 1])))
            if rho_max is not None:
                mask = curve[:, 1] <= float(rho_max) + 1e-9
                curve = curve[mask] if np.any(mask) else curve
            return _sanitize_generatrix(curve)

        rho_limit = rho_max if rho_max is not None else self._auto_rho_limit()
        rhos = np.linspace(0.0, rho_limit, samples)
        xs: list[float] = []
        prev_x: float | None = None
        for rho in rhos:
            roots = _conic_roots_for_rho(self.coeffs, float(rho))
            if not roots:
                continue
            roots = sorted(roots)
            if prev_x is None:
                idx = 0 if branch <= 0 else -1
                chosen = roots[idx]
            else:
                chosen = min(roots, key=lambda val: abs(val - prev_x))
            xs.append(chosen)
            prev_x = chosen

        if not xs:
            return np.empty((0, 2), dtype=float)

        curve = np.column_stack((xs, rhos[: len(xs)]))
        return _sanitize_generatrix(curve)


@dataclass
class AxisymmetricCartesianSurface3D(SurfaceND):
    """Revolution of a Cartesian ovoid profile around the x-axis."""

    curve: SigmaCurve
    rho_max: float
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    surface_id: str = "axisymmetric_cartesian"
    max_distance: float = 1e4
    initial_step: float = 0.1
    tol: float = 1e-9
    n_exterior: float | None = None
    n_interior: float | None = None

    def __post_init__(self) -> None:
        SurfaceND.__init__(self, dimension=3, parameter_dimension=3, surface_id=self.surface_id)
        self.origin = np.asarray(self.origin, dtype=float)
        if self.origin.shape != (3,):
            raise ValueError("origin must be a 3D vector")
        self._z_func, _ = self.curve.lambdify()
        self.n_exterior = float(self.curve.n0 if self.n_exterior is None else self.n_exterior)
        self.n_interior = float(self.curve.ni if self.n_interior is None else self.n_interior)

    def _z(self, rho: float) -> float:
        rho_clamped = max(_EPS, min(self.rho_max, rho))
        return float(self._z_func(rho_clamped))

    def _dz_drho(self, rho: float) -> float:
        rho = max(_EPS, min(self.rho_max, rho))
        h = max(1e-6, 1e-4 * rho)
        z_forward = self._z(rho + h)
        z_backward = self._z(max(_EPS, rho - h))
        return (z_forward - z_backward) / (2.0 * h)

    def _evaluate(self, origin: np.ndarray, direction: np.ndarray, t: float) -> tuple[float, float, float]:
        point = origin + t * direction
        x = point[0]
        y = point[1]
        z = point[2]
        rho = np.hypot(y, z)
        if rho > self.rho_max + 10 * _EPS:
            return np.inf, rho, x
        target = self._z(rho)
        value = x - target
        return value, rho, x

    def _find_root(self, origin: np.ndarray, direction: np.ndarray) -> Optional[float]:
        t_low = 0.0
        val_low, rho_low, _ = self._evaluate(origin, direction, t_low)
        if abs(val_low) <= self.tol:
            return t_low

        step = self.initial_step
        t_high = t_low + step
        val_high, rho_high, _ = self._evaluate(origin, direction, t_high)
        max_iter = 200
        iterations = 0
        while (np.isinf(val_high) or val_low * val_high > 0.0) and t_high <= self.max_distance and iterations < max_iter:
            t_low, val_low = t_high, val_high
            step *= 1.5
            t_high = t_low + step
            val_high, rho_high, _ = self._evaluate(origin, direction, t_high)
            iterations += 1

        if np.isinf(val_high) or val_low * val_high > 0.0:
            return None

        for _ in range(80):
            t_mid = 0.5 * (t_low + t_high)
            val_mid, _, _ = self._evaluate(origin, direction, t_mid)
            if abs(val_mid) <= self.tol:
                return t_mid
            if val_low * val_mid <= 0.0:
                t_high, val_high = t_mid, val_mid
            else:
                t_low, val_low = t_mid, val_mid
        return 0.5 * (t_low + t_high)

    def solve_intersection(self, ray: RayND) -> SurfaceIntersection | None:
        if ray.dimension != 3:
            raise ValueError("AxisymmetricCartesianSurface3D expects a 3D ray")
        origin = ray.origin - self.origin
        direction = ray.direction
        t = self._find_root(origin, direction)
        if t is None or t < self.tol:
            return None
        point_local = origin + t * direction
        rho = np.hypot(point_local[1], point_local[2])
        phi = np.arctan2(point_local[2], point_local[1]) if rho > _EPS else 0.0
        params = np.array([point_local[0], rho, phi], dtype=float)
        normal_world = self.normal_from_parameters(params)
        return SurfaceIntersection(
            distance=float(t),
            parameters=params,
            point=self.origin + point_local,
            normal=normal_world,
            meta={"rho": float(rho), "phi": float(phi)},
        )

    def point_from_parameters(self, parameters: np.ndarray) -> np.ndarray:
        x, rho, phi = parameters
        y = rho * np.cos(phi)
        z = rho * np.sin(phi)
        return self.origin + np.array([x, y, z], dtype=float)

    def normal_from_parameters(self, parameters: np.ndarray) -> np.ndarray:
        x, rho, phi = parameters
        dz_drho = self._dz_drho(rho)
        if rho > _EPS:
            ny = -dz_drho * np.cos(phi)
            nz = -dz_drho * np.sin(phi)
        else:
            ny = 0.0
            nz = 0.0
        nx = 1.0
        normal_local = np.array([nx, ny, nz], dtype=float)
        return normalize(normal_local)

    def generatrix_polyline(self, samples: int = 256) -> np.ndarray:
        """Return (x, rho) samples describing the generating curve."""

        if samples < 2:
            raise ValueError("samples must be >= 2")
        rhos = np.linspace(0.0, self.rho_max, samples)
        xs = np.array([self._z(float(rho)) for rho in rhos], dtype=float)
        curve = np.column_stack((xs, rhos))
        return _sanitize_generatrix(curve)
