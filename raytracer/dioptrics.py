"""Cartesian dioptric surfaces and helper utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .geometry import Surface2D
from .rays import RayND, normalize
from .surfaces import SurfaceIntersection

_EPS_RHO = 1e-9
_MAX_RHO = 1e6


def sigma(z0: float, zi: float, rho: np.ndarray, n0: float, ni: float) -> tuple[np.ndarray, np.ndarray]:
    """Return axial and radial coordinates for a Cartesian ovoid profile."""
    rho = np.asarray(rho, dtype=float)
    z0 = -float(z0)
    zi = float(zi)
    n0 = float(n0)
    ni = float(ni)

    G = ((ni**2 * zi - n0**2 * z0) ** 2) / (ni * n0 * (ni * zi - n0 * z0) * (ni * z0 - n0 * zi))
    O = (ni * z0 - n0 * zi) / (zi * z0 * (ni - n0))
    T = (ni - n0) * (ni + n0) ** 2 / (4.0 * ni * n0 * zi * z0 * (ni * zi - n0 * z0))
    S = ((ni + n0) * (ni**2 * zi - n0**2 * z0)) / (2.0 * ni * n0 * zi * z0 * (ni * zi - n0 * z0))

    num = (O + T * rho**2) * rho**2
    radicand = 1.0 + (2.0 * S - (O**2) * G) * rho**2
    radicand = np.maximum(radicand, 0.0)
    den = 1.0 + S * rho**2 + np.sqrt(radicand)

    z = num / den
    r_sq = np.maximum(rho**2 - z**2, 0.0)
    r = np.sqrt(r_sq)
    return z, r


def _rho_for_radius(
    z0: float,
    zi: float,
    n0: float,
    ni: float,
    target_radius: float,
    *,
    t_shift: float = 0.0,
    tol: float = 1e-9,
) -> float:
    if target_radius <= 0.0:
        raise ValueError("target_radius must be positive")

    low = _EPS_RHO
    high = max(target_radius + 1.0, 1.0)

    def radius_at(rho: float) -> float:
        _, r = sigma(z0 - t_shift, zi - t_shift, np.array([rho]), n0, ni)
        return float(r[0])

    while radius_at(high) < target_radius:
        high *= 2.0
        if high > _MAX_RHO:
            raise RuntimeError("Unable to bracket radius for Cartesian dioptrique")

    for _ in range(120):
        mid = 0.5 * (low + high)
        r_mid = radius_at(mid)
        if abs(r_mid - target_radius) <= tol:
            return mid
        if r_mid < target_radius:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


@dataclass
class SigmaCurve:
    """Helper wrapper that exposes z(ρ) and r(ρ) evaluators."""

    z0: float
    zi: float
    n0: float
    ni: float
    t_shift: float = 0.0


    def lambdify(self) -> tuple[Callable[[float], float], Callable[[float], float]]:

        z0 = self.z0 - self.t_shift
        zi = self.zi - self.t_shift
        n0 = self.n0
        ni = self.ni

        def z_rho(rho: float) -> float:
            z, _ = sigma(z0, zi, np.array([rho]), n0, ni)
            return float(z[0] + self.t_shift)

        def r_rho(rho: float) -> float:
            _, r = sigma(z0, zi, np.array([rho]), n0, ni)
            return float(r[0])

        return z_rho, r_rho


def intersection_between_curves(
    z1_func: Callable[[float], float],
    r1_func: Callable[[float], float],
    z2_func: Callable[[float], float],
    r2_func: Callable[[float], float],
    *,
    tol: float = 1e-9,
    s1_0: float = 1.0,
    s2_0: float = 1.0,
) -> tuple[float, float]:
    def rho2_for_radius(target: float) -> float:
        low = _EPS_RHO
        high = max(s2_0, target + 1.0)
        f_low = r2_func(low) - target
        f_high = r2_func(high) - target
        while f_low * f_high > 0.0:
            high *= 2.0
            if high > _MAX_RHO:
                raise RuntimeError("Failed to bracket radius for second curve")
            f_high = r2_func(high) - target
        for _ in range(120):
            mid = 0.5 * (low + high)
            f_mid = r2_func(mid) - target
            if abs(f_mid) <= tol:
                return mid
            if f_low * f_mid <= 0.0:
                high = mid
                f_high = f_mid
            else:
                low = mid
                f_low = f_mid
        return 0.5 * (low + high)

    def f(rho1: float) -> float:
        r_target = r1_func(rho1)
        rho2 = rho2_for_radius(r_target)
        return z1_func(rho1) - z2_func(rho2)

    low = _EPS_RHO
    high = max(s1_0, low * 2.0)
    f_low = f(low)
    f_high = f(high)
    iter_guard = 0
    while f_low * f_high > 0.0:
        high *= 2.0
        f_high = f(high)
        iter_guard += 1
        if high > _MAX_RHO or iter_guard > 120:
            raise RuntimeError("Failed to bracket intersection between curves")

    for _ in range(120):
        mid = 0.5 * (low + high)
        f_mid = f(mid)
        if abs(f_mid) <= tol:
            rho1 = mid
            r_target = r1_func(rho1)
            rho2 = rho2_for_radius(r_target)
            return rho1, rho2
        if f_low * f_mid <= 0.0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    rho1 = 0.5 * (low + high)
    r_target = r1_func(rho1)
    rho2 = rho2_for_radius(r_target)
    return rho1, rho2


class SampledSurface2D(Surface2D):
    """Base class for surfaces defined via discrete sampling."""

    def __init__(self, *, surface_id: str) -> None:
        super().__init__(parameter_dimension=2, surface_id=surface_id)


@dataclass
class CartesianDioptrique(SampledSurface2D):
    """Piecewise-linear approximation of a Cartesian dioptric surface."""

    z0: float
    zi: float
    n_exterior: float
    n_interior: float
    rho_max: float | None = None
    aperture_radius: float | None = None
    samples: int = 400
    t_shift: float = 0.0
    surface_id: str = "cartesian_dioptrique"

    def __post_init__(self) -> None:
        SampledSurface2D.__init__(self, surface_id=self.surface_id)
        if self.samples < 2:
            raise ValueError("samples must be >= 2")
        self.z0 = float(self.z0)
        self.zi = float(self.zi)
        self.n_exterior = float(self.n_exterior)
        self.n_interior = float(self.n_interior)
        self.t_shift = float(self.t_shift)
        if self.rho_max is None and self.aperture_radius is None:
            raise ValueError("Either rho_max or aperture_radius must be provided")
        if self.aperture_radius is not None:
            self.rho_max = _rho_for_radius(
                self.z0,
                self.zi,
                self.n_exterior,
                self.n_interior,
                self.aperture_radius,
                t_shift=self.t_shift,
            )
        else:
            self.rho_max = float(self.rho_max)
        self._build_polyline()

    def _build_polyline(self) -> None:
        rho = np.linspace(_EPS_RHO, self.rho_max, self.samples)
        z_local, r = sigma(
            self.z0 - self.t_shift,
            self.zi - self.t_shift,
            rho,
            self.n_exterior,
            self.n_interior,
        )
        z = z_local + self.t_shift
        self._edge_radius = float(r[-1])

        axis_z = sigma(
            self.z0 - self.t_shift,
            self.zi - self.t_shift,
            np.array([_EPS_RHO]),
            self.n_exterior,
            self.n_interior,
        )[0][0] + self.t_shift
        self._axis_point = np.array([[axis_z, 0.0]], dtype=np.float64)

        upper = np.column_stack((z, r))
        lower = np.column_stack((z[::-1], -r[::-1]))
        polyline = np.vstack([self._axis_point, upper, lower])

        self._polyline = polyline.astype(np.float64)
        self._centroid = np.mean(self._polyline, axis=0)

        self._upper_segment = np.vstack([self._axis_point, upper])
        lower_segment = np.vstack([self._axis_point, lower[::-1]])
        lower_segment[0] = self._axis_point[0]
        self._lower_segment = lower_segment

    def solve_intersection(self, ray: RayND) -> SurfaceIntersection | None:
        if ray.dimension != 2:
            raise ValueError("CartesianDioptrique expects a 2D ray")
        best_distance = np.inf
        best_point: Optional[np.ndarray] = None
        best_normal: Optional[np.ndarray] = None
        best_meta: Optional[dict[str, float]] = None
        origin = ray.origin
        direction = ray.direction

        pts = self._polyline
        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            segment = p1 - p0
            seg_norm = np.linalg.norm(segment)
            if seg_norm <= 1e-12:
                continue
            denom = _cross(direction, segment)
            if abs(denom) < 1e-12:
                continue
            diff = p0 - origin
            t = _cross(diff, segment) / denom
            u = _cross(diff, direction) / denom
            if t <= 1e-8 or u < 0.0 or u > 1.0:
                continue
            if t >= best_distance:
                continue
            hit_point = origin + t * direction
            tangent = segment / seg_norm
            normal = np.array([tangent[1], -tangent[0]])
            midpoint = (p0 + p1) / 2.0
            if np.dot(normal, midpoint - self._centroid) < 0.0:
                normal = -normal
            best_distance = t
            best_point = hit_point
            best_normal = normalize(normal)
            best_meta = {"segment_index": float(i), "segment_parameter": float(u)}

        if best_point is None or best_normal is None:
            return None

        return SurfaceIntersection(
            distance=float(best_distance),
            parameters=np.asarray(best_point, dtype=float),
            point=np.asarray(best_point, dtype=float),
            normal=best_normal,
            meta=best_meta,
        )

    def point_from_parameters(self, parameters: np.ndarray) -> np.ndarray:
        return np.asarray(parameters, dtype=float)

    def normal_from_parameters(self, parameters: np.ndarray) -> np.ndarray:
        point = np.asarray(parameters, dtype=float)
        pts = self._polyline
        distances = np.linalg.norm(pts - point, axis=1)
        idx = int(np.argmin(distances))
        if idx == len(pts) - 1:
            idx -= 1
        segment = pts[idx + 1] - pts[idx]
        seg_norm = np.linalg.norm(segment)
        if seg_norm <= 1e-12:
            return np.array([0.0, 1.0])
        tangent = segment / seg_norm
        normal = np.array([tangent[1], -tangent[0]])
        midpoint = (pts[idx] + pts[idx + 1]) / 2.0
        if np.dot(normal, midpoint - self._centroid) < 0.0:
            normal = -normal
        return normalize(normal)

    def polyline_segments(self) -> list[np.ndarray]:
        return [self._upper_segment, self._lower_segment]

    def polyline(self, samples: int = 512) -> np.ndarray:
        if samples == len(self._polyline):
            return self._polyline
        rho = np.linspace(_EPS_RHO, self.rho_max, samples)
        z_local, r = sigma(
            self.z0 - self.t_shift,
            self.zi - self.t_shift,
            rho,
            self.n_exterior,
            self.n_interior,
        )
        z = z_local + self.t_shift
        upper = np.column_stack((z, r))
        lower = np.column_stack((z[::-1], -r[::-1]))
        return np.vstack([self._axis_point, upper, lower])

    @property
    def edge_radius(self) -> float:
        return self._edge_radius


@dataclass
class CartesianSinglet:
    """Utility for constructing stigmatic lenses from Cartesian dioptriques."""

    z0: float
    zc: float
    zi: float
    n0: float
    n_lens: float
    n_out: float | None = None
    thickness: float = 0.0
    aperture_radius: float = 0.5
    samples: int = 400

    def __post_init__(self) -> None:
        self.n_out = float(self.n0 if self.n_out is None else self.n_out)
        self.thickness = float(self.thickness)
        if self.thickness < 0.0:
            raise ValueError("Lens thickness must be non-negative")

        self.front_curve = SigmaCurve(self.z0, self.zc, self.n0, self.n_lens, t_shift=0.0)
        virtual_object = self.zc - self.thickness
        self.back_curve = SigmaCurve(-virtual_object, self.zi, self.n_lens, self.n_out, t_shift=self.thickness)

        z1_func, r1_func = self.front_curve.lambdify()
        z2_func, r2_func = self.back_curve.lambdify()
        rho1_edge = _rho_for_radius(self.z0, self.zc, self.n0, self.n_lens, self.aperture_radius, t_shift=0.0)
        rho2_edge = _rho_for_radius(-virtual_object, self.zi, self.n_lens, self.n_out, self.aperture_radius, t_shift=self.thickness)

        rho_i1, rho_i2 = intersection_between_curves(
            z1_func,
            r1_func,
            z2_func,
            r2_func,
            s1_0=rho1_edge,
            s2_0=rho2_edge,
        )
        self.rho_front = rho_i1
        self.rho_back = rho_i2

        self.front_surface = CartesianDioptrique(
            z0=self.z0,
            zi=self.zc,
            n_exterior=self.n0,
            n_interior=self.n_lens,
            rho_max=self.rho_front,
            samples=self.samples,
            surface_id="cartesian_front",
        )
        self.back_surface = CartesianDioptrique(
            z0=-virtual_object,
            zi=self.zi,
            n_exterior=self.n_lens,
            n_interior=self.n_out,
            rho_max=self.rho_back,
            t_shift=self.thickness,
            samples=self.samples,
            surface_id="cartesian_back",
        )

    def surfaces(self) -> tuple[CartesianDioptrique, CartesianDioptrique]:
        return self.front_surface, self.back_surface
