"""Surfaces of revolution defined by a sag profile: conic base + asphere terms.

The sag of the surface at radial height ``h`` is

    z(h) = c h^2 / (1 + sqrt(1 - (1+K) c^2 h^2)) + sum_j A_j h^(2j+4)

where ``c`` is the vertex curvature (1/R), ``K`` the conic constant, and
``A_j = coefficients[j]`` the even polynomial coefficients starting at h^4.
This single convention covers both reference prescriptions:

- US 7,557,996 (Table 3A): K = 0, C1..C6 on h^4 .. h^14
- US 7,151,592 (EUV, Table 2): conic K plus A..E on h^4 .. h^12

:class:`AsphereProfile` is that sag law on its own (shared with the 3-D
sequential propagation, which positions it by a vertex along z);
:class:`ProfileSurface` places it as a traceable meridional surface, whose
implicit description is simply ``f_Sigma(x, y) = x - sag(|y|)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from ..math.transforms import RigidTransform
from .surface import Surface


@dataclass(frozen=True)
class AsphereProfile:
    """Immutable description of a conic + even-asphere profile."""

    curvature: float = 0.0
    conic: float = 0.0
    coefficients: tuple[float, ...] = ()

    @classmethod
    def plane(cls) -> "AsphereProfile":
        return cls()

    @classmethod
    def sphere(cls, radius: float) -> "AsphereProfile":
        return cls.from_radius(radius)

    @classmethod
    def from_radius(
        cls,
        radius: float,
        conic: float = 0.0,
        coefficients: tuple[float, ...] | list[float] = (),
    ) -> "AsphereProfile":
        """Build a profile from a vertex radius; ``radius == 0`` means plane."""

        curvature = 0.0 if radius == 0.0 or not np.isfinite(radius) else 1.0 / radius
        return cls(curvature=curvature, conic=float(conic), coefficients=tuple(coefficients))

    @property
    def radius(self) -> float:
        """Vertex radius of curvature (0.0 denotes a plane)."""

        return 0.0 if self.curvature == 0.0 else 1.0 / self.curvature

    @property
    def is_plane(self) -> bool:
        return self.curvature == 0.0 and not any(self.coefficients)

    def sag(self, h: ArrayLike) -> np.ndarray:
        return self.sag_and_slope(h)[0]

    def sag_and_slope(self, h: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(z, dz/dh)`` evaluated at radial height(s) *h*."""

        h = np.asarray(h, dtype=float)
        c = self.curvature
        if c == 0.0:
            base = np.zeros_like(h)
            slope = np.zeros_like(h)
        else:
            k = self.conic
            q2 = np.maximum(1.0 - (1.0 + k) * c * c * h * h, 1e-30)
            q = np.sqrt(q2)
            base = c * h * h / (1.0 + q)
            slope = c * h / q
        poly = np.zeros_like(h)
        dpoly = np.zeros_like(h)
        # coefficients[j] multiplies h^(2j+4)
        for j, coefficient in enumerate(self.coefficients, 2):
            if coefficient == 0.0:
                continue
            poly += coefficient * h ** (2 * j)
            dpoly += 2 * j * coefficient * h ** (2 * j - 1)
        return base + poly, slope + dpoly

    def with_coefficients(
        self, coefficients: tuple[float, ...] | list[float], conic: float | None = None
    ) -> "AsphereProfile":
        """Return a copy with new asphere coefficients (and optionally conic)."""

        return AsphereProfile(
            curvature=self.curvature,
            conic=self.conic if conic is None else float(conic),
            coefficients=tuple(coefficients),
        )


@dataclass
class ProfileSurface(Surface):
    """Meridional cut of a surface of revolution defined by a sag profile.

    The local frame puts the optical axis along +x, so a point at transverse
    height ``h`` sits at local ``(sag(h), h)`` and the implicit description is
    ``f_Sigma(x, y) = x - sag(|y|)``, with gradient
    ``(1, -slope * sign(y))``. This is the 2-D counterpart of a
    :class:`~raytracer.design.rows.SurfaceRow` and shares its sag mathematics
    (spheres, conics, aspheres) exactly.
    """

    profile: AsphereProfile = field(default_factory=AsphereProfile)
    vertex: np.ndarray = field(default_factory=lambda: np.zeros(2))
    angle: float = 0.0
    semidiameter: float = 10.0
    surface_id: str = "face"
    n_exterior: float = 1.0
    n_interior: float = 1.5
    interaction: str = "refract"  # "refract" | "reflect" | "config"

    #: Convention: the stored normal points toward -x (the exterior side).
    normal_sign: float = -1.0

    def __post_init__(self) -> None:
        Surface.__init__(
            self,
            surface_id=self.surface_id,
            n_exterior=self.n_exterior,
            n_interior=self.n_interior,
        )
        self.vertex = np.asarray(self.vertex, dtype=float)
        self.frame = RigidTransform.from_angle_2d(origin=self.vertex, angle=self.angle)
        if self.profile.is_plane:
            # A flat face is the degenerate quadratic f_Sigma = x, so the
            # solver reaches it in closed form -- including correctly
            # reporting no hit for a ray running parallel to it.
            self.quadratic_form = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    # -- implicit description ----------------------------------------------

    def implicit(self, point: np.ndarray) -> float:
        sag, _ = self.profile.sag_and_slope(abs(point[1]))
        return float(point[0] - float(sag))

    def implicit_gradient(self, point: np.ndarray) -> np.ndarray:
        _, slope = self.profile.sag_and_slope(abs(point[1]))
        return np.array([1.0, -float(slope) * np.sign(point[1])], dtype=float)

    def newton_seed(self, origin: np.ndarray, direction: np.ndarray) -> float | None:
        """Where the ray crosses the vertex plane -- the same guess the 3-D
        sequential propagation uses."""

        if abs(direction[0]) < 1e-14:
            return 0.0
        return (0.0 - origin[0]) / direction[0]

    def within_aperture(self, point: np.ndarray) -> bool:
        return abs(point[1]) <= self.semidiameter + 1e-9

    # -- drawing ------------------------------------------------------------

    def polyline(self, samples: int = 256) -> np.ndarray:
        h = np.linspace(-self.semidiameter, self.semidiameter, samples)
        sag = self.profile.sag(np.abs(h))
        pts_local = np.column_stack([sag, h])
        return np.array([self.frame.to_world(p) for p in pts_local])


__all__ = ["AsphereProfile", "ProfileSurface"]
