"""Rotationally-symmetric surface profiles: conic base + even asphere terms.

The sag of a surface at radial height ``h`` is

    z(h) = c h^2 / (1 + sqrt(1 - (1+K) c^2 h^2)) + sum_j A_j h^(2j+4)

where ``c`` is the vertex curvature (1/R), ``K`` the conic constant, and
``A_j = coefficients[j]`` the even polynomial coefficients starting at h^4.
This single convention covers both reference prescriptions:

- US 7,557,996 (Table 3A): K = 0, C1..C6 on h^4 .. h^14
- US 7,151,592 (EUV, Table 2): conic K plus A..E on h^4 .. h^12
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


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


__all__ = ["AsphereProfile"]
