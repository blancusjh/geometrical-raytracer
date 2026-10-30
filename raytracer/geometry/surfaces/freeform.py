"""Freeform surfaces: no symmetry assumed.

:class:`FreeformSurface` is the XY-polynomial freeform on a conic base,

    s(x, y) = conic(r; R, k) + Σ c_ij (x / ρ)^i (y / ρ)^j,

with ``ρ = norm_radius`` making the coefficients dimensionless-friendly
(mm, for ``ρ`` in mm).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike

from ._value import SurfaceValue
from .base import Surface
from .conics import conic_sag_and_slope


class FreeformSurface(SurfaceValue, Surface):
    """XY-polynomial departure ``{(i, j): c_ij}`` on a conic base ``(R, k)``."""

    _params = ("R", "k", "coefficients", "norm_radius")

    def __init__(
        self,
        R: float = np.inf,
        k: float = 0.0,
        coefficients: Mapping[tuple[int, int], float] | None = None,
        norm_radius: float = 1.0,
    ) -> None:
        R = float(R)
        if R == 0.0:
            raise ValueError("R = 0 is a point; a flat base is R = inf")
        if not float(norm_radius) > 0.0:
            raise ValueError("norm_radius must be positive")
        terms = {}
        for (i, j), value in dict(coefficients or {}).items():
            if int(i) < 0 or int(j) < 0:
                raise ValueError("polynomial exponents are non-negative")
            if float(value) != 0.0:
                terms[(int(i), int(j))] = float(value)
        self.R, self.k, self.norm_radius = R, float(k), float(norm_radius)
        self._terms = tuple(sorted(terms.items()))
        self._freeze()

    @property
    def coefficients(self) -> dict[tuple[int, int], float]:
        return dict(self._terms)

    def _key(self) -> tuple:
        return (self.R, self.k, self._terms, self.norm_radius)

    def sag_gradient(self, x: ArrayLike, y: ArrayLike):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        r = np.hypot(x, y)
        c = 0.0 if np.isinf(self.R) else 1.0 / self.R
        s, slope = conic_sag_and_slope(r, c, self.k)
        safe_r = np.where(r == 0.0, 1.0, r)
        dsdx = np.where(r > 0.0, slope * x / safe_r, 0.0)
        dsdy = np.where(r > 0.0, slope * y / safe_r, 0.0)
        u, v, rho = x / self.norm_radius, y / self.norm_radius, self.norm_radius
        for (i, j), a in self._terms:
            s = s + a * u**i * v**j
            if i:
                dsdx = dsdx + a * i * u ** (i - 1) * v**j / rho
            if j:
                dsdy = dsdy + a * j * u**i * v ** (j - 1) / rho
        return s, dsdx, dsdy


__all__ = ["FreeformSurface"]
