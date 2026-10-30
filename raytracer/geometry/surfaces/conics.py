"""Conic surfaces of revolution: the quadrics with a vertex.

A conic of vertex radius ``R`` (curvature ``c = 1/R``) and conic constant
``k`` is the surface

    c r² − 2 z + c (1 + k) z² = 0,      r² = x² + y²,

whose vertex sheet is the sag ``s(r) = c r² / (1 + √(1 − (1 + k) c² r²))``.
The value of ``k`` names the family, and each family is a subclass:

====================  ==================  ===========================
family                constant            class
====================  ==================  ===========================
hyperboloid           k < −1              :class:`Hyperboloid`
paraboloid            k = −1              :class:`Paraboloid`
ellipsoid (prolate)   −1 < k < 0          :class:`Ellipsoid`
sphere                k = 0               :class:`Sphere` ⊂ :class:`Ellipsoid`
ellipsoid (oblate)    k > 0               :class:`Ellipsoid`
plane                 c = 0               :class:`Plane`
====================  ==================  ===========================

A sphere is the ellipsoid of equal semi-axes, so it is a subclass of
:class:`Ellipsoid`. A plane is the ``c = 0`` member of *every* family, so
it is a subclass of :class:`Conic` itself, not of any one family.

Being a quadric, every conic is intersected in closed form.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..intersections import SurfaceHits, solve_quadric
from ._value import SurfaceValue
from .base import SurfaceOfRevolution


def conic_sag_and_slope(r: ArrayLike, c: float, k: float) -> tuple[np.ndarray, np.ndarray]:
    """``(s, ds/dr)`` of the conic vertex sheet (the shared formula)."""

    r = np.asarray(r, dtype=float)
    if c == 0.0:
        return np.zeros_like(r), np.zeros_like(r)
    q = np.sqrt(np.maximum(1.0 - (1.0 + k) * c * c * r * r, 1e-30))
    return c * r * r / (1.0 + q), c * r / q


class Conic(SurfaceValue, SurfaceOfRevolution):
    """General conic of revolution, vertex radius ``R`` and conic constant ``k``.

    ``R = inf`` is the flat member (prefer :class:`Plane`). ``R = 0`` is
    rejected: it is a point, not a surface.
    """

    _params = ("R", "k")

    def __init__(self, R: float, k: float = 0.0) -> None:
        R = float(R)
        if R == 0.0:
            raise ValueError("R = 0 is a point, not a surface; a flat surface is Plane() (or R = inf)")
        self.R = R
        self.k = float(k)
        self._check()
        self._freeze()

    def _check(self) -> None:
        """Family constraint on ``k``; the general conic accepts any value."""

    @classmethod
    def from_curvature(cls, c: float, k: float = 0.0) -> "Conic":
        return Plane() if c == 0.0 else cls(1.0 / c, k)

    @property
    def c(self) -> float:
        return 0.0 if np.isinf(self.R) else 1.0 / self.R

    @property
    def curvature(self) -> float:
        return self.c

    @property
    def radius(self) -> float:
        return self.R

    @property
    def is_plane(self) -> bool:
        return self.c == 0.0

    @property
    def max_radius(self) -> float:
        c, k = self.c, self.k
        if c == 0.0 or 1.0 + k <= 0.0:
            return np.inf
        return 1.0 / (abs(c) * np.sqrt(1.0 + k))

    def classify(self) -> "Conic":
        """The same surface as an instance of its most specific family."""

        if self.c == 0.0:
            return Plane()
        if self.k == 0.0:
            return Sphere(self.R)
        if self.k == -1.0:
            return Paraboloid(self.R)
        if self.k > -1.0:
            return Ellipsoid(self.R, self.k)
        return Hyperboloid(self.R, self.k)

    def sag_and_slope(self, r: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        return conic_sag_and_slope(r, self.c, self.k)

    def normal(self, points: ArrayLike) -> np.ndarray:
        """Unit normals from the quadric's own gradient, exact up to an ellipsoid's equator.

        ``∇(c r² − 2z + c(1+k) z²)`` turned to the +z side: ``(−c x, −c y, 1 − c(1+k) z)``.
        The sag slope diverges where the vertex sheet turns vertical; this does not.
        """

        p = np.asarray(points, dtype=float)
        c, ck = self.c, self.c * (1.0 + self.k)
        n = np.stack([-c * p[..., 0], -c * p[..., 1], 1.0 - ck * p[..., 2]], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def intersect(self, origins, directions, alive=None, *, mode="nearest", t_min=1e-9) -> SurfaceHits:
        o, d = origins, directions
        if alive is None:
            alive = np.ones(o.shape[0], dtype=bool)
        c, k = self.c, self.k
        if c == 0.0:
            return _plane_hits(o, d, alive, mode, t_min)
        ck = c * (1.0 + k)
        a = c * (d[:, 0] ** 2 + d[:, 1] ** 2) + ck * d[:, 2] ** 2
        b = 2.0 * (c * (o[:, 0] * d[:, 0] + o[:, 1] * d[:, 1]) + ck * o[:, 2] * d[:, 2] - d[:, 2])
        cc = c * (o[:, 0] ** 2 + o[:, 1] ** 2) + ck * o[:, 2] ** 2 - 2.0 * o[:, 2]

        def on_sheet(p: np.ndarray) -> np.ndarray:
            # The vertex sheet is where 1 - (1+k) c z = sqrt(1 - (1+k) c² r²) >= 0.
            return 1.0 - ck * p[:, 2] >= -1e-12

        return solve_quadric(a, b, cc, on_sheet, o, d, alive, mode=mode, t_min=t_min)


def _plane_hits(o, d, alive, mode, t_min) -> SurfaceHits:
    dz = d[:, 2]
    parallel = alive & (np.abs(dz) < 1e-14)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(parallel, np.nan, -o[:, 2] / np.where(dz == 0.0, 1.0, dz))
    missed = np.zeros(o.shape[0], dtype=bool)
    if mode == "forward":
        missed = alive & ~parallel & (t <= t_min)
    return SurfaceHits(t, parallel, np.zeros_like(parallel), missed)


class Plane(Conic):
    """The flat surface ``z = 0``: the ``c = 0`` member of every conic family."""

    _params = ()

    def __init__(self) -> None:
        self.R = np.inf
        self.k = 0.0
        self._freeze()


class Ellipsoid(Conic):
    """Ellipsoid of revolution, ``k > −1`` (prolate for ``k < 0``, oblate for ``k > 0``)."""

    def _check(self) -> None:
        if not self.k > -1.0:
            raise ValueError(f"an ellipsoid has k > -1, got k = {self.k}")

    @classmethod
    def from_semi_axes(cls, axial: float, transverse: float) -> "Ellipsoid":
        """From the semi-axis along z (signed: its sign is the sign of R) and the transverse one."""

        a, b = float(axial), float(transverse)
        return cls(b * b / a, (b * b) / (a * a) - 1.0)

    @property
    def semi_axes(self) -> tuple[float, float]:
        """``(axial, transverse)``: ``a = R / (1 + k)``, ``b = R / √(1 + k)``."""

        return self.R / (1.0 + self.k), abs(self.R) / np.sqrt(1.0 + self.k)


class Sphere(Ellipsoid):
    """Sphere of radius ``R``: the ellipsoid with equal semi-axes (``k = 0``)."""

    _params = ("R",)

    def __init__(self, R: float) -> None:
        super().__init__(R, 0.0)


class Paraboloid(Conic):
    """Paraboloid of revolution (``k = −1``), given by ``R`` or by its focal distance ``f = R/2``.

    ``f`` is the geometric distance from the vertex to the focus of the
    surface — the focal length of a paraboloidal *mirror* — not the focal
    length of a lens made with it.
    """

    _params = ("R",)

    def __init__(self, R: float | None = None, *, f: float | None = None) -> None:
        if (R is None) == (f is None):
            raise TypeError("give exactly one of R or f (f = R / 2)")
        super().__init__(2.0 * f if R is None else R, -1.0)

    @property
    def f(self) -> float:
        return self.R / 2.0


class Hyperboloid(Conic):
    """Hyperboloid of revolution (one sheet of it), ``k < −1``."""

    def _check(self) -> None:
        if not self.k < -1.0:
            raise ValueError(f"a hyperboloid has k < -1, got k = {self.k}")


__all__ = ["Conic", "Plane", "Ellipsoid", "Sphere", "Paraboloid", "Hyperboloid", "conic_sag_and_slope"]
