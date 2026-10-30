"""Aspheres: a conic base plus a polynomial departure from it.

Every asphere here is ``s(r) = conic(r; R, k) + departure(r)``, so each has
a :attr:`~Asphere.base` conic. A conic is the zero-departure member of
*every* asphere family, so specialization among them is a lattice, not a
tree: :class:`Conic` is not made a subclass of any one asphere family, and
:meth:`Asphere.base` / ``from_conic`` express the relation instead.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import eval_jacobi

from ._value import SurfaceValue
from .base import SurfaceOfRevolution
from .conics import Conic, conic_sag_and_slope


class Asphere(SurfaceValue, SurfaceOfRevolution):
    """A conic base ``(R, k)`` plus a departure term."""

    R: float
    k: float

    @property
    def c(self) -> float:
        return 0.0 if np.isinf(self.R) else 1.0 / self.R

    @property
    def base(self) -> Conic:
        """The conic this asphere departs from (as its most specific family)."""

        return Conic(self.R, self.k).classify() if not np.isinf(self.R) else Conic.from_curvature(0.0)

    @abstractmethod
    def departure_and_slope(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(departure(r), d departure / dr)``."""

    def sag_and_slope(self, r: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(r, dtype=float)
        s, ds = conic_sag_and_slope(r, self.c, self.k)
        p, dp = self.departure_and_slope(r)
        return s + p, ds + dp

    @property
    def max_radius(self) -> float:
        return Conic(self.R, self.k).max_radius if not np.isinf(self.R) else np.inf


def _read_radius(R: float) -> float:
    R = float(R)
    if R == 0.0:
        raise ValueError("R = 0 is a point, not a surface; a flat base is R = inf")
    return R


class EvenAsphere(Asphere):
    """``s(r) = conic(r) + Σ a_p r^p`` over even powers ``p ≥ 2``.

    ``coefficients`` maps each power to its coefficient (mm^(1−p)), e.g.
    ``{4: 1e-7, 6: -2e-10}``. The ``r²`` term is allowed, as in the
    standard definition; it adds ``2 a_2`` to the vertex curvature.
    """

    _params = ("R", "k", "coefficients")

    def __init__(self, R: float, k: float = 0.0, coefficients: Mapping[int, float] | None = None) -> None:
        terms = {}
        for power, value in dict(coefficients or {}).items():
            power = int(power)
            if power < 2 or power % 2:
                raise ValueError(f"an even asphere has even powers >= 2, got r^{power}")
            if float(value) != 0.0:
                terms[power] = float(value)
        self.R = _read_radius(R)
        self.k = float(k)
        self._terms = tuple(sorted(terms.items()))
        self._freeze()

    @property
    def coefficients(self) -> dict[int, float]:
        return dict(self._terms)

    def _key(self) -> tuple:
        return (self.R, self.k, self._terms)

    @classmethod
    def from_conic(cls, conic: Conic, coefficients: Mapping[int, float] | None = None) -> "EvenAsphere":
        return cls(conic.R, conic.k, coefficients)

    def departure_and_slope(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        poly = np.zeros_like(r)
        dpoly = np.zeros_like(r)
        for power, a in self._terms:
            poly += a * r**power
            dpoly += power * a * r ** (power - 1)
        return poly, dpoly

    @property
    def curvature(self) -> float:
        return self.c + 2.0 * dict(self._terms).get(2, 0.0)


class QTypeAsphere(Asphere):
    """Forbes Q-type asphere on a conic base (the "Q-con" form).

    ``s(r) = conic(r; R, k) + u⁴ Σ_m a_m Q_m(u²)`` with ``u = r / norm_radius``
    and ``Q_m(x) = P_m^(0,4)(2x − 1)``, the Jacobi polynomials that make the
    terms orthogonal over the normalization disk (G. W. Forbes, *Opt.
    Express* 15, 5218 (2007); 19, 9923 (2011)). The departure is ``O(r⁴)``,
    so the vertex curvature is the base curvature.

    The best-fit-sphere form ("Q-bfs") is a different family with a
    different basis; it is not implemented yet.
    """

    _params = ("R", "k", "coefficients", "norm_radius")

    def __init__(self, R: float, k: float = 0.0, coefficients: Sequence[float] = (), norm_radius: float = 1.0) -> None:
        if not float(norm_radius) > 0.0:
            raise ValueError("norm_radius must be positive")
        self.R = _read_radius(R)
        self.k = float(k)
        self.coefficients = tuple(float(a) for a in coefficients)
        self.norm_radius = float(norm_radius)
        self._freeze()

    def departure_and_slope(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = r / self.norm_radius
        x = u * u
        t = 2.0 * x - 1.0
        q = np.zeros_like(r)
        dq_dx = np.zeros_like(r)
        for m, a in enumerate(self.coefficients):
            if a == 0.0:
                continue
            q += a * eval_jacobi(m, 0.0, 4.0, t)
            if m > 0:
                # d/dt P_m^(a,b) = (m + a + b + 1)/2 · P_{m-1}^(a+1, b+1); dt/dx = 2.
                dq_dx += a * (m + 5.0) * eval_jacobi(m - 1, 1.0, 5.0, t)
        departure = x * x * q
        # d/dr [x² q(x)] = (2x q + x² q') · dx/dr, with dx/dr = 2u / norm_radius.
        slope = (2.0 * x * q + x * x * dq_dx) * 2.0 * u / self.norm_radius
        return departure, slope


__all__ = ["Asphere", "EvenAsphere", "QTypeAsphere"]
