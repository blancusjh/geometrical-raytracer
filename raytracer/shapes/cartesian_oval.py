"""Rotationally-symmetric Cartesian oval: the GOTS closed-form parametrization.

Descartes' construction gives the surface satisfying
``n0 * dist(object, surface) + ni * dist(surface, image) = const`` (Fermat's
principle applied to two conjugate points), i.e. a perfectly stigmatic
refracting surface between an object at ``z0`` (index ``n0``) and an image at
``zi`` (index ``ni``). ``raytracer/shapes/fermat_oval.py`` describes this
surface implicitly via ``F(z, r) = 0``. This module instead provides the equivalent
**closed-form parametric** surface derived in the author's thesis work
(``vecdiff/CartesianSurfaces.py`` / ``vecdiff/GOTS_parameters.py`` at
github.com/blancusjh/vecdiff, transcribed verbatim from the source), using the
GOTS parameters (G, O, T, S).

Important convention: the thesis surface is parametrized by an auxiliary
variable ``rho`` — the distance from the vertex to the surface point along a
"radial" construction — *not* the physical radial height. The physical height
is recovered as ``r(rho) = sqrt(rho**2 - z(rho)**2)`` (verified numerically:
plugging ``(z(rho), r(rho))`` into the Fermat implicit form used by
``ovoid2d.fermat_ovoid_F`` gives a residual at machine precision for every
``rho``). Two APIs follow from this:

- :func:`cartesian_oval_parametric_curve` — direct ``rho -> (z, r)``, used for
  2-D drawing (a parametric curve needs no inversion).
- :func:`cartesian_oval_sag_and_slope` — the usual optics ``sag(h) -> z``
  convention (radial *height* to axial *sag*), obtained by inverting
  ``r(rho) = h`` with a vectorized Newton solve. This is what plugs into the
  3-D sequential propagation, which expects ``profile.sag_and_slope(h)``
  exactly like :class:`raytracer.shapes.profile.AsphereProfile` (duck-typed,
  see ``raytracer/propagation/sequential.py``).

``r(rho)`` increases from 0, peaks, then turns back toward the axis (closing
the oval); only the monotonic branch near the vertex is usable as a lens
surface. :func:`max_usable_height` returns that peak height.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike


def gots_params(n0: float, z0: float, ni: float, zi: float) -> tuple[float, float, float, float]:
    """Return the (G, O, T, S) parameters for the object/image conjugate pair.

    ``n0``/``z0`` describe the object-side medium and axial position, ``ni``/
    ``zi`` the image side. Signs follow the usual optics convention: ``z0`` is
    negative for an object to the left of the vertex, ``zi`` positive for a
    real image to the right.
    """

    G = (ni**2 * zi - n0**2 * z0) ** 2 / (
        ni * n0 * (ni * zi - n0 * z0) * (ni * z0 - n0 * zi)
    )
    O = (ni * z0 - n0 * zi) / (zi * z0 * (ni - n0))
    T = (ni - n0) * (ni + n0) ** 2 / (4 * ni * n0 * zi * z0 * (ni * zi - n0 * z0))
    S = (ni + n0) * (ni**2 * zi - n0**2 * z0) / (2 * ni * n0 * zi * z0 * (ni * zi - n0 * z0))
    return G, O, T, S


def _z_and_dzdrho(rho: np.ndarray, G: float, O: float, T: float, S: float) -> tuple[np.ndarray, np.ndarray]:
    """GOTS closed form ``(z(rho), dz/drho)``, transcribed verbatim from the thesis source."""

    e = 2.0 * S - O**2 * G
    q = np.sqrt(np.maximum(1.0 + e * rho * rho, 1e-30))

    rho2 = rho * rho
    n = (O + T * rho2) * rho2
    d = 1.0 + S * rho2 + q

    dn = 2.0 * O * rho + 4.0 * T * rho**3
    dd = 2.0 * S * rho + e * rho / q

    z = n / d
    dz = (dn * d - n * dd) / (d * d)
    return z, dz


def cartesian_oval_parametric_curve(
    rho: ArrayLike, G: float, O: float, T: float, S: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(z, r)`` points on the oval for parameter *rho* (2-D drawing)."""

    rho = np.asarray(rho, dtype=float)
    z, _ = _z_and_dzdrho(rho, G, O, T, S)
    r = np.sqrt(np.maximum(rho * rho - z * z, 0.0))
    return z, r


def max_usable_height(
    G: float, O: float, T: float, S: float, *, rho_max_search: float = 1e4, samples: int = 20000
) -> float:
    """Height at which ``r(rho)`` peaks before the oval curves back to the axis.

    Only ``h`` below this value has a unique, physically meaningful vertex-side
    surface point — this is the natural bound on a usable clear aperture.
    """

    rho = np.linspace(1e-9, rho_max_search, samples)
    _, r = cartesian_oval_parametric_curve(rho, G, O, T, S)
    turn = np.where(np.diff(r) < 0.0)[0]
    if turn.size == 0:
        return float(np.max(r))
    lo, hi = rho[max(turn[0] - 1, 0)], rho[turn[0] + 1]

    def dr_drho(rho_val: float) -> float:
        eps = 1e-6 * max(abs(rho_val), 1.0)
        _, r0 = cartesian_oval_parametric_curve(rho_val - eps, G, O, T, S)
        _, r1 = cartesian_oval_parametric_curve(rho_val + eps, G, O, T, S)
        return float((r1 - r0) / (2.0 * eps))

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if dr_drho(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    _, r_peak = cartesian_oval_parametric_curve(np.array([0.5 * (lo + hi)]), G, O, T, S)
    return float(r_peak[0])


def cartesian_oval_sag_and_slope(
    h: ArrayLike,
    G: float,
    O: float,
    T: float,
    S: float,
    *,
    max_newton: int = 30,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``(z, dz/dh)`` at radial height(s) *h* (standard sag convention).

    Inverts ``r(rho) = h`` with a vectorized Newton solve (the oval is only
    given in closed form as ``rho -> (z, r)``, not directly as ``h -> z``),
    then converts the parametric derivative via the chain rule:
    ``dz/dh = (dz/drho) / (dr/drho)``, with
    ``dr/drho = (rho - z * dz/drho) / r``.

    Valid for ``0 <= h < max_usable_height(G, O, T, S)``.
    """

    h = np.atleast_1d(np.asarray(h, dtype=float))
    rho = h.copy()  # r(rho) <= rho everywhere on the vertex branch: a safe start
    tiny = 1e-12

    for _ in range(max_newton):
        z, dzdrho = _z_and_dzdrho(rho, G, O, T, S)
        r = np.sqrt(np.maximum(rho * rho - z * z, 0.0))
        safe_r = np.where(r < tiny, tiny, r)
        drdrho = (rho - z * dzdrho) / safe_r
        residual = r - h
        safe_drdrho = np.where(np.abs(drdrho) < tiny, tiny, drdrho)
        step = residual / safe_drdrho
        rho = rho - step
        if np.max(np.abs(step)) < tol:
            break

    z, dzdrho = _z_and_dzdrho(rho, G, O, T, S)
    r = np.sqrt(np.maximum(rho * rho - z * z, 0.0))
    safe_r = np.where(r < tiny, tiny, r)
    drdrho = (rho - z * dzdrho) / safe_r
    safe_drdrho = np.where(np.abs(drdrho) < tiny, tiny, drdrho)
    dzdh = np.where(h < tiny, 0.0, dzdrho / safe_drdrho)
    z_at_h = np.where(h < tiny, 0.0, z)
    return z_at_h, dzdh


@dataclass(frozen=True)
class CartesianOvalProfile:
    """3-D rotationally-symmetric Cartesian oval, stigmatic between (n0, z0) and (ni, zi).

    Duck-typed to the same contract as :class:`raytracer.shapes.profile.AsphereProfile`
    (``.sag(h)``, ``.sag_and_slope(h)``, ``.curvature``, ``.radius``, ``.is_plane``),
    so it plugs directly into ``SurfaceRow.profile`` and the 3-D
    ``SequentialTracer`` without any change to the tracer itself — the Newton
    intersection in ``raytracer/propagation/sequential.py`` only ever calls
    ``profile.sag_and_slope(h)``.

    Unlike the 2-D ``CartesianOvalSurface`` (implicit, solved by bisection on
    Fermat's equation), this evaluates the closed-form GOTS parametrization —
    exact for every radius up to :attr:`max_usable_height`.
    """

    n0: float
    z0: float
    ni: float
    zi: float
    G: float = field(init=False, repr=False)
    O: float = field(init=False, repr=False)
    T: float = field(init=False, repr=False)
    S: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        g, o, t, s = gots_params(self.n0, self.z0, self.ni, self.zi)
        object.__setattr__(self, "G", g)
        object.__setattr__(self, "O", o)
        object.__setattr__(self, "T", t)
        object.__setattr__(self, "S", s)

    @property
    def curvature(self) -> float:
        """Vertex curvature: ``z(h) ~ (O/2) h**2`` near the axis, so ``c = O``."""

        return self.O

    @property
    def radius(self) -> float:
        return 0.0 if self.curvature == 0.0 else 1.0 / self.curvature

    @property
    def is_plane(self) -> bool:
        return False

    @property
    def max_usable_height(self) -> float:
        """Largest radial height with a unique vertex-branch surface point."""

        return max_usable_height(self.G, self.O, self.T, self.S)

    def sag(self, h: ArrayLike) -> np.ndarray:
        return self.sag_and_slope(h)[0]

    def sag_and_slope(self, h: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        return cartesian_oval_sag_and_slope(h, self.G, self.O, self.T, self.S)


__all__ = [
    "gots_params",
    "cartesian_oval_parametric_curve",
    "cartesian_oval_sag_and_slope",
    "max_usable_height",
    "CartesianOvalProfile",
]
