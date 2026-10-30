"""Cartesian surfaces: the perfectly stigmatic refracting surfaces of Descartes.

A Cartesian surface (a "Cartesian oval" of revolution) is the surface
satisfying

    n0 · |P − A| + ni · |P − B| = const

for an axial object point ``A`` at ``z0`` in a medium of index ``n0`` and
its image ``B`` at ``zi`` in index ``ni`` — Fermat's principle made into a
shape. It images ``A`` onto ``B`` with no aberration at any aperture.
"Cartesian" names Descartes, not a coordinate system.

Both conjugates are measured **from the vertex**, in the surface's own
frame; where the vertex sits in the world is the pose of the element that
carries the surface. Either conjugate may be ``±inf`` (collimated light),
in which case the surface degenerates to the corresponding conic.

Three descriptions of the same surface live here and agree to machine
precision (the tests assert it):

- the closed-form parametrization ``ρ → (z, r)`` with the GOTS parameters
  (G, O, T, S) of Silva-Lora & Torres (2020), transcribed from the
  author's thesis code (github.com/blancusjh/vecdiff);
- the sag ``s(r)``, by inverting ``r(ρ) = h`` with Newton — what the
  tracer uses;
- the implicit Descartes condition, :func:`cartesian_oval_implicit`.

Only the branch from the vertex to :attr:`CartesianSurface.max_radius`,
where ``r(ρ)`` peaks and the oval turns back toward the axis, is a usable
lens surface.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._value import SurfaceValue
from .base import SurfaceOfRevolution


def gots_params(n0: float, z0: float, ni: float, zi: float) -> tuple[float, float, float, float]:
    """Return the (G, O, T, S) parameters for the object/image conjugate pair.

    ``n0``/``z0`` describe the object-side medium and axial position, ``ni``/
    ``zi`` the image side. Signs follow the usual optics convention: ``z0`` is
    negative for an object to the left of the vertex, ``zi`` positive for a
    real image to the right.

    Either conjugate may be ``±inf``, meaning collimated light on that side;
    the parameters are then the exact limits of Eqs. (3)-(6) of Silva-Lora &
    Torres (2020) and the surface degenerates to the corresponding Cartesian
    conic (e.g. the collimating hyperbola for a real object into a denser
    medium). With *both* conjugates infinite the surface is the plane
    ``z = 0``, and all four parameters are zero.
    """

    z0_inf, zi_inf = np.isinf(z0), np.isinf(zi)
    if z0_inf and zi_inf:
        return 0.0, 0.0, 0.0, 0.0
    if zi_inf:
        G = -(ni**2) / n0**2
        O = -n0 / (z0 * (ni - n0))
        return G, O, 0.0, 0.0
    if z0_inf:
        G = -(n0**2) / ni**2
        O = ni / (zi * (ni - n0))
        return G, O, 0.0, 0.0

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


def cartesian_oval_implicit(z, r, z0: float, zi: float, n0: float, ni: float):
    """``f_Sigma(z, r)``: zero exactly on the oval, from Descartes' condition.

    ``f_Sigma = n0*sqrt((z-z0)^2 + r^2) + ni*sqrt((z-zi)^2 + r^2)
                - (n0*|z0| + ni*|zi|)``

    i.e. the optical path from the object point to the surface and on to the
    image point, minus the constant that path must equal. That constancy *is*
    stigmatism: every ray from ``z0`` arrives at ``zi`` in phase.

    ``r`` may be signed (it enters squared), so the meridional curve below the
    axis is covered by the same expression.
    """

    to_object = np.sqrt((z - z0) ** 2 + r**2)
    to_image = np.sqrt((z - zi) ** 2 + r**2)
    return n0 * to_object + ni * to_image - (n0 * abs(z0) + ni * abs(zi))


def cartesian_oval_implicit_gradient(z, r, z0: float, zi: float, n0: float, ni: float) -> np.ndarray:
    """``grad f_Sigma = (df/dz, df/dr)`` — the surface normal direction.

    Signed in ``r``, so it needs no branch fix-up below the axis.
    """

    eps = 1e-12
    to_object = np.sqrt((z - z0) ** 2 + r**2) + eps
    to_image = np.sqrt((z - zi) ** 2 + r**2) + eps
    return np.array(
        [
            n0 * (z - z0) / to_object + ni * (z - zi) / to_image,
            n0 * r / to_object + ni * r / to_image,
        ],
        dtype=float,
    )


class CartesianSurface(SurfaceValue, SurfaceOfRevolution):
    """The stigmatic surface imaging ``z0`` (index ``n0``) onto ``zi`` (index ``ni``).

    ``z0`` and ``zi`` are measured from the vertex; ``z0 < 0`` is a real
    object to the left, ``zi > 0`` a real image to the right.

    The indices are parameters of the *shape*. The surface is stigmatic in
    use only where the media around it actually have these indices — at
    the design wavelength.
    """

    _params = ("n0", "z0", "ni", "zi")

    def __init__(self, n0: float, z0: float, ni: float, zi: float) -> None:
        self.n0, self.z0, self.ni, self.zi = float(n0), float(z0), float(ni), float(zi)
        if self.n0 == self.ni:
            raise ValueError("a Cartesian surface needs an index step: n0 == ni")
        for z in (self.z0, self.zi):
            if z == 0.0:
                raise ValueError("a conjugate at the vertex makes the GOTS parameters singular")
        self.gots = gots_params(self.n0, self.z0, self.ni, self.zi)
        self._freeze()

    @property
    def G(self) -> float:
        return self.gots[0]

    @property
    def O(self) -> float:  # noqa: E743 - the parameter's name in the paper
        return self.gots[1]

    @property
    def T(self) -> float:
        return self.gots[2]

    @property
    def S(self) -> float:
        return self.gots[3]

    @property
    def curvature(self) -> float:
        """Vertex curvature: ``z(h) ≈ (O/2) h²`` near the axis, so ``c = O``."""

        return self.gots[1]

    @property
    def is_plane(self) -> bool:
        return all(v == 0.0 for v in self.gots)

    @property
    def max_radius(self) -> float:
        """Largest radial height with a unique vertex-branch surface point."""

        if "_max_radius" not in self.__dict__:
            object.__setattr__(self, "_max_radius", max_usable_height(*self.gots))
        return self.__dict__["_max_radius"]

    def sag_and_slope(self, r: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(r, dtype=float)
        s, ds = cartesian_oval_sag_and_slope(r, *self.gots)
        return s.reshape(r.shape), ds.reshape(r.shape)

    def meridian(self, rho: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Points ``(z, r)`` of the meridional curve from the closed-form parametrization."""

        return cartesian_oval_parametric_curve(rho, *self.gots)


__all__ = [
    "CartesianSurface",
    "gots_params",
    "max_usable_height",
    "cartesian_oval_sag_and_slope",
    "cartesian_oval_parametric_curve",
    "cartesian_oval_implicit",
    "cartesian_oval_implicit_gradient",
]
