"""The Cartesian oval: a perfectly stigmatic refracting surface.

Descartes' construction gives the surface satisfying
``n0 * dist(object, surface) + ni * dist(surface, image) = const`` — Fermat's
principle applied to two conjugate points — i.e. a surface imaging an object
at ``z0`` (index ``n0``) exactly onto an image at ``zi`` (index ``ni``), with
no spherical aberration at any aperture.

This one surface has *both* descriptions the intersection solver understands,
and both live here:

- **Implicit**: :func:`cartesian_oval_implicit` is the equation above written
  as ``f_Sigma(z, r) = 0``, with :func:`cartesian_oval_implicit_gradient` its
  gradient. This is what :class:`CartesianOvalSurface` is hit-tested by.
- **Parametric**: the closed-form ``rho -> (z, r)`` derived in the author's
  thesis work (``vecdiff/CartesianSurfaces.py`` / ``vecdiff/GOTS_parameters.py``
  at github.com/blancusjh/vecdiff, transcribed verbatim), using the GOTS
  parameters (G, O, T, S) — see :func:`cartesian_oval_parametric_curve`. Used
  for drawing (a parametric curve needs no inversion) and, via
  :meth:`CartesianOvalSurface.parametric`, as an independent check on the
  implicit solve.

The two agree to machine precision: substituting the parametric ``(z, r)``
into the implicit form leaves a residual at the 1e-16 level, which the tests
assert directly.

Note the parametrization's convention: ``rho`` is an auxiliary construction
length from the vertex, *not* the physical radial height. The height is
recovered as ``r(rho) = sqrt(rho**2 - z(rho)**2)``. A third API follows for
the 3-D sequential propagation, which wants the usual optics ``sag(h) -> z``
convention: :func:`cartesian_oval_sag_and_slope` inverts ``r(rho) = h`` with a
vectorized Newton solve, so :class:`CartesianOvalProfile` is duck-typed to the
same contract as :class:`raytracer.surfaces.profile.AsphereProfile`.

``r(rho)`` increases from 0, peaks, then turns back toward the axis (closing
the oval); only the monotonic branch near the vertex is usable as a lens
surface. :func:`max_usable_height` returns that peak height.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from ..math.transforms import RigidTransform
from .surface import Surface


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


@dataclass(frozen=True)
class CartesianOvalProfile:
    """3-D rotationally-symmetric Cartesian oval, stigmatic between (n0, z0) and (ni, zi).

    Duck-typed to the same contract as :class:`raytracer.surfaces.profile.AsphereProfile`
    (``.sag(h)``, ``.sag_and_slope(h)``, ``.curvature``, ``.radius``, ``.is_plane``),
    so it plugs directly into ``SurfaceRow.profile`` and the 3-D
    ``SequentialTracer`` without any change to the tracer itself — the Newton
    intersection in ``raytracer/propagation/sequential.py`` only ever calls
    ``profile.sag_and_slope(h)``.

    This is the *parametric* route to the same surface
    :class:`CartesianOvalSurface` reaches implicitly: it evaluates the
    closed-form GOTS parametrization, exact for every radius up to
    :attr:`max_usable_height`.
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


@dataclass
class CartesianOvalSurface(Surface):
    """The meridional Cartesian oval as a traceable 2-D surface.

    Local axes: ``x`` is the optical axis (the ``z`` of the formulas above),
    ``y`` the signed radial coordinate (their ``r``).

    Offers both descriptions. Hit-testing uses the implicit one — the
    defining equation is exact and its bracket-and-bisect solve cannot be
    thrown off by a bad initial guess — while :meth:`parametric` exposes the
    closed-form GOTS curve, used for drawing and as an independent check that
    the two descriptions agree.
    """

    z0: float
    zi: float
    n_exterior: float = 1.0  # n0
    n_interior: float = 1.5  # ni
    origin: np.ndarray = field(default_factory=lambda: np.zeros(2))
    angle: float = 0.0
    surface_id: str = "cartesian_oval"
    semidiameter: float | None = None

    def __post_init__(self) -> None:
        Surface.__init__(
            self,
            surface_id=self.surface_id,
            n_exterior=self.n_exterior,
            n_interior=self.n_interior,
        )
        self.origin = np.asarray(self.origin, dtype=float)
        self.frame = RigidTransform.from_angle_2d(origin=self.origin, angle=self.angle)
        self._gots = gots_params(self.n_exterior, self.z0, self.n_interior, self.zi)

    # -- implicit description (used for hit testing) -----------------------

    def implicit(self, point: np.ndarray) -> float:
        return float(
            cartesian_oval_implicit(
                point[0], point[1], self.z0, self.zi, self.n_exterior, self.n_interior
            )
        )

    def implicit_gradient(self, point: np.ndarray) -> np.ndarray:
        return cartesian_oval_implicit_gradient(
            point[0], point[1], self.z0, self.zi, self.n_exterior, self.n_interior
        )

    def within_aperture(self, point: np.ndarray) -> bool:
        return self.semidiameter is None or abs(point[1]) <= self.semidiameter

    # -- parametric description (drawing, and cross-checking the implicit) --

    def parametric(self, t: float) -> np.ndarray:
        """``P(t) = (z(|t|), sign(t) * r(|t|))`` over the whole meridian.

        Signing the parameter covers both branches of the oval with one
        continuous parametrization, so a ray arriving below the axis needs no
        special case.
        """

        z, r = cartesian_oval_parametric_curve(np.array([abs(t)]), *self._gots)
        return np.array([float(z[0]), float(np.sign(t) * r[0])], dtype=float)

    def parametric_seed(self, origin: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
        """Seed the parametric solve from where the ray crosses the vertex plane."""

        lam = (0.0 - origin[0]) / direction[0] if abs(direction[0]) > 1e-14 else 1.0
        height = origin[1] + lam * direction[1]
        return float(lam), float(height if abs(height) > 1e-9 else 1e-6)

    # -- drawing ------------------------------------------------------------

    def _rho_max(self) -> float:
        """The closure point of the oval: the first ``rho > 0`` past the vertex
        where ``r(rho) = 0``, i.e. ``rho**2 - z(rho)**2 = 0``.

        Near the vertex ``z(rho)`` is higher-order in ``rho`` (paraxial sag),
        so ``g(rho) = rho**2 - z(rho)**2`` starts positive; the oval closes
        where ``g`` first turns negative (``z`` catches up with ``rho``).
        """

        def g(rho: float) -> float:
            z, _ = cartesian_oval_parametric_curve(np.array([rho]), *self._gots)
            return rho * rho - float(z[0]) ** 2

        lo, hi = 1e-6, 1.0
        glo = g(lo)  # > 0 just past the vertex
        for _ in range(64):
            if g(hi) <= 0.0:
                break
            hi *= 2.0
        else:
            # never closes within range: fall back to a sane radius to "see" the curve
            return hi

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            gm = g(mid)
            if abs(gm) < 1e-10 or (hi - lo) < 1e-9:
                return mid
            if np.sign(gm) == np.sign(glo):
                lo, glo = mid, gm
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def as_points(self, samples: int = 600) -> np.ndarray:
        """The closed meridional outline, drawn from the parametric form."""

        rho_max = self._rho_max()
        if not np.isfinite(rho_max) or rho_max <= 1e-9:
            return np.empty((0, 2), dtype=float)

        rho = np.linspace(0.0, rho_max, samples)
        z, r = cartesian_oval_parametric_curve(rho, *self._gots)
        upper = np.column_stack((z, r))
        lower = np.column_stack((z[::-1], -r[::-1]))
        pts_local = np.vstack((upper, lower))
        return np.array([self.frame.to_world(p) for p in pts_local], dtype=float)

    def polyline(self, samples: int = 512) -> np.ndarray:
        return self.as_points(samples=samples)


__all__ = [
    "gots_params",
    "cartesian_oval_parametric_curve",
    "cartesian_oval_sag_and_slope",
    "cartesian_oval_implicit",
    "cartesian_oval_implicit_gradient",
    "max_usable_height",
    "CartesianOvalProfile",
    "CartesianOvalSurface",
]
