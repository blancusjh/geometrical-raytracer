"""The Silva-Lora & Torres aplanatism condition for stigmatic trains.

A :class:`~raytracer.design.stigmatic.StigmaticTrain` is rigorously
stigmatic for its end conjugates whatever its intermediate conjugates are.
Aplanatism — stigmatism that survives a transverse displacement of the
object point, i.e. freedom from coma as well as spherical aberration — is
the additional Abbe sine condition, and for trains of Cartesian surfaces it
has a closed form (A. Silva-Lora and R. Torres, "Aplanatism in stigmatic
optical systems," J. Opt. Soc. Am. A 37, 2020):

- Per surface, the normal at the hit point crosses the axis at a distance
  ``V_k C_k`` from the vertex, given in the shape parameters by the paper's
  Eq. (24) (:func:`normal_axis_crossing`); paraxially it tends to the
  vertex radius ``1/O_k``.
- The ratio of the object- and image-space sines factors over the surfaces
  (their Eq. 14): ``sin u_0 / sin u_N = (n_N/n_0) · g_t · M``, with ``g_t``
  the product of surface magnifications (Eqs. 29-30, ``StigmaticTrain.gt``)
  and ``M`` the product of per-surface terms built from ``V_k C_k``
  (Eq. 25, :func:`aplanatism_map`). ``M = 1`` paraxially for every train
  (their Appendix A); the train is *aplanatic* exactly when ``M = 1`` for
  every ray (their Eq. 19), measured as the RMS of ``M - 1`` (Eq. 31).

:func:`aplanatism_report` traces a real meridional fan, evaluates ``M`` ray
by ray from the traced hit points, and cross-checks the closed form against
the independently traced sine ratio — the identity above is asserted by the
tests at the 1e-9 level, so the formula and the tracer vouch for each other.

:func:`aplanatic_image_surface` locates the surface the paper draws dashed
in its figures: where the images of transversally displaced object points
actually form. For each field height the emergent bundle is reduced to the
point minimizing the summed squared distance to the ray lines; aplanatism
makes those points sharp, and their axial locus is generally *curved* — its
:attr:`~ImageSurface.sagitta` is the flat-field residual an optimizer can
drive to zero (:mod:`raytracer.optimize.aplanat`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...design.stigmatic import StigmaticTrain
from ...propagation.fields import FieldPoint, PupilSampling, trace_from_object, trace_pupil
from ...propagation.sequential import SequentialTracer


def normal_axis_crossing(profile, rho):
    """``V_k C_k``: axis crossing of the surface normal, per the paper's Eq. (24).

    *profile* is a :class:`~raytracer.surfaces.cartesian_oval.CartesianOvalProfile`
    (only its G, O, S parameters enter); *rho* is the vertex-surface distance
    of the hit point. Paraxially (``rho -> 0``) this is the vertex radius
    ``1/O``. The expression assumes ``G != 0`` and ``O != 0`` (the paper's
    Appendix B divides by ``G_k O_k``); the report below uses the geometric
    equivalent :func:`_inverse_vc_geometric`, which has no such blind spots.
    """

    rho = np.asarray(rho, dtype=float)
    G, O, S = profile.G, profile.O, profile.S
    if G == 0.0 or O == 0.0:
        raise ValueError(
            "Eq. (24) is singular for G = 0 or O = 0; use the geometric "
            "normal-axis crossing instead"
        )
    q = np.sqrt(1.0 + (2.0 * S - G * O**2) * rho * rho)
    inverse = (-(2.0 * S / G - O**2) + (2.0 * S / G) * q) / (O * q)
    with np.errstate(divide="ignore"):
        return 1.0 / inverse


def _inverse_vc_geometric(profile, h):
    """``1/V_k C_k`` from the sag and slope at radial height *h*.

    The normal at ``(z(h), h)`` crosses the axis at ``z + h/z'``, so
    ``1/VC = z' / (z z' + h)`` — a form that stays finite for planes
    (``z' = 0``) and needs no shape-parameter conditioning. At ``h = 0``
    the limit is the vertex curvature ``O``.
    """

    h = np.asarray(h, dtype=float)
    z, dz = profile.sag_and_slope(h)
    tiny = 1e-12
    return np.where(h < tiny, profile.curvature, dz / (z * dz + np.where(h < tiny, 1.0, h)))


def aplanatism_map(train: StigmaticTrain, rho_or_heights, *, from_heights: bool = False):
    """Per-ray ``M`` of the paper's Eq. (25) — 1 for every ray iff aplanatic.

    ``rho_or_heights`` is ``(n_rays, n_surfaces)``: per ray, either the
    vertex-surface distances ``rho_k`` (default, the paper's variable) or
    the radial heights ``h_k`` of the hit points (``from_heights=True``).
    Each surface contributes
    ``(n_{k+1}/n_k) * (b_k - 1/VC) / (a_k - 1/VC)`` with
    ``a_k = 1/(d_k - ζ_k)`` and ``b_k = 1/(d_{k+1} - ζ_k)`` (zero for an
    infinite conjugate); a plane surface contributes exactly 1.
    """

    values = np.atleast_2d(np.asarray(rho_or_heights, dtype=float))
    if values.shape[1] != train.n_surfaces:
        raise ValueError(
            f"need one column per surface ({train.n_surfaces}), got {values.shape[1]}"
        )

    result = np.ones(values.shape[0])
    for k, profile in enumerate(train.profiles):
        if profile.is_plane:
            continue
        if from_heights:
            h = values[:, k]
        else:
            # Invert rho -> h on the vertex branch: h = r(rho), and the
            # geometric 1/VC wants the height. rho and h agree to O(rho^3),
            # but exactness matters here, so recover h = sqrt(rho^2 - z^2)
            # iterating the sag once (z depends on h only weakly).
            rho = values[:, k]
            h = rho.copy()
            for _ in range(3):
                z, _ = profile.sag_and_slope(h)
                h = np.sqrt(np.maximum(rho * rho - z * z, 0.0))
        inverse_vc = _inverse_vc_geometric(profile, h)
        za = train.conjugates[k] - train.vertices[k]
        zb = train.conjugates[k + 1] - train.vertices[k]
        a = 0.0 if np.isinf(za) else 1.0 / za
        b = 0.0 if np.isinf(zb) else 1.0 / zb
        result *= (
            (train.indices[k + 1] / train.indices[k])
            * (b - inverse_vc)
            / (a - inverse_vc)
        )
    return result


@dataclass
class AplanatismReport:
    """Ray-by-ray aplanatism diagnosis of a stigmatic train.

    All arrays have one entry per requested ray; ``valid`` marks the rays
    that traced. ``map_values`` is the paper's ``M``; ``sine_ratio`` is the
    traced ``sin u_0 / sin u_N``; ``offense`` is the real-ray offense
    against the sine condition, ``sine_ratio / ((n_N/n_0) g_t) - 1``, which
    equals ``M - 1`` when the closed form and the tracer agree —
    ``formula_residual`` is the per-ray gap between the two routes.
    """

    train: StigmaticTrain
    na_object_sine: float
    sines_object: np.ndarray
    valid: np.ndarray
    sine_ratio: np.ndarray
    map_values: np.ndarray
    offense: np.ndarray
    formula_residual: np.ndarray
    gt: float

    @property
    def map_rms(self) -> float:
        """The paper's criterion, Eq. (31): RMS of ``M - 1`` over traced rays."""

        deviations = self.map_values[self.valid] - 1.0
        return float(np.sqrt(np.mean(deviations**2))) if deviations.size else np.nan

    @property
    def offense_rms(self) -> float:
        values = self.offense[self.valid]
        return float(np.sqrt(np.mean(values**2))) if values.size else np.nan

    def is_aplanatic(self, tol: float = 1e-8) -> bool:
        return bool(self.valid.any()) and self.map_rms < tol

    def summary(self, *, tol: float = 1e-8) -> str:
        lines = [
            f"traced {int(self.valid.sum())}/{self.valid.size} rays to sin u = "
            f"{self.na_object_sine:g}",
            f"(M - 1) RMS: {self.map_rms:.3g}   (real-ray offense RMS: "
            f"{self.offense_rms:.3g})",
            f"gt = {self.gt:+.6f}",
            f"-> {'aplanatic' if self.is_aplanatic(tol=tol) else 'NOT aplanatic'} "
            f"(tol {tol:g})",
        ]
        return "\n".join(lines)


def aplanatism_report(
    train: StigmaticTrain,
    *,
    na_object_sine: float,
    samples: int = 25,
    tracer: SequentialTracer | None = None,
) -> AplanatismReport:
    """Trace a meridional fan and evaluate the aplanatism condition on it.

    ``samples`` rays leave the object point with sines spread over
    ``(0, na_object_sine]``. Rays that fail to trace (outside the usable
    branch of a strong surface, or vignetted) stay in the arrays as NaN with
    ``valid`` false, so the ray count is stable for optimizers.
    """

    tracer = tracer or SequentialTracer(train.to_system())
    n0, nn = train.indices[0], train.indices[-1]
    gt = train.gt
    sines = np.linspace(na_object_sine / samples, na_object_sine, samples)

    valid = np.zeros(samples, dtype=bool)
    sine_ratio = np.full(samples, np.nan)
    map_values = np.full(samples, np.nan)

    vertices = np.asarray(train.vertices)
    heights = np.full((samples, train.n_surfaces), np.nan)
    for i, sine in enumerate(sines):
        slope = sine / np.sqrt(1.0 - sine * sine)
        result = trace_from_object(tracer, (0.0, 0.0), (0.0, slope), keep_path=True)
        if not result.ok:
            continue
        direction = result.direction / np.linalg.norm(result.direction)
        if abs(direction[1]) < 1e-15:
            continue
        valid[i] = True
        sine_ratio[i] = sine / direction[1]
        # path rows: start, one per surface, image point
        heights[i] = np.abs(result.path[1:-1, 1])

    map_values[valid] = aplanatism_map(train, heights[valid], from_heights=True)

    with np.errstate(invalid="ignore"):
        offense = sine_ratio / ((nn / n0) * gt) - 1.0
        formula_residual = sine_ratio / ((nn / n0) * gt * map_values) - 1.0

    return AplanatismReport(
        train=train,
        na_object_sine=na_object_sine,
        sines_object=sines,
        valid=valid,
        sine_ratio=sine_ratio,
        map_values=map_values,
        offense=offense,
        formula_residual=formula_residual,
        gt=gt,
    )


@dataclass
class ImageSurface:
    """Best-focus image points for a sweep of object heights.

    ``points`` are the least-squares convergence points of the emergent
    bundles, one per height; ``blur_rms_um`` is each bundle's RMS transverse
    distance to that point (how sharp the focus actually is there);
    ``sagitta`` is the axial departure of the locus from the first point's
    plane — the quantity a flat-field constraint drives to zero.
    """

    heights: np.ndarray
    points: np.ndarray
    blur_rms_um: np.ndarray
    valid: np.ndarray

    @property
    def sagitta(self) -> np.ndarray:
        return self.points[:, 2] - self.points[0, 2]

    @property
    def max_sagitta(self) -> float:
        values = self.sagitta[self.valid]
        return float(np.max(np.abs(values))) if values.size else np.nan


def _closest_point_to_rays(origins: np.ndarray, directions: np.ndarray):
    """Point minimizing the summed squared distance to the ray lines."""

    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    projectors = np.eye(3)[None, :, :] - directions[:, :, None] * directions[:, None, :]
    matrix = projectors.sum(axis=0)
    rhs = np.einsum("nij,nj->i", projectors, origins)
    point = np.linalg.solve(matrix, rhs)
    residuals = np.einsum("nij,nj->ni", projectors, origins - point[None, :])
    blur_rms = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    return point, blur_rms


def aplanatic_image_surface(
    train: StigmaticTrain,
    heights,
    *,
    na_object_sine: float,
    sampling: PupilSampling | None = None,
    tracer: SequentialTracer | None = None,
) -> ImageSurface:
    """The surface where the images of displaced object points form.

    For each object height a pupil-filling cone (aimed at the front vertex,
    since a bare train has no stop) is traced and the emergent rays reduced
    to their least-squares convergence point. This is the dashed locus of
    the paper's figures; include ``0.0`` as the first height so
    :attr:`ImageSurface.sagitta` is measured against the axial image.
    """

    tracer = tracer or SequentialTracer(train.to_system())
    sampling = sampling or PupilSampling(kind="rings", radial=4, azimuth=12)
    heights = np.asarray(heights, dtype=float)

    points = np.full((heights.size, 3), np.nan)
    blur = np.full(heights.size, np.nan)
    valid = np.zeros(heights.size, dtype=bool)
    for i, h in enumerate(heights):
        chief_slope = (0.0 - h) / (0.0 - train.conjugates[0])
        pupil = trace_pupil(
            tracer,
            FieldPoint(y=float(h)),
            na_object_sine=na_object_sine,
            sampling=sampling,
            chief_slope=chief_slope,
        )
        if pupil.valid.sum() < 3:
            continue
        points[i], blur_mm = _closest_point_to_rays(
            pupil.image_points, pupil.directions
        )
        blur[i] = blur_mm * 1e3
        valid[i] = True

    return ImageSurface(heights=heights, points=points, blur_rms_um=blur, valid=valid)


__all__ = [
    "AplanatismReport",
    "ImageSurface",
    "aplanatic_image_surface",
    "aplanatism_map",
    "aplanatism_report",
    "normal_axis_crossing",
]
