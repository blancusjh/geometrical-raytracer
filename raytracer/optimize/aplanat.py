"""Drive a stigmatic train to aplanatism — the paper's optimization, made local.

Silva-Lora & Torres (JOSA A, 2020) obtain their aplanatic SOL examples by
searching the intermediate conjugates ``d_1..d_{N-1}`` at fixed indices,
vertices, and end conjugates, accepting a system when the RMS of ``M - 1``
(their Eq. 31) falls below tolerance — in the paper by uniform random
sampling of the ``d_k``. Here the same objective is handed to
``scipy.optimize.least_squares``: each residual is one ray's ``M - 1``,
evaluated by :func:`raytracer.analysis.aberrations.aplanatism.aplanatism_report`
on real traced rays, so the Jacobian the solver builds is the physically
meaningful sensitivity of each ray's sine-condition offense to each free
conjugate.

The optional flat-field term appends the sagitta of the aplanatic image
surface (the paper's dashed locus) at chosen field heights, weighted by
``flat_field_weight`` — driving the system toward aplanatism *on a plane*
rather than on the naturally curved surface. Intermediate conjugates that
are ``±inf`` (collimated spaces, including those flanking a flat physical
surface) are structural and never varied.

Failed rays and invalid parameter sets (a conjugate crossing a vertex, a
bundle that no longer traces) contribute a large finite penalty instead of
NaN, so the solver backs away from singular walls on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from ..analysis.aberrations.aplanatism import (
    AplanatismReport,
    ImageSurface,
    aplanatic_image_surface,
    aplanatism_report,
)
from ..design.stigmatic import StigmaticTrain

_PENALTY = 1e3


def _free_indices(train: StigmaticTrain) -> list[int]:
    """Positions (into the intermediate-conjugate list) that are finite."""

    intermediates = train.conjugates[1:-1]
    return [i for i, d in enumerate(intermediates) if not np.isinf(d)]


def _with_free_values(train: StigmaticTrain, free: list[int], x: np.ndarray) -> StigmaticTrain:
    intermediates = list(train.conjugates[1:-1])
    for position, value in zip(free, x):
        intermediates[position] = float(value)
    return train.with_conjugates(intermediates)


def aplanatism_residuals(
    train: StigmaticTrain,
    *,
    na_object_sine: float,
    samples: int = 15,
    field_heights=None,
    flat_field_weight: float = 0.0,
) -> np.ndarray:
    """Residual vector: per-ray ``M - 1``, plus weighted image-surface sagitta.

    Fixed length ``samples + len(field_heights)`` regardless of how many
    rays survive, with penalties in the failed slots — the shape stability
    ``least_squares`` requires.
    """

    n_fields = 0 if field_heights is None else len(field_heights)
    try:
        report = aplanatism_report(
            train, na_object_sine=na_object_sine, samples=samples
        )
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return np.full(samples + n_fields, _PENALTY)

    residuals = np.where(report.valid, report.map_values - 1.0, _PENALTY)

    if n_fields:
        try:
            surface = aplanatic_image_surface(
                train, field_heights, na_object_sine=na_object_sine
            )
            sagitta = np.where(surface.valid, surface.sagitta, _PENALTY)
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            sagitta = np.full(n_fields, _PENALTY)
        residuals = np.concatenate([residuals, flat_field_weight * sagitta])

    return residuals


@dataclass
class AplanatFit:
    """Outcome of :func:`optimize_aplanat`.

    ``before``/``after`` are full ray-traced aplanatism reports for the
    initial and optimized trains; ``image_surface`` is the optimized
    train's best-focus locus when field heights were given (None otherwise).
    """

    train: StigmaticTrain
    result: OptimizeResult
    before: AplanatismReport
    after: AplanatismReport
    image_surface: ImageSurface | None = None

    def summary(self) -> str:
        lines = [
            f"free conjugates: {len(self.result.x)}  ->  "
            f"d = {tuple(round(float(v), 4) for v in self.result.x)}",
            f"(M - 1) RMS: {self.before.map_rms:.3g}  ->  {self.after.map_rms:.3g}",
            f"gt: {self.before.gt:+.6f}  ->  {self.after.gt:+.6f}",
        ]
        if self.image_surface is not None:
            lines.append(
                f"image-surface max sagitta: {self.image_surface.max_sagitta:.4g} mm"
            )
        return "\n".join(lines)


def optimize_aplanat(
    train: StigmaticTrain,
    *,
    na_object_sine: float,
    samples: int = 15,
    field_heights=None,
    flat_field_weight: float = 0.0,
    bounds: tuple | None = None,
    **least_squares_kwargs,
) -> AplanatFit:
    """Vary the finite intermediate conjugates until the train is aplanatic.

    ``train`` supplies the fixed structure (indices, vertices, end
    conjugates) and the starting intermediates. ``field_heights`` +
    ``flat_field_weight`` add the flat-field term; weight 0 reproduces the
    paper's plain aplanatism search. ``bounds`` and any extra keyword
    arguments go to ``scipy.optimize.least_squares`` untouched.

    Returns an :class:`AplanatFit`; nothing raises on a poor fit — read
    ``fit.after.map_rms`` (and ``fit.image_surface.max_sagitta``) and judge
    against the tolerance your application needs, the way the paper accepts
    systems only below 1e-10.
    """

    free = _free_indices(train)
    if not free:
        raise ValueError(
            "train has no finite intermediate conjugates to vary "
            "(collimated-space conjugates are structural)"
        )
    x0 = np.array([train.conjugates[1:-1][i] for i in free], dtype=float)

    def objective(x: np.ndarray) -> np.ndarray:
        try:
            candidate = _with_free_values(train, free, x)
        except ValueError:
            return np.full(
                samples + (0 if field_heights is None else len(field_heights)),
                _PENALTY,
            )
        return aplanatism_residuals(
            candidate,
            na_object_sine=na_object_sine,
            samples=samples,
            field_heights=field_heights,
            flat_field_weight=flat_field_weight,
        )

    kwargs = {"x_scale": np.maximum(np.abs(x0), 1.0), **least_squares_kwargs}
    if bounds is not None:
        kwargs["bounds"] = bounds
    result = least_squares(objective, x0, **kwargs)

    optimized = _with_free_values(train, free, result.x)
    fit = AplanatFit(
        train=optimized,
        result=result,
        before=aplanatism_report(train, na_object_sine=na_object_sine, samples=samples),
        after=aplanatism_report(
            optimized, na_object_sine=na_object_sine, samples=samples
        ),
    )
    if field_heights is not None:
        fit.image_surface = aplanatic_image_surface(
            optimized, field_heights, na_object_sine=na_object_sine
        )
    return fit


__all__ = ["AplanatFit", "aplanatism_residuals", "optimize_aplanat"]
