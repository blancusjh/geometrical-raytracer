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
from .constraints import PENALTY, Aplanatism, Constraint, EvaluationContext, FlatImageSurface

_PENALTY = PENALTY


def _free_indices(train: StigmaticTrain) -> list[int]:
    """Positions (into the intermediate-conjugate list) that are finite."""

    intermediates = train.conjugates[1:-1]
    return [i for i, d in enumerate(intermediates) if not np.isinf(d)]


def _with_free_values(train: StigmaticTrain, free: list[int], x: np.ndarray) -> StigmaticTrain:
    intermediates = list(train.conjugates[1:-1])
    for position, value in zip(free, x):
        intermediates[position] = float(value)
    return train.with_conjugates(intermediates)


def _constraints_for(field_heights, flat_field_weight: float) -> list[Constraint]:
    constraints: list[Constraint] = [Aplanatism()]
    if field_heights is not None:
        constraints.append(
            FlatImageSurface(heights=tuple(field_heights), weight=flat_field_weight)
        )
    return constraints


def constraint_residuals(
    train: StigmaticTrain,
    constraints,
    *,
    na_object_sine: float,
    samples: int = 15,
) -> np.ndarray:
    """Concatenated residual blocks of *constraints* on one candidate train.

    Fixed length regardless of how many rays survive (failed slots carry
    :data:`~raytracer.optimize.constraints.PENALTY`) — the shape stability
    ``least_squares`` requires. One shared :class:`EvaluationContext` keeps
    every constraint reading the same cached traces.
    """

    context = EvaluationContext(train, na_object_sine=na_object_sine, samples=samples)
    total = sum(c.size(context) for c in constraints)
    try:
        return np.concatenate([c.residuals(context) for c in constraints])
    except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
        return np.full(total, _PENALTY)


def aplanatism_residuals(
    train: StigmaticTrain,
    *,
    na_object_sine: float,
    samples: int = 15,
    field_heights=None,
    flat_field_weight: float = 0.0,
) -> np.ndarray:
    """Residual vector: per-ray ``M - 1``, plus weighted image-surface sagitta.

    The original fixed pairing, now expressed through the constraint
    system: ``[Aplanatism(), FlatImageSurface(field_heights, weight)]``.
    """

    return constraint_residuals(
        train,
        _constraints_for(field_heights, flat_field_weight),
        na_object_sine=na_object_sine,
        samples=samples,
    )


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


def optimize_train(
    train: StigmaticTrain,
    constraints,
    *,
    na_object_sine: float,
    samples: int = 15,
    bounds: tuple | None = None,
    **least_squares_kwargs,
) -> AplanatFit:
    """Vary the finite intermediate conjugates to satisfy *constraints*.

    ``constraints`` is any list of
    :class:`~raytracer.optimize.constraints.Constraint` objects — plug in
    :class:`~raytracer.optimize.constraints.Aplanatism`,
    :class:`~raytracer.optimize.constraints.Distortion`,
    :class:`~raytracer.optimize.constraints.FlatImageSurface`,
    :class:`~raytracer.optimize.constraints.TargetMagnification`, or your
    own subclass, in any combination; their residual blocks form one
    least-squares objective. ``train`` supplies the fixed structure and the
    starting intermediates; ``bounds`` and extra keyword arguments go to
    ``scipy.optimize.least_squares`` untouched.

    Returns an :class:`AplanatFit`; nothing raises on a poor fit — read
    ``fit.after.map_rms`` (and ``fit.image_surface``) and judge against the
    tolerance your application needs. The fit's ``image_surface`` covers
    the union of every field height any constraint asked about.
    """

    free = _free_indices(train)
    if not free:
        raise ValueError(
            "train has no finite intermediate conjugates to vary "
            "(collimated-space conjugates are structural)"
        )
    constraints = list(constraints)
    if not constraints:
        raise ValueError("pass at least one constraint")
    x0 = np.array([train.conjugates[1:-1][i] for i in free], dtype=float)
    probe = EvaluationContext(train, na_object_sine=na_object_sine, samples=samples)
    total = sum(c.size(probe) for c in constraints)

    def objective(x: np.ndarray) -> np.ndarray:
        try:
            candidate = _with_free_values(train, free, x)
        except ValueError:
            return np.full(total, _PENALTY)
        return constraint_residuals(
            candidate, constraints, na_object_sine=na_object_sine, samples=samples
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
    heights = sorted(
        {0.0}
        | {
            float(h)
            for c in constraints
            for h in getattr(c, "heights", ())
        }
    )
    aims = {float(getattr(c, "aim_z", 0.0)) for c in constraints if getattr(c, "heights", ())}
    if len(heights) > 1:
        fit.image_surface = aplanatic_image_surface(
            optimized, heights, na_object_sine=na_object_sine,
            aim_z=aims.pop() if len(aims) == 1 else 0.0,
        )
    return fit


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
    """The original single-purpose entry point: aplanatism, optionally flat.

    Equivalent to :func:`optimize_train` with ``[Aplanatism()]`` plus a
    :class:`~raytracer.optimize.constraints.FlatImageSurface` when
    ``field_heights`` is given — kept because "make this train aplanatic"
    is the common case and reads better without constraint plumbing.
    """

    return optimize_train(
        train,
        _constraints_for(field_heights, flat_field_weight),
        na_object_sine=na_object_sine,
        samples=samples,
        bounds=bounds,
        **least_squares_kwargs,
    )


__all__ = [
    "AplanatFit",
    "aplanatism_residuals",
    "constraint_residuals",
    "optimize_aplanat",
    "optimize_train",
]
