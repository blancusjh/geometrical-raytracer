"""Pluggable design constraints for stigmatic-train optimization.

Every member of a :class:`~raytracer.design.stigmatic.StigmaticTrain`
family is already exactly stigmatic; a *constraint* is one further property
the optimizer buys with the free conjugates, expressed as a residual vector
that is zero exactly when the property holds. Constraints compose: pass any
list to :func:`raytracer.optimize.aplanat.optimize_train` and their
residual blocks are concatenated into one least-squares objective.

Built in:

- :class:`Aplanatism` — the sine condition, ``M - 1`` per traced ray
  (Silva-Lora & Torres); the constraint that removes coma.
- :class:`FlatImageSurface` — the sagitta of the aplanatic image surface at
  chosen field heights; zero means the image of the neighbourhood forms on
  the plane of the axial image.
- :class:`Distortion` — departure of the actual image height from the
  linear map ``g_t * h`` at chosen field heights; zero means the system
  images a grid as a grid.
- :class:`TargetMagnification` — pins ``g_t`` itself, so a microscope stays
  a 20x microscope while the other constraints fight over the conjugates.

Writing a new one is subclassing :class:`Constraint` with a ``size`` and a
``residuals`` — the :class:`EvaluationContext` hands every constraint the
same cached traces (one aplanatism fan, one image-surface locus per
distinct field tuple), so adding a constraint that reuses those costs no
extra rays.

Image positions are measured on the least-squares convergence points of
cones aimed at an axial reference point — the trains carry no aperture
stop, so "chief ray through ``aim_z``" is the reference convention, with
the front vertex (``aim_z = 0``) as the default. Systems whose natural
pupil sits elsewhere (a projector's, deep inside the barrel) should say so
via each field constraint's ``aim_z``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..analysis.aberrations.aplanatism import (
    AplanatismReport,
    ImageSurface,
    aplanatic_image_surface,
    aplanatism_report,
)
from ..design.stigmatic import StigmaticTrain

#: Finite stand-in residual for a ray or bundle that failed to trace, so
#: the solver backs away from singular walls instead of seeing NaN.
PENALTY = 1e3


class EvaluationContext:
    """One candidate train plus lazily cached traces shared by constraints."""

    def __init__(
        self, train: StigmaticTrain, *, na_object_sine: float, samples: int
    ) -> None:
        self.train = train
        self.na_object_sine = na_object_sine
        self.samples = samples
        self._report: AplanatismReport | None = None
        self._surfaces: dict[tuple[float, ...], ImageSurface] = {}

    def report(self) -> AplanatismReport:
        if self._report is None:
            self._report = aplanatism_report(
                self.train, na_object_sine=self.na_object_sine, samples=self.samples
            )
        return self._report

    def image_surface(
        self, heights: tuple[float, ...], *, aim_z: float = 0.0
    ) -> ImageSurface:
        key = (heights, aim_z)
        if key not in self._surfaces:
            self._surfaces[key] = aplanatic_image_surface(
                self.train, heights, na_object_sine=self.na_object_sine,
                aim_z=aim_z,
            )
        return self._surfaces[key]


class Constraint:
    """One property to drive to zero; subclasses define its residual block."""

    weight: float = 1.0

    def size(self, context: EvaluationContext) -> int:
        """Residual count — must not depend on how many rays survive."""

        raise NotImplementedError

    def residuals(self, context: EvaluationContext) -> np.ndarray:
        raise NotImplementedError


@dataclass
class Aplanatism(Constraint):
    """The Abbe sine condition: one residual ``M - 1`` per fan ray."""

    weight: float = 1.0

    def size(self, context: EvaluationContext) -> int:
        return context.samples

    def residuals(self, context: EvaluationContext) -> np.ndarray:
        report = context.report()
        return self.weight * np.where(
            report.valid, report.map_values - 1.0, PENALTY
        )


@dataclass
class FlatImageSurface(Constraint):
    """Zero sagitta of the aplanatic image surface at the given heights.

    ``heights`` must start at 0.0 — the axial image is the plane the rest
    of the field is asked to land on. ``aim_z`` places the reference pupil
    (see :func:`~raytracer.analysis.aberrations.aplanatism.aplanatic_image_surface`).
    """

    heights: tuple[float, ...] = (0.0,)
    weight: float = 1.0
    aim_z: float = 0.0

    def __post_init__(self) -> None:
        self.heights = tuple(float(h) for h in self.heights)
        if not self.heights or self.heights[0] != 0.0:
            raise ValueError(
                f"FlatImageSurface heights must start at 0.0 (the reference "
                f"plane is the axial image), got {self.heights}"
            )

    def size(self, context: EvaluationContext) -> int:
        return len(self.heights)

    def residuals(self, context: EvaluationContext) -> np.ndarray:
        surface = context.image_surface(self.heights, aim_z=self.aim_z)
        return self.weight * np.where(surface.valid, surface.sagitta, PENALTY)


@dataclass
class Distortion(Constraint):
    """Zero relative distortion, ``y(h) / (g_t h) - 1``, at the given heights.

    ``heights`` must be nonzero (the axial point carries no distortion
    information). The reference magnification is each candidate's own
    ``g_t``, so the constraint measures departure from linearity, not from
    a frozen number — pair it with :class:`TargetMagnification` to pin the
    scale itself. ``aim_z`` places the reference pupil.
    """

    heights: tuple[float, ...] = field(default_factory=tuple)
    weight: float = 1.0
    aim_z: float = 0.0

    def __post_init__(self) -> None:
        self.heights = tuple(float(h) for h in self.heights)
        if not self.heights or any(h == 0.0 for h in self.heights):
            raise ValueError(
                f"Distortion heights must be nonzero, got {self.heights}"
            )

    def size(self, context: EvaluationContext) -> int:
        return len(self.heights)

    def residuals(self, context: EvaluationContext) -> np.ndarray:
        surface = context.image_surface(self.heights, aim_z=self.aim_z)
        ideal = context.train.gt * np.asarray(self.heights)
        with np.errstate(invalid="ignore", divide="ignore"):
            relative = surface.points[:, 1] / ideal - 1.0
        return self.weight * np.where(surface.valid, relative, PENALTY)


@dataclass
class TargetMagnification(Constraint):
    """Pin the transverse magnification ``g_t`` to a design value."""

    value: float = -1.0
    weight: float = 1.0

    def size(self, context: EvaluationContext) -> int:
        return 1

    def residuals(self, context: EvaluationContext) -> np.ndarray:
        return np.array([self.weight * (context.train.gt - self.value)])


@dataclass
class AxialColor(Constraint):
    """Zero axial color: other wavelengths must focus on the design plane.

    Needs a train with at least one dispersive medium — the residuals are
    the axial focus shifts (mm) of a marginal ray retraced at each probe
    wavelength through the *real* dispersive media, relative to the design
    image plane. The optimizer can genuinely null them: the
    intermediate conjugates set the power distribution over the glasses,
    which is exactly the freedom a classic achromat spends. The surface
    shapes stay rigorously stigmatic at the design wavelength only.

    Residuals are millimetres; against the dimensionless ``M - 1`` block a
    ``weight`` of order 0.01-1 balances the two — start small and raise it
    until the color budget is met.
    """

    wavelengths: tuple[float, ...] = (0.48613, 0.65627)
    weight: float = 1.0
    #: The probe ray's sine, as a fraction of the evaluation NA (0.7 is the
    #: classic "zonal" compromise between paraxial and marginal color).
    marginal_fraction: float = 0.7

    def __post_init__(self) -> None:
        self.wavelengths = tuple(float(w) for w in self.wavelengths)

    def size(self, context: EvaluationContext) -> int:
        return len(self.wavelengths)

    def residuals(self, context: EvaluationContext) -> np.ndarray:
        from ..design.system import OpticalSystem
        from ..propagation.fields import trace_from_object
        from ..propagation.sequential import SequentialTracer

        train = context.train
        if not train.is_dispersive:
            raise TypeError(
                "AxialColor needs at least one dispersive medium; a train of "
                "constant-index media has no chromatic aberration to constrain"
            )
        sine = self.marginal_fraction * context.na_object_sine
        slope = sine / np.sqrt(1.0 - sine * sine)
        shifts = np.full(len(self.wavelengths), PENALTY)
        for i, wavelength in enumerate(self.wavelengths):
            system = OpticalSystem(
                train.rows(),
                object_space=train.media[0],
                object_z=train.conjugates[0],
                wavelength_um=wavelength,
            )
            result = trace_from_object(
                SequentialTracer(system), (0.0, 0.0), (0.0, slope)
            )
            direction = result.direction / np.linalg.norm(result.direction)
            if not result.ok or abs(direction[1]) < 1e-15:
                continue
            crossing = result.image_point[2] - result.image_point[1] * (
                direction[2] / direction[1]
            )
            shifts[i] = crossing - train.conjugates[-1]
        return self.weight * shifts


__all__ = [
    "PENALTY",
    "Constraint",
    "EvaluationContext",
    "Aplanatism",
    "FlatImageSurface",
    "Distortion",
    "TargetMagnification",
    "AxialColor",
]
