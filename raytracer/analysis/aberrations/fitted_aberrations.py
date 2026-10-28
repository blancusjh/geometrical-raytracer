"""Historical finite-aperture Zernike fits, not classical Seidel surface sums.

These effective coefficients mix higher orders and depend on the fitted
aperture and field interval. They omit distortion and are not validated by
our third-order reference contracts. Use seidel.seidel_coefficients for
analytic surface contributions and their explicit normalization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...propagation.fields import FieldPoint, PupilSampling, chief_ray_slopes, trace_pupil
from ...propagation.sequential import SequentialTracer
from .zernike import fit_transverse


@dataclass
class FittedLowOrderCoefficients:
    """Per-field low-order wavefront coefficients (waves) and their fits.

    ``*_waves`` arrays are the raw per-field values (one per sampled
    field height) of the corresponding pure monomial's coefficient;
    ``*_coefficient`` is the fitted Seidel coefficient — the field-scaling
    law's leading term, in waves, at the given wavelength.
    """

    field_heights_mm: np.ndarray
    spherical_waves: np.ndarray
    coma_waves: np.ndarray
    astigmatism_waves: np.ndarray
    defocus_waves: np.ndarray

    spherical_coefficient: float
    coma_coefficient: float
    astigmatism_coefficient: float
    field_curvature_coefficient: float
    defocus_offset_waves: float

    def summary(self) -> str:
        return (
            f"spherical (field-independent):  {self.spherical_coefficient:+.4f} waves\n"
            f"coma (per unit field height):   {self.coma_coefficient:+.4f} waves/mm\n"
            f"astigmatism (per h^2):          {self.astigmatism_coefficient:+.4f} waves/mm^2\n"
            f"field curvature (per h^2):      {self.field_curvature_coefficient:+.4f} waves/mm^2\n"
            f"(design defocus offset:         {self.defocus_offset_waves:+.4f} waves)"
        )


def _mode_coefficient(expansion, n: int, m: int) -> float:
    for mode, coefficient in zip(expansion.modes, expansion.coefficients_mm):
        if (mode.n, mode.m) == (n, m):
            return float(coefficient)
    return 0.0


def fitted_low_order_coefficients(
    tracer: SequentialTracer,
    field_heights_mm,
    *,
    na_object_sine: float,
    na_image: float,
    wavelength_mm: float | None = None,
    n_image: float = 1.0,
    sampling: PupilSampling | None = None,
    stop_index: int | None = None,
) -> FittedLowOrderCoefficients:
    """Fit Seidel-style coefficients from wavefronts traced at several fields.

    ``field_heights_mm`` must contain at least three points, spanning the
    field of interest, with at least two distinct ``y**2`` values — the
    field-curvature/defocus split is a two-parameter fit in ``y**2`` and is
    not identifiable otherwise. A field height of exactly 0 is fine and
    helps anchor the spherical/defocus fits.

    ``wavelength_mm`` defaults to the traced system's own
    ``wavelength_um``; pass it only to override that deliberately, since a
    value inconsistent with the system actually being traced would silently
    rescale every "waves" figure below. Each field is independently traced
    and its wavefront reconstructed from
    transverse ray aberrations via :func:`~.zernike.fit_transverse`
    (Hamilton's relation) rather than the OPL/reference-sphere route — the
    latter needs a ``reference_radius`` tuned to the system's own exit-pupil
    geometry (right for the EUV six-mirror notebook's scale, badly wrong
    for e.g. the US7557996 objective's), while the ray-slope route needs
    no such tuning and works unchanged across systems.

    Each field's chief ray is solved with the general 2-D solver
    :func:`~raytracer.propagation.fields.chief_ray_slopes` and passed into
    :func:`~raytracer.propagation.fields.trace_pupil` explicitly, rather
    than left for ``trace_pupil`` to solve internally: its default 1-D
    ``chief_ray_slope`` scans a fixed, comparatively coarse slope bracket
    that can come up empty for a huge finite-object stand-in for infinity
    (as in the Cooke triplet/double-Gauss examples), where the true chief
    ray sits in a needle-thin slice of that bracket.
    """

    tracer.system.require_axial_coordinates()

    field_heights_mm = np.asarray(field_heights_mm, dtype=float)
    if field_heights_mm.size < 3 or np.unique(field_heights_mm**2).size < 2:
        raise ValueError(
            "fitted_low_order_coefficients needs at least 3 field heights with at least 2 "
            f"distinct y**2 values to identify the field-scaling laws; got "
            f"{field_heights_mm.size} height(s): {field_heights_mm}"
        )
    if wavelength_mm is None:
        if tracer.system.wavelength_um is None:
            raise ValueError("system.wavelength_um is unset; pass wavelength_mm explicitly")
        wavelength_mm = tracer.system.wavelength_um * 1e-3
    sampling = sampling or PupilSampling(kind="rings", radial=10, azimuth=48)

    spherical, coma, astig, defocus = [], [], [], []
    for y in field_heights_mm:
        field = FieldPoint(y=float(y))
        chief_slopes = chief_ray_slopes(tracer, field, stop_index=stop_index)
        pupil = trace_pupil(
            tracer,
            field,
            na_object_sine=na_object_sine,
            sampling=sampling,
            chief_slope=chief_slopes,
            stop_index=stop_index,
        )
        expansion = fit_transverse(
            pupil,
            na_image=na_image,
            n_image=n_image,
            wavelength_mm=wavelength_mm,
            max_order=4,
        )
        to_waves = 1.0 / wavelength_mm
        spherical.append(6.0 * np.sqrt(5.0) * _mode_coefficient(expansion, 4, 0) * to_waves)
        coma.append(6.0 * np.sqrt(2.0) * _mode_coefficient(expansion, 3, -1) * to_waves)
        astig.append(np.sqrt(6.0) * _mode_coefficient(expansion, 2, 2) * to_waves)
        defocus.append(2.0 * np.sqrt(3.0) * _mode_coefficient(expansion, 2, 0) * to_waves)

    spherical = np.asarray(spherical)
    coma = np.asarray(coma)
    astig = np.asarray(astig)
    defocus = np.asarray(defocus)
    y = field_heights_mm
    y2 = y**2

    spherical_coefficient = float(np.mean(spherical))
    coma_coefficient = float(np.sum(y * coma) / np.sum(y * y)) if np.any(y) else 0.0
    astigmatism_coefficient = float(np.sum(y2 * astig) / np.sum(y2 * y2)) if np.any(y2) else 0.0

    design_matrix = np.column_stack([np.ones_like(y), y2])
    defocus_offset, field_curvature_coefficient = np.linalg.lstsq(
        design_matrix, defocus, rcond=None
    )[0]

    return FittedLowOrderCoefficients(
        field_heights_mm=field_heights_mm,
        spherical_waves=spherical,
        coma_waves=coma,
        astigmatism_waves=astig,
        defocus_waves=defocus,
        spherical_coefficient=spherical_coefficient,
        coma_coefficient=coma_coefficient,
        astigmatism_coefficient=astigmatism_coefficient,
        field_curvature_coefficient=float(field_curvature_coefficient),
        defocus_offset_waves=float(defocus_offset),
    )


__all__ = ["FittedLowOrderCoefficients", "fitted_low_order_coefficients"]
