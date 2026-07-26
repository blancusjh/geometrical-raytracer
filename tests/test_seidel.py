"""Seidel-style coefficients deduced from the traced wavefront.

Two sanity checks rather than tight regressions, since there's no external
reference dataset for this metric (unlike US7557996's EE80/astigmatism
values in test_us7557996.py):

- the well-corrected US7557996 objective should show small (sub-wave)
  coefficients, consistent with its already-known ~0.1 wave RMS
  (test_zernike_regression in test_us7557996.py);
- the unaspherized Cooke triplet should show coma and astigmatism growing
  monotonically with field, the textbook Seidel signature, since nothing
  in this simple three-element design corrects for it.
"""

from pathlib import Path

import numpy as np
import pytest

from raytracer.sequential import OpticalSystem, SequentialTracer, solve_object_plane
from raytracer.analysis import seidel_coefficients

US7557996_CSV = Path(__file__).resolve().parents[1] / "raytracer" / "data" / (
    "US7557996_Fig3_Table3_prescription.csv"
)
TRIPLET_CSV = Path(__file__).resolve().parents[1] / "raytracer" / "data" / (
    "cooke_triplet_prescription.csv"
)


def test_well_corrected_lens_gives_small_coefficients():
    system = OpticalSystem.from_prescription(US7557996_CSV)
    tracer = SequentialTracer(system)
    solve_object_plane(tracer)

    result = seidel_coefficients(
        tracer, np.linspace(0.0, 67.0, 4),
        na_object_sine=1.2 / 4.0, na_image=1.2,
        wavelength_mm=system.wavelength_um * 1e-3, n_image=system.n_image,
    )
    for value in (
        result.spherical_coefficient,
        result.coma_coefficient,
        result.astigmatism_coefficient,
        result.field_curvature_coefficient,
    ):
        assert np.isfinite(value)
    assert abs(result.spherical_coefficient) < 5.0


def test_unaspherized_triplet_shows_growing_coma_and_astigmatism():
    system = OpticalSystem.from_prescription(TRIPLET_CSV)
    tracer = SequentialTracer(system)
    conjugate = solve_object_plane(tracer)

    na_object_sine = 6.5 / abs(conjugate.object_z)  # ~entrance pupil / object distance
    na_image = na_object_sine / abs(conjugate.magnification)
    fields_deg = np.linspace(0.0, 8.0, 6)
    fields_mm = np.abs(conjugate.object_z) * np.tan(np.deg2rad(fields_deg))

    result = seidel_coefficients(
        tracer, fields_mm, na_object_sine=na_object_sine, na_image=na_image,
        wavelength_mm=0.5876e-3, n_image=system.n_image,
    )

    coma_magnitude = np.abs(result.coma_waves)
    astig_magnitude = np.abs(result.astigmatism_waves)
    assert np.all(np.diff(coma_magnitude) >= -1e-9)
    assert np.all(np.diff(astig_magnitude) >= -1e-9)
    assert coma_magnitude[-1] > coma_magnitude[0]
    assert astig_magnitude[-1] > astig_magnitude[0]
    assert np.isfinite(result.spherical_coefficient)


def test_rejects_too_few_fields_instead_of_fabricating_a_fit():
    """The defocus/field-curvature split is a two-parameter fit in y**2;
    with fewer points lstsq happily returns a minimum-norm answer that
    looks plausible and means nothing."""

    system = OpticalSystem.from_prescription(TRIPLET_CSV)
    tracer = SequentialTracer(system)
    conjugate = solve_object_plane(tracer)
    na = 6.5 / abs(conjugate.object_z)

    for fields in ([100.0], [100.0, 200.0], [50.0, -50.0]):  # too few, or degenerate y**2
        with pytest.raises(ValueError, match="at least 3 field heights"):
            seidel_coefficients(
                tracer, fields, na_object_sine=na,
                na_image=na / abs(conjugate.magnification), wavelength_mm=0.5876e-3,
            )


def test_wavelength_defaults_to_the_traced_system():
    system = OpticalSystem.from_prescription(TRIPLET_CSV)
    tracer = SequentialTracer(system)
    conjugate = solve_object_plane(tracer)
    na = 6.5 / abs(conjugate.object_z)
    fields = np.abs(conjugate.object_z) * np.tan(np.deg2rad(np.linspace(0.0, 6.0, 4)))
    kwargs = dict(na_object_sine=na, na_image=na / abs(conjugate.magnification))

    explicit = seidel_coefficients(tracer, fields, wavelength_mm=0.5876e-3, **kwargs)
    defaulted = seidel_coefficients(tracer, fields, **kwargs)
    assert defaulted.spherical_coefficient == pytest.approx(explicit.spherical_coefficient)
