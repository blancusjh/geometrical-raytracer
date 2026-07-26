"""Axial/lateral color and the Abbe-number dispersion model."""

from pathlib import Path

import numpy as np
import pytest

from raytracer.core.materials import AbbeMaterial
from raytracer.sequential import OpticalSystem, SequentialTracer, solve_object_plane
from raytracer.analysis import axial_color, lateral_color

TRIPLET_CSV = Path(__file__).resolve().parents[1] / "raytracer" / "data" / (
    "cooke_triplet_prescription.csv"
)
US7557996_CSV = Path(__file__).resolve().parents[1] / "raytracer" / "data" / (
    "US7557996_Fig3_Table3_prescription.csv"
)

F_LINE, D_LINE, C_LINE = 0.48613, 0.58756, 0.65627


def test_abbe_material_matches_nd_and_principal_dispersion():
    glass = AbbeMaterial("TEST", nd=1.6, vd=50.0)
    assert glass.index(D_LINE) == pytest.approx(1.6)
    assert glass.index(F_LINE) - glass.index(C_LINE) == pytest.approx((1.6 - 1.0) / 50.0)
    # Normal dispersion: index decreases with increasing wavelength.
    assert glass.index(F_LINE) > glass.index(D_LINE) > glass.index(C_LINE)


@pytest.fixture(scope="module")
def triplet_tracer():
    system = OpticalSystem.from_prescription(TRIPLET_CSV)
    tracer = SequentialTracer(system)
    solve_object_plane(tracer)
    return tracer


def test_axial_color_zero_at_reference_wavelength(triplet_tracer):
    result = axial_color(
        triplet_tracer.system, [F_LINE, D_LINE, C_LINE], reference_wavelength_um=D_LINE
    )
    assert np.all(np.isfinite(result.focus_z_mm))
    shift = result.focus_shift_mm
    assert shift[1] == pytest.approx(0.0, abs=1e-9)  # d-line is the reference
    assert shift[0] != pytest.approx(0.0, abs=1e-6)  # F and C should differ from d
    assert shift[2] != pytest.approx(0.0, abs=1e-6)


def test_lateral_color_zero_at_reference_wavelength(triplet_tracer):
    conjugate = solve_object_plane(triplet_tracer)
    y_field = abs(conjugate.object_z) * np.tan(np.deg2rad(6.0))
    result = lateral_color(
        triplet_tracer.system, [F_LINE, D_LINE, C_LINE],
        field_height_mm=y_field, reference_wavelength_um=D_LINE,
    )
    assert np.all(np.isfinite(result.image_height_mm))
    color = result.lateral_color_um
    assert color[1] == pytest.approx(0.0, abs=1e-9)
    assert color[0] != pytest.approx(0.0, abs=1e-3)
    assert color[2] != pytest.approx(0.0, abs=1e-3)


def test_constant_index_system_shows_no_axial_color():
    """A system built entirely from ConstantIndex materials (like
    US7557996's DUV glasses) has no real dispersion, so tracing it at
    different nominal wavelengths should give the identical focus."""

    system = OpticalSystem.from_prescription(US7557996_CSV)
    tracer = SequentialTracer(system)
    solve_object_plane(tracer)

    result = axial_color(system, [F_LINE, D_LINE, C_LINE], reference_wavelength_um=D_LINE)
    np.testing.assert_allclose(result.focus_shift_mm, 0.0, atol=1e-9)
