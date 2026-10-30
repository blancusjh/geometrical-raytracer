"""Axial/lateral color and the Abbe-number dispersion model."""

from pathlib import Path

import numpy as np
import pytest

from raytracer.analysis import axial_color, lateral_color
from raytracer.io import read_prescription
from raytracer.materials import AbbeMaterial
from raytracer.tracing import SequentialTracer, solve_object_plane

TRIPLET_CSV = Path(__file__).resolve().parents[1] / "data" / (
    "optical_systems/photographic/cooke_triplet_prescription.csv"
)
US7557996_CSV = Path(__file__).resolve().parents[1] / "data" / (
    "optical_systems/lithography/US7557996_Fig3_Table3_prescription.csv"
)

F_LINE, D_LINE, C_LINE = 0.48613, 0.58756, 0.65627

SINGLET_CSV = Path(__file__).resolve().parents[1] / "data" / (
    "optical_systems/elements/sf11_singlet_prescription.csv"
)
SINGLET_OBJECT_Z = -1.0e5   # the far object both example lenses were solved for
SINGLET_NA = 6.25 / abs(SINGLET_OBJECT_Z)


def test_abbe_material_matches_nd_and_principal_dispersion():
    glass = AbbeMaterial("TEST", nd=1.6, vd=50.0)
    assert glass.index(D_LINE) == pytest.approx(1.6)
    assert glass.index(F_LINE) - glass.index(C_LINE) == pytest.approx((1.6 - 1.0) / 50.0)
    # Normal dispersion: index decreases with increasing wavelength.
    assert glass.index(F_LINE) > glass.index(D_LINE) > glass.index(C_LINE)


@pytest.fixture(scope="module")
def triplet_tracer():
    tracer = read_prescription(TRIPLET_CSV).tracer()
    solve_object_plane(tracer)
    return tracer


def singlet_tracer() -> SequentialTracer:
    return read_prescription(SINGLET_CSV).tracer(object_z=SINGLET_OBJECT_Z)


def test_axial_color_zero_at_reference_wavelength(triplet_tracer):
    result = axial_color(
        triplet_tracer, [F_LINE, D_LINE, C_LINE], reference_wavelength_um=D_LINE
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
        triplet_tracer, [F_LINE, D_LINE, C_LINE],
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

    tracer = read_prescription(US7557996_CSV).tracer()
    solve_object_plane(tracer)

    result = axial_color(tracer, [F_LINE, D_LINE, C_LINE], reference_wavelength_um=D_LINE)
    np.testing.assert_allclose(result.focus_shift_mm, 0.0, atol=1e-9)


def test_chromatic_spots_share_one_reference_frame():
    """Each wavelength's spot must be referred to a common origin, or the
    colour separation the class exists to show would be subtracted out."""

    from raytracer.analysis import chromatic_spots

    result = chromatic_spots(
        singlet_tracer(), [F_LINE, D_LINE, C_LINE], na_object_sine=SINGLET_NA,
        reference_wavelength_um=D_LINE,
    )
    assert len(result.offsets_um) == 3
    # A strongly chromatic singlet: the reference wavelength is by far the
    # tightest, and the extreme wavelengths are decisively worse.
    rms = result.rms_radius_um
    assert rms[1] < rms[0] and rms[1] < rms[2]
    # Pooling every colour must be worse than the best single colour.
    assert result.polychromatic_rms_um > rms.min()


def test_chromatic_spots_rejects_unsampled_reference():
    from raytracer.analysis import chromatic_spots

    with pytest.raises(ValueError, match="not among the sampled wavelengths"):
        chromatic_spots(
            singlet_tracer(), [F_LINE, C_LINE], na_object_sine=SINGLET_NA,
            reference_wavelength_um=0.2,
        )


def test_axial_color_probe_survives_a_very_distant_object():
    """A fixed differential *slope* overflows the aperture once the object
    is far away; the probe must be specified by ray height instead."""

    tracer = singlet_tracer()  # 100 m: a 1e-4 slope would be 10 mm high
    result = axial_color(tracer, [F_LINE, D_LINE, C_LINE], reference_wavelength_um=D_LINE)
    assert np.all(np.isfinite(result.focus_z_mm))
    # A dense flint singlet focuses blue decidedly shorter than red.
    assert result.focus_z_mm[0] < result.focus_z_mm[2]


def test_tracer_at_wavelength_leaves_the_original_untouched():
    """The per-wavelength tracer is a new tracer of the same system and object
    plane, and the original keeps its wavelength."""

    tracer = singlet_tracer()
    at_f = tracer.at_wavelength(F_LINE)
    assert at_f is not tracer
    assert at_f.system is tracer.system
    assert at_f.object_z == tracer.object_z == SINGLET_OBJECT_Z
    assert at_f.wavelength_um == F_LINE
    assert tracer.wavelength_um == pytest.approx(0.5876)
    # The glass index is evaluated at each tracer's own wavelength.
    lens = tracer.system.step_index("L1.front")
    assert at_f._n[lens][1] > tracer._n[lens][1]
    with pytest.raises(AttributeError):
        tracer.system.name = "changed"
