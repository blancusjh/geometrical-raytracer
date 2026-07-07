"""Full-system regression against the US 7,557,996 reference outputs."""

from pathlib import Path

import numpy as np
import pytest

from raytracer.sequential import (
    FieldPoint,
    OpticalSystem,
    ParaxialModel,
    PupilSampling,
    SequentialTracer,
    differential_conjugates,
    solve_object_plane,
    trace_pupil,
)

CSV = Path(__file__).resolve().parents[1] / "raytracer" / "data" / (
    "US7557996_Fig3_Table3_prescription.csv"
)

NA_IMAGE = 1.2
REDUCTION = 4.0


@pytest.fixture(scope="module")
def system() -> OpticalSystem:
    return OpticalSystem.from_prescription(CSV)


@pytest.fixture(scope="module")
def tracer(system) -> SequentialTracer:
    tracer = SequentialTracer(system)
    solve_object_plane(tracer)
    return tracer


def test_prescription_loads(system):
    assert len(system) == 48
    assert system.wavelength_um == pytest.approx(0.193368)
    assert system.stop_index == 34
    assert system.mirror_indices == [6, 9]
    assert system.image_z == pytest.approx(1423.077700, abs=1e-6)
    assert system.n_image == pytest.approx(1.59667693)


def test_recovered_conjugates_match_reference(tracer):
    solution = differential_conjugates(tracer)
    assert solution.object_z == pytest.approx(-76.876406, abs=1e-4)
    assert solution.magnification == pytest.approx(0.2500000135, abs=1e-7)


def test_paraxial_matrix_agrees_with_differential(system, tracer):
    exact = differential_conjugates(tracer)
    model = ParaxialModel(system)
    analytic = model.solve_object_plane()
    assert analytic.object_z == pytest.approx(exact.object_z, abs=1e-3)
    assert analytic.magnification == pytest.approx(exact.magnification, abs=1e-6)


def test_field_traces_match_reference_image_heights(tracer):
    # Reference: object y=56 -> image y=14.000000; y=62 -> 15.499997; y=67 -> 16.749999
    from raytracer.sequential import chief_ray_slope, trace_from_object

    expected = {56.0: 14.000000, 62.0: 15.499997, 67.0: 16.749999}
    for y_obj, y_img in expected.items():
        slope = chief_ray_slope(tracer, FieldPoint(y=y_obj))
        result = trace_from_object(tracer, (0.0, y_obj), (0.0, slope))
        assert result.ok
        assert result.image_point[1] == pytest.approx(y_img, abs=5e-6)


@pytest.mark.slow
def test_pupil_na_and_ray_count(tracer):
    pupil = trace_pupil(
        tracer,
        FieldPoint(y=62.0),
        na_object_sine=NA_IMAGE / REDUCTION,
        sampling=PupilSampling(kind="rings", radial=12, azimuth=72),
    )
    n_image = tracer.system.n_image
    na = n_image * np.hypot(pupil.directions[:, 0], pupil.directions[:, 1])
    assert pupil.valid.sum() == 754  # reference: unvignetted rays
    assert na.max() == pytest.approx(1.197703, abs=1e-4)


@pytest.mark.slow
def test_rms_spot_radius_in_reference_band(tracer):
    # Reference RMS spot radii: 0.0922, 0.0862, 0.0812 um at the three fields.
    expected = {56.0: 0.092218766, 62.0: 0.086225812, 67.0: 0.081181224}
    for y_obj, rms_ref in expected.items():
        pupil = trace_pupil(
            tracer,
            FieldPoint(y=y_obj),
            na_object_sine=NA_IMAGE / REDUCTION,
            sampling=PupilSampling(kind="rings", radial=10, azimuth=48),
        )
        points = pupil.image_points[:, :2]
        weights = pupil.weights
        centroid = np.sum(points * weights[:, None], axis=0)
        relative_um = (points - centroid) * 1e3
        rms = np.sqrt(np.sum(weights * np.sum(relative_um**2, axis=1)))
        assert rms == pytest.approx(rms_ref, rel=2e-3)
