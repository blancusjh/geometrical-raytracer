"""Sanity regression for the reconstructed double-Gauss prescription.

See ``examples/notebooks/double_gauss.ipynb`` for the source citation
(rayopt-notebooks' "four element double gauss, intermediate optical
design") and the column-mapping derivation. As with the Cooke triplet,
there is no external reference dataset to regress against, so these
checks are sanity bounds rather than tight regressions.
"""

from pathlib import Path

import numpy as np
import pytest

from raytracer.design import OpticalSystem
from raytracer.propagation import (
    FieldPoint,
    ParaxialModel,
    SequentialTracer,
    chief_ray_slope,
    solve_object_plane,
    trace_from_object,
)

CSV = Path(__file__).resolve().parents[1] / "data" / "optical_systems/photographic/double_gauss_prescription.csv"


@pytest.fixture(scope="module")
def system() -> OpticalSystem:
    return OpticalSystem.from_prescription(CSV)


@pytest.fixture(scope="module")
def tracer(system) -> SequentialTracer:
    return SequentialTracer(system)


def test_prescription_loads(system):
    assert len(system) == 9
    assert system.stop_index == 4
    assert system.mirror_indices == []
    assert system.image_z == pytest.approx(113.18)
    assert system.n_image == pytest.approx(1.0)


def test_paraxial_focal_length_in_expected_range(tracer):
    solve_object_plane(tracer)
    efl = ParaxialModel(tracer.system).effective_focal_length()
    # ~100mm class design (rayopt-notebooks reports a 23.348mm optical
    # track and an 89.832mm back focal distance for this prescription).
    assert 90.0 < efl < 110.0


def chief_distortion_ppm(tracer, object_z, stop_index, magnification, angle_deg):
    y = abs(object_z) * np.tan(np.deg2rad(angle_deg))
    z_stop = tracer.system.vertices[stop_index]
    estimate = -y / (z_stop - object_z)
    slope = chief_ray_slope(
        tracer, FieldPoint(y=y), stop_index=stop_index,
        bracket=(estimate - 0.02, estimate + 0.02), scan=101,
    )
    result = trace_from_object(tracer, (0.0, y), (0.0, slope))
    ideal_y = y * magnification
    return float((result.image_point[1] / ideal_y - 1) * 1e6)


def test_chief_ray_distortion_grows_with_field(tracer):
    conjugate = solve_object_plane(tracer)
    ppm = [
        chief_distortion_ppm(
            tracer, conjugate.object_z, tracer.system.stop_index,
            conjugate.magnification, angle_deg,
        )
        for angle_deg in (5.0, 10.0, 15.0, 20.0)
    ]
    for p in ppm:
        assert np.isfinite(p)
    magnitudes = [abs(p) for p in ppm]
    assert magnitudes == sorted(magnitudes)


def test_less_relative_distortion_than_triplet_at_matching_field(tracer):
    """The symmetric stop position should give markedly less distortion
    than the Cooke triplet's asymmetric one, at the same field angle."""

    from raytracer.design import OpticalSystem as _OpticalSystem

    triplet_csv = Path(__file__).resolve().parents[1] / "data" / "optical_systems/photographic/cooke_triplet_prescription.csv"
    triplet_system = _OpticalSystem.from_prescription(triplet_csv)
    triplet_tracer = SequentialTracer(triplet_system)
    triplet_conjugate = solve_object_plane(triplet_tracer)

    conjugate = solve_object_plane(tracer)
    angle_deg = 6.0
    dg_ppm = abs(chief_distortion_ppm(
        tracer, conjugate.object_z, tracer.system.stop_index,
        conjugate.magnification, angle_deg,
    ))
    triplet_ppm = abs(chief_distortion_ppm(
        triplet_tracer, triplet_conjugate.object_z, triplet_system.stop_index,
        triplet_conjugate.magnification, angle_deg,
    ))
    assert dg_ppm < triplet_ppm
