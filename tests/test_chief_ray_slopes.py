"""The chief-ray solver, checked against the definition of a chief ray.

``chief_ray_slopes`` solves for both transverse slopes jointly, assuming no
symmetry. Its answer must cross the stop at its centre, keep ``s_x = 0``
for a meridional field, and — through the system's rotational symmetry,
checked independently rather than assumed by the solver — give the same
radial image height for a field point on a rotated axis.
"""

from pathlib import Path

import numpy as np
import pytest

from raytracer.io import read_prescription
from raytracer.tracing import (
    FieldPoint,
    SequentialTracer,
    chief_ray_slopes,
    solve_object_plane,
    trace_from_object,
)

CSV = Path(__file__).resolve().parents[1] / "data" / (
    "optical_systems/lithography/US7557996_Fig3_Table3_prescription.csv"
)


@pytest.fixture(scope="module")
def tracer() -> SequentialTracer:
    tracer = read_prescription(CSV).tracer()
    solve_object_plane(tracer)
    return tracer


@pytest.mark.parametrize("y", [56.0, 62.0, 67.0])
def test_meridional_chief_ray_crosses_the_stop_centre(tracer, y):
    sx, sy = chief_ray_slopes(tracer, FieldPoint(y=y))
    assert sx == pytest.approx(0.0, abs=1e-9)
    result = trace_from_object(tracer, (0.0, y), (sx, sy))
    at_stop = result.trajectory[0, tracer.system.stop_index + 1, :2]
    assert np.hypot(*at_stop) < 1e-8


def test_off_axis_field_matches_rotated_meridional_case(tracer):
    # Rotational symmetry: a field point at radius r on a rotated axis
    # should give the same radial image height as the pure-y case at the
    # same radius. This is a property of the (rotationally symmetric)
    # system under test, verified independently here — the solver itself
    # makes no such assumption.
    r = 62.0
    angle = np.deg2rad(37.0)
    field = FieldPoint(x=r * np.sin(angle), y=r * np.cos(angle))
    sx, sy = chief_ray_slopes(tracer, field)
    result = trace_from_object(tracer, (field.x, field.y), (sx, sy))
    radial_image_height = np.hypot(*result.position[0, :2])

    meridional = FieldPoint(y=r)
    _, sy_ref = chief_ray_slopes(tracer, meridional)
    ref_result = trace_from_object(tracer, (0.0, r), (0.0, sy_ref))

    assert radial_image_height == pytest.approx(abs(ref_result.position[0, 1]), rel=1e-6)


def test_warm_start_matches_cold_start(tracer):
    field = FieldPoint(x=8.0, y=60.0)
    cold_sx, cold_sy = chief_ray_slopes(tracer, field)
    warm_sx, warm_sy = chief_ray_slopes(
        tracer, field, initial_guess=(cold_sx * 0.9, cold_sy * 0.9)
    )
    assert warm_sx == pytest.approx(cold_sx, abs=1e-8)
    assert warm_sy == pytest.approx(cold_sy, abs=1e-8)


def test_raises_for_field_beyond_aperture(tracer):
    with pytest.raises(RuntimeError):
        chief_ray_slopes(tracer, FieldPoint(y=1000.0))
