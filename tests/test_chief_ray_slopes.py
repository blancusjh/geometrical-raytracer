"""General 2-D chief-ray solver, cross-checked against the 1-D solver.

``chief_ray_slopes`` solves for both transverse slopes jointly instead of
assuming ``s_x = 0`` and exploiting rotational symmetry, so it must agree
exactly with ``chief_ray_slope`` in the degenerate on-axis-y case, and must
give a physically consistent answer for a genuinely off-axis (x, y) field
point via the system's actual rotational symmetry (checked independently,
not assumed by the solver itself).
"""

from pathlib import Path

import numpy as np
import pytest

from raytracer.sequential import (
    FieldPoint,
    OpticalSystem,
    SequentialTracer,
    chief_ray_slope,
    chief_ray_slopes,
    solve_object_plane,
    trace_from_object,
)

CSV = Path(__file__).resolve().parents[1] / "raytracer" / "data" / (
    "US7557996_Fig3_Table3_prescription.csv"
)


@pytest.fixture(scope="module")
def tracer() -> SequentialTracer:
    system = OpticalSystem.from_prescription(CSV)
    tracer = SequentialTracer(system)
    solve_object_plane(tracer)
    return tracer


@pytest.mark.parametrize("y", [56.0, 62.0, 67.0])
def test_matches_1d_solver_on_axis(tracer, y):
    field = FieldPoint(y=y)
    sy_1d = chief_ray_slope(tracer, field)
    sx_2d, sy_2d = chief_ray_slopes(tracer, field)
    assert sx_2d == pytest.approx(0.0, abs=1e-9)
    assert sy_2d == pytest.approx(sy_1d, abs=1e-8)


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
    radial_image_height = np.hypot(*result.image_point[:2])

    meridional = FieldPoint(y=r)
    sy_ref = chief_ray_slope(tracer, meridional)
    ref_result = trace_from_object(tracer, (0.0, r), (0.0, sy_ref))

    assert radial_image_height == pytest.approx(abs(ref_result.image_point[1]), rel=1e-6)


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
