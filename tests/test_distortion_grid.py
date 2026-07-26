"""Square-grid distortion tracing (``raytracer.analysis.aberrations.distortion``)."""

from pathlib import Path

import numpy as np
import pytest

from raytracer.sequential import OpticalSystem, SequentialTracer, solve_object_plane
from raytracer.analysis import distortion_grid

CSV = Path(__file__).resolve().parents[1] / "raytracer" / "data" / "cooke_triplet_prescription.csv"


@pytest.fixture(scope="module")
def tracer() -> SequentialTracer:
    system = OpticalSystem.from_prescription(CSV)
    return SequentialTracer(system)


@pytest.fixture(scope="module")
def grid(tracer):
    conjugate = solve_object_plane(tracer)
    half = abs(conjugate.object_z) * np.tan(np.deg2rad(6.0))
    return distortion_grid(tracer, magnification=conjugate.magnification, half_field=half, n=5)


def test_grid_shapes(grid):
    assert grid.object_points.shape == (5, 5, 2)
    assert grid.ideal_points.shape == (5, 5, 2)
    assert grid.actual_points.shape == (5, 5, 2)
    assert grid.valid.shape == (5, 5)


def test_ideal_points_are_the_linear_map(grid):
    np.testing.assert_allclose(grid.ideal_points, grid.object_points * grid.magnification)


def test_fully_valid_within_the_working_field(grid):
    assert grid.valid_fraction == 1.0


def test_center_point_is_undistorted(grid):
    center = grid.object_points.shape[0] // 2
    np.testing.assert_allclose(grid.actual_points[center, center], [0.0, 0.0], atol=1e-9)


def test_rectangular_half_field_accepted(tracer):
    conjugate = solve_object_plane(tracer)
    half_x = abs(conjugate.object_z) * np.tan(np.deg2rad(4.0))
    half_y = abs(conjugate.object_z) * np.tan(np.deg2rad(6.0))
    result = distortion_grid(
        tracer, magnification=conjugate.magnification,
        half_field=(half_x, half_y), n=5,
    )
    assert result.object_points[0, 0, 0] == pytest.approx(-half_x)
    assert result.object_points[0, 0, 1] == pytest.approx(-half_y)


def test_off_origin_center_shifts_the_grid_but_not_the_ideal_map(tracer):
    """A ring-field system's usable field is an off-axis annulus, not a
    disk around the origin — the grid must be able to sit anywhere in
    object space, while the ideal map stays the single linear
    magnification (always through the origin) regardless of where the
    grid itself is centered."""

    conjugate = solve_object_plane(tracer)
    half = abs(conjugate.object_z) * np.tan(np.deg2rad(1.0))
    center = (0.0, abs(conjugate.object_z) * np.tan(np.deg2rad(6.0)))
    result = distortion_grid(
        tracer, magnification=conjugate.magnification,
        half_field=half, center=center, n=5,
    )
    np.testing.assert_allclose(result.object_points[2, 2], center)
    np.testing.assert_allclose(
        result.ideal_points, result.object_points * conjugate.magnification
    )
    assert result.valid_fraction == 1.0
