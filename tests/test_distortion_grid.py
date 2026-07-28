"""Square-grid distortion tracing (``raytracer.analysis.aberrations.distortion``)."""

from pathlib import Path

import numpy as np
import pytest

from raytracer.design import OpticalSystem
from raytracer.propagation import SequentialTracer, solve_object_plane
from raytracer.analysis import distortion_grid

CSV = Path(__file__).resolve().parents[1] / "data" / "optical_systems/photographic/cooke_triplet_prescription.csv"


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


def test_deviation_is_actual_minus_ideal(grid):
    np.testing.assert_allclose(grid.deviation_mm, grid.actual_points - grid.ideal_points)


def test_max_distortion_matches_the_worst_grid_point(grid):
    radius_um = np.linalg.norm(grid.deviation_mm, axis=-1) * 1e3
    assert grid.max_distortion_um == pytest.approx(np.nanmax(radius_um))
    assert grid.max_distortion_um > 0.0


def test_relative_distortion_excludes_the_axial_point(grid):
    """The axial point has zero ideal height, so dividing by it would make
    the relative figure infinite for every system."""

    ideal_height = np.linalg.norm(grid.ideal_points, axis=-1)
    center = grid.object_points.shape[0] // 2
    assert ideal_height[center, center] == 0.0
    assert np.isfinite(grid.max_relative_distortion_percent)

    radius = np.linalg.norm(grid.deviation_mm, axis=-1)
    off_axis = ideal_height > 0.0
    expected = (radius[off_axis] / ideal_height[off_axis]).max() * 100.0
    assert grid.max_relative_distortion_percent == pytest.approx(expected)


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


def test_chief_ray_distortion_matches_field_metrics(tracer):
    """The cheap chief-ray-only sweep must agree with the value
    field_metrics derives from a full pupil trace."""

    from raytracer.analysis import chief_ray_distortion, field_metrics

    conjugate = solve_object_plane(tracer)
    heights = abs(conjugate.object_z) * np.tan(np.deg2rad([2.0, 4.0, 6.0]))

    swept = chief_ray_distortion(
        tracer, heights, magnification=conjugate.magnification
    )
    full = field_metrics(
        tracer, heights, na_object_sine=6.5 / abs(conjugate.object_z),
        magnification=conjugate.magnification,
    )
    for cheap, reference in zip(swept, full):
        assert cheap["chief_ray_distortion_um"] == pytest.approx(
            reference["chief_ray_distortion_um"], rel=1e-6
        )


def test_chief_ray_distortion_accepts_field_angles(tracer):
    from raytracer.analysis import chief_ray_distortion

    conjugate = solve_object_plane(tracer)
    by_angle = chief_ray_distortion(
        tracer, [4.0], magnification=conjugate.magnification, field_unit="deg"
    )
    height = abs(conjugate.object_z) * np.tan(np.deg2rad(4.0))
    by_height = chief_ray_distortion(
        tracer, [height], magnification=conjugate.magnification
    )
    assert by_angle[0]["object_height_mm"] == pytest.approx(height)
    assert by_angle[0]["chief_ray_distortion_um"] == pytest.approx(
        by_height[0]["chief_ray_distortion_um"]
    )


def test_chief_ray_distortion_rejects_an_unknown_field_unit(tracer):
    from raytracer.analysis import chief_ray_distortion

    with pytest.raises(ValueError, match="field_unit"):
        chief_ray_distortion(tracer, [1.0], magnification=1.0, field_unit="radians")
