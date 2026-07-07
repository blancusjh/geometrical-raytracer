"""Pure camera math: round trips, anchored zoom, aspect lock (no GL)."""

import numpy as np
import pytest

from raytracer.viz.gl.camera import Camera
from raytracer.viz.gl.renderers import auto_exposure, quad_indices, segment_quads


def test_world_screen_round_trip():
    camera = Camera(center=[3.0, -2.0], half_height=5.0, viewport=(800, 600))
    points = np.array([[0.0, 0.0], [3.0, -2.0], [-7.5, 4.25]])
    recovered = camera.screen_to_world(camera.world_to_screen(points))
    assert recovered == pytest.approx(points, abs=1e-12)


def test_center_maps_to_viewport_middle():
    camera = Camera(center=[1.0, 2.0], half_height=4.0, viewport=(640, 480))
    assert camera.world_to_screen([1.0, 2.0]) == pytest.approx([320.0, 240.0])


def test_aspect_is_locked():
    camera = Camera(center=[0.0, 0.0], half_height=5.0, viewport=(1200, 600))
    # One world unit maps to the same number of pixels in x and y.
    origin = camera.world_to_screen([0.0, 0.0])
    dx = camera.world_to_screen([1.0, 0.0]) - origin
    dy = camera.world_to_screen([0.0, 1.0]) - origin
    assert abs(dx[0]) == pytest.approx(abs(dy[1]), abs=1e-12)


def test_zoom_keeps_cursor_point_fixed():
    camera = Camera(center=[2.0, 1.0], half_height=6.0, viewport=(800, 600))
    cursor = np.array([123.0, 456.0])
    anchor_before = camera.screen_to_world(cursor)
    camera.zoom_about(1.35, cursor)
    anchor_after = camera.screen_to_world(cursor)
    assert anchor_after == pytest.approx(anchor_before, abs=1e-10)
    camera.zoom_about(1.0 / 1.35, cursor)
    assert camera.half_height == pytest.approx(6.0, abs=1e-12)


def test_pan_moves_content_with_cursor():
    camera = Camera(center=[0.0, 0.0], half_height=5.0, viewport=(800, 600))
    world_point = np.array([1.0, 1.0])
    before = camera.world_to_screen(world_point)
    camera.pan_pixels(50.0, -30.0)
    after = camera.world_to_screen(world_point)
    assert after - before == pytest.approx([50.0, -30.0], abs=1e-12)


def test_resize_preserves_world_per_pixel():
    camera = Camera(center=[0.0, 0.0], half_height=5.0, viewport=(800, 600))
    ppw = camera.pixels_per_world
    camera.resize((1600, 900))
    assert camera.pixels_per_world == pytest.approx(ppw, abs=1e-12)


def test_from_limits_contains_rect():
    camera = Camera.from_limits((-8.0, 1.0), (-4.0, 4.0), (1200, 800))
    x0, x1, y0, y1 = camera.view_rect
    assert x0 <= -8.0 and x1 >= 1.0 and y0 <= -4.0 and y1 >= 4.0


def test_quad_tessellation_layout():
    segments = np.array([[0.0, 0.0, 1.0, 0.0], [2.0, 2.0, 3.0, 4.0]])
    starts, ends, corners = segment_quads(segments)
    assert starts.shape == (8, 2) and ends.shape == (8, 2) and corners.shape == (8, 2)
    assert np.all(starts[:4] == [0.0, 0.0])
    assert np.all(ends[4:] == [3.0, 4.0])
    indices = quad_indices(2)
    assert indices.tolist() == [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]


def test_auto_exposure_targets_percentile():
    accum = np.zeros((10, 10, 4))
    accum[:5, :, 0] = 2.0  # half the pixels carry energy 2.0
    exposure = auto_exposure(accum, percentile=99.0)
    # Exponential map: 1 - exp(-exposure * 2.0) ~= 0.98
    assert 1.0 - np.exp(-exposure * 2.0) == pytest.approx(0.98, abs=1e-6)


def test_tone_map_monotone_bounded():
    x = np.linspace(0.0, 50.0, 1000)
    mapped = 1.0 - np.exp(-x)
    assert np.all(np.diff(mapped) >= 0.0)
    # Bounded to [0, 1]; 1.0 is reached only asymptotically (float rounding).
    assert mapped.min() >= 0.0 and mapped.max() <= 1.0


def test_wavelength_to_rgb_sanity():
    from raytracer.viz.color import wavelength_to_rgb

    red = wavelength_to_rgb(650.0)
    green = wavelength_to_rgb(540.0)
    blue = wavelength_to_rgb(460.0)
    assert red.argmax() == 0
    assert green.argmax() == 1
    assert blue.argmax() == 2
    assert np.all(red >= 0.0) and np.all(red <= 1.0)
