"""Apertures: regions of a surface's local (x, y) plane (``raytracer.geometry.apertures``)."""

import numpy as np
import pytest

from raytracer.geometry.apertures import Annulus, Aperture, Circle, Polygon, Rectangle


def test_circle_contains_and_slack():
    c = Circle(2.0)
    assert c.contains(np.array([0.0, 2.0, 2.0 + 1e-9, 1.5]), np.array([0.0, 0.0, 0.0, 1.5])).tolist() == [
        True, True, False, False,
    ]
    assert c.contains(2.0 + 1e-9, 0.0, slack=1e-8)
    assert c.extent == 2.0


def test_decentred_circle():
    c = Circle(1.0, center=(3.0, 4.0))
    assert c.contains(3.0, 4.9) and not c.contains(0.0, 0.0)
    assert c.extent == pytest.approx(6.0)
    np.testing.assert_allclose(np.hypot(*(c.boundary(64) - [3.0, 4.0]).T), 1.0)


def test_annulus_excludes_its_hole():
    a = Annulus(1.0, 2.0)
    r = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    assert a.contains(r, np.zeros_like(r)).tolist() == [False, True, True, True, False]
    assert a.extent == 2.0
    with pytest.raises(ValueError):
        Annulus(2.0, 1.0)


def test_rectangle_is_named_by_width_and_height():
    r = Rectangle(4.0, 2.0)
    assert r.contains(1.9, 0.9) and not r.contains(2.1, 0.0) and not r.contains(0.0, 1.1)
    assert r.extent == pytest.approx(np.hypot(2.0, 1.0))
    assert r.boundary().shape == (5, 2)
    off = Rectangle(2.0, 2.0, center=(5.0, 0.0))
    assert off.contains(5.5, 0.5) and not off.contains(0.0, 0.0)


def test_polygon_contains_either_orientation():
    triangle = [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]
    for vertices in (triangle, triangle[::-1]):
        p = Polygon(vertices)
        assert p.contains(1.0, 1.0)
        assert not p.contains(3.0, 2.0)
    p = Polygon(triangle)
    # Slack grows the region by a distance from the edges.
    assert not p.contains(2.0, -0.05) and p.contains(2.0, -0.05, slack=0.1)
    assert p.extent == pytest.approx(4.0)
    with pytest.raises(ValueError):
        Polygon([(0.0, 0.0), (1.0, 0.0)])


def test_square_polygon_equals_rectangle():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-2, 2, 500), rng.uniform(-2, 2, 500)
    square = Polygon([(-1.0, -0.5), (1.0, -0.5), (1.0, 0.5), (-1.0, 0.5)])
    np.testing.assert_array_equal(square.contains(x, y), Rectangle(2.0, 1.0).contains(x, y))


def test_coerce_reads_a_number_as_a_circle_radius():
    assert Aperture.coerce(3.0) == Circle(3.0)
    assert Aperture.coerce(None) is None
    r = Rectangle(1.0, 2.0)
    assert Aperture.coerce(r) is r
    with pytest.raises(TypeError):
        Aperture.coerce("big")


def test_apertures_are_values():
    assert Circle(1.0) == Circle(1.0) and hash(Circle(1.0)) == hash(Circle(1.0))
    assert Circle(1.0) != Circle(1.0, center=(0.1, 0.0))
    assert Rectangle(1.0, 2.0) != Rectangle(2.0, 1.0)
    with pytest.raises(ValueError):
        Circle(0.0)
    with pytest.raises(ValueError):
        Rectangle(1.0, -1.0)
