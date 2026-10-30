"""Sag/slope ground truths for the surfaces of revolution.

(Was ``test_profile.py``: the shared ``AsphereProfile`` became the surface
hierarchy of :mod:`raytracer.geometry.surfaces`, one class per family.)
"""

import numpy as np
import pytest

from raytracer.geometry.surfaces import Conic, EvenAsphere, Paraboloid, Plane, Sphere
from raytracer.io.prescription_csv import surface_from_columns


def _even(coefficients) -> dict[int, float]:
    """The prescription columns C1..C6 are the r^4..r^14 terms."""

    return dict(zip((4, 6, 8, 10, 12, 14), coefficients))


def test_plane_is_zero():
    plane = Plane()
    h = np.linspace(0, 100, 11)
    sag, slope = plane.sag_and_slope(h)
    assert np.all(sag == 0.0)
    assert np.all(slope == 0.0)
    assert plane.is_plane and plane.radius == np.inf and plane.curvature == 0.0


def test_sphere_sag_matches_circle_equation():
    R = -200.0
    sphere = Sphere(R)
    h = np.linspace(0.0, 50.0, 101)
    sag = sphere.profile(h)
    # Exact circle: z = R - sign(R)*sqrt(R^2 - h^2) with vertex at z=0.
    exact = R + np.sign(-R) * np.sqrt(R * R - h * h)
    assert sag == pytest.approx(exact, abs=1e-12)
    # The 3-D sag is the same circle in every azimuth.
    phi = np.linspace(0.0, 2 * np.pi, 101)
    assert sphere.sag(h * np.cos(phi), h * np.sin(phi)) == pytest.approx(exact, abs=1e-12)


def test_slope_matches_finite_difference():
    surface = EvenAsphere(
        -236.15461,
        0.0,
        _even((-1.163364e-7, -9.570513e-13, 4.236584e-17, -7.733258e-22, 2.136062e-26, 1.705507e-30)),
    )
    h = np.linspace(1.0, 75.0, 200)
    _, slope = surface.sag_and_slope(h)
    eps = 1e-6
    numeric = (surface.profile(h + eps) - surface.profile(h - eps)) / (2 * eps)
    assert slope == pytest.approx(numeric, rel=1e-7)


def test_conic_slope_matches_finite_difference():
    # EUV M2-like surface: strong conic constant.
    surface = EvenAsphere(
        -2742.4833,
        -207.7484,
        _even((1.0280e-09, 2.5446e-14, 6.3412e-18, -2.2982e-21, 3.2649e-25)),
    )
    h = np.linspace(1.0, 120.0, 200)
    _, slope = surface.sag_and_slope(h)
    eps = 1e-5
    numeric = (surface.profile(h + eps) - surface.profile(h - eps)) / (2 * eps)
    assert slope == pytest.approx(numeric, rel=1e-6)


def test_parabola_sag_is_h2_over_2R():
    R = -2000.0
    parabola = Paraboloid(R)
    assert parabola == Conic(R, -1.0).classify()
    assert parabola.f == R / 2.0
    h = np.linspace(0.0, 300.0, 50)
    # For K=-1 the conic sag reduces exactly to c h^2 / 2.
    assert parabola.profile(h) == pytest.approx(h * h / (2 * R), abs=1e-12)


def test_asphere_base_is_its_conic():
    surface = EvenAsphere(-150.0, -0.5, {4: 1e-7})
    assert surface.base == Conic(-150.0, -0.5).classify()
    h = np.linspace(0.0, 20.0, 11)
    assert surface.profile(h) - surface.base.profile(h) == pytest.approx(1e-7 * h**4, abs=1e-15)
    # An r^2 term is part of the definition and adds 2 a_2 to the vertex curvature.
    assert EvenAsphere(100.0, 0.0, {2: 1e-3}).curvature == pytest.approx(0.01 + 2e-3)
    with pytest.raises(ValueError):
        EvenAsphere(100.0, 0.0, {3: 1e-3})


def test_prescription_columns_pick_the_most_specific_surface():
    assert surface_from_columns(0.0, 0.0, {}) == Plane()
    assert surface_from_columns(80.0, 0.0, {}) == Sphere(80.0)
    assert surface_from_columns(80.0, -1.0, {}) == Paraboloid(80.0)
    assert type(surface_from_columns(80.0, -2.0, {})).__name__ == "Hyperboloid"
    assert type(surface_from_columns(80.0, 0.3, {})).__name__ == "Ellipsoid"
    asphere = surface_from_columns(80.0, 0.0, {4: 1e-6, 6: 0.0})
    assert isinstance(asphere, EvenAsphere) and asphere.coefficients == {4: 1e-6}
