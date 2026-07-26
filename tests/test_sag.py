"""Sag/slope ground truths for the shared asphere profile."""

import numpy as np
import pytest

from raytracer.shapes.profile import AsphereProfile


def test_plane_is_zero():
    plane = AsphereProfile.plane()
    h = np.linspace(0, 100, 11)
    sag, slope = plane.sag_and_slope(h)
    assert np.all(sag == 0.0)
    assert np.all(slope == 0.0)


def test_sphere_sag_matches_circle_equation():
    R = -200.0
    profile = AsphereProfile.sphere(R)
    h = np.linspace(0.0, 50.0, 101)
    sag = profile.sag(h)
    # Exact circle: z = R - sign(R)*sqrt(R^2 - h^2) with vertex at z=0.
    exact = R + np.sign(-R) * np.sqrt(R * R - h * h)
    assert sag == pytest.approx(exact, abs=1e-12)


def test_slope_matches_finite_difference():
    profile = AsphereProfile.from_radius(
        -236.15461,
        conic=0.0,
        coefficients=(-1.163364e-7, -9.570513e-13, 4.236584e-17,
                      -7.733258e-22, 2.136062e-26, 1.705507e-30),
    )
    h = np.linspace(1.0, 75.0, 200)
    _, slope = profile.sag_and_slope(h)
    eps = 1e-6
    numeric = (profile.sag(h + eps) - profile.sag(h - eps)) / (2 * eps)
    assert slope == pytest.approx(numeric, rel=1e-7)


def test_conic_slope_matches_finite_difference():
    # EUV M2-like surface: strong conic constant.
    profile = AsphereProfile.from_radius(
        -2742.4833,
        conic=-207.7484,
        coefficients=(1.0280e-09, 2.5446e-14, 6.3412e-18, -2.2982e-21, 3.2649e-25),
    )
    h = np.linspace(1.0, 120.0, 200)
    _, slope = profile.sag_and_slope(h)
    eps = 1e-5
    numeric = (profile.sag(h + eps) - profile.sag(h - eps)) / (2 * eps)
    assert slope == pytest.approx(numeric, rel=1e-6)


def test_parabola_sag_is_h2_over_2R():
    R = -2000.0
    profile = AsphereProfile.from_radius(R, conic=-1.0)
    h = np.linspace(0.0, 300.0, 50)
    # For K=-1 the conic sag reduces exactly to c h^2 / 2.
    assert profile.sag(h) == pytest.approx(h * h / (2 * R), abs=1e-12)
