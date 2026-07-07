"""Analytic ground truths for reflection, refraction, and Fresnel laws."""

import numpy as np
import pytest

from raytracer.core.physics import fresnel_coefficients, reflect, refract, snell


def test_snell_angles_match_analytic():
    n1, n2 = 1.0, 1.5
    theta_i = np.deg2rad(30.0)
    direction = np.array([np.sin(theta_i), -np.cos(theta_i)])
    normal = np.array([0.0, 1.0])

    result = snell(direction, normal, n1, n2)
    theta_t_expected = np.arcsin(n1 / n2 * np.sin(theta_i))
    assert result.cos_incident == pytest.approx(np.cos(theta_i), abs=1e-12)
    assert result.cos_transmitted == pytest.approx(np.cos(theta_t_expected), abs=1e-12)


def test_refraction_direction_is_unit_and_in_plane():
    n1, n2 = 1.0, 1.5
    direction = np.array([0.6, -0.8])
    normal = np.array([0.0, 1.0])

    out = refract(direction, normal, n1, n2)
    assert out is not None
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-12)
    # Tangential component scales by n1/n2 (Snell in vector form).
    assert out[0] == pytest.approx(direction[0] * n1 / n2, abs=1e-12)


def test_tir_onset_angle_exact():
    n1, n2 = 1.5, 1.0
    theta_c = np.arcsin(n2 / n1)
    normal = np.array([0.0, 1.0])

    just_below = theta_c - 1e-9
    just_above = theta_c + 1e-9
    d_below = np.array([np.sin(just_below), -np.cos(just_below)])
    d_above = np.array([np.sin(just_above), -np.cos(just_above)])

    assert refract(d_below, normal, n1, n2) is not None
    assert refract(d_above, normal, n1, n2) is None


def test_reflection_preserves_angle():
    direction = np.array([0.6, -0.8])
    normal = np.array([0.0, 1.0])
    out = reflect(direction, normal)
    assert out == pytest.approx(np.array([0.6, 0.8]), abs=1e-12)


def test_reflection_3d():
    direction = np.array([1.0, 0.0, -1.0]) / np.sqrt(2)
    normal = np.array([0.0, 0.0, 1.0])
    out = reflect(direction, normal)
    assert out == pytest.approx(np.array([1.0, 0.0, 1.0]) / np.sqrt(2), abs=1e-12)


def test_fresnel_energy_conservation():
    direction = np.array([0.6, -0.8])
    normal = np.array([0.0, 1.0])
    coeffs = fresnel_coefficients(direction, normal, 1.0, 1.5)
    assert coeffs.r_s + coeffs.t_s == pytest.approx(1.0, abs=1e-12)
    assert coeffs.r_p + coeffs.t_p == pytest.approx(1.0, abs=1e-12)


def test_fresnel_normal_incidence():
    direction = np.array([0.0, -1.0])
    normal = np.array([0.0, 1.0])
    n1, n2 = 1.0, 1.5
    coeffs = fresnel_coefficients(direction, normal, n1, n2)
    expected = ((n1 - n2) / (n1 + n2)) ** 2
    assert coeffs.r_s == pytest.approx(expected, abs=1e-12)
    assert coeffs.r_p == pytest.approx(expected, abs=1e-12)


def test_fresnel_brewster_angle():
    n1, n2 = 1.0, 1.5
    theta_b = np.arctan(n2 / n1)
    direction = np.array([np.sin(theta_b), -np.cos(theta_b)])
    normal = np.array([0.0, 1.0])
    coeffs = fresnel_coefficients(direction, normal, n1, n2)
    assert coeffs.r_p == pytest.approx(0.0, abs=1e-12)


def test_fresnel_tir_reflects_everything():
    n1, n2 = 1.5, 1.0
    theta = np.arcsin(n2 / n1) + 0.05
    direction = np.array([np.sin(theta), -np.cos(theta)])
    normal = np.array([0.0, 1.0])
    coeffs = fresnel_coefficients(direction, normal, n1, n2)
    assert coeffs.reflectance == 1.0
    assert coeffs.transmittance == 0.0
