"""Analytic ground truths for reflection, refraction, and Fresnel laws."""

import numpy as np
import pytest

from raytracer.physics.radiometry import fresnel_coefficients
from raytracer.physics.refraction import reflect, reflect_batch, refract, refract_batch


def test_refraction_angle_matches_analytic_snell():
    n1, n2 = 1.0, 1.5
    theta_i = np.deg2rad(30.0)
    direction = np.array([np.sin(theta_i), -np.cos(theta_i)])
    normal = np.array([0.0, 1.0])

    out = refract(direction, normal, n1, n2)
    theta_t_expected = np.arcsin(n1 / n2 * np.sin(theta_i))
    expected = np.array([np.sin(theta_t_expected), -np.cos(theta_t_expected)])
    assert out == pytest.approx(expected, abs=1e-12)


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


def test_reflect_batch_matches_scalar_reflect():
    directions = np.array([[0.6, -0.8], [1.0, 0.0], [-0.3, -0.95]])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    normal = np.array([0.0, 1.0])
    normals = np.tile(normal, (len(directions), 1))

    batch = reflect_batch(directions, normals)
    for d, out in zip(directions, batch):
        assert out == pytest.approx(reflect(d, normal), abs=1e-12)


def test_refract_batch_matches_scalar_refract():
    n1, n2 = 1.0, 1.5
    thetas = np.deg2rad([5.0, 20.0, 40.0])
    directions = np.column_stack([np.sin(thetas), -np.cos(thetas)])
    normal = np.array([0.0, 1.0])
    normals = np.tile(normal, (len(directions), 1))

    out, tir = refract_batch(directions, normals, n1, n2)
    assert not tir.any()
    for d, o in zip(directions, out):
        assert o == pytest.approx(refract(d, normal, n1, n2), abs=1e-12)


def test_refract_batch_flags_tir():
    n1, n2 = 1.5, 1.0
    theta_c = np.arcsin(n2 / n1)
    thetas = np.array([theta_c - 1e-3, theta_c + 1e-3])
    directions = np.column_stack([np.sin(thetas), -np.cos(thetas)])
    normal = np.array([0.0, 1.0])
    normals = np.tile(normal, (len(directions), 1))

    _, tir = refract_batch(directions, normals, n1, n2)
    assert not tir[0]
    assert tir[1]


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
