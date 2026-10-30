"""Analytic ground truths for reflection, refraction, and Fresnel laws.

The laws live in :mod:`raytracer.interfaces` (vector form, batches of 3-D
rays) and the Fresnel power split in :mod:`raytracer.coatings`
(``Uncoated``). The plane of incidence here is the x-z plane, with the
surface normal along z.
"""

import numpy as np
import pytest

from raytracer.coatings import AR, IdealBeamSplitter, PerfectMirror, Uncoated
from raytracer.interfaces import reflect, refract

UP = np.array([[0.0, 0.0, 1.0]])  # normal facing a ray that travels toward -z


def incoming(theta_rad) -> np.ndarray:
    """Unit directions in the x-z plane, travelling toward -z at angle theta from the normal."""

    theta = np.atleast_1d(theta_rad)
    return np.column_stack([np.sin(theta), np.zeros_like(theta), -np.cos(theta)])


def test_refraction_angle_matches_analytic_snell():
    n1, n2 = 1.0, 1.5
    theta_i = np.deg2rad(30.0)
    out, tir = refract(incoming(theta_i), UP, n1, n2)
    assert not tir.any()
    theta_t = np.arcsin(n1 / n2 * np.sin(theta_i))
    assert out[0] == pytest.approx(incoming(theta_t)[0], abs=1e-12)


def test_refraction_direction_is_unit_and_in_plane():
    n1, n2 = 1.0, 1.5
    direction = np.array([[0.6, 0.0, -0.8]])
    out, _ = refract(direction, UP, n1, n2)
    assert np.linalg.norm(out[0]) == pytest.approx(1.0, abs=1e-12)
    assert out[0, 1] == 0.0  # stays in the plane of incidence
    # Tangential component scales by n1/n2 (the vector form of the refraction law).
    assert out[0, 0] == pytest.approx(direction[0, 0] * n1 / n2, abs=1e-12)


def test_refraction_in_an_oblique_plane_of_incidence():
    """3-D: the transmitted ray stays in the plane of the incident ray and the normal."""

    d = np.array([[0.3, -0.4, -np.sqrt(1 - 0.25)]])
    out, _ = refract(d, UP, 1.0, 1.7)
    plane_normal = np.cross(d[0], UP[0])
    assert out[0] @ plane_normal == pytest.approx(0.0, abs=1e-15)
    sin_i = np.linalg.norm(np.cross(d[0], UP[0]))
    sin_t = np.linalg.norm(np.cross(out[0], UP[0]))
    assert 1.0 * sin_i == pytest.approx(1.7 * sin_t, abs=1e-14)


def test_tir_onset_angle_exact():
    n1, n2 = 1.5, 1.0
    theta_c = np.arcsin(n2 / n1)
    _, tir = refract(incoming([theta_c - 1e-9, theta_c + 1e-9]), np.repeat(UP, 2, axis=0), n1, n2)
    assert not tir[0]
    assert tir[1]


def test_reflection_preserves_angle():
    out = reflect(np.array([[0.6, 0.0, -0.8]]), UP)
    assert out[0] == pytest.approx([0.6, 0.0, 0.8], abs=1e-12)


def test_reflection_3d():
    direction = np.array([[1.0, 0.0, -1.0]]) / np.sqrt(2)
    out = reflect(direction, UP)
    assert out[0] == pytest.approx(np.array([1.0, 0.0, 1.0]) / np.sqrt(2), abs=1e-12)


def test_reflection_ignores_the_normal_orientation():
    d = np.array([[0.2, -0.3, -np.sqrt(1 - 0.13)]])
    assert reflect(d, UP) == pytest.approx(reflect(d, -UP), abs=1e-15)


def test_batch_rows_are_independent():
    """Each row of a batch is the same as tracing that ray alone."""

    n1, n2 = 1.0, 1.5
    thetas = np.deg2rad([5.0, 20.0, 40.0])
    directions = incoming(thetas)
    normals = np.repeat(UP, len(thetas), axis=0)
    batch, tir = refract(directions, normals, n1, n2)
    assert not tir.any()
    for d, o in zip(directions, batch):
        single, _ = refract(d[None, :], UP, n1, n2)
        assert o == pytest.approx(single[0], abs=1e-15)
    reflected = reflect(directions, normals)
    for d, o in zip(directions, reflected):
        assert o == pytest.approx(reflect(d[None, :], UP)[0], abs=1e-15)


def test_refract_batch_flags_tir():
    n1, n2 = 1.5, 1.0
    theta_c = np.arcsin(n2 / n1)
    _, tir = refract(incoming([theta_c - 1e-3, theta_c + 1e-3]), np.repeat(UP, 2, axis=0), n1, n2)
    assert not tir[0]
    assert tir[1]


def test_fresnel_energy_conservation():
    coeffs = Uncoated().power(np.array([0.8]), 1.0, 1.5)
    assert coeffs.Rs[0] + coeffs.Ts[0] == pytest.approx(1.0, abs=1e-12)
    assert coeffs.Rp[0] + coeffs.Tp[0] == pytest.approx(1.0, abs=1e-12)
    assert coeffs.R[0] + coeffs.T[0] == pytest.approx(1.0, abs=1e-12)


def test_fresnel_normal_incidence():
    n1, n2 = 1.0, 1.5
    coeffs = Uncoated().power(np.array([1.0]), n1, n2)
    expected = ((n1 - n2) / (n1 + n2)) ** 2
    assert coeffs.Rs[0] == pytest.approx(expected, abs=1e-12)
    assert coeffs.Rp[0] == pytest.approx(expected, abs=1e-12)


def test_fresnel_brewster_angle():
    n1, n2 = 1.0, 1.5
    theta_b = np.arctan(n2 / n1)
    coeffs = Uncoated().power(np.array([np.cos(theta_b)]), n1, n2)
    assert coeffs.Rp[0] == pytest.approx(0.0, abs=1e-12)
    assert coeffs.Rs[0] > 0.1


def test_fresnel_is_reciprocal():
    """Stokes: the same pair of angles gives the same reflectance from either side."""

    theta_i = np.deg2rad(25.0)
    theta_t = np.arcsin(np.sin(theta_i) / 1.5)
    there = Uncoated().power(np.array([np.cos(theta_i)]), 1.0, 1.5)
    back = Uncoated().power(np.array([np.cos(theta_t)]), 1.5, 1.0)
    assert there.Rs[0] == pytest.approx(back.Rs[0], abs=1e-14)
    assert there.Rp[0] == pytest.approx(back.Rp[0], abs=1e-14)


def test_fresnel_tir_reflects_everything():
    n1, n2 = 1.5, 1.0
    theta = np.arcsin(n2 / n1) + 0.05
    coeffs = Uncoated().power(np.array([np.cos(theta)]), n1, n2)
    assert coeffs.R[0] == 1.0
    assert coeffs.T[0] == 0.0


def test_coatings_split_power_as_named():
    cos_i = np.array([1.0, 0.7])
    assert AR().power(cos_i, 1.0, 1.5).T == pytest.approx([1.0, 1.0])
    assert PerfectMirror().power(cos_i, 1.0, 1.5).R == pytest.approx([1.0, 1.0])
    split = IdealBeamSplitter(0.3, 0.6).power(cos_i, 1.0, 1.0)
    assert split.R == pytest.approx([0.3, 0.3])
    assert split.T == pytest.approx([0.6, 0.6])
    # The sequential default each coating states.
    assert Uncoated().sequential_event == "transmit"
    assert PerfectMirror().sequential_event == "reflect"
    assert IdealBeamSplitter().sequential_event is None
    with pytest.raises(ValueError):
        IdealBeamSplitter(0.7, 0.7)
