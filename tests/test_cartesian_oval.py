"""Shared GOTS Cartesian-oval math (raytracer.shapes.cartesian_oval)."""

import numpy as np
import pytest

from raytracer.shapes.cartesian_oval import (
    CartesianOvalProfile,
    cartesian_oval_parametric_curve,
    cartesian_oval_sag_and_slope,
    gots_params,
    max_usable_height,
)
from raytracer.shapes.fermat_oval import fermat_oval_F

N0, Z0, NI, ZI = 1.0, -30.0, 1.7, 10.0


def test_parametric_curve_satisfies_fermat_condition():
    """(z(rho), r(rho)) must lie exactly on the two-point Fermat oval."""

    G, O, T, S = gots_params(N0, Z0, NI, ZI)
    rho = np.linspace(0.01, 5.0, 25)
    z, r = cartesian_oval_parametric_curve(rho, G, O, T, S)
    residual = [fermat_oval_F(zz, rr, Z0, ZI, N0, NI) for zz, rr in zip(z, r)]
    assert np.max(np.abs(residual)) < 1e-8


def test_sag_and_slope_matches_finite_difference():
    G, O, T, S = gots_params(N0, Z0, NI, ZI)
    h = np.linspace(0.0, 4.0, 20)
    _, dz = cartesian_oval_sag_and_slope(h, G, O, T, S)
    eps = 1e-6
    z_plus, _ = cartesian_oval_sag_and_slope(h + eps, G, O, T, S)
    z_minus, _ = cartesian_oval_sag_and_slope(np.maximum(h - eps, 0.0), G, O, T, S)
    denom = np.where(h < eps, eps, 2.0 * eps)
    finite_diff = (z_plus - z_minus) / denom
    assert np.max(np.abs(dz[1:] - finite_diff[1:])) < 1e-5


def test_sag_at_height_satisfies_fermat_condition():
    """sag_and_slope(h) inverts rho->r(rho) correctly: z(h) must also solve Fermat."""

    G, O, T, S = gots_params(N0, Z0, NI, ZI)
    h = np.linspace(0.0, 4.0, 15)
    z, _ = cartesian_oval_sag_and_slope(h, G, O, T, S)
    residual = [fermat_oval_F(zz, hh, Z0, ZI, N0, NI) for zz, hh in zip(z, h)]
    assert np.max(np.abs(residual)) < 1e-8


def test_max_usable_height_is_the_r_of_rho_peak():
    G, O, T, S = gots_params(N0, Z0, NI, ZI)
    h_max = max_usable_height(G, O, T, S)
    assert h_max > 0
    # Just inside the domain must invert cleanly; the derivative blows up at
    # the peak itself, so stay comfortably below it.
    z, dz = cartesian_oval_sag_and_slope(np.array([0.9 * h_max]), G, O, T, S)
    assert np.isfinite(z).all() and np.isfinite(dz).all()


def test_profile_curvature_matches_paraxial_limit():
    """Vertex curvature should equal O: z(h) ~ (O/2) h**2 near the axis."""

    profile = CartesianOvalProfile(n0=N0, z0=Z0, ni=NI, zi=ZI)
    h = 1e-4
    z = float(profile.sag(np.array([h]))[0])
    assert z == pytest.approx(0.5 * profile.curvature * h * h, rel=1e-3)
