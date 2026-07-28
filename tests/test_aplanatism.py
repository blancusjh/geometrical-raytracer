"""Aplanatism condition (``raytracer.analysis.aberrations.aplanatism``).

Two independent routes to the same physics are held against each other:
the closed-form map M of Silva-Lora & Torres (Eqs. 24-25), evaluated from
the shape parameters at the traced hit points, and the sine ratio measured
directly on the exact traced rays. Their identity
``sin u_0 / sin u_N = (n_N/n_0) g_t M`` is asserted at the 1e-9 level, and
the mirror-symmetric singlet provides an exact aplanat (M = 1 to machine
precision) as the analytic truth.
"""

import numpy as np
import pytest

from raytracer.analysis import (
    aplanatic_image_surface,
    aplanatism_map,
    aplanatism_report,
    normal_axis_crossing,
)
from raytracer.analysis.aberrations.aplanatism import _inverse_vc_geometric
from raytracer.design import StigmaticTrain
from raytracer.propagation import PupilSampling
from raytracer.surfaces.cartesian_oval import CartesianOvalProfile

INF = float("inf")


# -- V_k C_k (Eq. 24) --------------------------------------------------------


def test_normal_axis_crossing_tends_to_the_paraxial_radius():
    profile = CartesianOvalProfile(n0=1.0, z0=-45.0, ni=1.62, zi=33.0)
    vc = normal_axis_crossing(profile, np.array([1e-9]))
    assert vc[0] == pytest.approx(1.0 / profile.O, rel=1e-9)


@pytest.mark.parametrize(
    "profile",
    [
        CartesianOvalProfile(n0=1.0, z0=-45.0, ni=1.62, zi=33.0),
        CartesianOvalProfile(n0=1.5, z0=-12.0, ni=1.0, zi=80.0),
        CartesianOvalProfile(n0=1.0, z0=-60.0, ni=1.5, zi=INF),  # collimator: S = 0
    ],
)
def test_eq24_equals_the_geometric_normal_crossing(profile):
    """The paper's Eq. (24) and the direct geometric construction (normal at
    the hit point extended to the axis) are the same quantity."""

    h = np.linspace(1e-3, 0.6 * min(profile.max_usable_height, 20.0), 30)
    z, _ = profile.sag_and_slope(h)
    rho = np.sqrt(h * h + z * z)
    vc_eq24 = normal_axis_crossing(profile, rho)
    vc_geometric = 1.0 / _inverse_vc_geometric(profile, h)
    np.testing.assert_allclose(vc_eq24, vc_geometric, rtol=1e-10)


def test_eq24_refuses_its_singular_cases():
    plane = CartesianOvalProfile(n0=1.5, z0=INF, ni=1.0, zi=-INF)
    with pytest.raises(ValueError, match="singular"):
        normal_axis_crossing(plane, np.array([1.0]))


def test_aplanatism_map_validates_its_shape():
    train = StigmaticTrain.symmetric_singlet(n=1.5, d0=-30.0, thickness=40.0)
    with pytest.raises(ValueError, match="per surface"):
        aplanatism_map(train, np.ones((5, 3)))


# -- the exact aplanat -------------------------------------------------------


@pytest.fixture(scope="module")
def symmetric():
    return StigmaticTrain.symmetric_singlet(
        n=1.5, d0=-30.0, thickness=40.0, semidiameter=3.5
    )


@pytest.fixture(scope="module")
def asymmetric():
    """Same end conjugates and thickness as ``symmetric``; only the
    intermediate conjugate moves. Stigmatic, but not aplanatic."""

    return StigmaticTrain.singlet(
        n=1.5, d0=-30.0, d1=300.0, d2=70.0, thickness=40.0, semidiameter=3.5
    )


def test_symmetric_singlet_is_exactly_aplanatic(symmetric):
    report = aplanatism_report(symmetric, na_object_sine=0.10, samples=15)
    assert report.valid.all()
    assert report.map_rms < 1e-12
    assert report.is_aplanatic(tol=1e-10)
    assert report.gt == pytest.approx(1.0)
    np.testing.assert_allclose(report.sine_ratio[report.valid], 1.0, atol=1e-12)


def test_asymmetric_singlet_is_not_aplanatic(asymmetric):
    report = aplanatism_report(asymmetric, na_object_sine=0.10, samples=15)
    assert report.valid.all()
    assert report.map_rms > 1e-4
    assert not report.is_aplanatic()


def test_closed_form_matches_the_traced_sine_ratio(asymmetric):
    """sin u_0 / sin u_N = (n_N/n_0) g_t M, per ray, formula vs exact trace."""

    report = aplanatism_report(asymmetric, na_object_sine=0.10, samples=21)
    assert report.valid.all()
    assert np.max(np.abs(report.formula_residual[report.valid])) < 1e-9
    # And the two aggregate figures are the same number when the identity holds.
    assert report.offense_rms == pytest.approx(report.map_rms, rel=1e-6)


def test_summary_names_the_verdict(symmetric):
    text = aplanatism_report(symmetric, na_object_sine=0.08, samples=9).summary()
    assert "aplanatic" in text and "NOT" not in text


# -- the aplanatic image surface ---------------------------------------------


def test_axial_image_point_is_the_design_conjugate(symmetric):
    surface = aplanatic_image_surface(
        symmetric, [0.0, 0.5], na_object_sine=0.08
    )
    assert surface.valid.all()
    assert surface.points[0, 2] == pytest.approx(symmetric.conjugates[-1], abs=1e-9)
    assert surface.blur_rms_um[0] < 1e-9
    assert surface.sagitta[0] == 0.0


def test_aplanat_blur_grows_quadratically_where_coma_would_be_linear(
    symmetric, asymmetric
):
    """Coma is the field-linear blur; the sine condition removes it. At the
    per-field best-focus point the aplanat's residual (astigmatism) scales as
    h^2 while the plain stigmatic singlet's coma scales as h."""

    heights = np.array([0.0, 0.05, 0.1])
    sampling = PupilSampling(kind="rings", radial=6, azimuth=24)

    blur_aplanat = aplanatic_image_surface(
        symmetric, heights, na_object_sine=0.08, sampling=sampling
    ).blur_rms_um
    blur_stigmatic = aplanatic_image_surface(
        asymmetric, heights, na_object_sine=0.08, sampling=sampling
    ).blur_rms_um

    assert blur_aplanat[2] / blur_aplanat[1] == pytest.approx(4.0, abs=0.3)
    assert blur_stigmatic[2] / blur_stigmatic[1] == pytest.approx(2.0, abs=0.2)
