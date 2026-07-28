"""Stigmatic ovoid lens trains (``raytracer.design.stigmatic``).

The train is stigmatic *by construction* for any intermediate conjugates, so
the tests here are analytic identities: infinite-conjugate GOTS limits, the
mirror symmetry of swapped conjugates, the telescoping magnification, and
end-to-end stigmatism of traced bundles at the 1e-9 um level.
"""

import numpy as np
import pytest

from raytracer.analysis import spot_data
from raytracer.design import StigmaticTrain
from raytracer.propagation import (
    FieldPoint,
    PupilSampling,
    SequentialTracer,
    trace_from_object,
    trace_pupil,
)
from raytracer.surfaces.cartesian_oval import CartesianOvalProfile, gots_params

INF = float("inf")


# -- GOTS limits for infinite conjugates ------------------------------------


def test_gots_image_at_infinity_matches_the_finite_limit():
    n0, z0, ni = 1.0, -60.0, 1.5
    limit = gots_params(n0, z0, ni, INF)
    far = gots_params(n0, z0, ni, 1e10)
    np.testing.assert_allclose(limit, far, rtol=1e-6, atol=1e-10)
    G, O, T, S = limit
    assert O == pytest.approx(-n0 / (z0 * (ni - n0)))
    assert G == pytest.approx(-(ni**2) / n0**2)
    assert T == 0.0 and S == 0.0


def test_gots_object_at_infinity_matches_the_finite_limit():
    n0, ni, zi = 1.0, 1.5, 40.0
    limit = gots_params(n0, -INF, ni, zi)
    far = gots_params(n0, -1e10, ni, zi)
    np.testing.assert_allclose(limit, far, rtol=1e-6, atol=1e-10)
    _, O, T, S = limit
    assert O == pytest.approx(ni / (zi * (ni - n0)))
    assert T == 0.0 and S == 0.0


def test_gots_doubly_collimated_is_the_plane():
    assert gots_params(1.5, INF, 1.0, -INF) == (0.0, 0.0, 0.0, 0.0)
    profile = CartesianOvalProfile(n0=1.5, z0=INF, ni=1.0, zi=-INF)
    assert profile.is_plane
    np.testing.assert_allclose(profile.sag(np.array([0.0, 2.0, 5.0])), 0.0)


def test_collimating_surface_emits_parallel_rays():
    """Object at -60 into glass with the image conjugate at infinity: every
    emergent ray direction is parallel to the axis, at any real aperture."""

    from raytracer.design import OpticalSystem, SurfaceRow
    from raytracer.optics.materials import AIR, ConstantIndex

    profile = CartesianOvalProfile(n0=1.0, z0=-60.0, ni=1.5, zi=INF)
    row = SurfaceRow(
        profile=profile, thickness=30.0, material_after=ConstantIndex("N1.5", 1.5)
    )
    system = OpticalSystem([row], object_space=AIR, object_z=-60.0)
    tracer = SequentialTracer(system)
    for sine in (0.05, 0.2, 0.35):
        slope = sine / np.sqrt(1.0 - sine * sine)
        result = trace_from_object(tracer, (0.0, 0.0), (0.0, slope))
        assert result.ok
        d = result.direction / np.linalg.norm(result.direction)
        assert abs(d[1]) < 1e-12


# -- mirror identity ---------------------------------------------------------


def test_mirrored_conjugates_negate_the_sag():
    """The surface imaging (n1, -z1) -> (n0, -z0) is the mirror image of the
    one imaging (n0, z0) -> (n1, z1): its sag is the negated sag."""

    forward = CartesianOvalProfile(n0=1.0, z0=-30.0, ni=1.5, zi=20.0)
    mirrored = CartesianOvalProfile(n0=1.5, z0=-20.0, ni=1.0, zi=30.0)
    h = np.linspace(0.0, 3.0, 40)
    np.testing.assert_allclose(mirrored.sag(h), -forward.sag(h), atol=1e-12)


# -- validation --------------------------------------------------------------


def test_rejects_mismatched_lengths_and_bad_vertices():
    with pytest.raises(ValueError, match="indices"):
        StigmaticTrain(indices=(1.0, 1.5), vertices=(0.0, 5.0), conjugates=(-10.0, 3.0, 20.0))
    with pytest.raises(ValueError, match="conjugates"):
        StigmaticTrain(indices=(1.0, 1.5, 1.0), vertices=(0.0, 5.0), conjugates=(-10.0, 20.0))
    with pytest.raises(ValueError, match="increasing"):
        StigmaticTrain(indices=(1.0, 1.5, 1.0), vertices=(0.0, -5.0), conjugates=(-10.0, 3.0, 20.0))
    with pytest.raises(ValueError, match="ζ_0 = 0"):
        StigmaticTrain(indices=(1.0, 1.5, 1.0), vertices=(1.0, 5.0), conjugates=(-10.0, 3.0, 20.0))
    with pytest.raises(ValueError, match="finite"):
        StigmaticTrain(indices=(1.0, 1.5, 1.0), vertices=(0.0, 5.0), conjugates=(-INF, 3.0, 20.0))
    with pytest.raises(ValueError, match="index step"):
        StigmaticTrain(indices=(1.0, 1.0, 1.5), vertices=(0.0, 5.0), conjugates=(-10.0, 3.0, 20.0))
    with pytest.raises(ValueError, match="singular"):
        StigmaticTrain(indices=(1.0, 1.5, 1.0), vertices=(0.0, 5.0), conjugates=(-10.0, 5.0, 20.0))


# -- magnification -----------------------------------------------------------


def test_gt_matches_the_product_of_surface_magnifications():
    train = StigmaticTrain.singlet(n=1.5, d0=-60.0, d1=300.0, d2=60.0, thickness=8.0)
    g = train.surface_magnifications
    assert train.gt == pytest.approx(g[0] * g[1])


def test_gt_with_a_collimated_interior_pairs_the_infinities():
    """d1 = inf appears once as a numerator and once as a denominator; the
    limit of the pair is 1, so gt stays finite and equals the value at a
    huge finite stand-in."""

    train = StigmaticTrain(
        indices=(1.0, 1.5, 1.0), vertices=(0.0, 8.0), conjugates=(-60.0, INF, 70.0)
    )
    nearly = StigmaticTrain(
        indices=(1.0, 1.5, 1.0), vertices=(0.0, 8.0), conjugates=(-60.0, 1e12, 70.0)
    )
    assert np.isfinite(train.gt)
    assert train.gt == pytest.approx(nearly.gt, rel=1e-9)


def test_symmetric_singlet_has_unit_magnitude_magnification():
    train = StigmaticTrain.symmetric_singlet(n=1.5, d0=-30.0, thickness=40.0)
    assert train.conjugates == (-30.0, 20.0, 70.0)
    assert abs(train.gt) == pytest.approx(1.0)


# -- end-to-end stigmatism ---------------------------------------------------


def _rms_spot_um(train: StigmaticTrain, *, na: float) -> float:
    tracer = SequentialTracer(train.to_system())
    pupil = trace_pupil(
        tracer,
        FieldPoint(y=0.0),
        na_object_sine=na,
        sampling=PupilSampling(kind="rings", radial=8, azimuth=32),
        chief_slope=0.0,
    )
    assert pupil.valid.sum() > 0.8 * pupil.valid.size
    return spot_data(pupil).rms_radius_um


def test_omega_singlet_is_stigmatic_for_any_intermediate_conjugate():
    for d1 in (300.0, -300.0, 120.0):
        train = StigmaticTrain.singlet(
            n=1.5, d0=-60.0, d1=d1, d2=60.0, thickness=8.0, semidiameter=10.0
        )
        assert _rms_spot_um(train, na=0.12) < 1e-9


def test_symmetric_singlet_is_stigmatic():
    train = StigmaticTrain.symmetric_singlet(
        n=1.5, d0=-30.0, thickness=40.0, semidiameter=3.0
    )
    assert train.max_semidiameter > 3.0
    assert _rms_spot_um(train, na=0.1) < 1e-9


def test_flat_interior_surface_train_is_stigmatic():
    """A plane can sit inside an exactly stigmatic train only in collimated
    light; with d1 = d2 = inf the middle surface degenerates to the plane and
    the train still images d0 to d3 stigmatically."""

    train = StigmaticTrain(
        indices=(1.0, 1.5, 1.7, 1.0),
        vertices=(0.0, 8.0, 12.0),
        conjugates=(-60.0, INF, INF, 70.0),
        semidiameter=6.0,
    )
    assert [p.is_plane for p in train.profiles] == [False, True, False]
    assert _rms_spot_um(train, na=0.08) < 1e-9
