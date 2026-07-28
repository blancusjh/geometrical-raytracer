"""Aplanatism optimization (``raytracer.optimize.aplanat``).

The convergence test reproduces the paper's 4-surface cemented-triplet
configuration (Silva-Lora & Torres 2020, Fig. 3a: ζ = 0, 15, 25, 35 mm,
n = 1, 1.517122, 1.670591, 1.851280, 1, conjugates -100 -> 90 mm, 20 mm
aperture) and drives the three intermediate conjugates from a deliberately
poor start to the isoplanatic regime.
"""

import numpy as np
import pytest

from raytracer.analysis import aplanatic_image_surface
from raytracer.design import StigmaticTrain
from raytracer.optimize import aplanatism_residuals, optimize_aplanat

INF = float("inf")

CEMENTED = dict(
    indices=(1.0, 1.517122, 1.670591, 1.851280, 1.0),
    vertices=(0.0, 15.0, 25.0, 35.0),
    semidiameter=10.0,
)


def test_cemented_triplet_reaches_the_isoplanatic_regime():
    train = StigmaticTrain(conjugates=(-100.0, -150.0, -600.0, 450.0, 90.0), **CEMENTED)
    fit = optimize_aplanat(
        train,
        na_object_sine=0.195,
        samples=15,
        bounds=(-3000, 3000),
        diff_step=1e-4,
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
    )
    assert fit.after.map_rms < 3e-5
    assert fit.after.map_rms < fit.before.map_rms / 50.0
    # The closed form is measured on real rays; the offense agrees with it.
    assert fit.after.offense_rms == pytest.approx(fit.after.map_rms, rel=1e-3)


def test_all_infinite_intermediates_leave_nothing_to_vary():
    train = StigmaticTrain(
        indices=(1.0, 1.5, 1.0),
        vertices=(0.0, 8.0),
        conjugates=(-60.0, INF, 70.0),
    )
    with pytest.raises(ValueError, match="no finite intermediate"):
        optimize_aplanat(train, na_object_sine=0.1)


def test_residual_vector_has_stable_length_and_carries_the_sagitta():
    train = StigmaticTrain.symmetric_singlet(
        n=1.5, d0=-30.0, thickness=40.0, semidiameter=3.5
    )
    heights = [0.0, 0.8]
    residuals = aplanatism_residuals(
        train,
        na_object_sine=0.08,
        samples=9,
        field_heights=heights,
        flat_field_weight=2.0,
    )
    assert residuals.shape == (9 + 2,)
    surface = aplanatic_image_surface(train, heights, na_object_sine=0.08)
    np.testing.assert_allclose(residuals[9:], 2.0 * surface.sagitta, rtol=1e-9)
    # The exact aplanat's map residuals are numerically zero...
    assert np.max(np.abs(residuals[:9])) < 1e-12
    # ...while its image surface is genuinely curved: that is what the
    # flat-field term exists to trade against.
    assert abs(residuals[10]) > 0.1


def test_flat_field_weight_changes_the_optimum():
    """With a heavy flat-field term the optimizer must leave the plain
    aplanatism optimum: the two objectives are in tension."""

    train = StigmaticTrain(conjugates=(-100.0, -150.0, -600.0, 450.0, 90.0), **CEMENTED)
    plain = optimize_aplanat(
        train, na_object_sine=0.15, samples=9,
        bounds=(-3000, 3000), diff_step=1e-4, max_nfev=25,
    )
    flat = optimize_aplanat(
        train, na_object_sine=0.15, samples=9,
        field_heights=[0.0, 1.5, 3.0], flat_field_weight=5.0,
        bounds=(-3000, 3000), diff_step=1e-4, max_nfev=25,
    )
    assert flat.image_surface is not None
    assert np.isfinite(flat.image_surface.max_sagitta)
    assert not np.allclose(flat.result.x, plain.result.x)
