"""Pluggable constraints (``raytracer.optimize.constraints``)."""

import numpy as np
import pytest

from raytracer.design import StigmaticTrain
from raytracer.optimize import (
    Aplanatism,
    Constraint,
    Distortion,
    EvaluationContext,
    FlatImageSurface,
    TargetMagnification,
    constraint_residuals,
    optimize_train,
)

CEMENTED = dict(
    indices=(1.0, 1.517122, 1.670591, 1.851280, 1.0),
    vertices=(0.0, 15.0, 25.0, 35.0),
    semidiameter=10.0,
)


@pytest.fixture(scope="module")
def symmetric():
    return StigmaticTrain.symmetric_singlet(
        n=1.5, d0=-30.0, thickness=40.0, semidiameter=3.5
    )


def test_residual_blocks_concatenate_in_order(symmetric):
    constraints = [
        Aplanatism(),
        FlatImageSurface(heights=(0.0, 1.0), weight=3.0),
        Distortion(heights=(1.0,), weight=2.0),
        TargetMagnification(value=1.0),
    ]
    residuals = constraint_residuals(
        symmetric, constraints, na_object_sine=0.08, samples=7
    )
    assert residuals.shape == (7 + 2 + 1 + 1,)
    # The exact aplanat: sine-condition block is numerically zero...
    assert np.max(np.abs(residuals[:7])) < 1e-12
    # ...its magnification is exactly the +1 target...
    assert residuals[-1] == pytest.approx(0.0, abs=1e-9)
    # ...but the field-dependent blocks are genuinely nonzero: curved image
    # surface and visible pincushion.
    assert abs(residuals[8]) > 0.1  # 3 * sagitta(1 mm)
    assert abs(residuals[9]) > 0.02  # 2 * relative distortion(1 mm)


def test_distortion_matches_the_locus_geometry(symmetric):
    heights = (0.8, 1.2)
    context = EvaluationContext(symmetric, na_object_sine=0.08, samples=5)
    residuals = Distortion(heights=heights).residuals(context)
    surface = context.image_surface(heights)
    expected = surface.points[:, 1] / (symmetric.gt * np.asarray(heights)) - 1.0
    np.testing.assert_allclose(residuals, expected)
    # The symmetric singlet is visibly distorted, and more so at the larger
    # field. (The sign is convention-dependent: this measure references
    # cones aimed at the front vertex, not a chief ray through a stop.)
    assert abs(residuals[1]) > abs(residuals[0]) > 0.01


def test_constraint_validation():
    with pytest.raises(ValueError, match="start at 0.0"):
        FlatImageSurface(heights=(1.0, 2.0))
    with pytest.raises(ValueError, match="nonzero"):
        Distortion(heights=(0.0, 1.0))


def test_context_caches_shared_traces(symmetric):
    context = EvaluationContext(symmetric, na_object_sine=0.08, samples=5)
    assert context.report() is context.report()
    assert context.image_surface((0.0, 1.0)) is context.image_surface((0.0, 1.0))


def test_custom_constraint_plugs_in(symmetric):
    class BackFocalDistance(Constraint):
        """Toy plug-in: pin the image plane a set distance past the vertex."""

        def __init__(self, minimum: float) -> None:
            self.minimum = minimum

        def size(self, context) -> int:
            return 1

        def residuals(self, context) -> np.ndarray:
            gap = context.train.conjugates[-1] - context.train.vertices[-1]
            return np.array([min(0.0, gap - self.minimum)])

    residuals = constraint_residuals(
        symmetric, [BackFocalDistance(50.0)], na_object_sine=0.08, samples=5
    )
    assert residuals.shape == (1,)
    assert residuals[0] == pytest.approx(-(50.0 - 30.0))


def test_distortion_constraint_actually_reduces_distortion():
    """Adding the Distortion block must cut the starting train's distortion
    by a large factor and end below the plain-aplanatism optimum, while
    keeping the train aplanatic to a usable level."""

    start = StigmaticTrain(conjugates=(-100.0, -150.0, -600.0, 450.0, 90.0), **CEMENTED)
    fields = (1.5, 3.0)
    common = dict(na_object_sine=0.15, samples=11, bounds=(-3000, 3000),
                  diff_step=1e-4, xtol=1e-14, ftol=1e-14, gtol=1e-14)

    def worst_distortion(train):
        context = EvaluationContext(train, na_object_sine=0.15, samples=5)
        return float(np.max(np.abs(Distortion(heights=fields).residuals(context))))

    plain = optimize_train(start, [Aplanatism()], **common)
    both = optimize_train(
        start, [Aplanatism(), Distortion(heights=fields, weight=5.0)], **common
    )

    assert worst_distortion(both.train) < worst_distortion(start) / 5.0
    assert worst_distortion(both.train) < worst_distortion(plain.train)
    assert both.after.map_rms < 1e-3
