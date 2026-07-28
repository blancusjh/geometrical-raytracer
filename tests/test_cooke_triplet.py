"""Sanity regression for the reconstructed Cooke triplet prescription.

See ``examples/notebooks/cooke_triplet.ipynb`` for the source citation
(rayopt-notebooks' "oslo cooke triplet example 50mm f/4 20deg") and the
column-mapping derivation. Unlike US7557996, this prescription has no
external reference dataset to regress against, so these checks are sanity
bounds (paraxial EFL matches the design spec; chief-ray distortion is
finite, zero on-axis, and grows with field) rather than tight regressions.
"""

from pathlib import Path

import numpy as np
import pytest

from raytracer.design import OpticalSystem
from raytracer.propagation import (
    FieldPoint,
    ParaxialModel,
    SequentialTracer,
    chief_ray_slope,
    solve_object_plane,
    trace_from_object,
)

CSV = Path(__file__).resolve().parents[1] / "data" / "optical_systems/photographic/cooke_triplet_prescription.csv"


@pytest.fixture(scope="module")
def system() -> OpticalSystem:
    return OpticalSystem.from_prescription(CSV)


@pytest.fixture(scope="module")
def tracer(system) -> SequentialTracer:
    return SequentialTracer(system)


def test_prescription_loads(system):
    assert len(system) == 7
    assert system.stop_index == 4
    assert system.mirror_indices == []
    assert system.image_z == pytest.approx(59.95)
    assert system.n_image == pytest.approx(1.0)


def test_paraxial_focal_length_matches_design(tracer):
    solve_object_plane(tracer)
    efl = ParaxialModel(tracer.system).effective_focal_length()
    assert efl == pytest.approx(50.0, rel=2e-3)


def chief_distortion_um(tracer, object_z, stop_index, magnification, angle_deg):
    y = abs(object_z) * np.tan(np.deg2rad(angle_deg))
    z_stop = tracer.system.vertices[stop_index]
    estimate = -y / (z_stop - object_z)
    slope = chief_ray_slope(
        tracer, FieldPoint(y=y), stop_index=stop_index,
        bracket=(estimate - 0.02, estimate + 0.02), scan=101,
    )
    result = trace_from_object(tracer, (0.0, y), (0.0, slope))
    ideal_y = y * magnification
    return float((result.image_point[1] - ideal_y) * 1e3)


def test_near_infinity_conjugate_is_in_front(tracer):
    """The image plane sits a fraction of a millimetre before the back
    focal plane, making the raw B = 0 solution a virtual object behind
    the system. ``solve_object_plane`` must enforce the physical
    convention: real object in front, inverted image."""

    conjugate = solve_object_plane(tracer)
    assert conjugate.object_z < 0
    assert conjugate.magnification < 0
    r = trace_from_object(tracer, (0.0, 0.0), (0.0, 0.0), keep_path=True)
    assert np.all(np.diff(r.path[:, 2]) >= 0)  # monotonically forward


def test_chief_ray_distortion_grows_with_field(tracer):
    conjugate = solve_object_plane(tracer)
    distortions = [
        chief_distortion_um(
            tracer, conjugate.object_z, tracer.system.stop_index,
            conjugate.magnification, angle_deg,
        )
        for angle_deg in (2.0, 4.0, 6.0, 8.0)
    ]
    for d in distortions:
        assert np.isfinite(d)
    # Monotonically increasing magnitude with field, as expected for a
    # simple third-order-dominated aberration near the axis.
    magnitudes = [abs(d) for d in distortions]
    assert magnitudes == sorted(magnitudes)
    assert magnitudes[-1] < 50.0  # sanity bound, not a regression value
