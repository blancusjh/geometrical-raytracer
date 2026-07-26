"""Analytic conjugate tests for the sequential 3-D tracer (EUV notebook §4)."""

import numpy as np
import pytest

from raytracer.design import OpticalSystem, SurfaceRow
from raytracer.physics.materials import ConstantIndex
from raytracer.propagation import SequentialTracer, TraceStatus


def test_parabola_focuses_collimated_beam_to_machine_precision():
    # Parabola R=-2000, K=-1, vertex at z=1000; focus at zv + R/2 = 0.
    rows = [
        SurfaceRow.mirror(radius=-2000.0, thickness=-1000.0, conic=-1.0),
    ]
    system = OpticalSystem(rows)
    # Vertex at z=0 by construction; place image plane at focus z = R/2 = -1000.
    tracer = SequentialTracer(system)

    ys = np.linspace(-300.0, 300.0, 13)
    origins = np.column_stack([np.zeros_like(ys), ys, np.full_like(ys, -3000.0)])
    directions = np.tile([0.0, 0.0, 1.0], (len(ys), 1))
    batch = tracer.trace_batch(origins, directions)

    assert np.all(batch.status == TraceStatus.OK)
    assert np.ptp(batch.image_points[:, 1]) < 1e-10
    assert np.ptp(batch.image_points[:, 0]) < 1e-12
    assert batch.image_points[:, 1] == pytest.approx(0.0, abs=1e-10)


def test_ellipsoid_images_focus_onto_focus():
    # Prolate ellipsoid a=1000, b=600: R=-b^2/a, K=-(c/a)^2, foci at z=-a±c
    # for a vertex at z=0 opening toward -z.
    a, b = 1000.0, 600.0
    c = np.sqrt(a * a - b * b)
    rows = [SurfaceRow.mirror(radius=-(b * b) / a, thickness=-(a - c), conic=-(c / a) ** 2)]
    system = OpticalSystem(rows)
    tracer = SequentialTracer(system)
    # Near focus at z = -(a - c); far focus at z = -(a + c).
    near_focus = -(a - c)
    far_focus = -(a + c)
    system.rows[0].thickness = far_focus  # image plane at the far focus
    system._rebuild()

    thetas = np.linspace(-0.25, 0.25, 15)
    origins = np.tile([0.0, 0.0, near_focus], (len(thetas), 1))
    directions = np.column_stack(
        [np.zeros_like(thetas), np.sin(thetas), np.cos(thetas)]
    )
    batch = tracer.trace_batch(origins, directions)

    assert np.all(batch.status == TraceStatus.OK)
    assert np.ptp(batch.image_points[:, 1]) < 1e-9
    assert batch.image_points[:, 1] == pytest.approx(0.0, abs=1e-9)


def test_sphere_newton_matches_closed_form():
    # Refracting sphere: Newton intersection must match the quadratic solution.
    R = 150.0
    n_glass = ConstantIndex("GLASS", 1.5)
    rows = [SurfaceRow.refracting(radius=R, thickness=10.0, material=n_glass)]
    system = OpticalSystem(rows)
    tracer = SequentialTracer(system)

    y0 = 40.0
    result = tracer.trace([0.0, y0, -100.0], [0.0, 0.0, 1.0], keep_path=True)
    assert result.ok
    hit = result.path[1]
    # Closed form: intersection of the line x=0, y=y0 with the sphere centred
    # at (0, 0, R): z = R - sqrt(R^2 - y0^2).
    z_exact = R - np.sqrt(R * R - y0 * y0)
    assert hit[2] == pytest.approx(z_exact, abs=1e-11)
    assert hit[1] == pytest.approx(y0, abs=1e-12)


def test_vignetting_status_and_surface():
    rows = [
        SurfaceRow.refracting(
            radius=0.0, thickness=10.0, material=ConstantIndex("AIR", 1.0), semidiameter=5.0
        )
    ]
    system = OpticalSystem(rows)
    tracer = SequentialTracer(system)
    batch = tracer.trace_batch(
        np.array([[0.0, 2.0, -10.0], [0.0, 8.0, -10.0]]),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    )
    assert batch.status[0] == TraceStatus.OK
    assert batch.status[1] == TraceStatus.VIGNETTED
    assert batch.failed_surface[1] == 0
    assert np.isnan(batch.image_points[1]).all()


def test_tir_status():
    # Object space IS the glass: a glass-to-air interface beyond the
    # critical angle measured inside the dense medium.
    glass = ConstantIndex("GLASS", 1.5)
    rows = [
        SurfaceRow.refracting(radius=0.0, thickness=10.0, material=ConstantIndex("AIR", 1.0)),
    ]
    system = OpticalSystem(rows, object_space=glass)
    tracer = SequentialTracer(system)
    theta = np.arcsin(1.0 / 1.5) + 0.05  # beyond critical inside the glass
    direction = [0.0, np.sin(theta), np.cos(theta)]
    result = tracer.trace([0.0, 0.0, -1.0], direction)
    assert result.status == TraceStatus.TIR
    assert result.failed_surface == 0


def test_opl_through_flat_glass_slab():
    glass = ConstantIndex("GLASS", 1.5)
    thickness = 20.0
    rows = [
        SurfaceRow.refracting(radius=0.0, thickness=thickness, material=glass),
        SurfaceRow.refracting(radius=0.0, thickness=30.0, material=ConstantIndex("AIR", 1.0)),
    ]
    system = OpticalSystem(rows)
    tracer = SequentialTracer(system, restart_offset=0.0)
    result = tracer.trace([0.0, 0.0, -10.0], [0.0, 0.0, 1.0])
    assert result.ok
    expected = 1.0 * 10.0 + 1.5 * thickness + 1.0 * 30.0
    assert result.opl == pytest.approx(expected, abs=1e-9)
