"""Analytic conjugate tests for the sequential 3-D tracer (EUV notebook §4)."""

import numpy as np
import pytest

from raytracer import AIR, OpticalSystem, TableRow
from raytracer.elements import Lens, Mirror, Sensor, Stop
from raytracer.geometry.surfaces import Ellipsoid, Paraboloid, Plane, Sphere
from raytracer.materials import ConstantIndex
from raytracer.system import Gap
from raytracer.tracing import SequentialTracer, TraceStatus


def test_parabola_focuses_collimated_beam_to_machine_precision():
    # Parabola R=-2000, K=-1, vertex at z=0; focus at R/2 = -1000, where the
    # image plane sits (the mirror row's gap is -1000).
    rows = [TableRow(Paraboloid(-2000.0), -1000.0, None, 400.0, "mirror")]
    system = OpticalSystem.from_table(rows)
    tracer = SequentialTracer(system)
    assert tracer.image_z == pytest.approx(-1000.0)

    ys = np.linspace(-300.0, 300.0, 13)
    origins = np.column_stack([np.zeros_like(ys), ys, np.full_like(ys, -3000.0)])
    directions = np.tile([0.0, 0.0, 1.0], (len(ys), 1))
    trace = tracer.trace(origins, directions)

    assert np.all(trace.status == TraceStatus.OK)
    assert np.ptp(trace.end[:, 1]) < 1e-10
    assert np.ptp(trace.end[:, 0]) < 1e-12
    assert trace.end[:, 1] == pytest.approx(0.0, abs=1e-10)


def test_ellipsoid_images_focus_onto_focus():
    # Prolate ellipsoid a=1000, b=600: R=-b^2/a, K=-(c/a)^2, foci at z=-a±c
    # for a vertex at z=0 opening toward -z.
    a, b = 1000.0, 600.0
    c = np.sqrt(a * a - b * b)
    surface = Ellipsoid.from_semi_axes(-a, b)
    assert surface.R == pytest.approx(-(b * b) / a)
    assert surface.k == pytest.approx(-((c / a) ** 2))
    near_focus = -(a - c)
    far_focus = -(a + c)
    # Mirror at the origin, sensor at the far focus, stated path.
    system = OpticalSystem(
        [Mirror("M", surface=surface, aperture=500.0), Sensor("IMG", z=far_focus)],
        path=["M:reflect", "IMG"],
    )
    tracer = SequentialTracer(system)

    thetas = np.linspace(-0.25, 0.25, 15)
    origins = np.tile([0.0, 0.0, near_focus], (len(thetas), 1))
    directions = np.column_stack([np.zeros_like(thetas), np.sin(thetas), np.cos(thetas)])
    trace = tracer.trace(origins, directions)

    assert np.all(trace.status == TraceStatus.OK)
    assert np.ptp(trace.end[:, 1]) < 1e-9
    assert trace.end[:, 1] == pytest.approx(0.0, abs=1e-9)


def test_sphere_intersection_matches_closed_form():
    # Refracting sphere: the traced hit must be the exact circle point.
    R = 150.0
    n_glass = ConstantIndex("GLASS", 1.5)
    rows = [TableRow(Sphere(R), 10.0, n_glass, 60.0)]
    system = OpticalSystem.from_table(rows)
    tracer = SequentialTracer(system)

    y0 = 40.0
    result = tracer.trace_ray([0.0, y0, -100.0], [0.0, 0.0, 1.0])
    assert result.ok
    hit = result.path[1]
    # Closed form: intersection of the line x=0, y=y0 with the sphere centred
    # at (0, 0, R): z = R - sqrt(R^2 - y0^2).
    z_exact = R - np.sqrt(R * R - y0 * y0)
    assert hit[2] == pytest.approx(z_exact, abs=1e-11)
    assert hit[1] == pytest.approx(y0, abs=1e-12)


def test_vignetting_status_and_step():
    # A real diaphragm clips: a prescription row of air on both sides changes
    # nothing and is not an element, so it cannot.
    system = OpticalSystem.compose(Stop("S", aperture=5.0), Gap(10.0), Sensor("IMG"))
    tracer = SequentialTracer(system)
    trace = tracer.trace(
        np.array([[0.0, 2.0, -10.0], [0.0, 8.0, -10.0]]),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    )
    assert trace.status[0] == TraceStatus.OK
    assert trace.status[1] == TraceStatus.VIGNETTED
    assert trace.failed_step[1] == 0
    assert np.isnan(trace.end[1]).all()


def test_lens_rim_vignettes_at_its_face():
    glass = ConstantIndex("GLASS", 1.5)
    system = OpticalSystem.compose(
        Lens("W", front=Plane(), back=Plane(), thickness=5.0, material=glass, semi_diameter=5.0),
        Gap(10.0),
        Sensor("IMG"),
    )
    trace = SequentialTracer(system).trace(
        np.array([[0.0, 2.0, -10.0], [0.0, 8.0, -10.0]]),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    )
    assert trace.status[1] == TraceStatus.VIGNETTED
    assert trace.failed_step[1] == system.step_index("W.front")


def test_tir_status():
    # Object space IS the glass: the system is immersed in glass and a layer
    # of air follows the plane. Beyond the critical angle measured inside
    # the dense medium, the first boundary totally reflects.
    glass = ConstantIndex("GLASS", 1.5)
    rows = [TableRow(Plane(), 10.0, AIR, 50.0)]
    system = OpticalSystem.from_table(rows, ambient=glass)
    assert system.path[0].face.medium_minus is glass
    tracer = SequentialTracer(system)
    theta = np.arcsin(1.0 / 1.5) + 0.05  # beyond critical inside the glass
    direction = [0.0, np.sin(theta), np.cos(theta)]
    result = tracer.trace_ray([0.0, 0.0, -1.0], direction)
    assert result.status == TraceStatus.TIR
    assert result.failed_step == 0


def test_opl_through_flat_glass_slab():
    glass = ConstantIndex("GLASS", 1.5)
    thickness = 20.0
    rows = [
        TableRow(Plane(), thickness, glass, 10.0),
        TableRow(Plane(), 30.0, AIR, 10.0),
    ]
    system = OpticalSystem.from_table(rows)
    tracer = SequentialTracer(system, restart_offset=0.0)
    result = tracer.trace_ray([0.0, 0.0, -10.0], [0.0, 0.0, 1.0])
    assert result.ok
    expected = 1.0 * 10.0 + 1.5 * thickness + 1.0 * 30.0
    assert result.opl == pytest.approx(expected, abs=1e-9)


def test_opl_is_signed_behind_a_virtual_stop():
    """A stop placed behind the previous surface (as patent tables do) must
    not add optical path: the signed ray parameter cancels on the way back."""

    glass = ConstantIndex("GLASS", 1.5)
    rows = [
        TableRow(Plane(), 20.0, glass, 10.0),
        TableRow(Plane(), -5.0, AIR, 10.0),  # back face; the stop sits 5 mm before it
        TableRow(Plane(), 35.0, AIR, 10.0, kind="stop"),
    ]
    system = OpticalSystem.from_table(rows)
    tracer = SequentialTracer(system, restart_offset=0.0)
    result = tracer.trace_ray([0.0, 0.0, -10.0], [0.0, 0.0, 1.0])
    assert result.ok
    assert tracer.image_z == pytest.approx(50.0)
    assert result.opl == pytest.approx(10.0 + 1.5 * 20.0 + 30.0, abs=1e-9)


def test_a_path_back_to_a_passed_boundary_stops_the_ray():
    """Two faces that cross: beyond the crossing the second face lies behind
    the ray, so the stated path is no longer physical there."""

    glass = ConstantIndex("G", 1.5)
    system = OpticalSystem.compose(
        Lens("A", front=Plane(), back=Plane(), thickness=2.0, material=glass, semi_diameter=5.0),
        Gap(1.0),
        Lens("B", front=Sphere(-5.0), back=Plane(), thickness=4.0, material=glass, semi_diameter=4.8),
        Gap(20.0),
        Sensor("IMG"),
    )
    tracer = SequentialTracer(system)
    on_axis = tracer.trace_ray([0.0, 0.0, -5.0], [0.0, 0.0, 1.0])
    beyond = tracer.trace_ray([0.0, 4.5, -5.0], [0.0, 0.0, 1.0])
    assert on_axis.ok
    assert beyond.status is TraceStatus.BEHIND and beyond.failed_step == 2


def test_a_stop_may_sit_behind_the_ray():
    """Prescriptions put the stop plane at a curved vertex; it is a reference plane, not a wall."""

    glass = ConstantIndex("G", 1.5)
    system = OpticalSystem.from_table(
        [
            TableRow(Sphere(20.0), 3.0, glass, 8.0),
            TableRow(Sphere(-20.0), 0.0, AIR, 8.0),
            TableRow(Plane(), 30.0, AIR, 6.0, kind="stop"),
        ]
    )
    result = SequentialTracer(system).trace_ray([0.0, 5.0, -5.0], [0.0, 0.0, 1.0])
    assert result.ok


def test_tracer_replace_keeps_the_settings_it_is_not_given():
    glass = ConstantIndex("G", 1.5)
    system = OpticalSystem.compose(Lens("A", front=Sphere(20.0), back=Plane(), thickness=2.0, material=glass, semi_diameter=5.0), Gap(30.0), Sensor("IMG"))
    tracer = SequentialTracer(system, wavelength_um=0.55, object_z=-100.0, clip_apertures=False)
    moved = tracer.replace(object_z=-200.0)
    assert (moved.wavelength_um, moved.object_z, moved.clip_apertures) == (0.55, -200.0, False)
    assert tracer.at_wavelength(0.65).object_z == -100.0
