"""Lenses and sensors at work: refraction to a focus, detection, and the
agreement of the two engines.

(Was the 2-D ``optics`` elements and ``Screen``, plus the 2-D bridge from a
design to the branching tracer. In 3-D a screen is a :class:`Sensor` on a
plane with a rectangular aperture, and no bridge is needed: the
non-sequential tracer traces the same :class:`OpticalSystem` the sequential
one does. Meridional (``fan="y"``) bundles reproduce the old 2-D pictures.)
"""

import numpy as np
import pytest

from raytracer import OpticalSystem, TableRow
from raytracer.elements import Lens, Sensor
from raytracer.geometry.apertures import Rectangle
from raytracer.geometry.surfaces import Plane, Sphere
from raytracer.materials import ConstantIndex
from raytracer.sources import CollimatedSource, PointSource, Rays
from raytracer.tracing import (
    NonSequentialTracer,
    SequentialTracer,
    solve_object_plane,
    trace_from_object,
)


def single_ray(origin, direction) -> Rays:
    return Rays([origin], [direction], np.nan, 1.0)


def test_sensor_records_and_misses_outside_its_aperture():
    screen = Sensor("S", aperture=Rectangle(1.0, 2.0))  # the old segment from y=-1 to y=1
    system = OpticalSystem([screen])
    tree = NonSequentialTracer(system).trace(single_ray([0.0, 0.5, -2.0], [0.0, 0.0, 1.0]))
    hits = tree.hits("S")
    assert len(hits) == 1
    assert hits.points[0] == pytest.approx([0.0, 0.5, 0.0])
    assert hits.local_points[0, 1] == pytest.approx(0.5)  # 3/4 of the way along the segment
    assert hits.opl[0] == pytest.approx(2.0)

    tree = NonSequentialTracer(system).trace(single_ray([0.0, 2.0, -2.0], [0.0, 0.0, 1.0]))
    assert "S" not in tree.detectors
    assert tree.escaped.all()


def test_lens_face_matches_sphere_geometry():
    R, n = 150.0, 1.5
    lens = Lens(
        "L", front=Sphere(R), back=Plane(), thickness=20.0,
        material=ConstantIndex("G", n), semi_diameter=60.0, z=10.0,
    )
    y0 = 40.0
    tree = NonSequentialTracer(OpticalSystem([lens])).trace(single_ray([0.0, y0, -100.0], [0.0, 0.0, 1.0]))
    assert tree.face[0] == "L.front"
    z_exact = 10.0 + R - np.sqrt(R * R - y0 * y0)
    assert tree.end[0] == pytest.approx([0.0, y0, z_exact], abs=1e-10)


def test_single_refracting_face_paraxial_focus():
    """Air -> glass through a convex face: f (from the vertex) = n R / (n - 1)."""

    R, n = 50.0, 1.5
    f_expected = n * R / (n - 1.0)  # 150 mm, inside the glass
    lens = Lens(
        "L", front=Sphere(R), back=Plane(), thickness=200.0,
        material=ConstantIndex("G", n), semi_diameter=10.0,
    )
    source = CollimatedSource((0.0, 0.0, -20.0), beam_radius=0.5, rays=11, fan="y")
    tree = NonSequentialTracer(OpticalSystem([lens]), split=False).trace(source)

    inside = (tree.generation == 2) & (np.abs(tree.direction[:, 1]) > 1e-12)  # skip the axial ray
    assert inside.sum() >= 8
    t = -tree.start[inside, 1] / tree.direction[inside, 1]
    crossings = tree.start[inside, 2] + t * tree.direction[inside, 2]
    assert np.mean(crossings) == pytest.approx(f_expected, rel=1e-3)


def biconvex(name="L1", **placement) -> Lens:
    return Lens(
        name, front=Sphere(60.0), back=Sphere(-60.0), thickness=8.0,
        material=ConstantIndex("G", 1.5), semi_diameter=15.0, **placement,
    )


def test_lens_focuses_a_parallel_fan_onto_a_sensor():
    # Thin-lens estimate f ~ R / (2 (n-1)) = 60 mm; sensor near focus.
    screen = Sensor("screen", aperture=Rectangle(40.0, 40.0), z=58.0)
    system = OpticalSystem([biconvex(), screen])
    source = CollimatedSource((0.0, 0.0, -30.0), beam_radius=8.0, rays=41, fan="y")
    tree = NonSequentialTracer(system, split=False).trace(source)

    hits = tree.hits("screen")
    assert len(hits) >= 35  # nearly every ray lands on the screen
    assert np.ptp(hits.local_points[:, 1]) < 3.0  # concentrated near focus (spherical aberration aside)
    values, edges = hits.histogram(axis=1, bins=64, range=(-20.0, 20.0))
    assert values.sum() == pytest.approx(hits.power.sum())
    peak = np.argmax(values)
    assert abs((edges[peak] + edges[peak + 1]) / 2) < 4.0  # peak near the axis


def test_sensor_absorbs_rays():
    screen = Sensor("screen", aperture=Rectangle(20.0, 20.0), z=5.0)
    source = PointSource((0.0, 0.0, 0.0), half_angle_deg=15.0, rays=9, fan="y")
    tree = NonSequentialTracer(OpticalSystem([screen]), max_interactions=5).trace(source)
    # Every ray ends on the sensor: no children are spawned.
    assert (tree.generation == 1).all()
    assert (tree.parent == -1).all()
    assert len(tree.hits("screen")) == 9
    assert tree.hits("screen").power.sum() == pytest.approx(1.0)


def test_sequential_and_non_sequential_land_rays_on_the_same_image_point():
    """A cemented doublet: the meridional ray the sequential tracer follows
    must reach the same image point non-sequentially (split=False), since
    both read the same placed elements."""

    glass = ConstantIndex("BK7", 1.5168)
    flint = ConstantIndex("F2", 1.62)
    air = ConstantIndex("AIR", 1.0)
    rows = [
        TableRow(Sphere(60.0), 8.0, glass, 15.0),
        TableRow(Sphere(-45.0), 3.0, flint, 15.0),
        TableRow(Sphere(-120.0), 95.0, air, 15.0),
    ]
    system = OpticalSystem.from_table(rows)
    assert system.face("L1.back").names == ("L1.back", "L2.front")  # cemented
    tracer = SequentialTracer(system)
    solve_object_plane(tracer)

    y_obj, slope = 5.0, -0.005
    reference = trace_from_object(tracer, (0.0, y_obj), (0.0, slope))
    assert reference.ok

    direction = np.array([0.0, slope, 1.0]) / np.hypot(1.0, slope)
    tree = NonSequentialTracer(system, split=False).trace(single_ray([0.0, y_obj, tracer.object_z], direction))
    hits = tree.hits("IMG")
    assert len(hits) == 1
    assert hits.points[0] == pytest.approx(reference.image_point, abs=1e-9)
    # Both accumulate the same optical path (the sequential tracer's restart
    # offset adds 1e-8 mm per boundary, hence the tolerance).
    assert hits.opl[0] == pytest.approx(reference.opl, abs=1e-6)
