"""2-D elements, instruments, and the design bridge."""

import numpy as np
import pytest

from raytracer.optics import Lens, ParallelSource, PointSource, Screen
from raytracer.optics.profiles import LineSegment, ProfileSurface
from raytracer.optics.ray import Ray
from raytracer.propagation import BranchingTracer, TraceConfig
from raytracer.shapes.profile import AsphereProfile


def test_segment_intersection_and_miss():
    seg = LineSegment(p0=[0.0, -1.0], p1=[0.0, 1.0])
    hit = seg.hit(Ray(origin=[-2.0, 0.5], direction=[1.0, 0.0]))
    assert hit is not None
    assert hit.distance == pytest.approx(2.0)
    assert hit.point == pytest.approx([0.0, 0.5])
    assert seg.hit(Ray(origin=[-2.0, 2.0], direction=[1.0, 0.0])) is None


def test_profile_face_matches_sphere_geometry():
    R = 150.0
    face = ProfileSurface(
        profile=AsphereProfile.sphere(R), vertex=[10.0, 0.0], semidiameter=60.0
    )
    y0 = 40.0
    hit = face.hit(Ray(origin=[-100.0, y0], direction=[1.0, 0.0]))
    assert hit is not None
    z_exact = 10.0 + R - np.sqrt(R * R - y0 * y0)
    assert hit.point == pytest.approx([z_exact, y0], abs=1e-10)


def test_single_refracting_face_paraxial_focus():
    """Plano-convex air->glass face: f (from vertex) = n R / (n - 1)."""

    R, n = 50.0, 1.5
    face = ProfileSurface(
        profile=AsphereProfile.sphere(R), vertex=[0.0, 0.0],
        semidiameter=10.0, n_exterior=1.0, n_interior=n, interaction="refract",
    )
    source = ParallelSource(
        origin=np.array([-20.0, 0.0]), direction=np.array([1.0, 0.0]),
        width=1.0, samples=11,
    )
    tree = BranchingTracer([face], TraceConfig(max_generations=2)).trace([source])
    f_expected = n * R / (n - 1.0)  # 150 mm, measured from the vertex

    crossings = []
    for node in tree.nodes():
        if node.generation != 2:
            continue
        origin, direction = node.ray.origin, node.ray.direction
        if abs(direction[1]) < 1e-12:
            continue  # axial ray
        t = -origin[1] / direction[1]
        crossings.append(origin[0] + t * direction[0])
    assert len(crossings) >= 8
    assert np.mean(crossings) == pytest.approx(f_expected, rel=1e-3)


def test_lens2d_traces_and_screen_records():
    lens = Lens.from_radii(
        r1=60.0, r2=-60.0, thickness=8.0, semidiameter=15.0, n=1.5,
        vertex=(0.0, 0.0), name="L1",
    )
    # Thin-lens estimate f ~ R / (2 (n-1)) = 60 mm; screen near focus.
    screen = Screen([58.0, -20.0], [58.0, 20.0])
    source = ParallelSource(
        origin=np.array([-30.0, 0.0]), direction=np.array([1.0, 0.0]),
        width=16.0, samples=41,
    )
    tracer = BranchingTracer(
        [*lens.surfaces(), screen], TraceConfig(max_generations=4)
    )
    tracer.trace([source])

    assert len(screen.hits) >= 35  # nearly every ray lands on the screen
    spread = np.ptp([hit.point[1] for hit in screen.hits])
    assert spread < 3.0  # concentrated near focus (spherical aberration aside)
    edges, values = screen.irradiance(bins=64)
    assert values.sum() == pytest.approx(sum(h.intensity for h in screen.hits))
    peak_bin = np.argmax(values)
    # Peak near the axis (middle of the screen).
    assert abs((edges[peak_bin] + edges[peak_bin + 1]) / 2 - 20.0) < 4.0


def test_screen_absorbs_rays():
    screen = Screen([5.0, -10.0], [5.0, 10.0])
    source = PointSource(
        origin=np.array([0.0, 0.0]), axis_direction=np.array([1.0, 0.0]),
        aperture=np.deg2rad(30.0), samples=9,
    )
    tree = BranchingTracer([screen], TraceConfig(max_generations=5)).trace([source])
    # All rays terminate at the screen: no children spawned.
    assert all(not node.children for node in tree.nodes())
    assert len(screen.hits) == 9


def test_design_bridge_matches_sequential_trace():
    """A dioptric doublet sliced to 2-D must land rays exactly where the
    sequential 3-D propagation does (meridional rays are common to both)."""

    from raytracer.design import OpticalSystem, SurfaceRow, to_branching_surfaces
    from raytracer.physics.materials import ConstantIndex
    from raytracer.propagation import SequentialTracer, solve_object_plane, trace_from_object

    glass = ConstantIndex("BK7", 1.5168)
    flint = ConstantIndex("F2", 1.62)
    air = ConstantIndex("AIR", 1.0)
    rows = [
        SurfaceRow.refracting(radius=60.0, thickness=8.0, material=glass, semidiameter=15.0),
        SurfaceRow.refracting(radius=-45.0, thickness=3.0, material=flint, semidiameter=15.0),
        SurfaceRow.refracting(radius=-120.0, thickness=95.0, material=air, semidiameter=15.0),
    ]
    system = OpticalSystem(rows)
    tracer3d = SequentialTracer(system)
    solve_object_plane(tracer3d)

    # Sequential meridional ray from an off-axis object point.
    y_obj, slope = 5.0, -0.005

    reference = trace_from_object(tracer3d, (0.0, y_obj), (0.0, slope))
    assert reference.ok

    # Same ray through the 2-D slice.
    surfaces = to_branching_surfaces(system)
    screen = Screen([system.image_z, -30.0], [system.image_z, 30.0])
    from raytracer.optics.sources import RaySeed, Source

    class SingleRay(Source):
        def emit(self):
            direction = np.array([1.0, slope]) / np.hypot(1.0, slope)
            return [RaySeed(origin=np.array([system.object_z, y_obj]), direction=direction)]

    tracer2d = BranchingTracer(
        [*surfaces, screen], TraceConfig(max_generations=10)
    )
    tracer2d.trace([SingleRay()])
    assert len(screen.hits) == 1
    assert screen.hits[0].point[1] == pytest.approx(reference.image_point[1], abs=1e-6)
