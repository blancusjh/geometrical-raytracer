"""Elements: values with a name and a pose (immutability, replace, placement,
faces), and lenses and sensors at work (refraction to a focus, detection,
the agreement of the two engines)."""

import numpy as np
import pytest

from raytracer import OpticalSystem, TableRow
from raytracer.coatings import AR, Absorbing, IdealBeamSplitter, PerfectMirror, Uncoated
from raytracer.elements import (
    BLADE,
    BeamSplitter,
    DETECTOR,
    EDGE,
    Lens,
    Mirror,
    OPTICAL,
    Prism,
    Sensor,
    Stop,
)
from raytracer.geometry import Pose, Rotation
from raytracer.geometry.apertures import Circle, Rectangle
from raytracer.geometry.solids import Box
from raytracer.geometry.surfaces import Plane, Sphere
from raytracer.interfaces import Interface
from raytracer.materials import ConstantIndex
from raytracer.analysis import detected_on
from raytracer.rays import Rays, Status
from raytracer.sources import CollimatedSource, PointSource
from raytracer.tracing import (
    NonSequentialTracer,
    SequentialTracer,
    solve_object_plane,
    trace_from_object,
)


GLASS = ConstantIndex("G", 1.5)


def lens(**placement) -> Lens:
    return Lens("L", front=Sphere(50.0), back=Sphere(-50.0), thickness=5.0, material=GLASS,
                semi_diameter=10.0, **placement)


# -- values -----------------------------------------------------------------------


def test_elements_are_immutable():
    element = lens()
    with pytest.raises(AttributeError):
        element.thickness = 6.0
    with pytest.raises(AttributeError):
        element.name = "other"
    with pytest.raises(AttributeError):
        element.front.coating = AR()


def test_replace_returns_a_modified_copy_and_keeps_the_pose():
    element = lens(z=3.0)
    thicker = element.replace(thickness=6.0)
    assert thicker.thickness == 6.0 and element.thickness == 5.0
    assert thicker.pose == element.pose
    assert thicker.name == "L"
    assert element.replace(name="L2").name == "L2"
    moved = element.replace(z=7.0)
    assert moved.pose.position[2] == 7.0 and moved.thickness == 5.0
    with pytest.raises(TypeError):
        element.replace(radius=3.0)


def test_equal_elements_are_equal_values():
    assert lens() == lens()
    assert hash(lens()) == hash(lens())
    assert lens() != lens(z=1.0)
    assert lens() != lens().replace(thickness=5.5)


def test_constructor_validation():
    with pytest.raises(ValueError):
        Lens("L", front=Plane(), back=Plane(), thickness=0.0, material=GLASS, semi_diameter=1.0)
    with pytest.raises(ValueError):
        Lens("L", front=Plane(), back=Plane(), thickness=1.0, material=GLASS, semi_diameter=-1.0)
    with pytest.raises(ValueError):
        Sensor("a.b")  # '.' separates element and face names
    with pytest.raises(ValueError):
        Sensor("a:b")  # ':' separates a path step and its event
    with pytest.raises(TypeError):
        Sensor("")
    with pytest.raises(TypeError):
        Prism("P", solid=Plane(), material=GLASS)


# -- placement ---------------------------------------------------------------------


def test_placement_keywords():
    assert lens().pose.is_identity()
    assert lens(z=4.0).pose == Pose((0.0, 0.0, 4.0))
    assert lens(pos=(1.0, 2.0, 3.0)).pose == Pose((1.0, 2.0, 3.0))
    assert lens(rot=(10.0, 0.0, 0.0)).pose.rotation == Rotation.euler(10.0, 0.0, 0.0)
    assert lens(rot=(30.0, (0.0, 1.0, 0.0))).pose.rotation == Rotation.axis_angle((0.0, 1.0, 0.0), 30.0)
    np.testing.assert_allclose(lens(axis=(0.0, 1.0, 0.0)).pose.axis, [0.0, 1.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(Mirror("M", surface=Plane(), aperture=1.0, normal=(1, 0, 1)).pose.axis,
                               np.array([1.0, 0.0, 1.0]) / np.sqrt(2), atol=1e-15)
    pose = Pose((0.0, 1.0, 0.0), Rotation.euler(0.0, 0.0, 45.0))
    assert lens(pose=pose).pose is pose
    with pytest.raises(TypeError):
        lens(z=1.0, pos=(0.0, 0.0, 1.0))
    with pytest.raises(TypeError):
        lens(rot=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0))


def test_placed_translated_rotated():
    element = lens(z=10.0)
    assert element.translated((1.0, 0.0, 0.0)).pose.position.tolist() == [1.0, 0.0, 10.0]
    turned = element.rotated(90.0, (1.0, 0.0, 0.0), about=(0.0, 0.0, 5.0))
    np.testing.assert_allclose(turned.pose.position, [0.0, -5.0, 5.0], atol=1e-14)
    np.testing.assert_allclose(turned.pose.axis, [0.0, -1.0, 0.0], atol=1e-14)
    pose = Pose((3.0, 0.0, 0.0))
    assert element.placed(pose).pose == pose
    assert element.placed(pose).replace(pose=element.pose) == element


# -- faces --------------------------------------------------------------------------


def test_lens_faces():
    element = lens()
    front, back, edge = element.faces()
    assert (front.name, back.name, edge.name) == ("front", "back", "edge")
    assert front.pose.is_identity() and back.pose == Pose((0.0, 0.0, 5.0))
    assert (front.bulk_side, back.bulk_side) == (+1, -1)  # the glass lies between them
    assert front.role == back.role == OPTICAL and edge.role == EDGE
    assert front.extent == Circle(10.0)
    sag = 50.0 - np.sqrt(50.0**2 - 10.0**2)
    assert edge.wall.radius == 10.0
    assert edge.pose.position[2] == pytest.approx(sag)  # the rim starts where it meets the front face
    assert edge.wall.length == pytest.approx(5.0 - 2 * sag)
    assert element.edge_thickness == pytest.approx(5.0 - 2 * sag)


def test_bare_surfaces_are_uncoated_interfaces():
    element = lens()
    assert element.front == Interface(Sphere(50.0))
    assert isinstance(element.front.coating, Uncoated)
    coated = lens().replace(front=Interface(Sphere(50.0), coating=AR(), clear_aperture=8.0))
    assert isinstance(coated.faces()[0].interface.coating, AR)
    assert coated.front.clear_aperture == Circle(8.0)


def test_mirror_and_beam_splitter_faces():
    (face,) = Mirror("M", surface=Sphere(-100.0), aperture=Rectangle(4.0, 2.0)).faces()
    assert isinstance(face.interface.coating, PerfectMirror)
    assert face.bulk_side == 0 and face.extent == Rectangle(4.0, 2.0)
    (face,) = Mirror("M", surface=Plane(), aperture=3.0, coating=IdealBeamSplitter(0.9, 0.1)).faces()
    assert face.interface.coating == IdealBeamSplitter(0.9, 0.1)
    (face,) = BeamSplitter("BS", aperture=3.0).faces()
    assert face.interface.coating == IdealBeamSplitter()
    assert face.interface.surface == Plane()


def test_stop_and_sensor_faces():
    (blade,) = Stop("S", aperture=2.0).faces()
    assert blade.role == BLADE and blade.hole == Circle(2.0)
    assert blade.extent == Circle(4.0)  # the blade reaches twice the opening by default
    assert isinstance(blade.interface.coating, Absorbing)
    assert not Stop("S", aperture=2.0).virtual and Stop("S", aperture=2.0, virtual=True).virtual
    (detector,) = Sensor("D", aperture=Rectangle(5.0, 5.0)).faces()
    assert detector.role == DETECTOR and detector.extent == Rectangle(5.0, 5.0)
    assert Sensor("D").faces()[0].extent is None  # unbounded


def test_prism_faces_are_its_facets():
    prism = Prism("P", solid=Box(2.0, 2.0, 2.0), material=GLASS, coatings={1: AR()})
    faces = prism.faces()
    assert [f.name for f in faces] == [f"facet{i}" for i in range(6)]
    assert all(f.bulk_side == -1 for f in faces)  # the glass is inside, each facet's +z outward
    assert isinstance(faces[1].interface.coating, AR)
    assert isinstance(faces[0].interface.coating, Uncoated)


def test_sensor_records_and_misses_outside_its_aperture():
    sensor = Sensor("S", aperture=Rectangle(1.0, 2.0))  # y from -1 to 1
    system = OpticalSystem([sensor])
    ray = NonSequentialTracer(system).trace(Rays([0.0, 0.5, -2.0], [0.0, 0.0, 1.0]))
    assert ray.detected[0] and ray.boundary[0] == "S"
    assert ray.position[0] == pytest.approx([0.0, 0.5, 0.0])
    assert ray.opl[0] == pytest.approx(2.0)

    ray = NonSequentialTracer(system).trace(Rays([0.0, 2.0, -2.0], [0.0, 0.0, 1.0]))
    assert ray.status.tolist() == [Status.ESCAPED]


def test_lens_face_matches_sphere_geometry():
    R, n = 150.0, 1.5
    lens = Lens(
        "L", front=Sphere(R), back=Plane(), thickness=20.0,
        material=ConstantIndex("G", n), semi_diameter=60.0, z=10.0,
    )
    y0 = 40.0
    hit = NonSequentialTracer(OpticalSystem([lens])).trace(Rays([0.0, y0, -100.0], [0.0, 0.0, 1.0])).at(1)
    assert hit.boundary[0] == "L.front"
    z_exact = 10.0 + R - np.sqrt(R * R - y0 * y0)
    assert hit.position[0] == pytest.approx([0.0, y0, z_exact], abs=1e-10)


def test_single_refracting_face_paraxial_focus():
    """Air -> glass through a convex face: f (from the vertex) = n R / (n - 1)."""

    R, n = 50.0, 1.5
    f_expected = n * R / (n - 1.0)  # 150 mm, inside the glass
    lens = Lens(
        "L", front=Sphere(R), back=Plane(), thickness=200.0,
        material=ConstantIndex("G", n), semi_diameter=10.0,
    )
    source = CollimatedSource((0.0, 0.0, -20.0), beam_radius=0.5, rays=11, fan="y")
    in_glass = NonSequentialTracer(OpticalSystem([lens]), split=False).trace(source).at(1)

    inside = np.abs(in_glass.direction[:, 1]) > 1e-12  # skip the axial ray
    assert inside.sum() >= 8
    start, direction = in_glass.position[inside], in_glass.direction[inside]
    crossings = start[:, 2] - start[:, 1] / direction[:, 1] * direction[:, 2]
    assert np.mean(crossings) == pytest.approx(f_expected, rel=1e-3)


def biconvex(name="L1", **placement) -> Lens:
    return Lens(
        name, front=Sphere(60.0), back=Sphere(-60.0), thickness=8.0,
        material=ConstantIndex("G", 1.5), semi_diameter=15.0, **placement,
    )


def test_lens_focuses_a_parallel_fan_onto_a_sensor():
    # Thin-lens estimate f ~ R / (2 (n-1)) = 60 mm; sensor near focus.
    sensor = Sensor("IMG", aperture=Rectangle(40.0, 40.0), z=58.0)
    system = OpticalSystem([biconvex(), sensor])
    source = CollimatedSource((0.0, 0.0, -30.0), beam_radius=8.0, rays=41, fan="y")
    rays = NonSequentialTracer(system, split=False).trace(source)

    points, power = detected_on(system, rays, "IMG")
    assert len(points) >= 35  # nearly every ray lands on the sensor
    assert np.ptp(points[:, 1]) < 3.0  # concentrated near focus (spherical aberration aside)
    values, edges = np.histogram(points[:, 1], bins=64, range=(-20.0, 20.0), weights=power)
    assert values.sum() == pytest.approx(power.sum())
    peak = np.argmax(values)
    assert abs((edges[peak] + edges[peak + 1]) / 2) < 4.0  # peak near the axis


def test_sensor_absorbs_rays():
    sensor = Sensor("IMG", aperture=Rectangle(20.0, 20.0), z=5.0)
    source = PointSource((0.0, 0.0, 0.0), half_angle_deg=15.0, rays=9, fan="y")
    rays = NonSequentialTracer(OpticalSystem([sensor]), max_interactions=5).trace(source)
    # Every ray ends on the sensor, its first boundary: nothing splits off.
    assert len(rays) == 9 and rays.trajectory.shape[1] == 2
    assert rays.detected.all() and (rays.parent == -1).all()
    assert rays.power.sum() == pytest.approx(1.0)


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
    assert reference.detected[0]

    direction = np.array([0.0, slope, 1.0]) / np.hypot(1.0, slope)
    ray = NonSequentialTracer(system, split=False).trace(Rays([0.0, y_obj, tracer.object_z], direction))
    assert ray.detected[0] and ray.boundary[0] == "IMG"
    assert ray.position[0] == pytest.approx(reference.position[0], abs=1e-9)
    # Both accumulate the same optical path (the sequential tracer's restart
    # offset adds 1e-8 mm per boundary, hence the tolerance).
    assert ray.opl[0] == pytest.approx(reference.opl[0], abs=1e-6)
