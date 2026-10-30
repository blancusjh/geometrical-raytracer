"""Elements as values: immutability, replace, placement, faces (``raytracer.elements``)."""

import numpy as np
import pytest

from raytracer.coatings import AR, Absorbing, IdealBeamSplitter, PerfectMirror, Uncoated
from raytracer.elements import (
    BLADE,
    DETECTOR,
    EDGE,
    OPTICAL,
    BeamSplitter,
    Lens,
    Mirror,
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
    assert (edge.wall.z_front, edge.wall.z_back) == pytest.approx((sag, 5.0 - sag))
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
