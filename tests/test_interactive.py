"""Headless tests of the interactive model layer (no GL)."""

import numpy as np
import pytest

from raytracer import OpticalSystem
from raytracer.elements import Lens, Sensor
from raytracer.geometry.apertures import Rectangle
from raytracer.geometry.surfaces import Sphere
from raytracer.materials import ConstantIndex
from raytracer.sources import PointSource
from raytracer.viz.interactive import InteractiveSession, editable_for
from raytracer.viz.drawing import LensBodyItem, RaySegmentsItem, SensorItem


@pytest.fixture()
def session():
    lens = Lens(
        "lens", front=Sphere(60.0), back=Sphere(-60.0), thickness=9.0,
        material=ConstantIndex("N1.5168", 1.5168), semi_diameter=16.0,
    )
    sensor = Sensor("IMG", aperture=Rectangle(30.0, 30.0), z=171.4)
    source = PointSource((0.0, 0.0, -90.0), half_angle_deg=9.0, rays=51, fan="y")
    return InteractiveSession(sources=[source], system=OpticalSystem([lens, sensor]), split=False, max_interactions=5)


def test_retrace_builds_drawing_and_hits(session):
    drawing = session.retrace()
    kinds = {type(item) for item in drawing.items}
    assert {LensBodyItem, SensorItem, RaySegmentsItem} <= kinds
    assert len(session.hits("IMG")) > 40
    sensor_item = next(i for i in drawing.items if isinstance(i, SensorItem))
    assert sensor_item.hits is not None and len(sensor_item.hits) > 40


def test_drag_source_handle_moves_focus(session):
    session.retrace()
    y_before = np.median(session.hits("IMG").point[:, 1])

    editable = session.editables[0]  # the point source
    handle = editable.handles[0]
    session.drag_handle(handle, np.array([-90.0, 3.0]))  # raise the source (z, y)

    y_after = np.median(session.hits("IMG").point[:, 1])
    # Imaging inverts: raising the object lowers the image.
    assert y_after < y_before - 1.0


def test_drag_lens_rebuilds_the_system(session):
    session.retrace()
    old_system = session.system
    lens_editable = next(e for e in session.editables if e.label == "lens")
    session.drag_handle(lens_editable.handles[0], np.array([5.0, 2.0]))
    assert session.system is not old_system  # a new value, the old one untouched
    assert old_system.pose_of("lens").position == pytest.approx([0.0, 0.0, 0.0])
    assert session.system.pose_of("lens").position == pytest.approx([0.0, 2.0, 5.0])
    assert lens_editable.handles[0].get() == pytest.approx([5.0, 2.0])


def test_parameter_cycle_and_adjust_retraces(session):
    session.retrace()
    first = session.selected
    assert first is not None
    second = session.cycle_selection()
    assert second != first

    lens_editable = next(e for e in session.editables if e.label == "lens")
    r1 = next(p for p in lens_editable.parameters if p.name == "r1")
    old_radius = r1.get()
    r1.set(old_radius * 0.8)  # stronger curvature -> shorter focal length
    session.retrace()
    assert session.system["lens"].front.surface.radius == pytest.approx(old_radius * 0.8)


def test_lens_rebuild_preserves_position_and_updates_the_rim(session):
    session.drag_handle(next(e for e in session.editables if e.label == "lens").handles[0], np.array([4.0, 0.0]))
    editable = next(e for e in session.editables if e.label == "lens")
    thickness = next(p for p in editable.parameters if p.name == "thickness")
    thickness.set(15.0)
    lens = session.system["lens"]
    assert lens.thickness == pytest.approx(15.0)
    assert session.system.pose_of("lens").position[2] == pytest.approx(4.0)
    # the rim follows the new geometry
    assert lens.edge_thickness == pytest.approx(15.0 + Sphere(-60.0).profile(16.0) - Sphere(60.0).profile(16.0), abs=1e-9)


def test_tilting_a_lens_turns_it_in_the_drawing_plane(session):
    editable = next(e for e in session.editables if e.label == "lens")
    angle = next(p for p in editable.parameters if p.name == "angle_deg")
    angle.set(10.0)
    axis = session.system.pose_of("lens").axis
    assert axis == pytest.approx([0.0, np.sin(np.radians(10.0)), np.cos(np.radians(10.0))])
    session.retrace()


def test_dragging_a_sensor_end_turns_and_resizes_it(session):
    session.retrace()
    editable = next(e for e in session.editables if e.label == "IMG")
    p0, p1 = (h.get() for h in editable.handles)
    assert p0 == pytest.approx([171.4, -15.0]) and p1 == pytest.approx([171.4, 15.0])
    session.drag_handle(editable.handles[1], np.array([181.4, 15.0]))
    pose = session.system.pose_of("IMG")
    assert pose.position == pytest.approx([0.0, 0.0, 176.4])
    # still facing the incoming light, its normal now perpendicular to the dragged segment
    assert pose.axis == pytest.approx(np.array([0.0, -10.0, 30.0]) / np.hypot(10.0, 30.0))
    assert session.system["IMG"].aperture.height == pytest.approx(np.hypot(10.0, 30.0))
    assert editable.handles[1].get() == pytest.approx([181.4, 15.0])


def test_pick_handle_radius(session):
    picked = session.pick_handle(np.array([-90.5, 0.2]), radius_world=1.0)
    assert picked is not None
    assert picked[0].label == "source"
    assert session.pick_handle(np.array([-50.0, 30.0]), radius_world=1.0) is None


def test_status_reports_selection(session):
    text = session.status()
    assert "Tab" in text and "=" in text


def test_editable_for_rejects_unknown():
    with pytest.raises(TypeError):
        editable_for(object())


def test_editing_an_element_needs_its_session(session):
    with pytest.raises(TypeError):
        editable_for(session.system["lens"])
