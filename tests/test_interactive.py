"""Headless tests of the interactive model layer (no GL)."""

import numpy as np
import pytest

from raytracer.nonseq import Lens2D, PointSource2D, Screen2D, TraceConfig
from raytracer.viz.interactive import InteractiveSession, editable_for


@pytest.fixture()
def session():
    lens = Lens2D.from_radii(
        r1=60.0, r2=-60.0, thickness=9.0, semidiameter=16.0, n=1.5168,
        vertex=(0.0, 0.0), name="lens",
    )
    screen = Screen2D([171.4, -15.0], [171.4, 15.0])
    source = PointSource2D(
        origin=np.array([-90.0, 0.0]),
        axis_direction=np.array([1.0, 0.0]),
        aperture=np.deg2rad(18.0),
        samples=51,
    )
    return InteractiveSession(
        sources=[source], elements=[lens], screens=[screen],
        trace_config=TraceConfig(max_generations=5),
    )


def test_retrace_builds_scene_and_hits(session):
    scene = session.retrace()
    assert scene.items
    screen = session.screens[0]
    assert len(screen.hits) > 40


def test_drag_source_handle_moves_focus(session):
    session.retrace()
    screen = session.screens[0]
    y_before = np.median([hit.point[1] for hit in screen.hits])

    editable = session.editables[0]  # the point source
    handle = editable.handles[0]
    session.drag_handle(handle, np.array([-90.0, 3.0]))  # raise the source

    y_after = np.median([hit.point[1] for hit in screen.hits])
    # Imaging inverts: raising the object lowers the image.
    assert y_after < y_before - 1.0


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
    assert lens_editable.obj.front.profile.radius == pytest.approx(old_radius * 0.8)


def test_lens_rebuild_preserves_position_and_updates_rims(session):
    lens = session.elements[0]
    editable = next(e for e in session.editables if e.label == "lens")
    thickness = next(p for p in editable.parameters if p.name == "thickness")
    thickness.set(15.0)
    axis_gap = lens.back.vertex[0] - lens.front.vertex[0]
    assert axis_gap == pytest.approx(15.0)
    # Rims follow the new geometry.
    rim_dx = lens.rims[0].p1[0] - lens.rims[0].p0[0]
    assert rim_dx == pytest.approx(
        15.0 + lens.back.profile.sag(16.0) - lens.front.profile.sag(16.0), abs=1e-9
    )


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
