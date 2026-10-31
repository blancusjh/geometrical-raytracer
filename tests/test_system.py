"""The optical system: placed elements, the media between them, and the path.

The media on each side of every boundary and the sequential path are
derived from the placed elements (``raytracer.system``); these tests state
what must be derived for the configurations a designer actually builds:
cemented contact, immersion, a double-pass lens, a virtual stop, folded
and beam-splitting layouts, and prescription tables in both directions.
"""

from pathlib import Path

import numpy as np
import pytest

from raytracer import AIR, OpticalSystem, Pose, Rays, Rotation, TableRow
from raytracer.elements import BeamSplitter, Lens, Mirror, Sensor, Stop
from raytracer.geometry.apertures import Circle
from raytracer.geometry.surfaces import Plane, Sphere
from raytracer.io import read_prescription
from raytracer.materials import ConstantIndex
from raytracer.system import Assembly, Gap
from raytracer.rays import Status
from raytracer.tracing import NonSequentialTracer, ParaxialModel, SequentialTracer

DATA = Path(__file__).resolve().parents[1] / "data" / "optical_systems"
COOKE = DATA / "photographic/cooke_triplet_prescription.csv"
US7557996 = DATA / "lithography/US7557996_Fig3_Table3_prescription.csv"

G1, G2 = ConstantIndex("G1", 1.5), ConstantIndex("G2", 1.7)
WATER = ConstantIndex("WATER", 1.33)


def meridional_rays(z=-100.0):
    o = np.array([[0.0, 3.0, z], [1.0, -2.0, z], [0.0, 0.0, z]])
    d = np.array([[0.0, -0.02, 1.0], [0.01, 0.03, 1.0], [0.0, 0.01, 1.0]])
    return o, d / np.linalg.norm(d, axis=1, keepdims=True)


# -- what the system derives ------------------------------------------------------


def test_separate_lenses_in_contact_share_one_boundary():
    l1 = Lens("L1", surfaces=[Sphere(40.0), Sphere(-30.0)], thicknesses=[6.0], materials=[G1], semi_diameter=10.0)
    l2 = Lens("L2", surfaces=[Sphere(-30.0), Plane()], thicknesses=[2.0], materials=[G2], semi_diameter=10.0, z=6.0)
    system = OpticalSystem([l1, l2, Sensor("IMG", z=60.0)])
    contact = system.face("L1.2")
    assert contact is system.face("L2.1")
    assert set(contact.names) == {"L1.2", "L2.1"}
    assert (contact.medium_minus, contact.medium_plus) == (G1, G2)
    optical = [f for f in system.faces if f.role != "edge"]
    assert len(optical) == 4  # front, the contact face, back, sensor
    assert [s.event for s in system.path] == ["transmit"] * 3 + ["detect"]
    # The element order does not matter: the path follows the axis.
    shuffled = OpticalSystem([l2, Sensor("IMG", z=60.0), l1])
    assert [s.face.pose.position[2] for s in shuffled.path] == [0.0, 6.0, 8.0, 60.0]


def test_compose_places_touching_lenses_in_contact():
    l1 = Lens("L1", surfaces=[Sphere(40.0), Sphere(-30.0)], thicknesses=[6.0], materials=[G1], semi_diameter=10.0)
    l2 = Lens("L2", surfaces=[Sphere(-30.0), Plane()], thicknesses=[2.0], materials=[G2], semi_diameter=10.0)
    system = OpticalSystem.compose(l1, l2, Gap(50.0), Sensor("IMG"))
    assert system.pose_of("L2").position.tolist() == [0.0, 0.0, 6.0]
    assert system.face("L2.1").medium_minus == G1
    with pytest.raises(ValueError, match="share the contact surface"):
        OpticalSystem.compose(l1, l2.replace(surfaces=[Sphere(-31.0), Plane()]), Sensor("IMG"))


def test_overlapping_bodies_are_rejected():
    lens = Lens("A", surfaces=[Sphere(40.0), Plane()], thicknesses=[6.0], materials=[G1], semi_diameter=10.0)
    with pytest.raises(ValueError, match="overlap"):
        OpticalSystem([lens, lens.replace(name="B", thicknesses=[3.0])]).faces


def test_immersion_layer_and_a_detector_at_its_back():
    """A liquid is a lens of liquid: in contact with the glass in front of it,
    and with the sensor at its back face, so the image space is the liquid."""

    system = OpticalSystem([
        Lens("L", surfaces=[Sphere(10.0), Plane()], thicknesses=[5.0], materials=[G1], semi_diameter=4.0),
        Lens("W", surfaces=[Plane(), Plane()], thicknesses=[2.0], materials=[WATER], semi_diameter=4.0, z=5.0),
        Sensor("IMG", z=7.0),
    ])
    assert (system.face("W.1").medium_minus, system.face("W.1").medium_plus) == (G1, WATER)
    image = system.face("IMG")
    assert set(image.names) == {"W.2", "IMG"} and image.role == "detector"
    assert len(system.path) == 3 and system.path[-1].event == "detect"
    tracer = SequentialTracer(system)
    assert tracer.n_image == pytest.approx(1.33)
    # An immersion objective: the system may also be immersed as a whole.
    immersed = system.replace(ambient=WATER)
    assert immersed.face("L.1").medium_minus == WATER
    assert SequentialTracer(immersed).n_object == pytest.approx(1.33)


def test_double_pass_lens_from_a_prescription_table():
    """Lens, flat mirror, back through the same lens: the table's repeated
    surfaces are one physical lens, visited twice, and the folded trace
    equals the unfolded one (the mirror image of the lens behind a gap of 2d)."""

    R1, R2, t, d, s = 80.0, -200.0, 6.0, 30.0, 40.0
    folded = OpticalSystem.from_table([
        TableRow(Sphere(R1), t, G2, 15.0),
        TableRow(Sphere(R2), d, AIR, 15.0),
        TableRow(Plane(), -d, None, 15.0, "mirror"),
        TableRow(Sphere(R2), -t, G2, 15.0),
        TableRow(Sphere(R1), -s, AIR, 15.0),
    ])
    assert [e.name for e in folded.elements] == ["L1", "M1", "IMG"]
    assert [s.name for s in folded.path] == ["L1.1", "L1.2", "M1", "L1.2", "L1.1", "IMG"]
    assert folded.path[1].face is folded.path[3].face
    assert folded.path[2].event == "reflect"

    unfolded = OpticalSystem.from_table([
        TableRow(Sphere(R1), t, G2, 15.0),
        TableRow(Sphere(R2), 2 * d, AIR, 15.0),
        TableRow(Sphere(-R2), t, G2, 15.0),
        TableRow(Sphere(-R1), s, AIR, 15.0),
    ])
    o, dirs = meridional_rays()
    a = SequentialTracer(folded, restart_offset=0.0).trace(Rays(o, dirs))
    b = SequentialTracer(unfolded, restart_offset=0.0).trace(Rays(o, dirs))
    assert a.detected.all() and b.detected.all()
    np.testing.assert_allclose(a.position[:, :2], b.position[:, :2], atol=1e-12)
    np.testing.assert_allclose(a.opl, b.opl, atol=1e-10)
    assert ParaxialModel(SequentialTracer(folded)).effective_focal_length() == pytest.approx(
        ParaxialModel(SequentialTracer(unfolded)).effective_focal_length(), rel=1e-12
    )


def test_double_pass_is_one_surface_with_one_clear_aperture():
    """US 7,557,996 crosses its CaF2 lens twice and lists a clear
    semi-diameter per pass; the physical surface is one, clear over the
    wider of the two."""

    system = read_prescription(US7557996).system
    first, second = system.path[4], system.path[8]
    assert first.face is second.face
    assert first.face.clear_aperture == Circle(67.241)
    # The last lens holds the HIINDEX1 glass and the HIINDEX2 liquid against it, one element; the
    # face between them keeps its own 18.135 mm, not its lens's 58.621 mm.
    last = system["L21"]
    assert [m.name for m in last.materials] == ["HIINDEX1", "HIINDEX2"] and last.semi_diameter == 58.621
    cemented = system.face("L21.2")
    assert cemented.names == ("L21.2",) and cemented.clear_aperture == Circle(18.135)
    assert set(system.face("IMG").names) == {"IMG", "L21.3"}  # the wafer at the back of the liquid


def test_virtual_stop_inside_a_lens():
    """A table may put the stop plane inside a lens: it is then a reference
    plane only — the sequential trace clips there, light does not."""

    rows = [
        TableRow(Sphere(50.0), 10.0, G1, 12.0),
        TableRow(Plane(), -4.0, AIR, 12.0),
        TableRow(Plane(), 44.0, AIR, 3.0, kind="stop"),
    ]
    system = OpticalSystem.from_table(rows)
    assert system["STO"].virtual
    assert system.stop == "STO" and system.stop_index == 2
    assert system.pose_of("STO").position[2] == pytest.approx(6.0)

    rays = Rays([[0.0, 1.0, -20.0], [0.0, 8.0, -20.0]], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], np.nan, 1.0)
    trace = SequentialTracer(system).trace(rays)
    assert trace.status.tolist() == [Status.DETECTED, Status.VIGNETTED]
    assert trace.boundary[1] == system.path[1].name  # clipped at the stop, step 2
    passed = NonSequentialTracer(system, split=False).trace(rays)
    assert np.isnan(passed.at("STO").position).all()  # light passes a virtual stop untouched
    assert (passed.detected & (passed.boundary == "IMG")).sum() == 2


def test_a_real_stop_blocks_light_non_sequentially():
    system = OpticalSystem.compose(Stop("S", aperture=2.0), Gap(10.0), Sensor("IMG"))
    rays = Rays([[0.0, 1.0, -5.0], [0.0, 3.0, -5.0], [0.0, 5.0, -5.0]], [[0, 0, 1]] * 3, np.nan, 1.0)
    rays = NonSequentialTracer(system).trace(rays)
    assert rays.boundary.tolist() == ["IMG", "S", "IMG"]  # through the opening, on the blade, around it
    assert rays.status.tolist() == [Status.DETECTED, Status.ABSORBED, Status.DETECTED]
    assert len(rays) == 3  # nothing splits off a blade
    assert rays.position[rays.detected, 1].tolist() == [1.0, 5.0]


# -- the path ---------------------------------------------------------------------


def test_path_inference_along_a_tilted_common_axis():
    axis = np.array([0.0, np.sin(0.3), np.cos(0.3)])
    lens = Lens("L", surfaces=[Sphere(40.0), Sphere(-40.0)], thicknesses=[4.0], materials=[G1], semi_diameter=8.0, axis=axis)
    sensor = Sensor("IMG", pos=50.0 * axis, axis=axis)
    stop = Stop("S", aperture=5.0, pos=-5.0 * axis, axis=axis)
    system = OpticalSystem([sensor, lens, stop])
    assert [s.name for s in system.path] == ["S", "L.1", "L.2", "IMG"]
    assert [s.event for s in system.path] == ["stop", "transmit", "transmit", "detect"]
    assert system.stop_index == 0 and not system.has_stated_path


@pytest.mark.parametrize(
    "extra, message",
    [
        (Mirror("M", surface=Plane(), aperture=5.0, z=20.0), "reflects"),
        (BeamSplitter("BS", aperture=5.0, z=20.0), "reflects"),
        (Sensor("IMG", z=60.0, rot=(5.0, 0.0, 0.0)), "share one axis"),
        (Sensor("IMG", pos=(1.0, 0.0, 60.0)), "centred on one axis"),
    ],
    ids=["folded", "beam-splitter", "tilted", "decentred"],
)
def test_path_inference_refuses_what_it_cannot_order(extra, message):
    lens = Lens("L", surfaces=[Sphere(40.0), Sphere(-40.0)], thicknesses=[4.0], materials=[G1], semi_diameter=8.0)
    system = OpticalSystem([lens, extra])
    with pytest.raises(ValueError, match=message):
        system.path


def beam_splitter_bench():
    """A 45-degree beam splitter at z=20: the transmitted arm continues along
    +z, the reflected arm leaves along -y."""

    bs = BeamSplitter("BS", aperture=8.0, z=20.0, normal=(0.0, 1.0, 1.0))
    straight = Sensor("CAM_T", z=60.0)
    side = Sensor("CAM_R", pos=(0.0, -40.0, 20.0), normal=(0.0, -1.0, 0.0))
    return [bs, straight, side]


def test_stated_paths_choose_the_beam_splitter_branch():
    parts = beam_splitter_bench()
    reflect = OpticalSystem(parts, path=["BS:reflect", "CAM_R"])
    transmit = OpticalSystem(parts, path=["BS:transmit", "CAM_T"])
    assert [s.event for s in reflect.path] == ["reflect", "detect"]
    o = np.array([[0.0, 1.0, 0.0], [2.0, -1.0, 0.0]])
    d = np.tile([0.0, 0.0, 1.0], (2, 1))
    r = SequentialTracer(reflect).trace(Rays(o, d))
    t = SequentialTracer(transmit).trace(Rays(o, d))
    assert r.detected.all() and t.detected.all()
    # Reflection about the 45-degree plane swaps the roles of y and z.
    np.testing.assert_allclose(r.position, [[0.0, -40.0, 19.0], [2.0, -40.0, 21.0]], atol=1e-12)
    np.testing.assert_allclose(t.position, [[0.0, 1.0, 60.0], [2.0, -1.0, 60.0]], atol=1e-12)
    np.testing.assert_allclose(r.direction, np.tile([0.0, -1.0, 0.0], (2, 1)), atol=1e-12)
    # The non-sequential tracer finds both branches at once, half the power each.
    rays = NonSequentialTracer(OpticalSystem(parts)).trace(Rays(o, d, np.nan, 0.5))

    def on(camera):
        return rays[rays.detected & (rays.boundary == camera)]

    np.testing.assert_allclose(on("CAM_R").position, r.position, atol=1e-6)
    np.testing.assert_allclose(on("CAM_T").position, t.position, atol=1e-6)
    assert on("CAM_R").power.sum() == pytest.approx(0.5)


def test_stated_path_errors():
    parts = beam_splitter_bench()
    with pytest.raises(ValueError, match="say which branch"):
        OpticalSystem(parts, path=["BS", "CAM_R"]).path
    with pytest.raises(KeyError, match="no boundary named"):
        OpticalSystem(parts, path=["BX:reflect", "CAM_R"]).path
    with pytest.raises(ValueError, match="unknown event"):
        OpticalSystem(parts, path=["BS:bounce", "CAM_R"]).path
    lens = Lens("L", surfaces=[Sphere(40.0), Sphere(-40.0)], thicknesses=[4.0], materials=[G1], semi_diameter=8.0)
    with pytest.raises(ValueError, match="edge"):
        OpticalSystem([lens], path=["L.edge1"]).path
    with pytest.raises(ValueError, match="detector"):
        SequentialTracer(OpticalSystem(parts, path=["BS:reflect"]))


def test_folding_mirror_with_a_stated_path():
    lens = Lens("L", surfaces=[Sphere(50.0), Sphere(-50.0)], thicknesses=[5.0], materials=[G1], semi_diameter=10.0)
    fold = Mirror("FOLD", surface=Plane(), aperture=12.0, z=30.0, normal=(0.0, 1.0, 1.0))
    straight = OpticalSystem.compose(lens, Gap(100.0), Sensor("IMG"))
    # The folded twin: the mirror plane y + z = 30 turns the axis onto -y,
    # and the image plane with it.
    sensor = Sensor("IMG", pos=(0.0, 30.0 - 105.0, 30.0), normal=(0.0, -1.0, 0.0))
    folded = OpticalSystem([lens, fold, sensor], path=["L.1", "L.2", "FOLD:reflect", "IMG"])
    o, d = meridional_rays()
    a = SequentialTracer(straight, restart_offset=0.0).trace(Rays(o, d))
    b = SequentialTracer(folded, restart_offset=0.0).trace(Rays(o, d))
    assert a.detected.all() and b.detected.all()
    # Unfold: the mirror maps (x, y, z) to (x, 30 - z, 30 - y).
    np.testing.assert_allclose(b.position[:, 0], a.position[:, 0], atol=1e-11)
    np.testing.assert_allclose(30.0 - b.position[:, 2], a.position[:, 1], atol=1e-11)
    np.testing.assert_allclose(30.0 - b.position[:, 1], a.position[:, 2], atol=1e-11)
    np.testing.assert_allclose(b.opl, a.opl, atol=1e-10)


# -- prescription tables ---------------------------------------------------------------


@pytest.mark.parametrize("csv", [COOKE, US7557996], ids=["cooke", "us7557996"])
def test_table_round_trip(csv):
    system = read_prescription(csv).system
    rows = system.table()
    again = OpticalSystem.from_table(rows)
    assert again == system
    assert again.table() == rows


def test_table_rows_describe_the_path():
    rows = read_prescription(COOKE).system.table()
    assert [r.kind for r in rows] == ["refract"] * 4 + ["stop"] + ["refract"] * 2
    assert sum(r.gap for r in rows) == pytest.approx(59.95)
    assert rows[4].semi_diameter == pytest.approx(read_prescription(COOKE).system["STO"].aperture.radius)


def test_table_without_semi_diameter_falls_back_on_what_the_surfaces_allow():
    system = OpticalSystem.from_table([TableRow(Sphere(50.0), 5.0, G1), TableRow(Plane(), 20.0, AIR)])
    assert system["L1"].semi_diameter == 50.0  # a sphere's sag exists out to r = |R|
    with pytest.raises(ValueError, match="semi-diameter"):
        OpticalSystem.from_table([TableRow(Plane(), 5.0, G1), TableRow(Plane(), 20.0, AIR)])


def test_air_rows_that_change_nothing_are_dropped():
    rows = [
        TableRow(Plane(), 5.0, AIR, 10.0),  # air to air: nothing there
        TableRow(Sphere(50.0), 5.0, G1, 10.0),
        TableRow(Plane(), 20.0, AIR, 10.0),
    ]
    system = OpticalSystem.from_table(rows)
    assert [s.name for s in system.path] == ["L1.1", "L1.2", "IMG"]
    assert system.pose_of("L1").position[2] == 5.0


# -- rigid motions ---------------------------------------------------------------------


def test_a_moved_system_traces_identically_after_moving_the_rays():
    prescription = read_prescription(COOKE)
    system = prescription.system
    axis, angle, pivot, shift = (1.0, 2.0, -0.5), 37.0, (3.0, -4.0, 20.0), (10.0, 5.0, -7.0)
    moved = system.rotated(angle, axis, about=pivot).translated(shift)
    motion = Pose(shift) @ Pose(pivot) @ Pose(None, Rotation.axis_angle(axis, angle)) @ Pose(-np.asarray(pivot))
    for element in system.elements:
        assert moved.pose_of(element.name).isclose(motion @ system.pose_of(element.name), atol=1e-12)

    o, d = meridional_rays()
    a = prescription.tracer().trace(Rays(o, d))
    b = SequentialTracer(moved, wavelength_um=prescription.wavelength_um).trace(Rays(
        motion.to_world(o), motion.direction_to_world(d)
    ))
    assert a.detected.all() and b.detected.all()
    np.testing.assert_allclose(b.position, motion.to_world(a.position), atol=1e-11)
    np.testing.assert_allclose(b.direction, motion.direction_to_world(a.direction), atol=1e-13)
    np.testing.assert_allclose(b.opl, a.opl, atol=1e-10)


def test_an_assembly_moves_its_elements_together():
    system = read_prescription(COOKE).system
    pose = Pose((1.0, 2.0, 3.0), Rotation.euler(10.0, 20.0, 30.0))
    grouped = OpticalSystem([Assembly("triplet", system.components, pose=pose)], path=[s.name for s in system.path])
    for element in system.elements:
        assert grouped.pose_of(element.name).isclose(pose @ system.pose_of(element.name), atol=1e-12)
    o, d = meridional_rays()
    a = SequentialTracer(system, wavelength_um=0.5876).trace(Rays(o, d))
    b = SequentialTracer(grouped, wavelength_um=0.5876).trace(Rays(pose.to_world(o), pose.direction_to_world(d)))
    np.testing.assert_allclose(b.position, pose.to_world(a.position), atol=1e-11)


def test_systems_are_values():
    system = read_prescription(COOKE).system
    assert system == read_prescription(COOKE).system
    with pytest.raises(AttributeError):
        system.ambient = G1
    thicker = system.with_component("L2", system["L2"].replace(thicknesses=[1.5]))
    assert thicker["L2"].thickness == 1.5 and system["L2"].thickness == 1.0
    assert thicker != system
    with pytest.raises(KeyError):
        system.with_component("nope", system["L2"])
    with pytest.raises(ValueError, match="unique"):
        OpticalSystem([Sensor("A"), Sensor("A", z=1.0)])


def test_a_stop_whose_plane_runs_through_glass_is_virtual():
    """The Cooke triplet's stop sits at the F4 lens's curved back vertex: off
    axis its plane is inside the glass, so it is a reference plane, not a blade."""

    cooke = read_prescription(Path(__file__).resolve().parents[1] / "data/optical_systems/photographic/cooke_triplet_prescription.csv")
    gauss = read_prescription(Path(__file__).resolve().parents[1] / "data/optical_systems/photographic/double_gauss_prescription.csv")
    assert cooke.system["STO"].virtual
    assert not gauss.system["STO"].virtual


def test_table_mirror_without_semi_diameter_is_unbounded_and_stop_without_is_an_error():
    rows = [TableRow(Sphere(-100.0), -50.0, None, None, kind="mirror")]
    system = OpticalSystem.from_table(rows)
    assert system["M1"].aperture is None
    with pytest.raises(ValueError, match="stop"):
        OpticalSystem.from_table([TableRow(Plane(), 10.0, AIR, None, kind="stop"), TableRow(Sphere(50.0), 5.0, G1, 10.0), TableRow(Plane(), 20.0, AIR, 10.0)])
