"""Lens bodies in the drawing plane: outlines, inter-lens cuts, colour (``raytracer.viz.bodies``)."""

from pathlib import Path

import numpy as np
import pytest

from raytracer import OpticalSystem
from raytracer.elements import Lens, Mirror, Sensor
from raytracer.families import StigmaticTrain
from raytracer.geometry import Rotation
from raytracer.geometry.surfaces import Plane, Sphere
from raytracer.io import read_train
from raytracer.materials import ConstantIndex, default_materials
from raytracer.viz.bodies import body_outlines, body_overlaps, draw_system, index_color
from raytracer.viz.sag_drawing import face_section, is_meridional, lens_outline

DATA = Path(__file__).resolve().parents[1] / "data"
GLASS = ConstantIndex("G", 1.5)


def ultrawide_train():
    return read_train(
        DATA / "optical_systems/ultrawide/sf11_bk7_virtual_object.json",
        materials=default_materials(),
    )


def ultrawide_system():
    """The shipped virtual-object doublet: its lenses genuinely interpenetrate."""

    return ultrawide_train().system()


def outline_of(system, name):
    return next(b for b in body_outlines(system) if b.name == name)


def interior_collisions(a, b, resolution=600):
    """Grid points of the common bounding box lying inside both polygons."""

    from matplotlib.path import Path as PolygonPath

    z_all = np.concatenate([a[:, 0], b[:, 0]])
    h_all = np.concatenate([a[:, 1], b[:, 1]])
    z, h = np.meshgrid(
        np.linspace(z_all.min(), z_all.max(), resolution),
        np.linspace(h_all.min(), h_all.max(), resolution),
    )
    points = np.column_stack([z.ravel(), h.ravel()])
    inside_a = PolygonPath(a).contains_points(points)
    inside_b = PolygonPath(b).contains_points(points)
    return int((inside_a & inside_b).sum())


def test_every_lens_of_the_system_is_one_body():
    """The system states its bodies: one outline per lens, coloured by its index."""

    train = StigmaticTrain.singlet(n=1.5, d0=-60.0, d1=300.0, d2=60.0, thickness=8.0, semidiameter=8.0)
    bodies = body_outlines(train.system())
    assert [b.name for b in bodies] == ["L1"]
    assert bodies[0].n == pytest.approx(1.5)


def test_cemented_chain_yields_one_body_per_glass():
    """Air G1 G2 G3 Air across four surfaces: three lenses in contact."""

    train = StigmaticTrain(
        media=(1.0, 1.517122, 1.670591, 1.851280, 1.0),
        vertices=(0.0, 15.0, 25.0, 35.0),
        conjugates=(-100.0, -150.0, -600.0, 450.0, 90.0),
        semidiameter=10.0,
    )
    system = train.system()
    bodies = body_outlines(system)
    assert [b.name for b in bodies] == ["L1", "L2", "L3"]
    assert [round(b.n, 4) for b in bodies] == [1.5171, 1.6706, 1.8513]
    assert body_overlaps(system) == []  # contact is not an overlap
    # neighbours share the cemented face: the back edge of one is the front edge of the next
    np.testing.assert_allclose(bodies[0].polygon[240:480][::-1], bodies[1].polygon[:240], atol=1e-9)


def test_outline_is_a_closed_flat_rim_polygon_for_a_biconvex():
    train = StigmaticTrain.symmetric_singlet(n=1.5, d0=-30.0, thickness=40.0, semidiameter=3.5)
    system = train.system()
    polygon, (edge_front, edge_back), joined = lens_outline(system["L1"], system.pose_of("L1"))
    np.testing.assert_allclose(polygon[0], polygon[-1])  # closed
    assert not joined
    assert edge_front == pytest.approx(3.5) and edge_back == pytest.approx(3.5)
    # both faces stay inside the vertex span of the lens
    assert polygon[:, 0].min() > -5.0 and polygon[:, 0].max() < 45.0


def test_crossing_faces_join_at_the_intersection_height():
    """Faces curving toward each other meet below the rim; the outline
    must close at that sharp edge instead of drawing crossed curves."""

    train = StigmaticTrain.singlet(n=1.5, d0=-9.0, d1=-8.0, d2=25.0, thickness=1.0, semidiameter=3.7)
    system = train.system()
    lens = system["L1"]
    front, back = lens.front.surface, lens.back.surface
    h_probe = np.linspace(0, 3.7, 200)
    crossing = back.profile(h_probe) + 1.0 - front.profile(h_probe)
    assert (crossing <= 0).any()  # the faces genuinely cross below h = 3.7

    polygon, (edge, edge_back), joined = lens_outline(lens, system.pose_of("L1"))
    assert joined and edge == edge_back
    assert 0.0 < edge < 3.7
    assert edge == pytest.approx(2.553, abs=0.05)
    assert np.abs(polygon[:, 1]).max() == pytest.approx(edge, abs=1e-6)
    # at the join the two faces coincide
    z_front = front.profile(np.array([edge]))
    z_back = 1.0 + back.profile(np.array([edge]))
    assert z_front[0] == pytest.approx(z_back[0], abs=1e-6)


def test_outline_respects_the_usable_branch_of_an_ovoid():
    """A strong Cartesian surface must be clamped to its single-valued
    branch, never drawn onto the closing branch of the ovoid."""

    train = StigmaticTrain(
        media=(1.0, 1.78472, 1.0),
        vertices=(0.0, 7.0),
        conjugates=(8.0, 27.05, 70.0),  # the ultrawide's virtual-object lens
        semidiameter=22.0,
    )
    system = train.system()
    _, (edge_front, edge_back), _ = lens_outline(system["L1"], system.pose_of("L1"))
    assert edge_front <= train.surfaces[0].max_radius + 1e-9
    assert edge_back <= train.surfaces[1].max_radius + 1e-9
    assert edge_back < 22.0  # the clamp is active on the back face


def test_neighbouring_bodies_of_different_media_are_cut_apart():
    """The ultrawide's SF11 back face crosses the BK7 front face off-axis:
    two materials claim one region. The parts must be cut at the crossing
    height so their interiors are disjoint."""

    system = ultrawide_system()
    overlaps = body_overlaps(system)
    assert [(a, b) for a, b, _ in overlaps] == [("L1", "L2")]
    assert overlaps[0][2] == pytest.approx(5.8584, abs=1e-3)

    raw = [lens_outline(system[n], system.pose_of(n))[0] for n in ("L1", "L2")]
    assert interior_collisions(*raw) > 0  # the uncut solids interpenetrate

    cut = {b.name: b.polygon for b in body_outlines(system)}
    assert interior_collisions(cut["L1"], cut["L2"]) == 0


def test_no_part_claims_the_ambiguous_region():
    """The space between the crossed faces above the cut belongs to both
    solids, so the drawn parts must leave it hollow — while each part keeps
    its unambiguous glass: the SF11 body its full front face (its rim stays
    clear in front of the region), the BK7 body its back face over the band
    behind the SF11 back face, up to where that band closes (h* ≈ 8.02,
    where the two back faces meet)."""

    system = ultrawide_system()
    h_cut = body_overlaps(system)[0][2]
    cut = {b.name: b.polygon for b in body_outlines(system)}
    assert np.abs(cut["L2"][:, 1]).max() == pytest.approx(8.0215, abs=5e-3)
    assert np.abs(cut["L1"][:, 1]).max() == pytest.approx(22.0, abs=1e-9)

    from matplotlib.path import Path as PolygonPath

    l1, l2 = system["L1"], system["L2"]
    back1, front2 = l1.back.surface, l2.front.surface
    h = np.linspace(h_cut * 1.01, 0.98 * back1.max_radius, 40)
    z_lo = system.pose_of("L2").position[2] + front2.profile(h)
    z_hi = l1.thickness + back1.profile(h)
    probe = np.column_stack([0.5 * (z_lo + z_hi), h])  # zone midline
    for polygon in cut.values():
        assert not PolygonPath(polygon).contains_points(probe).any()


def test_valid_aperture_fan_refracts_only_on_drawn_glass():
    """A ±43° fan stays inside the valid aperture: every refraction lands
    on its face within the drawn part, no kink sits strictly inside a body,
    and no artifact warning is raised."""

    import warnings

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as PolygonPath

    train = ultrawide_train()
    system = train.system()
    fig, ax = plt.subplots()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            paths = draw_system(
                ax, train,
                fields=[np.sin(np.radians(a)) for a in (15.0, 30.0, 43.0)],
                rays=7, launch=-25.0,
            )
        assert paths is not None
        kinks = paths[:, 1:-1, :]  # lens-face hits of every traced ray

        extents, outlines = {}, {}
        for body in body_outlines(system):
            extents[f"{body.name}.front"], extents[f"{body.name}.back"] = body.extents
            outlines[body.name] = body.polygon
        for k, step in enumerate(system.path[:-1]):
            assert np.abs(kinks[:, k, 1]).max() <= extents[step.face.name] + 1e-9

        # no kink strictly inside a body: "inside" means contained AND more
        # than 5 µm from the boundary polyline
        def strictly_inside(polygon, pts, tol=5e-3):
            contained = PolygonPath(polygon).contains_points(pts)
            edge_a, edge_b = polygon[:-1], polygon[1:]
            span = edge_b - edge_a
            length2 = np.maximum((span * span).sum(axis=1), 1e-30)
            for idx in np.nonzero(contained)[0]:
                offset = pts[idx] - edge_a
                t = np.clip((offset * span).sum(axis=1) / length2, 0.0, 1.0)
                nearest = edge_a + t[:, None] * span
                if np.hypot(*(pts[idx] - nearest).T).min() > tol:
                    return True
            return False

        points = kinks[:, :, [2, 1]].reshape(-1, 2)
        for polygon in outlines.values():
            assert not strictly_inside(polygon, points)
    finally:
        plt.close(fig)


def test_rays_beyond_the_valid_aperture_warn():
    """A 55°-aim fan refracts where no medium is defined — draw_system says so."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        with pytest.warns(UserWarning, match="valid aperture"):
            draw_system(ax, ultrawide_train(), fields=(np.sin(np.radians(55.0)),), rays=3, launch=-30.0)
    finally:
        plt.close(fig)


def test_index_color_is_monotone_darker_with_density():
    assert index_color(1.0) is None
    light = index_color(1.3)
    dense = index_color(1.9)
    assert light is not None and dense is not None

    def brightness(color):
        r, g, b = (int(color[k:k + 2], 16) for k in (1, 3, 5))
        return r + g + b

    assert brightness(dense[0]) < brightness(light[0])  # denser -> darker fill
    assert brightness(light[1]) < brightness(light[0])  # edge darker than face


# -- the drawing plane is the y-z section of the 3-D model ----------------------


def biconvex(**placement):
    return Lens("L", front=Sphere(40.0), back=Sphere(-40.0), thickness=6.0, material=GLASS, semi_diameter=10.0, **placement)


def test_a_lens_tilted_in_the_yz_plane_is_drawn_as_its_exact_section():
    """Rotating a lens about x turns its drawn outline rigidly in (z, y)."""

    flat = lens_outline(biconvex(pos=(0.0, 0.0, 0.0)))[0]
    angle = 20.0
    tilted = biconvex(pos=(0.0, 3.0, 50.0), rot=Rotation.axis_angle((1.0, 0.0, 0.0), -angle))
    assert is_meridional(tilted.pose)
    drawn = lens_outline(tilted)[0]
    # the same outline turned by +angle (z toward +y) and moved to (z, y) = (50, 3)
    a = np.radians(angle)
    turn = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    expected = flat @ turn.T + np.array([50.0, 3.0])
    np.testing.assert_allclose(drawn, expected, atol=1e-12)


def test_a_decentred_section_is_the_chord_of_the_rim():
    """A lens displaced along x is cut off-centre: a shorter chord, the faces' sag along it."""

    lens = biconvex(pos=(6.0, 0.0, 0.0))
    polygon, (e_front, e_back), _ = lens_outline(lens)
    assert e_front == pytest.approx(8.0) and e_back == pytest.approx(8.0)  # sqrt(10² - 6²)
    # the front face's z on the chord is the sag at radius hypot(6, y)
    y = polygon[:240, 1]
    np.testing.assert_allclose(polygon[:240, 0], Sphere(40.0).profile(np.hypot(6.0, y)), atol=1e-12)


def test_a_fold_mirror_section_is_a_tilted_segment():
    mirror = Mirror("FM", surface=Plane(), aperture=5.0, pos=(0.0, 0.0, 30.0), normal=(0.0, 1.0, -1.0))
    (curve,) = face_section(mirror.surface, mirror.pose, mirror.aperture)
    ends = sorted([tuple(curve[0]), tuple(curve[-1])])
    np.testing.assert_allclose(ends[0], [30.0 - 5.0 / np.sqrt(2), -5.0 / np.sqrt(2)], atol=1e-12)
    np.testing.assert_allclose(ends[1], [30.0 + 5.0 / np.sqrt(2), 5.0 / np.sqrt(2)], atol=1e-12)
    np.testing.assert_allclose(np.diff(curve, axis=0) @ [1.0, -1.0], 0.0, atol=1e-12)  # a straight 45° segment


def test_an_element_turned_out_of_the_plane_is_projected():
    lens = biconvex(rot=Rotation.axis_angle((0.0, 1.0, 0.0), 30.0))
    assert not is_meridional(lens.pose)
    polygon = lens_outline(lens)[0]
    # the meridian (local x = 0) keeps its heights; its z is foreshortened by cos 30°
    assert np.abs(polygon[:, 1]).max() == pytest.approx(10.0)
    assert polygon[:, 0].max() == pytest.approx(np.cos(np.radians(30.0)) * 6.0, abs=1e-4)  # sampled vertex


def test_draw_system_draws_a_folded_system():
    """Lens, fold mirror, tilted lens and sensor after the fold: everything drawn, rays bent at the fold."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from raytracer.tracing import SequentialTracer

    system = OpticalSystem(
        [
            biconvex(z=0.0).replace(name="L1"),
            Mirror("FM", surface=Plane(), aperture=12.0, pos=(0.0, 0.0, 40.0), normal=(0.0, 1.0, -1.0)),
            biconvex(pos=(0.0, 20.0, 40.0), axis=(0.0, 1.0, 0.0)).replace(name="L2"),
            Sensor("IMG", aperture=20.0, pos=(0.0, 60.0, 40.0), axis=(0.0, 1.0, 0.0)),
        ],
        path=["L1.front", "L1.back", "FM:reflect", "L2.front", "L2.back", "IMG"],
    )
    tracer = SequentialTracer(system, object_z=-100.0)
    fig, ax = plt.subplots()
    try:
        paths = draw_system(ax, tracer, fields=(0.0, 3.0), na=0.04, rays=5)
        assert paths is not None
        # after the fold the rays travel along +y
        assert np.all(paths[:, -1, 1] > 50.0)
        assert len(ax.patches) == 2  # the two lens bodies
        assert sum(1 for line in ax.lines if line.get_color() == "#202020") == 1  # the mirror arc
    finally:
        plt.close(fig)
