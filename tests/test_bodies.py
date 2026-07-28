"""Lens-body deduction, outlines, and inter-lens cuts (``raytracer.viz.bodies``)."""

from pathlib import Path

import numpy as np
import pytest

from raytracer.design import StigmaticTrain
from raytracer.io import read_train
from raytracer.optics.materials import default_materials
from raytracer.viz.bodies import (
    body_outlines,
    body_overlaps,
    dense_gaps,
    draw_system,
    element_outline,
    index_color,
)

INF = float("inf")
DATA = Path(__file__).resolve().parents[1] / "data"


def ultrawide_system():
    """The shipped virtual-object doublet: its lenses genuinely interpenetrate."""

    train = read_train(
        DATA / "optical_systems/ultrawide/sf11_bk7_virtual_object.json",
        materials=default_materials(),
    )
    return train.to_system()


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


def test_enclosure_is_deduced_from_the_media_sequence():
    """Air -> S1 -> Glass -> S2 -> Air: S1 and S2 enclose the glass."""

    train = StigmaticTrain.singlet(
        n=1.5, d0=-60.0, d1=300.0, d2=60.0, thickness=8.0, semidiameter=8.0
    )
    gaps = dense_gaps(train.to_system())
    assert [(i, j) for i, j, _, _ in gaps] == [(0, 1)]
    assert gaps[0][2] == pytest.approx(1.5)


def test_cemented_chain_yields_one_body_per_glass():
    """Air G1 G2 G3 Air across four surfaces: three bodies sharing faces."""

    train = StigmaticTrain(
        media=(1.0, 1.517122, 1.670591, 1.851280, 1.0),
        vertices=(0.0, 15.0, 25.0, 35.0),
        conjugates=(-100.0, -150.0, -600.0, 450.0, 90.0),
        semidiameter=10.0,
    )
    gaps = dense_gaps(train.to_system())
    assert [(i, j) for i, j, _, _ in gaps] == [(0, 1), (1, 2), (2, 3)]
    assert [round(n, 4) for _, _, n, _ in gaps] == [1.5171, 1.6706, 1.8513]


def test_outline_is_a_closed_flat_rim_polygon_for_a_biconvex():
    train = StigmaticTrain.symmetric_singlet(
        n=1.5, d0=-30.0, thickness=40.0, semidiameter=3.5
    )
    polygon, (edge_front, edge_back), joined = element_outline(train.to_system(), 0, 1)
    np.testing.assert_allclose(polygon[0], polygon[-1])  # closed
    assert not joined
    assert edge_front == pytest.approx(3.5) and edge_back == pytest.approx(3.5)
    # both faces stay inside the vertex span of the lens
    assert polygon[:, 0].min() > -5.0 and polygon[:, 0].max() < 45.0


def test_crossing_faces_join_at_the_intersection_height():
    """Faces curving toward each other meet below the aperture; the outline
    must close at that sharp edge instead of drawing crossed curves."""

    train = StigmaticTrain.singlet(
        n=1.5, d0=-9.0, d1=-8.0, d2=25.0, thickness=1.0, semidiameter=3.7
    )
    system = train.to_system()
    front, back = system.rows[0].profile, system.rows[1].profile
    h_probe = np.linspace(0, 3.7, 200)
    crossing = back.sag(h_probe) + 1.0 - front.sag(h_probe)
    assert (crossing <= 0).any()  # the faces genuinely cross below h = 3.7

    polygon, (edge, edge_back), joined = element_outline(system, 0, 1)
    assert joined and edge == edge_back
    assert 0.0 < edge < 3.7
    assert edge == pytest.approx(2.553, abs=0.05)
    assert np.abs(polygon[:, 1]).max() == pytest.approx(edge, abs=1e-6)
    # at the join the two faces coincide
    z_front = system.vertices[0] + front.sag(np.array([edge]))
    z_back = system.vertices[1] + back.sag(np.array([edge]))
    assert z_front[0] == pytest.approx(z_back[0], abs=1e-6)


def test_outline_respects_the_usable_branch_of_an_ovoid():
    """A strong Cartesian surface must be clamped to its single-valued
    branch, never drawn onto the closing branch of the ovoid."""

    train = StigmaticTrain(
        media=(1.0, 1.78472, 1.0),
        vertices=(0.0, 7.0),
        conjugates=(8.0, 27.05, 70.0),   # the ultrawide's virtual-object lens
        semidiameter=22.0,
    )
    system = train.to_system()
    _, (edge_front, edge_back), _ = element_outline(system, 0, 1)
    assert edge_front <= 0.98 * train.profiles[0].max_usable_height + 1e-9
    assert edge_back <= 0.98 * train.profiles[1].max_usable_height + 1e-9


def test_neighbouring_bodies_of_different_media_are_cut_apart():
    """The ultrawide's SF11 back face crosses the BK7 front face off-axis:
    an ambiguity of the description for two different media. The parts must
    be cut at the crossing height so their interiors are disjoint."""

    system = ultrawide_system()
    overlaps = body_overlaps(system)
    assert [(back, front) for back, front, _ in overlaps] == [(1, 2)]
    assert overlaps[0][2] == pytest.approx(5.8584, abs=1e-3)

    raw = [element_outline(system, i, j)[0] for i, j in ((0, 1), (2, 3))]
    assert interior_collisions(*raw) > 0  # the uncut solids interpenetrate

    cut = {(i, j): poly for i, j, _, _, poly, _ in body_outlines(system)}
    assert interior_collisions(cut[(0, 1)], cut[(2, 3)]) == 0


def test_no_part_claims_the_ambiguous_region():
    """The space between the crossed faces above the cut belongs to both
    mathematical solids, so the drawn parts must leave it hollow: the
    downstream BK7 body caps entirely at the cut (a longer back face would
    drag its rim across the region), while the SF11 body keeps its full
    front face because its rim stays clear in front of the region."""

    system = ultrawide_system()
    h_cut = body_overlaps(system)[0][2]
    cut = {(i, j): poly for i, j, _, _, poly, _ in body_outlines(system)}
    assert np.abs(cut[(2, 3)][:, 1]).max() == pytest.approx(h_cut, abs=1e-9)
    assert np.abs(cut[(0, 1)][:, 1]).max() == pytest.approx(22.0, abs=1e-9)

    # the ambiguous annulus — between the crossed faces, above the cut —
    # is covered by neither part
    from matplotlib.path import Path as PolygonPath

    back1, front2 = system.rows[1].profile, system.rows[2].profile
    h = np.linspace(h_cut * 1.01, 0.98 * back1.max_usable_height, 40)
    z_lo = system.vertices[2] + front2.sag(h)
    z_hi = system.vertices[1] + back1.sag(h)
    probe = np.column_stack([0.5 * (z_lo + z_hi), h])  # zone midline
    for polygon in cut.values():
        assert not PolygonPath(polygon).contains_points(probe).any()


def test_steep_rays_refract_on_drawn_curves_never_inside_a_body():
    """A 55°-aim ray refracts at h ≈ 14.2 on the drawn front face and, past
    the cut, on dashed continuations of faces 1, 2 and 3 — every kink must
    lie on a drawn curve and no kink may sit inside a filled body."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as PolygonPath

    train = read_train(
        DATA / "optical_systems/ultrawide/sf11_bk7_virtual_object.json",
        materials=default_materials(),
    )
    system = train.to_system()
    fig, ax = plt.subplots()
    try:
        paths = draw_system(ax, train, fields=(np.sin(np.radians(55.0)),),
                            rays=3, launch=-30.0)
        assert paths is not None
        kinks = paths[:, 1:-1, :]  # surface hits of every traced ray

        outlines = {(i, j): poly
                    for i, j, _, _, poly, _ in body_outlines(system)}
        assert np.abs(outlines[(0, 1)][:, 1]).max() > 14.2  # steep hit covered

        # no refraction point sits strictly inside any filled body
        points = kinks[:, :, [2, 1]].reshape(-1, 2)
        for polygon in outlines.values():
            inside = PolygonPath(polygon).contains_points(points, radius=-1e-6)
            assert not inside.any()

        # dashed continuations start at each face's drawn extent and cover
        # the hits — the outermost is the BK7 back face's (h ≈ 8.0)
        dashed = [line for line in ax.lines
                  if line.get_linestyle() not in ("-", "None", ":")
                  and len(line.get_xdata()) > 2]
        assert dashed, "faces hit beyond their drawn extent must continue dashed"
        h_cut = body_overlaps(system)[0][2]
        assert min(np.abs(line.get_ydata()).min() for line in dashed) == (
            pytest.approx(h_cut, abs=1e-6)
        )
        hit3 = np.abs(kinks[:, 3, 1]).max()
        assert max(np.abs(line.get_ydata()).max() for line in dashed) == (
            pytest.approx(1.02 * hit3, rel=1e-6)
        )

        # every kink lies on a drawn curve: within its face's solid extent
        # (the 14.2 mm hit on the front face), or on a dashed continuation
        extents = {}
        for i, j, _, _, _, (e_front, e_back) in body_outlines(system):
            extents[i], extents[j] = e_front, e_back
        for k in range(kinks.shape[1]):
            for z, y in kinks[:, k, :][:, [2, 1]]:
                if abs(y) <= extents[k] + 1e-9:
                    continue  # refracts on the drawn solid face
                distances = [
                    np.hypot(line.get_xdata() - z, line.get_ydata() - y).min()
                    for line in dashed
                ]
                assert min(distances) < 0.12
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
