"""Lens-body deduction and outlines (``raytracer.viz.bodies``)."""

import numpy as np
import pytest

from raytracer.design import StigmaticTrain
from raytracer.viz.bodies import dense_gaps, element_outline, index_color

INF = float("inf")


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
    polygon, edge, joined = element_outline(train.to_system(), 0, 1)
    np.testing.assert_allclose(polygon[0], polygon[-1])  # closed
    assert not joined
    assert edge == pytest.approx(3.5)
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

    polygon, edge, joined = element_outline(system, 0, 1)
    assert joined
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
    _, edge, _ = element_outline(system, 0, 1)
    assert edge <= 0.98 * min(p.max_usable_height for p in train.profiles) + 1e-9


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
