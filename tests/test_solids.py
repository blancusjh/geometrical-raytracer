"""Solids: closed regions of space (``raytracer.geometry.solids``)."""

import numpy as np
import pytest

from raytracer.geometry.solids import Box, Cylinder, Mesh, Sphere


def unit(v):
    v = np.atleast_2d(np.asarray(v, dtype=float))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# -- containment -----------------------------------------------------------------


def test_sphere_contains():
    s = Sphere(2.0)
    pts = np.array([[0, 0, 0], [0, 0, 2.0], [1.5, 1.5, 0], [0, 0, -1.99]])
    assert s.contains(pts).tolist() == [True, True, False, True]


def test_cylinder_contains():
    c = Cylinder(1.0, 5.0)
    pts = np.array([[0, 0, 0], [0, 0, 5.0], [0, 0, 5.1], [0, 0, -0.1], [0.9, 0.0, 2.0], [0.8, 0.8, 2.0]])
    assert c.contains(pts).tolist() == [True, True, False, False, True, False]


def test_box_contains():
    b = Box(2.0, 4.0, 6.0)  # |x| <= 1, |y| <= 2, 0 <= z <= 6
    pts = np.array([[0, 0, 3], [1, 2, 6], [1.01, 0, 3], [0, 2.01, 3], [0, 0, -0.01], [0, 0, 6.01]])
    assert b.contains(pts).tolist() == [True, True, False, False, False, False]
    assert len(b.facets()) == 6


def test_box_facets_face_outward():
    b = Box(2.0, 4.0, 6.0)
    centre = np.array([0.0, 0.0, 3.0])
    normals = []
    for pose, outline in b.facets():
        assert (pose.position - centre) @ pose.axis > 0  # outward
        normals.append(pose.axis)
        assert np.allclose(pose.to_local(pose.position[None, :])[0], 0.0)
    np.testing.assert_allclose(np.sum(normals, axis=0), 0.0, atol=1e-14)


def test_mesh_prism_matches_its_outline():
    triangle = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)]  # counter-clockwise
    prism = Mesh.prism(triangle, 3.0)
    pts = np.array([[0.5, 0.5, 1.0], [1.5, 1.5, 1.0], [0.5, 0.5, 3.5], [0.5, 0.5, -0.5]])
    assert prism.contains(pts).tolist() == [True, False, False, False]
    assert len(prism.facets()) == 5


# -- first hit ----------------------------------------------------------------------


def test_sphere_first_hit_and_normal():
    s = Sphere(2.0)
    o = np.array([[-5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-5.0, 3.0, 0.0]])
    d = unit([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
    t, n = s.intersect(o, d)
    assert t[0] == pytest.approx(3.0) and n[0] == pytest.approx([-1, 0, 0])
    assert t[1] == pytest.approx(2.0) and n[1] == pytest.approx([0, 1, 0])  # from inside: the exit
    assert np.isinf(t[2])


def test_cylinder_first_hit_on_wall_and_caps():
    c = Cylinder(1.0, 5.0)
    o = np.array([[-3.0, 0.0, 2.0], [0.0, 0.0, -2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 7.0], [3.0, 3.0, 2.0]])
    d = unit([[1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, -1], [0, 0, 1]])
    t, n = c.intersect(o, d)
    assert t[0] == pytest.approx(2.0) and n[0] == pytest.approx([-1, 0, 0])
    assert t[1] == pytest.approx(2.0) and n[1] == pytest.approx([0, 0, -1])  # front cap
    assert t[2] == pytest.approx(3.0) and n[2] == pytest.approx([0, 0, 1])  # back cap from inside
    assert t[3] == pytest.approx(2.0) and n[3] == pytest.approx([0, 0, 1])
    assert np.isinf(t[4])


def test_box_first_hit_and_outward_normal():
    b = Box(2.0, 4.0, 6.0)
    o = np.array([[-5.0, 0.0, 3.0], [0.0, 0.0, 3.0], [0.0, 0.0, -1.0], [5.0, 5.0, 3.0]])
    d = unit([[1, 0, 0], [0, 1, 0], [0.1, 0.0, 1.0], [1, 0, 0]])
    t, n = b.intersect(o, d)
    assert t[0] == pytest.approx(4.0) and n[0] == pytest.approx([-1, 0, 0])
    assert t[1] == pytest.approx(2.0) and n[1] == pytest.approx([0, 1, 0])
    assert t[2] == pytest.approx(np.sqrt(1.01)) and n[2] == pytest.approx([0, 0, -1])
    assert np.isinf(t[3])


def test_first_hit_lies_on_the_boundary():
    """From inside, the first crossing is the exit: just before it the point is
    contained, just after it is not."""

    rng = np.random.default_rng(2)
    d = unit(rng.normal(size=(100, 3)))
    jitter = rng.uniform(-0.5, 0.5, (100, 3))
    for solid, centre in ((Sphere(1.5), (0, 0, 0)), (Cylinder(1.2, 4.0), (0, 0, 2)), (Box(2.0, 2.0, 4.0), (0, 0, 2))):
        start = jitter + centre
        t, _ = solid.intersect(start, d)
        assert np.isfinite(t).all()
        assert solid.contains(start + (t - 1e-9)[:, None] * d).all()
        assert not solid.contains(start + (t + 1e-9)[:, None] * d).any()


def test_solids_are_values():
    assert Box(1, 2, 3) == Box(1, 2, 3) and Box(1, 2, 3) != Box(3, 2, 1)
    assert hash(Cylinder(1, 2)) == hash(Cylinder(1, 2))
    with pytest.raises(ValueError):
        Sphere(-1.0)
