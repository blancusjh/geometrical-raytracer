"""Rotation and Pose conventions (``raytracer.geometry.transforms``).

The conventions are fixed once for the package: right-handed frames,
degrees, intrinsic x -> y' -> z'' Euler angles (``R = Rx @ Ry @ Rz``),
counter-clockwise positive angles, and a pose mapping local to world as
``world = R @ local + position``.
"""

import numpy as np
import pytest

from raytracer.geometry import Pose, Rotation

X, Y, Z = np.eye(3)


def Rx(a):
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# -- Rotation ------------------------------------------------------------------


def test_euler_is_intrinsic_xyz():
    a, b, c = 20.0, -35.0, 50.0
    R = Rotation.euler(a, b, c)
    np.testing.assert_allclose(R.matrix, Rx(a) @ Ry(b) @ Rz(c), atol=1e-15)
    # Intrinsic: about x, then about the *new* y, then about the newest z.
    first = Rotation.axis_angle(X, a)
    second = Rotation.axis_angle(first.apply(Y), b) @ first
    third = Rotation.axis_angle(second.apply(Z), c) @ second
    np.testing.assert_allclose(R.matrix, third.matrix, atol=1e-14)
    # ... equivalently, extrinsic z, y, x about the fixed axes.
    extrinsic = Rotation.axis_angle(X, a) @ Rotation.axis_angle(Y, b) @ Rotation.axis_angle(Z, c)
    np.testing.assert_allclose(R.matrix, extrinsic.matrix, atol=1e-14)


def test_euler_radians_option():
    np.testing.assert_allclose(
        Rotation.euler(0.1, 0.2, 0.3, degrees=False).matrix,
        Rotation.euler(*np.degrees([0.1, 0.2, 0.3])).matrix,
        atol=1e-15,
    )


def test_axis_angle_follows_the_right_hand_rule():
    # +90 about z takes x to y (counter-clockwise looking down from +z).
    np.testing.assert_allclose(Rotation.axis_angle(Z, 90.0).apply(X), Y, atol=1e-15)
    # +90 about x takes y to z; +90 about y takes z to x.
    np.testing.assert_allclose(Rotation.axis_angle(X, 90.0).apply(Y), Z, atol=1e-15)
    np.testing.assert_allclose(Rotation.axis_angle(Y, 90.0).apply(Z), X, atol=1e-15)
    # -90 about x tilts +z toward +y: the meridional tilt of an element's axis.
    np.testing.assert_allclose(Rotation.axis_angle(X, -90.0).apply(Z), Y, atol=1e-15)
    # The axis is fixed, its length irrelevant.
    k = np.array([1.0, 2.0, -2.0])
    R = Rotation.axis_angle(3 * k, 73.0)
    np.testing.assert_allclose(R.apply(k), k, atol=1e-14)
    np.testing.assert_allclose(R.matrix, Rotation.axis_angle(k, 73.0).matrix, atol=1e-15)


def test_rotations_are_proper_and_compose_right_to_left():
    A = Rotation.euler(10.0, 20.0, 30.0)
    B = Rotation.axis_angle((1.0, -1.0, 0.5), 40.0)
    v = np.array([0.3, -0.2, 0.9])
    np.testing.assert_allclose((A @ B).apply(v), A.apply(B.apply(v)), atol=1e-15)
    np.testing.assert_allclose((A @ A.inverse()).matrix, np.eye(3), atol=1e-15)
    assert np.linalg.det(A.matrix) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        Rotation(np.diag([1.0, 1.0, -1.0]))  # a reflection
    with pytest.raises(ValueError):
        Rotation(2 * np.eye(3))


def test_rotation_is_an_immutable_value():
    R = Rotation.euler(1.0, 2.0, 3.0)
    with pytest.raises(ValueError):
        R.matrix[0, 0] = 2.0
    with pytest.raises(AttributeError):
        R.matrix = np.eye(3)
    assert R == Rotation(R.matrix.copy())
    assert hash(R) == hash(Rotation(R.matrix.copy()))


@pytest.mark.parametrize(
    "direction",
    [(0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 2, 3), (0.2, -0.3, -1.0), (0, 0, -1)],
)
def test_align_z_takes_z_onto_the_direction(direction):
    v = np.asarray(direction, dtype=float)
    v /= np.linalg.norm(v)
    R = Rotation.align_z(direction)
    np.testing.assert_allclose(R.apply(Z), v, atol=1e-14)
    np.testing.assert_allclose(R.axis, v, atol=1e-14)


def test_align_z_is_the_smallest_rotation():
    """The rotation axis is perpendicular to both z and the target, so
    whatever is perpendicular to both stays put."""

    v = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    R = Rotation.align_z(v)
    fixed = np.cross(Z, v)
    np.testing.assert_allclose(R.apply(fixed), fixed, atol=1e-14)
    angle = np.degrees(np.arccos((np.trace(R.matrix) - 1.0) / 2.0))
    assert angle == pytest.approx(np.degrees(np.arccos(v[2])))


def test_align_z_half_turn_is_about_x():
    R = Rotation.align_z((0.0, 0.0, -1.0))
    np.testing.assert_allclose(R.matrix, Rotation.axis_angle(X, 180.0).matrix, atol=1e-15)
    assert Rotation.align_z((0.0, 0.0, 5.0)).is_identity()


def test_coerce_reads_every_shorthand():
    assert Rotation.coerce(None).is_identity()
    R = Rotation.euler(5.0, 6.0, 7.0)
    assert Rotation.coerce(R) is R
    assert Rotation.coerce((5.0, 6.0, 7.0)) == R
    assert Rotation.coerce((30.0, (0.0, 0.0, 1.0))) == Rotation.axis_angle(Z, 30.0)
    assert Rotation.coerce(R.matrix) == R
    with pytest.raises(TypeError):
        Rotation.coerce((1.0, 2.0))


# -- Pose ----------------------------------------------------------------------


def test_pose_maps_local_to_world():
    pose = Pose((1.0, 2.0, 3.0), Rotation.axis_angle(Z, 90.0))
    np.testing.assert_allclose(pose.to_world(X), [1.0, 3.0, 3.0], atol=1e-15)
    np.testing.assert_allclose(pose.direction_to_world(X), Y, atol=1e-15)
    pts = np.array([[0.1, 0.2, 0.3], [-1.0, 4.0, 2.0]])
    np.testing.assert_allclose(pose.to_local(pose.to_world(pts)), pts, atol=1e-14)
    np.testing.assert_allclose(pose.direction_to_local(pose.direction_to_world(pts)), pts, atol=1e-14)
    np.testing.assert_allclose(pose.axis, Z, atol=1e-15)


def test_pose_composition_is_parent_then_child():
    parent = Pose((10.0, 0.0, 0.0), Rotation.axis_angle(Z, 90.0))
    child = Pose((1.0, 0.0, 0.0), Rotation.axis_angle(X, 30.0))
    world = parent @ child
    p = np.array([0.2, -0.4, 0.7])
    np.testing.assert_allclose(world.to_world(p), parent.to_world(child.to_world(p)), atol=1e-14)
    np.testing.assert_allclose(world.position, [10.0, 1.0, 0.0], atol=1e-14)


def test_pose_inverse():
    pose = Pose((1.0, -2.0, 3.0), Rotation.euler(10.0, 20.0, 30.0))
    assert (pose @ pose.inverse()).isclose(Pose.identity(), atol=1e-14)
    assert (pose.inverse() @ pose).isclose(Pose.identity(), atol=1e-14)
    p = np.array([0.5, 0.6, 0.7])
    np.testing.assert_allclose(pose.inverse().to_world(p), pose.to_local(p), atol=1e-14)


def test_rotated_about_a_point():
    """Turning a pose about a world line moves its position around the pivot
    and turns its orientation by the same rotation."""

    pose = Pose((0.0, 0.0, 10.0))
    turned = pose.rotated(90.0, X, about=(0.0, 0.0, 5.0))
    np.testing.assert_allclose(turned.position, [0.0, -5.0, 5.0], atol=1e-14)
    np.testing.assert_allclose(turned.axis, [0.0, -1.0, 0.0], atol=1e-14)
    # The pivot itself does not move; rotating about the origin is the default.
    at_pivot = Pose((0.0, 0.0, 5.0)).rotated(90.0, X, about=(0.0, 0.0, 5.0))
    np.testing.assert_allclose(at_pivot.position, [0.0, 0.0, 5.0], atol=1e-14)
    np.testing.assert_allclose(pose.rotated(90.0, X).position, [0.0, -10.0, 0.0], atol=1e-14)
    # A rigid motion: distances between frames are preserved.
    other = Pose((3.0, 1.0, -2.0))
    a, b = pose.rotated(37.0, (1, 1, 0), (1, 2, 3)), other.rotated(37.0, (1, 1, 0), (1, 2, 3))
    assert np.linalg.norm(a.position - b.position) == pytest.approx(np.linalg.norm(pose.position - other.position))


def test_translated():
    pose = Pose((1.0, 2.0, 3.0), Rotation.euler(0.0, 10.0, 0.0))
    moved = pose.translated((0.5, 0.0, -1.0))
    np.testing.assert_allclose(moved.position, [1.5, 2.0, 2.0])
    assert moved.rotation == pose.rotation


def test_from_spec_reads_placement_keywords():
    assert Pose.from_spec(z=5.0) == Pose((0.0, 0.0, 5.0))
    assert Pose.from_spec(pos=(1, 2, 3), rot=(0, 0, 90)) == Pose((1, 2, 3), Rotation.euler(0, 0, 90))
    np.testing.assert_allclose(Pose.from_spec(axis=(0, 1, 0)).axis, Y, atol=1e-15)
    np.testing.assert_allclose(Pose.from_spec(normal=(1, 0, 0)).axis, X, atol=1e-15)
    p = Pose((1.0, 0.0, 0.0))
    assert Pose.from_spec(pose=p) is p
    with pytest.raises(TypeError):
        Pose.from_spec(pos=(0, 0, 1), z=1.0)
    with pytest.raises(TypeError):
        Pose.from_spec(rot=(0, 0, 1), axis=(0, 0, 1))
    with pytest.raises(TypeError):
        Pose.from_spec(pose=p, z=1.0)


def test_pose_is_an_immutable_value():
    pose = Pose((1.0, 2.0, 3.0))
    with pytest.raises(ValueError):
        pose.position[0] = 5.0
    with pytest.raises(AttributeError):
        pose.position = np.zeros(3)
    assert pose == Pose((1.0, 2.0, 3.0))
    assert len({pose, Pose((1.0, 2.0, 3.0))}) == 1
