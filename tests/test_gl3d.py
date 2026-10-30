"""Lathe mesh generation (pure numpy) + optional 3-D render smoke test."""

import numpy as np
import pytest

pytest.importorskip("vispy")

from raytracer.geometry import Pose, Rotation  # noqa: E402
from raytracer.geometry.apertures import Rectangle  # noqa: E402
from raytracer.geometry.surfaces import Plane, Sphere, ToroidalSurface  # noqa: E402
from raytracer.viz.gl3d import lathe, lathe_solid  # noqa: E402


def test_lathe_vertex_positions_follow_sag():
    surface = Sphere(-200.0)
    vertices, faces = lathe(surface, 50.0, pose=Pose((0.0, 0.0, 10.0)), radial_samples=8, angular_samples=16)
    assert vertices.shape == (8 * 16, 3)
    radii = np.hypot(vertices[:, 0], vertices[:, 1])
    expected_z = 10.0 + surface.profile(radii)
    assert vertices[:, 2] == pytest.approx(expected_z, abs=1e-5)
    assert radii.max() == pytest.approx(50.0, abs=1e-5)


def test_lathe_faces_are_valid_and_closed():
    vertices, faces = lathe(Plane(), 10.0, radial_samples=5, angular_samples=12)
    assert faces.min() >= 0
    assert faces.max() < vertices.shape[0]
    # A closed annulus band: 2 triangles per quad, (r-1)*ang quads.
    assert faces.shape == (2 * 4 * 12, 3)
    # No degenerate triangles away from the axis.
    v = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    ring_faces = areas[faces.min(axis=1) >= 12]  # skip the innermost ring
    assert np.all(ring_faces > 0.0)


def test_lathe_meshes_a_surface_without_rotational_symmetry():
    """The sag is read as s(x, y): a toroidal face keeps its two curvatures."""

    surface = ToroidalSurface(50.0, 100.0)
    vertices, _ = lathe(surface, 5.0, radial_samples=6, angular_samples=24)
    np.testing.assert_allclose(vertices[:, 2], surface.sag(vertices[:, 0], vertices[:, 1]), atol=1e-5)


def test_lathe_is_placed_by_the_pose():
    pose = Pose((1.0, 2.0, 3.0), Rotation.axis_angle((1.0, 0.0, 0.0), -30.0))
    local, _ = lathe(Sphere(80.0), 10.0, radial_samples=4, angular_samples=8)
    placed, _ = lathe(Sphere(80.0), 10.0, pose=pose, radial_samples=4, angular_samples=8)
    np.testing.assert_allclose(placed, pose.to_world(local.astype(float)), atol=1e-5)


def test_lathe_clips_to_a_rectangular_aperture():
    vertices, faces = lathe(Plane(), 10.0, radial_samples=12, angular_samples=48, aperture=Rectangle(8.0, 4.0))
    used = vertices[np.unique(faces)]
    assert np.abs(used[:, 0]).max() <= 4.0 + 1e-6 and np.abs(used[:, 1]).max() <= 2.0 + 1e-6


def test_lathe_solid_is_a_closed_body_between_two_extents():
    vertices, faces = lathe_solid(Sphere(40.0), Sphere(-40.0), 10.0, thickness=6.0, back_semidiameter=8.0,
                                  radial_samples=6, angular_samples=16)
    radii = np.hypot(vertices[:, 0], vertices[:, 1])
    assert radii[: 6 * 16].max() == pytest.approx(10.0, abs=1e-5)
    assert radii[6 * 16:].max() == pytest.approx(8.0, abs=1e-5)
    assert vertices[6 * 16, 2] == pytest.approx(6.0)  # back vertex
    # watertight: once the coincident axis vertices are merged, every edge
    # of a non-degenerate triangle is shared by exactly two triangles
    _, merged = np.unique(np.round(vertices.astype(float), 6), axis=0, return_inverse=True)
    tri = merged.ravel()[faces]
    tri = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 2] != tri[:, 0])]
    edges = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)


@pytest.mark.gpu
def test_viewer3d_snapshot_renders_content():
    from raytracer import OpticalSystem
    from raytracer.elements import Mirror
    from raytracer.geometry.surfaces import Paraboloid
    from raytracer.viz.gl3d import Viewer3D

    system = OpticalSystem([Mirror("M1", surface=Paraboloid(-2000.0), aperture=200.0)], path=["M1:reflect"])
    try:
        viewer = Viewer3D(size=(300, 220))
        viewer.add_system(system)
        image = viewer.snapshot()
    except Exception as error:
        pytest.skip(f"No usable OpenGL context: {error}")
    assert image.shape[0] > 0
    # The mirror mesh must be visible against the dark background.
    assert image[..., :3].std() > 1.0
