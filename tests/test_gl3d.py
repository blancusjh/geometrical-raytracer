"""Lathe mesh generation (pure numpy) + optional 3-D render smoke test."""

import numpy as np
import pytest

from raytracer.shapes.profile import AsphereProfile
from raytracer.viz.gl3d import lathe


def test_lathe_vertex_positions_follow_sag():
    profile = AsphereProfile.sphere(-200.0)
    vertices, faces = lathe(profile, 50.0, vertex_z=10.0, radial_samples=8,
                            angular_samples=16)
    assert vertices.shape == (8 * 16, 3)
    radii = np.hypot(vertices[:, 0], vertices[:, 1])
    expected_z = 10.0 + profile.sag(radii)
    assert vertices[:, 2] == pytest.approx(expected_z, abs=1e-5)
    assert radii.max() == pytest.approx(50.0, abs=1e-5)


def test_lathe_faces_are_valid_and_closed():
    profile = AsphereProfile.plane()
    vertices, faces = lathe(profile, 10.0, radial_samples=5, angular_samples=12)
    assert faces.min() >= 0
    assert faces.max() < vertices.shape[0]
    # A closed annulus band: 2 triangles per quad, (r-1)*ang quads.
    assert faces.shape == (2 * 4 * 12, 3)
    # No degenerate triangles away from the axis.
    v = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1
    )
    ring_faces = areas[faces.min(axis=1) >= 12]  # skip the innermost ring
    assert np.all(ring_faces > 0.0)


@pytest.mark.gpu
def test_viewer3d_snapshot_renders_content():
    pytest.importorskip("vispy")
    from raytracer.design import OpticalSystem, SurfaceRow
    from raytracer.viz.gl3d import Viewer3D

    rows = [SurfaceRow.mirror(radius=-2000.0, thickness=-1000.0, conic=-1.0)]
    system = OpticalSystem(rows)
    try:
        viewer = Viewer3D(size=(300, 220))
        viewer.add_system(system)
        image = viewer.snapshot()
    except Exception as error:
        pytest.skip(f"No usable OpenGL context: {error}")
    assert image.shape[0] > 0
    # The mirror mesh must be visible against the dark background.
    assert image[..., :3].std() > 1.0
