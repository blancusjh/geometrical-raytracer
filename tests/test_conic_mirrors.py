"""Geometric ground truths for conic mirrors: a :class:`Mirror` carrying a
conic surface, placed with a pose, traced non-sequentially."""

import numpy as np
import pytest

from raytracer.elements import Mirror, Sensor
from raytracer.geometry.apertures import Rectangle
from raytracer.geometry.surfaces import Ellipsoid, Paraboloid, Sphere
from raytracer.rays import Rays, Status
from raytracer.sources import CollimatedSource, PointSource
from raytracer.system import OpticalSystem
from raytracer.tracing import NonSequentialTracer


def distance_to_lines(point, origins, directions) -> np.ndarray:
    """Perpendicular distance from *point* to each ray line."""

    v = np.asarray(point) - origins
    return np.linalg.norm(np.cross(v, directions), axis=1)


def test_sphere_mirror_intersection_distance_and_normal():
    mirror = Mirror("M", surface=Sphere(2.0), aperture=1.9)
    ray = NonSequentialTracer(OpticalSystem([mirror])).trace(Rays([0.0, 0.0, -5.0], [0.0, 0.0, 1.0]))
    start, hit = ray.trajectory[0, 0], ray.trajectory[0, 1]
    assert hit == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert np.linalg.norm(hit - start) == pytest.approx(5.0, abs=1e-12)
    normal = mirror.surface.normal(hit[None, :])[0]
    assert abs(normal @ np.array([0.0, 0.0, 1.0])) == pytest.approx(1.0, abs=1e-12)
    # Reflected straight back.
    assert ray.at(1).direction[0] == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)


def test_ellipsoid_focus_to_focus_reflection():
    """Rays from one focus of an ellipsoidal mirror reflect through the other."""

    a, b = 4.0, 2.5
    c = np.sqrt(a**2 - b**2)
    # Vertex at the origin, opening toward -z: foci at z = -(a - c) and -(a + c).
    mirror = Mirror("M", surface=Ellipsoid.from_semi_axes(-a, b), aperture=b)
    near, far = np.array([0.0, 0.0, -(a - c)]), np.array([0.0, 0.0, -(a + c)])

    source = PointSource(near, half_angle_deg=75.0, rays=61)  # full 3-D bundle
    hit = NonSequentialTracer(OpticalSystem([mirror]), max_interactions=2).trace(source).at(1)

    reflected = hit.event == "reflect"
    assert reflected.sum() > 10
    d = distance_to_lines(far, hit.position[reflected], hit.direction[reflected])
    assert np.max(d) < 1e-7


def test_paraboloid_focuses_collimated_beam():
    """A collimated beam along the axis reflects through the paraboloid's focus."""

    f = 2.0
    mirror = Mirror("M", surface=Paraboloid(f=-f), aperture=3.0)  # opening toward -z
    source = CollimatedSource((0.0, 0.0, -6.0), beam_radius=2.5, rays=61)
    hit = NonSequentialTracer(OpticalSystem([mirror]), max_interactions=2).trace(source).at(1)

    reflected = hit.event == "reflect"
    assert reflected.sum() > 10
    focus = np.array([0.0, 0.0, -f])
    assert np.max(distance_to_lines(focus, hit.position[reflected], hit.direction[reflected])) < 1e-9


def test_aperture_clips_rays_beyond_the_mirror():
    """A ray that would meet the conic outside its aperture passes by."""

    mirror = Mirror("M", surface=Sphere(5.0), aperture=2.0, z=10.0)
    rays = Rays([[0.0, 1.0, 0.0], [0.0, 3.0, 0.0]], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], np.nan, 1.0)
    rays = NonSequentialTracer(OpticalSystem([mirror])).trace(rays)
    assert rays.at(1).boundary.tolist() == ["M", None]  # the second never meets the mirror
    assert rays.status[1] == Status.ESCAPED


def test_sensor_without_aperture_does_not_clip():
    """An unbounded detector (aperture=None) records every ray that crosses its plane."""

    sensor = Sensor("S", z=10.0)
    bounded = Sensor("S", z=10.0, aperture=Rectangle(2.0, 2.0))
    rays = Rays([[0.0, 0.5, 0.0], [0.0, 400.0, 0.0]], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], np.nan, 1.0)
    assert NonSequentialTracer(OpticalSystem([sensor])).trace(rays).detected.sum() == 2
    assert NonSequentialTracer(OpticalSystem([bounded])).trace(rays).detected.sum() == 1
