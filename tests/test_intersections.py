"""Where a ray meets a surface: ``Surface.intersect`` and its two solvers.

Every surface solves ``f(o + t d) = 0`` for a batch of rays in its own
frame: the conics in closed form (:func:`solve_quadric`), everything else
by Newton from the vertex plane (:func:`newton_graph`). The answer must not
depend on the route, so the Cartesian surface — the one surface with a
closed-form parametrization *and* an implicit Descartes condition besides
its sag — is the natural cross-check.
"""

import numpy as np
import pytest

from raytracer.elements import Lens, Sensor
from raytracer.geometry import Pose, Rotation
from raytracer.geometry.intersections import newton_graph
from raytracer.geometry.surfaces import CartesianSurface, EvenAsphere, Plane, Sphere, ToroidalSurface
from raytracer.geometry.surfaces.cartesian import cartesian_oval_implicit
from raytracer.materials import ConstantIndex
from raytracer.rays import Rays
from raytracer.system import OpticalSystem
from raytracer.tracing import SequentialTracer

Z0, ZI, N0, NI = -30.0, 10.0, 1.0, 1.7


def rays(origins, directions):
    o = np.atleast_2d(np.asarray(origins, dtype=float))
    d = np.atleast_2d(np.asarray(directions, dtype=float))
    return o, d / np.linalg.norm(d, axis=1, keepdims=True)


def test_quadratic_route_is_exact_for_a_sphere():
    """A sphere's f is quadratic, so the root is analytic, not iterated."""

    sphere = Sphere(2.0)
    o, d = rays([[0.0, 0.0, -5.0], [0.0, 1.0, -5.0]], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    hits = sphere.intersect(o, d)
    assert hits.ok.all()
    assert hits.t[0] == pytest.approx(5.0, abs=1e-15)
    assert hits.t[1] == pytest.approx(5.0 + 2.0 - np.sqrt(3.0), abs=1e-15)


def test_forward_mode_returns_the_first_crossing_ahead():
    """A chord of the vertex hemisphere meets it twice; ahead of the ray
    the near crossing is the hit, and from between the two the far one."""

    sphere = Sphere(2.0)  # centre (0, 0, 2); at z = 1 the chord is x = ±sqrt(3)
    o, d = rays([[-5.0, 0.0, 1.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    hits = sphere.intersect(o, d, mode="forward")
    assert hits.ok.all()
    assert hits.t[0] == pytest.approx(5.0 - np.sqrt(3.0), abs=1e-14)
    assert hits.t[1] == pytest.approx(np.sqrt(3.0), abs=1e-14)


def test_nearest_mode_accepts_a_surface_behind_the_ray():
    """The sequential convention keeps the root nearest the vertex plane
    whatever its sign (a stop placed behind the previous surface); the
    physical convention reports it missed."""

    o, d = rays([[0.0, 1.0, 3.0]], [[0.0, 0.0, 1.0]])
    for surface in (Plane(), Sphere(50.0)):
        nearest = surface.intersect(o, d, mode="nearest")
        assert nearest.ok[0] and nearest.t[0] == pytest.approx(-3.0 + surface.profile(1.0), abs=1e-14)
        assert surface.intersect(o, d, mode="forward").missed[0]


def test_newton_route_matches_sphere_geometry():
    """Newton on the sag, forced on a sphere, lands on the exact circle point."""

    R, y0 = 150.0, 40.0
    o, d = rays([[0.0, y0, -100.0]], [[0.0, 0.0, 1.0]])
    hits = newton_graph(Sphere(R), o, d, np.ones(1, dtype=bool))
    assert hits.ok[0]
    point = o[0] + hits.t[0] * d[0]
    assert point[2] == pytest.approx(R - np.sqrt(R * R - y0 * y0), abs=1e-12)
    assert hits.t[0] == pytest.approx(100.0 + point[2], abs=1e-12)


def test_implicit_residual_is_zero_at_the_returned_point():
    """Whatever route was taken, the answer satisfies f = z - s(x, y) = 0."""

    surfaces = [
        Sphere(150.0),
        EvenAsphere(-120.0, -0.8, {4: 2e-7, 6: -1e-11}),
        ToroidalSurface(80.0, -60.0),
        CartesianSurface(N0, Z0, NI, ZI),
    ]
    rng = np.random.default_rng(3)
    o = np.column_stack([rng.uniform(-3, 3, 20), rng.uniform(-3, 3, 20), np.full(20, -10.0)])
    d = np.column_stack([rng.uniform(-0.05, 0.05, 20), rng.uniform(-0.05, 0.05, 20), np.ones(20)])
    o, d = rays(o, d)
    for surface in surfaces:
        hits = surface.intersect(o, d)
        assert hits.ok.all(), surface
        points = o + hits.t[:, None] * d
        assert np.max(np.abs(surface.implicit(points))) < 1e-10, surface


def test_parallel_ray_misses_a_flat_face():
    """A ray running inside a plane never meets it."""

    o, d = rays([[0.0, 1.0, 0.0], [0.0, 1.0, -1.0]], [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    hits = Plane().intersect(o, d)
    assert hits.parallel.all()
    assert not hits.ok.any()


def test_a_ray_beyond_a_closed_conic_misses_it():
    """A sphere's vertex sheet ends at its equator: a ray passing outside misses."""

    o, d = rays([[0.0, 3.0, -5.0]], [[0.0, 0.0, 1.0]])
    hits = Sphere(2.0).intersect(o, d)
    assert hits.missed[0]


# ---------------------------------------------------------------------------
# The descriptions of the Cartesian surface must agree
# ---------------------------------------------------------------------------


def test_parametric_and_sag_descriptions_agree_on_the_cartesian_surface():
    """A ray from the object point, solved on the sag by Newton, lands on the
    closed-form parametrization (z(rho), r(rho)) and on the Descartes oval."""

    surface = CartesianSurface(N0, Z0, NI, ZI)
    heights = np.array([-2.5, -1.0, 0.5, 2.0, 3.0])
    targets = np.column_stack([np.zeros_like(heights), heights, np.zeros_like(heights)])
    o = np.tile([0.0, 0.0, Z0], (len(heights), 1))
    o, d = rays(o, targets - o)
    hits = surface.intersect(o, d)
    assert hits.ok.all()
    points = o + hits.t[:, None] * d
    r = np.hypot(points[:, 0], points[:, 1])
    rho = np.hypot(points[:, 2], r)
    z_param, r_param = surface.meridian(rho)
    np.testing.assert_allclose(z_param, points[:, 2], atol=1e-9)
    np.testing.assert_allclose(r_param, r, atol=1e-9)
    residual = cartesian_oval_implicit(points[:, 2], r, Z0, ZI, N0, NI)
    assert np.max(np.abs(residual)) < 1e-9


def test_parametric_points_satisfy_the_implicit_equation():
    """P(rho) lies on f_Sigma = 0 -- the two descriptions are of one surface."""

    surface = CartesianSurface(N0, Z0, NI, ZI)
    z, r = surface.meridian(np.array([0.5, 1.0, 2.0, 4.0, 5.0]))
    assert np.max(np.abs(cartesian_oval_implicit(z, r, Z0, ZI, N0, NI))) < 1e-10


def test_placed_hit_agrees_with_the_bare_solver():
    """A traced hit is the solver in the face's frame plus the pose: the
    tracer's world point maps back to the local root of Surface.intersect."""

    glass = ConstantIndex("G", NI)
    pose = Pose((1.0, 2.0, 5.0), Rotation.axis_angle((1.0, 1.0, 0.0), 7.0))
    lens = Lens(
        "L", front=CartesianSurface(N0, Z0, NI, ZI), back=Plane(), thickness=ZI,
        material=glass, semi_diameter=4.0, pose=pose,
    )
    sensor = Sensor("IMG", pose=pose @ Pose((0.0, 0.0, ZI + 5.0)))
    tracer = SequentialTracer(OpticalSystem([lens, sensor]))
    origin = pose.to_world([0.0, 0.5, -20.0])
    direction = pose.direction_to_world([0.0, 0.0, 1.0])
    result = tracer.trace(Rays(origin, direction))
    assert result.detected[0]
    world_hit = result.trajectory[0, 1]

    o_local, d_local = pose.to_local(origin[None, :]), pose.direction_to_local(direction[None, :])
    hits = lens.front.surface.intersect(o_local, d_local)
    assert pose.to_world(o_local[0] + hits.t[0] * d_local[0]) == pytest.approx(world_hit, abs=1e-12)
    normal = lens.front.surface.normal(pose.to_local(world_hit[None, :]))
    assert np.linalg.norm(normal) == pytest.approx(1.0)
