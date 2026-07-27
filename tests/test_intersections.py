"""The single intersection entry point, and its two methods.

``intersect_ray_with_surface`` must give the same answer whichever way a
surface chooses to describe itself, so the Cartesian oval — the one surface
that offers *both* an implicit form and a closed-form parametrization — is
the natural cross-check.
"""

import numpy as np
import pytest

from raytracer.math.intersections import intersect_ray_with_surface, solve_parametric
from raytracer.optics.ray import Ray
from raytracer.surfaces.cartesian_oval import CartesianOvalSurface, cartesian_oval_implicit
from raytracer.surfaces.conic import CircleSurface
from raytracer.surfaces.profile import AsphereProfile, ProfileSurface
from raytracer.surfaces.segment import LineSegment

Z0, ZI, N0, NI = -30.0, 10.0, 1.0, 1.7


def test_quadratic_route_is_exact_for_a_circle():
    """A circle's f_Sigma is quadratic, so the root is analytic, not iterated."""

    circle = CircleSurface(radius=2.0)
    point, lam = intersect_ray_with_surface(
        np.array([-5.0, 0.0]), np.array([1.0, 0.0]), circle
    )
    assert lam == pytest.approx(3.0, abs=1e-15)
    assert point == pytest.approx([-2.0, 0.0], abs=1e-15)


def test_smallest_positive_lambda_is_returned():
    """A ray through a circle meets it twice; the near face is the hit."""

    circle = CircleSurface(radius=2.0, center=np.array([0.0, 0.0]))
    _, lam = intersect_ray_with_surface(
        np.array([-5.0, 0.0]), np.array([1.0, 0.0]), circle
    )
    assert lam == pytest.approx(3.0)  # not 7.0, the far side


def test_newton_route_matches_sphere_geometry():
    """The implicit form of a sag profile, solved by Newton, hits the exact sphere."""

    R = 150.0
    face = ProfileSurface(profile=AsphereProfile.sphere(R), vertex=[0.0, 0.0], semidiameter=60.0)
    y0 = 40.0
    point, lam = intersect_ray_with_surface(
        np.array([-100.0, y0]), np.array([1.0, 0.0]), face
    )
    assert point[0] == pytest.approx(R - np.sqrt(R * R - y0 * y0), abs=1e-12)
    assert lam == pytest.approx(100.0 + point[0], abs=1e-12)


def test_implicit_residual_is_zero_at_the_returned_point():
    """Whatever route was taken, the answer must satisfy f_Sigma = 0."""

    surfaces = [
        CircleSurface(radius=2.0),
        ProfileSurface(profile=AsphereProfile.sphere(150.0), vertex=[0.0, 0.0], semidiameter=60.0),
        CartesianOvalSurface(z0=Z0, zi=ZI, n_exterior=N0, n_interior=NI),
    ]
    origins = [np.array([-5.0, 0.0]), np.array([-100.0, 40.0]), np.array([Z0, 1.0])]
    for surface, origin in zip(surfaces, origins):
        result = intersect_ray_with_surface(origin, np.array([1.0, 0.0]), surface)
        assert result is not None
        point, _ = result
        assert abs(surface.implicit(point)) < 1e-8


def test_parallel_ray_misses_a_flat_face():
    """A ray running along a plane never meets it (a = b = 0 in the quadratic)."""

    face = ProfileSurface(profile=AsphereProfile.plane(), vertex=[0.0, 0.0], semidiameter=10.0)
    assert intersect_ray_with_surface(
        np.array([-5.0, 1.0]), np.array([0.0, 1.0]), face
    ) is None


def test_segment_is_a_degenerate_quadratic():
    seg = LineSegment(p0=[0.0, -1.0], p1=[0.0, 1.0])
    point, lam = intersect_ray_with_surface(
        np.array([-2.0, 0.5]), np.array([1.0, 0.0]), seg
    )
    assert lam == pytest.approx(2.0, abs=1e-15)
    assert point == pytest.approx([0.0, 0.5], abs=1e-15)
    assert seg.fraction_along(point) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# The two methods must agree
# ---------------------------------------------------------------------------


def test_parametric_and_implicit_methods_agree_on_the_cartesian_oval():
    """Same ray, same surface, both descriptions: A + lambda u = P(t) must
    land where f_Sigma(A + lambda u) = 0 does."""

    oval = CartesianOvalSurface(z0=Z0, zi=ZI, n_exterior=N0, n_interior=NI)
    for height in (-2.5, -1.0, 0.5, 2.0, 3.0):
        origin = np.array([Z0, 0.0])
        target = np.array([0.0, height])
        direction = (target - origin) / np.linalg.norm(target - origin)

        implicit_hit = intersect_ray_with_surface(origin, direction, oval)
        assert implicit_hit is not None

        parametric_hit = solve_parametric(
            origin, direction, oval, oval.parametric_seed(origin, direction)
        )
        assert parametric_hit is not None

        assert parametric_hit[1] == pytest.approx(implicit_hit[1], rel=1e-7)
        assert parametric_hit[0] == pytest.approx(implicit_hit[0], abs=1e-7)


def test_parametric_points_satisfy_the_implicit_equation():
    """P(t) lies on f_Sigma = 0 -- the two descriptions are of one surface."""

    oval = CartesianOvalSurface(z0=Z0, zi=ZI, n_exterior=N0, n_interior=NI)
    for t in (-4.0, -1.0, 0.5, 2.0, 5.0):
        z, r = oval.parametric(t)
        assert abs(cartesian_oval_implicit(z, r, Z0, ZI, N0, NI)) < 1e-10


def test_surface_hit_agrees_with_the_bare_solver():
    """Surface.hit() is the solver plus frame/aperture/normal bookkeeping."""

    oval = CartesianOvalSurface(
        z0=Z0, zi=ZI, n_exterior=N0, n_interior=NI, origin=np.array([1.0, 2.0])
    )
    ray = Ray(origin=np.array([Z0 + 1.0, 2.5]), direction=np.array([1.0, 0.0]))
    hit = oval.hit(ray)
    assert hit is not None

    local_origin = oval.frame.to_local(ray.origin)
    local_direction = oval.frame.direction_to_local(ray.direction)
    _, lam = intersect_ray_with_surface(local_origin, local_direction, oval)
    assert hit.distance == pytest.approx(lam)
    # The stored normal is the normalized implicit gradient, in world axes.
    assert np.linalg.norm(hit.normal) == pytest.approx(1.0)
