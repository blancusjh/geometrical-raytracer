"""The surface hierarchy (``raytracer.geometry.surfaces``).

Inheritance follows mathematical specialization — a subclass is a subset
of its parent's family — so the isinstance relations are statements of
geometry and are tested as such, together with what every surface must
get right: its gradient (the normal, and the Newton derivative) and the
agreement of the closed-form and iterative intersections.
"""

import numpy as np
import pytest

from raytracer.geometry.intersections import newton_graph
from raytracer.geometry.surfaces import (
    Asphere,
    CartesianSurface,
    Conic,
    CylindricalSurface,
    Ellipsoid,
    EvenAsphere,
    FreeformSurface,
    Hyperboloid,
    Paraboloid,
    Plane,
    QTypeAsphere,
    Sphere,
    Surface,
    SurfaceOfRevolution,
    ToroidalSurface,
)

INF = float("inf")


# -- the hierarchy --------------------------------------------------------------


def test_sphere_is_an_ellipsoid_is_a_conic_is_a_surface_of_revolution():
    s = Sphere(10.0)
    for family in (Ellipsoid, Conic, SurfaceOfRevolution, Surface):
        assert isinstance(s, family)
    assert issubclass(Sphere, Ellipsoid)
    assert issubclass(Ellipsoid, Conic)
    assert issubclass(Conic, SurfaceOfRevolution)


def test_plane_is_a_conic_of_no_one_family():
    p = Plane()
    assert isinstance(p, Conic) and isinstance(p, SurfaceOfRevolution)
    for family in (Ellipsoid, Sphere, Paraboloid, Hyperboloid):
        assert not isinstance(p, family)
    assert p.is_plane and p.curvature == 0.0


def test_the_other_families():
    assert isinstance(Paraboloid(10.0), Conic) and not isinstance(Paraboloid(10.0), Ellipsoid)
    assert isinstance(Hyperboloid(10.0, -2.0), Conic) and not isinstance(Hyperboloid(10.0, -2.0), Ellipsoid)
    assert issubclass(CylindricalSurface, ToroidalSurface)
    assert not issubclass(ToroidalSurface, SurfaceOfRevolution)
    assert issubclass(CartesianSurface, SurfaceOfRevolution) and not issubclass(CartesianSurface, Conic)
    assert issubclass(EvenAsphere, Asphere) and issubclass(QTypeAsphere, Asphere)
    assert issubclass(Asphere, SurfaceOfRevolution) and not issubclass(Asphere, Conic)
    assert not issubclass(FreeformSurface, SurfaceOfRevolution)


def test_families_enforce_their_constant():
    with pytest.raises(ValueError):
        Ellipsoid(10.0, -1.5)
    with pytest.raises(ValueError):
        Hyperboloid(10.0, -0.5)
    with pytest.raises(ValueError):
        Conic(0.0)
    with pytest.raises(TypeError):
        Paraboloid()
    with pytest.raises(TypeError):
        Paraboloid(10.0, f=5.0)
    assert Paraboloid(f=5.0) == Paraboloid(10.0)
    assert Conic(25.0, -7.0).k == -7.0  # the general conic accepts any constant


@pytest.mark.parametrize(
    "conic, expected",
    [
        (Conic(INF, 0.3), Plane()),
        (Conic(40.0, 0.0), Sphere(40.0)),
        (Conic(40.0, -1.0), Paraboloid(40.0)),
        (Conic(40.0, -0.4), Ellipsoid(40.0, -0.4)),
        (Conic(40.0, 0.6), Ellipsoid(40.0, 0.6)),
        (Conic(-40.0, -3.0), Hyperboloid(-40.0, -3.0)),
    ],
)
def test_classify_returns_the_most_specific_family(conic, expected):
    specific = conic.classify()
    assert type(specific) is type(expected)
    assert specific == expected
    r = np.linspace(0.0, 5.0, 11)
    np.testing.assert_array_equal(specific.profile(r), conic.profile(r))


def test_surfaces_are_values():
    assert Sphere(10.0) == Sphere(10.0) and hash(Sphere(10.0)) == hash(Sphere(10.0))
    assert Sphere(10.0) != Conic(10.0, 0.0)  # same shape, different description
    assert Conic(10.0, 0.0).classify() == Sphere(10.0)
    s = EvenAsphere(50.0, -1.0, {4: 1e-6})
    with pytest.raises(AttributeError):
        s.R = 20.0
    assert s.replace(R=20.0) == EvenAsphere(20.0, -1.0, {4: 1e-6})
    with pytest.raises(TypeError):
        s.replace(radius=20.0)


def test_ellipsoid_semi_axes_round_trip():
    e = Ellipsoid.from_semi_axes(-7.0, 3.0)
    axial, transverse = e.semi_axes
    assert axial == pytest.approx(-7.0) and transverse == pytest.approx(3.0)
    assert e.max_radius == pytest.approx(3.0)  # the vertex sheet ends at the equator


# -- gradients ------------------------------------------------------------------

SURFACES = [
    Plane(),
    Sphere(-35.0),
    Ellipsoid(30.0, 0.7),
    Ellipsoid(30.0, -0.6),
    Paraboloid(-25.0),
    Hyperboloid(20.0, -2.5),
    Conic(45.0, -1.3),
    EvenAsphere(-60.0, -0.8, {4: 3e-6, 6: -2e-9, 8: 1e-12}),
    QTypeAsphere(40.0, -0.5, (2e-3, -1e-3, 5e-4, 2e-4), norm_radius=6.0),
    CartesianSurface(1.0, -30.0, 1.7, 10.0),
    ToroidalSurface(30.0, -45.0, -0.4),
    CylindricalSurface(22.0, axis="x"),
    CylindricalSurface(-22.0, 0.5, axis="y"),
    FreeformSurface(80.0, -0.3, {(2, 0): 0.05, (1, 2): -0.02, (0, 3): 0.01, (3, 1): 0.004}, norm_radius=5.0),
]


@pytest.mark.parametrize("surface", SURFACES, ids=lambda s: type(s).__name__)
def test_gradient_matches_finite_differences(surface):
    rng = np.random.default_rng(11)
    radius = rng.uniform(0.0, min(4.0, 0.9 * surface.max_radius), 40)  # inside the domain of the sag
    phi = rng.uniform(0.0, 2 * np.pi, 40)
    x, y = radius * np.cos(phi), radius * np.sin(phi)
    s, dsdx, dsdy = surface.sag_gradient(x, y)
    h = 1e-6
    fx = (surface.sag(x + h, y) - surface.sag(x - h, y)) / (2 * h)
    fy = (surface.sag(x, y + h) - surface.sag(x, y - h)) / (2 * h)
    np.testing.assert_allclose(dsdx, fx, atol=2e-8)
    np.testing.assert_allclose(dsdy, fy, atol=2e-8)
    np.testing.assert_array_equal(s, surface.sag(x, y))
    # The normal is the unit gradient of f = z - s, on the +z side.
    n = surface.normal(np.column_stack([x, y, s]))
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-15)
    assert (n[:, 2] > 0).all()
    np.testing.assert_allclose(n[:, 0] / n[:, 2], -dsdx, atol=1e-12)


@pytest.mark.parametrize("surface", SURFACES, ids=lambda s: type(s).__name__)
def test_vertex_curvature_matches_the_numerical_second_derivative(surface):
    numeric = Surface.vertex_curvature.fget(surface)
    np.testing.assert_allclose(surface.vertex_curvature, numeric, atol=1e-6)
    assert surface.sag(np.zeros(1), np.zeros(1))[0] == pytest.approx(0.0, abs=1e-15)


def test_toroid_specializations():
    r = np.linspace(-5.0, 5.0, 21)
    # Equal radii: the toroid is the sphere.
    np.testing.assert_allclose(ToroidalSurface(30.0, 30.0).sag(r, r[::-1] / 2), Sphere(30.0).sag(r, r[::-1] / 2), atol=1e-13)
    # A cylinder's generating lines are straight along its axis.
    cyl = CylindricalSurface(22.0, axis="x")
    np.testing.assert_array_equal(cyl.sag(r, np.full_like(r, 2.0)), cyl.sag(np.zeros_like(r), np.full_like(r, 2.0)))
    np.testing.assert_allclose(cyl.sag(np.zeros_like(r), r), Sphere(22.0).profile(r), atol=1e-14)
    assert cyl.vertex_curvature == (0.0, pytest.approx(1 / 22.0))
    assert CylindricalSurface(22.0, axis="y").vertex_curvature == (pytest.approx(1 / 22.0), 0.0)


# -- aspheres -------------------------------------------------------------------


def test_qtype_departure_is_fourth_order():
    """Forbes' Q-con terms start at u^4: the vertex curvature is the base's,
    and halving r divides the departure by 16 near the axis."""

    q = QTypeAsphere(40.0, -0.5, (2e-3, -1e-3, 5e-4, 2e-4), norm_radius=6.0)
    base = q.base
    assert base == Ellipsoid(40.0, -0.5)
    r = np.array([1e-3, 5e-4])
    departure = q.profile(r) - base.profile(r)
    assert departure[0] / departure[1] == pytest.approx(16.0, rel=1e-6)
    assert q.curvature == pytest.approx(base.curvature, rel=1e-6)
    # The first term is exactly a_0 (r / norm_radius)^4 (Q_0 = 1).
    flat = QTypeAsphere(INF, 0.0, (3e-3,), norm_radius=5.0)
    r = np.linspace(0.0, 5.0, 11)
    np.testing.assert_allclose(flat.profile(r), 3e-3 * (r / 5.0) ** 4, atol=1e-16)


# -- closed form versus Newton ---------------------------------------------------


@pytest.mark.parametrize(
    "conic",
    [Sphere(30.0), Sphere(-30.0), Ellipsoid(30.0, 0.8), Ellipsoid(-30.0, -0.5), Paraboloid(25.0),
     Hyperboloid(-20.0, -3.0), Conic(40.0, -1.6)],
    ids=repr,
)
def test_closed_form_conic_intersection_equals_newton(conic):
    rng = np.random.default_rng(5)
    n = 50
    o = np.column_stack([rng.uniform(-5, 5, n), rng.uniform(-5, 5, n), rng.uniform(-30, -5, n)])
    d = np.column_stack([rng.uniform(-0.2, 0.2, n), rng.uniform(-0.2, 0.2, n), np.ones(n)])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    closed = conic.intersect(o, d)
    newton = newton_graph(conic, o, d, np.ones(n, dtype=bool))
    both = closed.ok & newton.ok
    assert both.sum() > 0.9 * n
    np.testing.assert_allclose(closed.t[both], newton.t[both], atol=1e-9)
