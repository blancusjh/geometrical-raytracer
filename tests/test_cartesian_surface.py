"""The Cartesian (Descartes) surface: its GOTS mathematics, and stigmatism traced by both engines."""

import numpy as np
import pytest

from raytracer.elements import Lens, Sensor
from raytracer.geometry.surfaces import CartesianSurface, Plane
from raytracer.geometry.surfaces.cartesian import (
    cartesian_oval_implicit,
    cartesian_oval_parametric_curve,
    cartesian_oval_sag_and_slope,
    gots_params,
    max_usable_height,
)
from raytracer.materials import ConstantIndex
from raytracer.sources import PointSource
from raytracer.system import OpticalSystem
from raytracer.tracing import NonSequentialTracer, SequentialTracer, TraceStatus


N0, Z0, NI, ZI = 1.0, -30.0, 1.7, 10.0


def test_parametric_curve_satisfies_fermat_condition():
    """(z(rho), r(rho)) must lie exactly on the two-point Fermat oval."""

    G, O, T, S = gots_params(N0, Z0, NI, ZI)
    rho = np.linspace(0.01, 5.0, 25)
    z, r = cartesian_oval_parametric_curve(rho, G, O, T, S)
    residual = [cartesian_oval_implicit(zz, rr, Z0, ZI, N0, NI) for zz, rr in zip(z, r)]
    assert np.max(np.abs(residual)) < 1e-8


def test_sag_and_slope_matches_finite_difference():
    G, O, T, S = gots_params(N0, Z0, NI, ZI)
    h = np.linspace(0.0, 4.0, 20)
    _, dz = cartesian_oval_sag_and_slope(h, G, O, T, S)
    eps = 1e-6
    z_plus, _ = cartesian_oval_sag_and_slope(h + eps, G, O, T, S)
    z_minus, _ = cartesian_oval_sag_and_slope(np.maximum(h - eps, 0.0), G, O, T, S)
    denom = np.where(h < eps, eps, 2.0 * eps)
    finite_diff = (z_plus - z_minus) / denom
    assert np.max(np.abs(dz[1:] - finite_diff[1:])) < 1e-5


def test_sag_at_height_satisfies_fermat_condition():
    """sag_and_slope(h) inverts rho->r(rho) correctly: z(h) must also solve Fermat."""

    G, O, T, S = gots_params(N0, Z0, NI, ZI)
    h = np.linspace(0.0, 4.0, 15)
    z, _ = cartesian_oval_sag_and_slope(h, G, O, T, S)
    residual = [cartesian_oval_implicit(zz, hh, Z0, ZI, N0, NI) for zz, hh in zip(z, h)]
    assert np.max(np.abs(residual)) < 1e-8


def test_max_usable_height_is_the_r_of_rho_peak():
    G, O, T, S = gots_params(N0, Z0, NI, ZI)
    h_max = max_usable_height(G, O, T, S)
    assert h_max > 0
    # Just inside the domain must invert cleanly; the derivative blows up at
    # the peak itself, so stay comfortably below it.
    z, dz = cartesian_oval_sag_and_slope(np.array([0.9 * h_max]), G, O, T, S)
    assert np.isfinite(z).all() and np.isfinite(dz).all()


def test_profile_curvature_matches_paraxial_limit():
    """Vertex curvature should equal O: z(h) ~ (O/2) h**2 near the axis."""

    profile = CartesianSurface(n0=N0, z0=Z0, ni=NI, zi=ZI)
    h = 1e-4
    z = float(profile.profile(np.array([h]))[0])
    assert z == pytest.approx(0.5 * profile.curvature * h * h, rel=1e-3)


def test_surface_wraps_the_module_functions():
    """CartesianSurface is the GOTS math with value semantics: its parameters,
    usable radius and sag are the module functions' results."""

    surface = CartesianSurface(N0, Z0, NI, ZI)
    assert surface.gots == gots_params(N0, Z0, NI, ZI)
    assert (surface.G, surface.O, surface.T, surface.S) == surface.gots
    assert surface.max_radius == pytest.approx(max_usable_height(*surface.gots))
    h = np.linspace(0.0, 4.0, 9)
    z, dz = cartesian_oval_sag_and_slope(h, *surface.gots)
    s, ds = surface.sag_and_slope(h)
    np.testing.assert_array_equal(s, z)
    np.testing.assert_array_equal(ds, dz)
    # The 3-D sag is the profile swept about the axis.
    phi = np.deg2rad(40.0)
    np.testing.assert_allclose(surface.sag(h * np.cos(phi), h * np.sin(phi)), z, atol=1e-15)
    assert surface == CartesianSurface(N0, Z0, NI, ZI)
    with pytest.raises(AttributeError):
        surface.z0 = -20.0
    assert surface.replace(z0=-20.0).z0 == -20.0


Z0, ZI, N0, NI = -30.0, 10.0, 1.0, 1.7
GLASS = ConstantIndex("TEST_GLASS", NI)


def oval_lens(semi_diameter: float, *, extra: float = 0.0) -> OpticalSystem:
    """Air -> glass through the Cartesian surface; the sensor sits at the image
    point ``ZI`` (on the glass's flat back face when ``extra`` is 0)."""

    lens = Lens(
        "L", front=CartesianSurface(N0, Z0, NI, ZI), back=Plane(),
        thickness=ZI + extra, material=GLASS, semi_diameter=semi_diameter,
    )
    return OpticalSystem([lens, Sensor("IMG", z=ZI)], path=["L.front", "IMG"])


def test_semidiameter_clips_rays_beyond_the_aperture():
    tracer = SequentialTracer(oval_lens(2.0, extra=5.0))
    trace = tracer.trace(
        np.array([[0.0, 1.0, Z0], [0.0, 3.0, Z0]]),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    )
    assert trace.status[0] == TraceStatus.OK
    assert trace.status[1] == TraceStatus.VIGNETTED
    assert trace.failed_step[1] == 0


def test_drawn_curve_matches_the_traced_surface():
    """The closed-form meridian (what a drawing uses) is the sag the tracer
    intersects, and both lie on the Descartes oval."""

    surface = CartesianSurface(N0, Z0, NI, ZI)
    rho = np.linspace(1e-3, surface.max_radius, 64)  # rho >= r: still on the vertex branch
    z, r = surface.meridian(rho)
    assert r.max() > 0.75 * surface.max_radius  # not a degenerate placeholder curve
    np.testing.assert_allclose(surface.profile(r), z, atol=1e-9)
    assert np.max(np.abs(cartesian_oval_implicit(z, r, Z0, ZI, N0, NI))) < 1e-8


def test_axial_object_point_images_stigmatically():
    """Every ray from z0 reaches zi exactly, at any aperture the surface supports."""

    surface = CartesianSurface(N0, Z0, NI, ZI)
    semi = 0.8 * surface.max_radius
    tracer = SequentialTracer(oval_lens(semi), restart_offset=0.0)
    angles = np.arctan(np.linspace(-0.8, 0.8, 41) * semi / -Z0)
    origins = np.tile([0.0, 0.0, Z0], (len(angles), 1))
    phi = np.linspace(0.0, 2 * np.pi, len(angles))
    directions = np.column_stack(
        [np.sin(angles) * np.cos(phi), np.sin(angles) * np.sin(phi), np.cos(angles)]
    )
    trace = tracer.trace(origins, directions)
    assert trace.ok.all()
    assert np.max(np.abs(trace.end[:, :2])) < 1e-9
    # Fermat: every path from the object to the image has the same optical length.
    expected = N0 * abs(Z0) + NI * ZI
    assert np.max(np.abs(trace.opl - expected)) < 1e-9


def test_non_sequential_trace_finds_the_same_focus():
    surface = CartesianSurface(N0, Z0, NI, ZI)
    system = oval_lens(0.8 * surface.max_radius)
    half_angle = np.degrees(np.arctan(0.7 * surface.max_radius / -Z0))
    source = PointSource((0.0, 0.0, Z0), half_angle_deg=half_angle, rays=61)
    tree = NonSequentialTracer(system, split=False).trace(source)
    hits = tree.hits("IMG")
    assert len(hits) == 61
    assert np.max(np.abs(hits.point[:, :2])) < 1e-9
    assert hits.opl == pytest.approx(N0 * abs(Z0) + NI * ZI, abs=1e-9)
