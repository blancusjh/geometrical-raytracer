"""The Cartesian surface as an optical element: clipping, drawing, stigmatism.

(Was ``test_cartesian_oval_surface.py``, on the 2-D ``CartesianOvalSurface``
with media attached. Now the surface is geometry, a :class:`Lens` carries it,
and the media come from the lens and its surroundings.)
"""

import numpy as np
import pytest

from raytracer.elements import Lens, Sensor
from raytracer.geometry.surfaces import CartesianSurface, Plane
from raytracer.geometry.surfaces.cartesian import cartesian_oval_implicit
from raytracer.materials import ConstantIndex
from raytracer.sources import PointSource
from raytracer.system import OpticalSystem
from raytracer.tracing import NonSequentialTracer, SequentialTracer, TraceStatus

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
    batch = tracer.trace_batch(
        np.array([[0.0, 1.0, Z0], [0.0, 3.0, Z0]]),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    )
    assert batch.status[0] == TraceStatus.OK
    assert batch.status[1] == TraceStatus.VIGNETTED
    assert batch.failed_step[1] == 0


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
    batch = tracer.trace_batch(origins, directions)
    assert batch.valid.all()
    assert np.max(np.abs(batch.image_points[:, :2])) < 1e-9
    # Fermat: every path from the object to the image has the same optical length.
    expected = N0 * abs(Z0) + NI * ZI
    assert np.max(np.abs(batch.opl - expected)) < 1e-9


def test_non_sequential_trace_finds_the_same_focus():
    surface = CartesianSurface(N0, Z0, NI, ZI)
    system = oval_lens(0.8 * surface.max_radius)
    half_angle = np.degrees(np.arctan(0.7 * surface.max_radius / -Z0))
    source = PointSource((0.0, 0.0, Z0), half_angle_deg=half_angle, rays=61)
    tree = NonSequentialTracer(system, split=False).trace(source)
    hits = tree.hits("IMG")
    assert len(hits) == 61
    assert np.max(np.abs(hits.points[:, :2])) < 1e-9
    assert hits.opl == pytest.approx(N0 * abs(Z0) + NI * ZI, abs=1e-9)
