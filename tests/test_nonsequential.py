"""Non-sequential tracing: light goes wherever the physics sends it.

Ground truths: power is conserved through every split (Fresnel and coated),
total internal reflection keeps all of it, ``split=False`` follows exactly
the branch a sequential trace takes — row for row the same rays — and both
engines put image points in the same place.
"""

from pathlib import Path

import numpy as np
import pytest

from raytracer import OpticalSystem, Rays
from raytracer.coatings import AR
from raytracer.elements import BeamSplitter, Lens, Mirror, Prism, Sensor
from raytracer.geometry.apertures import Rectangle
from raytracer.geometry.solids import Mesh
from raytracer.geometry.surfaces import Plane, Sphere
from raytracer.io import read_prescription
from raytracer.materials import ConstantIndex
from raytracer.rays import Status
from raytracer.sources import CollimatedSource, PointSource
from raytracer.tracing import NonSequentialTracer

COOKE = Path(__file__).resolve().parents[1] / "data/optical_systems/photographic/cooke_triplet_prescription.csv"
GLASS = ConstantIndex("G", 1.5)
R_NORMAL = ((1.5 - 1.0) / (1.5 + 1.0)) ** 2  # Fresnel reflectance at normal incidence


def birth(rays: Rays) -> np.ndarray:
    """The number of the point each ray was born at."""

    return np.isfinite(rays.trajectory[:, :, 0]).argmax(axis=1)


def powers(rays: Rays) -> np.ndarray:
    """``(N, V)``: the power each ray carries on from each of its points (NaN where it has none)."""

    return np.stack([rays.at(j).power for j in range(rays.trajectory.shape[1])], axis=1)


def detected(rays: Rays, detector: str) -> Rays:
    return rays[rays.detected & (rays.boundary == detector)]


def splitter_bench(window: Lens) -> OpticalSystem:
    """A 50/50 splitter at z=20 folding to -y, the given element at z=40 on
    the straight arm, and a sensor on each arm."""

    return OpticalSystem([
        BeamSplitter("BS", aperture=10.0, z=20.0, normal=(0.0, 1.0, 1.0)),
        window,
        Sensor("CAM_T", aperture=Rectangle(100.0, 100.0), z=200.0),
        Sensor("CAM_R", aperture=Rectangle(100.0, 100.0), pos=(0.0, -60.0, 20.0), normal=(0.0, -1.0, 0.0)),
    ])


def test_power_is_conserved_through_a_beam_splitter_and_a_lens():
    lens = Lens("L", front=Sphere(80.0), back=Sphere(-80.0), thickness=6.0, material=GLASS,
                semi_diameter=8.0, z=40.0)
    source = CollimatedSource((0.0, 0.0, 0.0), beam_radius=4.0, rays=37, power=1.0)
    rays = NonSequentialTracer(splitter_bench(lens), max_interactions=40, min_power=1e-12).trace(source)
    assert np.nansum(rays.at(0).power) == pytest.approx(1.0)
    # Every split takes from a ray exactly the power the ray split off it carries away ...
    split_off = np.flatnonzero(rays.parent >= 0)
    parent, j, power = rays.parent[split_off], birth(rays)[split_off], powers(rays)
    np.testing.assert_allclose(power[parent, j - 1], power[parent, j] + power[split_off, j], rtol=1e-12)
    # ... so where the rays end (detected, escaped, absorbed) holds what came in, up to
    # the rays not split off below min_power.
    assert rays.power.sum() == pytest.approx(1.0, abs=1e-9)
    assert detected(rays, "CAM_T").power.sum() + detected(rays, "CAM_R").power.sum() < 1.0
    assert detected(rays, "CAM_R").power.sum() == pytest.approx(0.5)  # the splitter's direct reflection
    # Two uncoated surfaces at near-normal incidence cost about 2 R each way.
    assert detected(rays, "CAM_T").power.sum() == pytest.approx(0.5 * (1 - R_NORMAL) ** 2, abs=2e-3)


def test_fresnel_losses_of_a_window_sum_the_incoherent_series():
    """A flat window at normal incidence transmits T^2 (1 + R^2 + R^4 + ...)
    = (1 - R) / (1 + R): every internal reflection retraces the axis."""

    window = Lens("W", front=Plane(), back=Plane(), thickness=5.0, material=GLASS,
                  semi_diameter=8.0, z=40.0)
    source = CollimatedSource((0.0, 0.0, 0.0), beam_radius=3.0, rays=19)
    rays = NonSequentialTracer(splitter_bench(window), max_interactions=60, min_power=1e-14).trace(source)
    expected = 0.5 * (1.0 - R_NORMAL) / (1.0 + R_NORMAL)
    camera = detected(rays, "CAM_T")
    assert camera.power.sum() == pytest.approx(expected, rel=1e-10)
    np.testing.assert_allclose(camera.direction, np.tile([0.0, 0.0, 1.0], (len(camera), 1)), atol=1e-12)
    assert rays.power.sum() == pytest.approx(1.0, abs=1e-12)
    # An anti-reflection coating removes the loss.
    coated = window.replace(front=window.front.replace(coating=AR()), back=window.back.replace(coating=AR()))
    rays = NonSequentialTracer(splitter_bench(coated)).trace(source)
    assert detected(rays, "CAM_T").power.sum() == pytest.approx(0.5)


def right_angle_prism() -> Prism:
    """Legs on the planes x = 0 and y = 0, hypotenuse x + y = 10, 10 mm deep in z."""

    return Prism("P", solid=Mesh.prism([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)], 10.0), material=GLASS)


def test_total_internal_reflection_in_a_prism():
    """Light entering a leg normally meets the hypotenuse at 45 degrees, beyond
    the 41.8 degree critical angle of n = 1.5: all of it turns 90 degrees."""

    x = np.linspace(2.0, 4.0, 5)
    launched = Rays(np.column_stack([x, np.full_like(x, -5.0), np.full_like(x, 5.0)]), np.tile([0.0, 1.0, 0.0], (5, 1)))
    rays = NonSequentialTracer(OpticalSystem([right_angle_prism()]), max_interactions=20, min_power=1e-12).trace(launched)

    entered = rays.at(1).event == "transmit"
    assert entered.sum() == 5 and (rays.parent[entered] == -1).all()  # the launched rays go on into the glass
    hypotenuse = rays.at(2)
    np.testing.assert_allclose(hypotenuse.position[entered, 0] + hypotenuse.position[entered, 1], 10.0, atol=1e-12)
    # Only reflection, with the whole power: no transmission under TIR, so nothing splits off there.
    assert (hypotenuse.event[entered] == "reflect").all()
    np.testing.assert_allclose(powers(rays)[entered, 2], powers(rays)[entered, 1], rtol=1e-15)
    np.testing.assert_allclose(hypotenuse.direction[entered], np.tile([-1.0, 0.0, 0.0], (5, 1)), atol=1e-12)
    assert not ((rays.parent >= 0) & (birth(rays) == 2)).any()

    # Out through the other leg along -x, having lost only the two normal-incidence reflections.
    out = (rays.status == Status.ESCAPED) & (np.abs(rays.direction[:, 0] + 1.0) < 1e-12) & (rays.parent == -1)
    assert out.sum() == 5
    np.testing.assert_allclose(rays.power[out], (1.0 - R_NORMAL) ** 2, rtol=1e-12)
    assert rays.power.sum() == pytest.approx(5.0, abs=1e-9)  # five rays of unit power


def test_split_false_follows_the_sequential_branch():
    """Nothing splits off, power untouched, refracted through glass, reflected
    off mirrors and under TIR — the clean picture for drawing."""

    ray = NonSequentialTracer(OpticalSystem([right_angle_prism()]), split=False).trace(Rays([3.0, -5.0, 5.0], [0.0, 1.0, 0.0]))
    assert len(ray) == 1 and ray.parent.tolist() == [-1]
    assert [ray.at(j).event[0] for j in range(4)] == [None, "transmit", "reflect", "transmit"]
    assert powers(ray).tolist() == [[1.0] * 4]
    np.testing.assert_allclose(ray.direction[0], [-1.0, 0.0, 0.0], atol=1e-12)
    assert ray.status.tolist() == [Status.ESCAPED]

    mirror = Mirror("M", surface=Sphere(-100.0), aperture=20.0, z=100.0)  # centre of curvature at 0
    rays = NonSequentialTracer(OpticalSystem([mirror]), split=False).trace(
        PointSource((0.0, 0.0, 0.0), half_angle_deg=5.0, rays=7, fan="y")
    )
    hit = rays.at(1)
    assert len(rays) == 7 and (hit.event == "reflect").all()
    # Rays from the centre of curvature return to it.
    v = -hit.position
    assert np.max(np.linalg.norm(np.cross(v, hit.direction), axis=1)) < 1e-12


def test_sequential_and_non_sequential_agree_on_image_points():
    prescription = read_prescription(COOKE)
    tracer = prescription.tracer(object_z=-500.0)
    y_obj = 40.0
    slopes = np.linspace(-0.088, -0.072, 9)  # through the stop, short of the rims
    origins = np.tile([0.0, y_obj, -500.0], (len(slopes), 1))
    directions = np.column_stack([np.zeros_like(slopes), slopes, np.ones_like(slopes)])
    launched = Rays(origins, directions)
    sequential = tracer.trace(launched)
    assert sequential.detected.all()

    ns = NonSequentialTracer(prescription.system, wavelength_um=prescription.wavelength_um, split=False)
    rays = ns.trace(launched)
    # Row for row the same rays, ending in the same place.
    assert len(rays) == len(slopes) and rays.detected.all() and (rays.boundary == "IMG").all()
    np.testing.assert_allclose(rays.position, sequential.position, atol=1e-9)
    np.testing.assert_allclose(rays.opl, sequential.opl, atol=1e-6)


def test_rays_that_meet_nothing_escape():
    """A ray that meets nothing keeps only the point it started from; drawing it gives an open piece."""

    rays = NonSequentialTracer(OpticalSystem([Sensor("S", aperture=1.0)])).trace(Rays([0.0, 5.0, -1.0], [0.0, 0.0, 1.0]))
    assert rays.status.tolist() == [Status.ESCAPED]
    assert rays.trajectory.shape == (1, 1, 3) and not rays.detected.any()
    points, _, _, _ = rays.segments()
    assert points.shape == (1, 2, 3) and np.isnan(points[0, 1]).all()


def test_dispersive_media_need_a_wavelength():
    from raytracer.materials import sellmeier_glass

    lens = Lens("L", front=Sphere(50.0), back=Plane(), thickness=5.0, material=sellmeier_glass("N-BK7"),
                semi_diameter=10.0)
    rays = Rays([0.0, 1.0, -5.0], [0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="wavelength"):
        NonSequentialTracer(OpticalSystem([lens])).trace(rays)
    blue = NonSequentialTracer(OpticalSystem([lens]), split=False).trace(Rays([0.0, 1.0, -5.0], [0.0, 0.0, 1.0], 0.48613))
    red = NonSequentialTracer(OpticalSystem([lens]), wavelength_um=0.65627, split=False).trace(rays)
    assert red.wavelength_um.tolist() == [0.65627]  # the tracer states it for rays that do not
    # Blue bends more toward the axis than red.
    assert blue.at(1).direction[0, 1] < red.at(1).direction[0, 1] < 0.0


def test_a_beam_splitter_splits_even_without_split():
    """split=False follows the sequential branch; a beam splitter has none, so it still splits."""

    system = OpticalSystem([BeamSplitter("BS", aperture=10.0, rot=(0.0, -45.0, 0.0))])
    beam = CollimatedSource((0.0, 0.0, -20.0), beam_radius=1.0, rays=7)
    rays = NonSequentialTracer(system, split=False).trace(beam)
    split_off = rays.parent >= 0
    assert split_off.sum() == 7 and len(rays) == 14
    at_splitter = rays.at(1)
    assert set(at_splitter.event[~split_off]) == {"transmit"} and set(at_splitter.event[split_off]) == {"reflect"}
    assert at_splitter.power.sum() == pytest.approx(1.0)
