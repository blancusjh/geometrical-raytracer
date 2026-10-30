"""Non-sequential tracing: light goes wherever the physics sends it.

Ground truths: power is conserved through every split (Fresnel and coated),
total internal reflection keeps all of it, ``split=False`` follows exactly
the branch a sequential trace takes, and both engines put image points in
the same place.
"""

from pathlib import Path

import numpy as np
import pytest

from raytracer import OpticalSystem
from raytracer.coatings import AR
from raytracer.elements import BeamSplitter, Lens, Mirror, Prism, Sensor
from raytracer.geometry.apertures import Rectangle
from raytracer.geometry.solids import Mesh
from raytracer.geometry.surfaces import Plane, Sphere
from raytracer.io import read_prescription
from raytracer.materials import ConstantIndex
from raytracer.sources import CollimatedSource, PointSource, Rays
from raytracer.tracing import NonSequentialTracer, SequentialTracer
from raytracer.tracing.nonsequential import PRIMARY, REFLECTED, TRANSMITTED

COOKE = Path(__file__).resolve().parents[1] / "data/optical_systems/photographic/cooke_triplet_prescription.csv"
GLASS = ConstantIndex("G", 1.5)
R_NORMAL = ((1.5 - 1.0) / (1.5 + 1.0)) ** 2  # Fresnel reflectance at normal incidence


def leaves(tree) -> np.ndarray:
    """Segments with no children: where power left the tree."""

    has_child = np.zeros(len(tree), dtype=bool)
    has_child[tree.parent[tree.parent >= 0]] = True
    return ~has_child


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
    tree = NonSequentialTracer(splitter_bench(lens), max_interactions=40, min_power=1e-12).trace(source)
    assert tree.power[tree.generation == 1].sum() == pytest.approx(1.0)
    # Every split hands its parent's power to its children exactly (checked where
    # no child can have been pruned below min_power) ...
    for i in np.flatnonzero(~leaves(tree) & (tree.power > 1e-6)):
        assert tree.power[tree.parent == i].sum() == pytest.approx(tree.power[i], rel=1e-12)
    # ... so what leaves the tree (detected, escaped, absorbed) is what came in, up
    # to the branches pruned below min_power.
    assert tree.power[leaves(tree)].sum() == pytest.approx(1.0, abs=1e-9)
    detected = tree.hits("CAM_T").power.sum() + tree.hits("CAM_R").power.sum()
    assert detected < 1.0
    assert tree.hits("CAM_R").power.sum() == pytest.approx(0.5)  # the splitter's direct reflection
    # Two uncoated surfaces at near-normal incidence cost about 2 R each way.
    assert tree.hits("CAM_T").power.sum() == pytest.approx(0.5 * (1 - R_NORMAL) ** 2, abs=2e-3)


def test_fresnel_losses_of_a_window_sum_the_incoherent_series():
    """A flat window at normal incidence transmits T^2 (1 + R^2 + R^4 + ...)
    = (1 - R) / (1 + R): every internal reflection retraces the axis."""

    window = Lens("W", front=Plane(), back=Plane(), thickness=5.0, material=GLASS,
                  semi_diameter=8.0, z=40.0)
    source = CollimatedSource((0.0, 0.0, 0.0), beam_radius=3.0, rays=19)
    tree = NonSequentialTracer(splitter_bench(window), max_interactions=60, min_power=1e-14).trace(source)
    expected = 0.5 * (1.0 - R_NORMAL) / (1.0 + R_NORMAL)
    assert tree.hits("CAM_T").power.sum() == pytest.approx(expected, rel=1e-10)
    np.testing.assert_allclose(tree.hits("CAM_T").directions, np.tile([0.0, 0.0, 1.0], (len(tree.hits("CAM_T")), 1)), atol=1e-12)
    assert tree.power[leaves(tree)].sum() == pytest.approx(1.0, abs=1e-12)
    # An anti-reflection coating removes the loss.
    coated = window.replace(front=window.front.replace(coating=AR()), back=window.back.replace(coating=AR()))
    tree = NonSequentialTracer(splitter_bench(coated)).trace(source)
    assert tree.hits("CAM_T").power.sum() == pytest.approx(0.5)


def right_angle_prism() -> Prism:
    """Legs on the planes x = 0 and y = 0, hypotenuse x + y = 10, 10 mm deep in z."""

    return Prism("P", solid=Mesh.prism([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)], 10.0), material=GLASS)


def test_total_internal_reflection_in_a_prism():
    """Light entering a leg normally meets the hypotenuse at 45 degrees, beyond
    the 41.8 degree critical angle of n = 1.5: all of it turns 90 degrees."""

    x = np.linspace(2.0, 4.0, 5)
    rays = Rays(np.column_stack([x, np.full_like(x, -5.0), np.full_like(x, 5.0)]),
                np.tile([0.0, 1.0, 0.0], (5, 1)), np.nan, 1.0)
    tree = NonSequentialTracer(OpticalSystem([right_angle_prism()]), max_interactions=20, min_power=1e-12).trace(rays)

    entered = (tree.generation == 2) & (tree.kind == TRANSMITTED)
    assert entered.sum() == 5
    at_hypotenuse = np.flatnonzero(entered)
    np.testing.assert_allclose(tree.end[at_hypotenuse, 0] + tree.end[at_hypotenuse, 1], 10.0, atol=1e-12)
    for i in at_hypotenuse:
        children = np.flatnonzero(tree.parent == i)
        # Only the reflected child survives, with the whole power: no transmission under TIR.
        assert tree.kind[children].tolist() == [REFLECTED]
        assert tree.power[children[0]] == pytest.approx(tree.power[i])
        np.testing.assert_allclose(tree.direction[children[0]], [-1.0, 0.0, 0.0], atol=1e-12)

    # Out through the other leg along -x, having lost only the two normal-incidence reflections.
    out = tree.escaped & (np.abs(tree.direction[:, 0] + 1.0) < 1e-12) & (tree.generation == 4)
    assert out.sum() == 5
    np.testing.assert_allclose(tree.power[out], (1.0 - R_NORMAL) ** 2, rtol=1e-12)
    assert tree.power[leaves(tree)].sum() == pytest.approx(5.0, abs=1e-9)  # five rays of unit power


def test_split_false_follows_the_sequential_branch():
    """One child per hit, power untouched, refracted through glass, reflected
    off mirrors and under TIR — the clean picture for drawing."""

    tree = NonSequentialTracer(OpticalSystem([right_angle_prism()]), split=False).trace(
        Rays([[3.0, -5.0, 5.0]], [[0.0, 1.0, 0.0]], np.nan, 1.0)
    )
    assert tree.kind.tolist() == [PRIMARY, TRANSMITTED, REFLECTED, TRANSMITTED]
    assert tree.parent.tolist() == [-1, 0, 1, 2]
    assert tree.generation.tolist() == [1, 2, 3, 4]
    assert tree.power.tolist() == [1.0] * 4
    np.testing.assert_allclose(tree.direction[-1], [-1.0, 0.0, 0.0], atol=1e-12)
    assert tree.escaped.tolist() == [False, False, False, True]

    mirror = Mirror("M", surface=Sphere(-100.0), aperture=20.0, z=100.0)  # centre of curvature at 0
    tree = NonSequentialTracer(OpticalSystem([mirror]), split=False).trace(
        PointSource((0.0, 0.0, 0.0), half_angle_deg=5.0, rays=7, fan="y")
    )
    reflected = tree.kind == REFLECTED
    assert reflected.sum() == 7 and (tree.kind != TRANSMITTED).all()
    # Rays from the centre of curvature return to it.
    v = -tree.start[reflected]
    assert np.max(np.linalg.norm(np.cross(v, tree.direction[reflected]), axis=1)) < 1e-12


def test_sequential_and_non_sequential_agree_on_image_points():
    prescription = read_prescription(COOKE)
    tracer = prescription.tracer(object_z=-500.0)
    y_obj = 40.0
    slopes = np.linspace(-0.088, -0.072, 9)  # through the stop, short of the rims
    origins = np.tile([0.0, y_obj, -500.0], (len(slopes), 1))
    directions = np.column_stack([np.zeros_like(slopes), slopes, np.ones_like(slopes)])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    batch = tracer.trace_batch(origins, directions)
    assert batch.valid.all()

    ns = NonSequentialTracer(prescription.system, wavelength_um=prescription.wavelength_um, split=False)
    tree = ns.trace(Rays(origins, directions, np.nan, 1.0))
    hits = tree.hits("IMG")
    assert len(hits) == len(slopes)
    order = np.argsort(hits.points[:, 1])
    np.testing.assert_allclose(hits.points[order], batch.image_points[np.argsort(batch.image_points[:, 1])], atol=1e-9)
    np.testing.assert_allclose(np.sort(hits.opl), np.sort(batch.opl), atol=1e-6)


def test_rays_that_meet_nothing_escape():
    """Rays that meet nothing leave the system with face None, extended by
    escape_length for drawing."""

    tree = NonSequentialTracer(OpticalSystem([Sensor("S", aperture=1.0)]), escape_length=50.0).trace(
        Rays([[0.0, 5.0, -1.0]], [[0.0, 0.0, 1.0]], np.nan, 1.0)
    )
    assert tree.face == [None] and tree.escaped.all()
    assert tree.end[0] == pytest.approx([0.0, 5.0, 49.0])
    assert tree.detectors == {}
    assert tree.segments().shape == (1, 2, 3)


def test_dispersive_media_need_a_wavelength():
    from raytracer.materials import sellmeier_glass

    lens = Lens("L", front=Sphere(50.0), back=Plane(), thickness=5.0, material=sellmeier_glass("N-BK7"),
                semi_diameter=10.0)
    rays = Rays([[0.0, 1.0, -5.0]], [[0.0, 0.0, 1.0]], np.nan, 1.0)
    with pytest.raises(ValueError, match="wavelength"):
        NonSequentialTracer(OpticalSystem([lens])).trace(rays)
    blue = NonSequentialTracer(OpticalSystem([lens]), split=False).trace(
        Rays([[0.0, 1.0, -5.0]], [[0.0, 0.0, 1.0]], 0.48613, 1.0))
    red = NonSequentialTracer(OpticalSystem([lens]), wavelength_um=0.65627, split=False).trace(rays)
    # Blue bends more toward the axis than red.
    assert blue.direction[1, 1] < red.direction[1, 1] < 0.0
