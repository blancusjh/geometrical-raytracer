"""Ten independent validation contracts; see docs/geometrical_validation.md.

No parametrization hides additional tests. Related cases within a contract
share one physical proposition and identify their inputs in assertion messages.
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from raytracer.analysis.aberrations import distortion_map, parabasal_focus, seidel_coefficients
from raytracer.analysis.aberrations.chromatic import axial_color, chromatic_spots
from raytracer.analysis.aberrations.geometric import axial_intercepts, geometric_aberrations
from raytracer.analysis.tolerancing import (
    GeometricEvaluator,
    Perturbation,
    Tolerance,
    monte_carlo,
    sensitivity,
)
from raytracer.design import OpticalSystem, SurfaceRow, to_branching_surfaces
from raytracer.math.transforms import RigidTransform
from raytracer.optics.laws import reflect, refract
from raytracer.optics.materials import AIR, ConstantIndex, sellmeier_glass
from raytracer.optics.radiometry import fresnel_coefficients
from raytracer.optics.ray import Ray
from raytracer.propagation import (
    FieldPoint,
    ParaxialModel,
    PupilSampling,
    SequentialTracer,
    TraceStatus,
    solve_object_plane,
    trace_pupil,
)
from raytracer.propagation.aiming import aim_stop_targets
from raytracer.surfaces.apertures import CircularAperture, RectangularAperture
from raytracer.surfaces.profile import AsphereProfile

ROOT = Path(__file__).resolve().parents[1]
GLASS = ConstantIndex("test glass", 1.5)


def singlet(*, material=GLASS, wavelength_um=None):
    front = SurfaceRow.refracting(radius=50, thickness=5, material=material, semidiameter=8)
    front.is_stop = True
    back = SurfaceRow.refracting(radius=-50, thickness=50, material=AIR, semidiameter=8)
    return OpticalSystem([front, back], object_z=-100, wavelength_um=wavelength_um)


def collimated_bundle(heights, *, start_z=-200):
    heights = np.asarray(heights, dtype=float)
    origins = np.column_stack([np.zeros_like(heights), heights, np.full_like(heights, start_z)])
    directions = np.tile([0.0, 0.0, 1.0], (len(heights), 1))
    return origins, directions


def test_01_interface_laws_reciprocity_and_optical_length():
    """Snell, reflection, critical angle, R+T=1 and analytic slab OPL."""

    normal = np.array([0.0, 0.0, -1.0])
    for theta in np.deg2rad([0, 10, 35, 70]):
        incoming = np.array([np.sin(theta), 0, np.cos(theta)])
        outgoing = refract(incoming, normal, 1.0, 1.5)
        expected_sine = np.sin(theta) / 1.5
        assert_allclose(outgoing, [expected_sine, 0, np.sqrt(1 - expected_sine**2)], atol=2e-15)
        assert_allclose(reflect(incoming, normal), [np.sin(theta), 0, -np.cos(theta)], atol=2e-15)
        assert_allclose(refract(-outgoing, -normal, 1.5, 1), -incoming, atol=2e-15)
        coefficients = fresnel_coefficients(incoming, normal, 1, 1.5)
        assert_allclose(
            [coefficients.r_s + coefficients.t_s, coefficients.r_p + coefficients.t_p], 1
        )
    brewster = np.arctan(1.5)
    assert (
        fresnel_coefficients(np.array([np.sin(brewster), 0, np.cos(brewster)]), normal, 1, 1.5).r_p
        < 1e-28
    )
    critical = np.arcsin(1 / 1.5)
    assert (
        refract(np.array([np.sin(critical + 0.01), 0, np.cos(critical + 0.01)]), normal, 1.5, 1)
        is None
    )
    system = OpticalSystem(
        [
            SurfaceRow.refracting(radius=0, thickness=20, material=GLASS),
            SurfaceRow.refracting(radius=0, thickness=30, material=AIR),
        ]
    )
    theta = 0.3
    result = SequentialTracer(system).trace([0, 0, -10], [np.sin(theta), 0, np.cos(theta)])
    glass_cosine = np.sqrt(1 - (np.sin(theta) / 1.5) ** 2)
    assert result.ok
    assert_allclose(result.opl, 40 / np.cos(theta) + 30 / glass_cosine, atol=2e-12, rtol=0)


def test_02_surface_intersections_domains_and_forward_paths():
    """Closed-form sphere geometry, equator, misses, parallel and backward rays."""

    system = OpticalSystem([SurfaceRow.refracting(radius=10, thickness=20, material=AIR)])
    origins, directions = collimated_bundle([0, 5, 9.9, 10, 11], start_z=-10)
    result = SequentialTracer(system).trace_batch(origins, directions, keep_paths=True)
    assert np.all(result.valid[:4])
    assert result.status[4] == TraceStatus.NO_INTERSECTION
    assert_allclose(result.paths[:4, 1, 2], 10 - np.sqrt(100 - origins[:4, 1] ** 2), atol=3e-12)
    # A horizontal ray can intersect a curved sag surface.
    ray = SequentialTracer(system).trace([0, -12, 5], [0, 1, 0])
    assert_allclose(ray.path[1], [0, -np.sqrt(75), 5], atol=3e-12)
    # Its final direction is parallel to the image plane, which is diagnosed.
    assert ray.status == TraceStatus.PARALLEL
    plane = OpticalSystem([SurfaceRow.stop(thickness=10)])
    assert SequentialTracer(plane).trace([0, 0, 1], [0, 0, 1]).status == TraceStatus.BACKWARD
    with pytest.raises(ValueError, match="nonzero"):
        SequentialTracer(plane).trace([0, 0, -1], [0, 0, 0])
    assert np.isnan(AsphereProfile.sphere(10).sag(11))


def test_03_gaussian_optics_virtual_conjugates_and_dispersion():
    """Thick-lens power and Gaussian conjugates retain physical signs."""

    for wavelength in (0.48613, 0.58756, 0.65627):
        glass = sellmeier_glass("N-BK7")
        system = singlet(material=glass, wavelength_um=wavelength)
        n = glass.index(wavelength)
        power = (n - 1) * (1 / 50 - 1 / -50 + (n - 1) * 5 / (n * 50 * -50))
        assert_allclose(ParaxialModel(system).effective_focal_length(), 1 / power, rtol=2e-14)
        system.set_thickness(1, ParaxialModel(system).back_focal_z() - 5 - 0.01)
        expected = ParaxialModel(system).solve_object_plane()
        actual = solve_object_plane(SequentialTracer(system), assign=False)
        assert expected.object_z > 0 and actual.object_z > 0
        assert_allclose(actual.object_z, expected.object_z, rtol=2e-7)
        assert_allclose(actual.magnification, expected.magnification, rtol=2e-7)
    assert_allclose(sellmeier_glass("N-BK7").index(0.58756), 1.5168, atol=1e-5)
    # In different external media, image EFL includes the image index.
    immersion = OpticalSystem([SurfaceRow.refracting(radius=50, thickness=10, material=GLASS)])
    assert_allclose(ParaxialModel(immersion).effective_focal_length(), 150)
    glass = sellmeier_glass("N-BK7")
    blue = ParaxialModel(singlet(material=glass, wavelength_um=0.48613)).back_focal_z()
    red = ParaxialModel(singlet(material=glass, wavelength_um=0.65627)).back_focal_z()
    assert blue < red

    # Independent manufacturer values, rounded to five decimal places.
    reference_wavelengths = [0.48613, 0.58756, 0.65627, 1.014]
    for name, expected in [
        ("N-BK7", [1.52238, 1.51680, 1.51432, 1.50731]),
        ("N-F2", [1.63208, 1.62005, 1.61506, 1.60261]),
    ]:
        material = sellmeier_glass(name)
        assert_allclose(
            [material.index(wl) for wl in reference_wavelengths], expected, atol=6e-6, rtol=0
        )
        with pytest.raises(ValueError, match="outside supported"):
            material.index(0.2)
        with pytest.raises(ValueError):
            material.index(np.inf)
    infinite_object = singlet(material=glass, wavelength_um=0.58756)
    infinite_object.object_z = -np.inf
    color = axial_color(infinite_object, reference_wavelengths[:3], reference_wavelength_um=0.58756)
    assert_allclose(
        color.focus_z_mm,
        [
            ParaxialModel(infinite_object.at_wavelength(wl)).back_focal_z()
            for wl in reference_wavelengths[:3]
        ],
    )
    with pytest.raises(ValueError, match="exactly once"):
        axial_color(infinite_object, [0.48613, 0.65627], reference_wavelength_um=0.58756)

    # Oblique plane-parallel plate: exact Snell displacement and disk moments.
    plate = OpticalSystem(
        [
            SurfaceRow.refracting(
                radius=0, thickness=5, material=glass, semidiameter=1, is_stop=True
            ),
            SurfaceRow.refracting(radius=0, thickness=10, material=AIR),
        ],
        wavelength_um=0.58756,
    )
    wavelengths = np.array(reference_wavelengths[:3])
    spectrum = np.array([0.2, 0.5, 0.3])
    spots = chromatic_spots(
        plate,
        wavelengths,
        reference_wavelength_um=0.58756,
        field=FieldPoint.angle(x_deg=20),
        wavelength_weights=spectrum,
        sampling=PupilSampling(kind="rings", radial=5, azimuth=24),
    )
    theta = np.deg2rad(20)
    offsets = np.array(
        [5 * np.tan(np.arcsin(np.sin(theta) / glass.index(wl))) for wl in wavelengths]
    )
    offsets -= offsets[1]
    assert_allclose(spots.chief_offsets_um[:, 0], offsets * 1000, atol=1e-9)
    assert_allclose(spots.rms_radius_um, np.sqrt(0.5 + offsets**2) * 1000, atol=1e-8)
    assert_allclose(
        spots.polychromatic_rms_um, np.sqrt(0.5 + spectrum @ offsets**2) * 1000, atol=1e-8
    )
    variance = spectrum @ (offsets - spectrum @ offsets) ** 2
    assert_allclose(spots.centroid_rms_um, np.sqrt(0.5 + variance) * 1000, atol=1e-8)
    assert_allclose(spots.detected_spectral_weights, spectrum, atol=1e-14)


def test_04_exact_conic_foci_and_cartesian_optical_path():
    """Parabola/ellipse focus and Cartesian oval constant eikonal are analytic."""

    parabola = OpticalSystem([SurfaceRow.mirror(radius=-200, thickness=-100, conic=-1)])
    origins, directions = collimated_bundle(np.linspace(-30, 30, 31), start_z=-300)
    batch = SequentialTracer(parabola).trace_batch(origins, directions)
    assert batch.valid.all()
    assert_allclose(batch.image_points, np.tile([0, 0, -100], (31, 1)), atol=1e-11)
    assert np.ptp(batch.opl) < 1e-10
    ellipse = OpticalSystem([SurfaceRow.mirror(radius=-36, thickness=-180, conic=-0.64)])
    angles = np.linspace(-0.2, 0.2, 31)
    directions = np.column_stack([np.zeros(31), np.sin(angles), np.cos(angles)])
    batch = SequentialTracer(ellipse).trace_batch(np.tile([0, 0, -20], (31, 1)), directions)
    assert batch.valid.all()
    assert_allclose(batch.image_points, np.tile([0, 0, -180], (31, 1)), atol=2e-10)
    oval = SurfaceRow.cartesian_oval(
        n0=1, z0=-30, ni=1.7, zi=10, thickness=10, material=ConstantIndex("G", 1.7)
    )
    system = OpticalSystem([oval], object_z=-30)
    pupil = trace_pupil(SequentialTracer(system), FieldPoint(), na_object_sine=0.08, chief_slope=0)
    assert pupil.valid.all()
    assert np.max(np.linalg.norm(pupil.image_points[:, :2], axis=1)) < 1e-9
    assert np.ptp(pupil.opl) < 1e-9


def test_05_apertures_quadrature_and_geometric_throughput():
    """Apertures preserve media; quadrature integrates known disk moments."""

    system = OpticalSystem(
        [SurfaceRow.stop(thickness=10, semidiameter=1)], object_space=GLASS, object_z=-10
    )
    batch = SequentialTracer(system).trace_batch(*collimated_bundle([0.5, 1.5], start_z=-1))
    assert_allclose(system.n_after, [1.5])
    assert list(batch.status) == [TraceStatus.OK, TraceStatus.VIGNETTED]
    for kind in ("rings", "gauss"):
        sample = PupilSampling(kind=kind, radial=10, azimuth=48)
        points = sample.points()
        weights = sample.area_weights(points)
        assert_allclose(weights.sum(), 1, atol=1e-14)
        assert_allclose(weights @ np.sum(points**2, axis=1), 0.5, atol=2e-14)
        assert_allclose(weights @ points, [0, 0], atol=1e-14)
        if kind == "gauss":
            assert_allclose(weights @ np.sum(points**2, axis=1) ** 2, 1 / 3, atol=1e-14)
    annulus = CircularAperture(2, inner_radius=1)
    assert np.array_equal(annulus.contains([[0, 0], [1.5, 0], [2.5, 0]]), [False, True, False])
    annular_system = OpticalSystem([SurfaceRow.stop(thickness=5, aperture=annulus)])
    annular_pupil = trace_pupil(
        SequentialTracer(annular_system), FieldPoint.angle(), keep_paths=True
    )
    radii_squared = np.sum(annular_pupil.batch.paths[:, 1, :2] ** 2, axis=1)
    assert annular_pupil.valid.all()
    assert_allclose(annular_pupil.weights @ radii_squared, (1**2 + 2**2) / 2, atol=1e-13)
    assert annular_pupil.geometric_throughput == pytest.approx(1)
    rectangle = RectangularAperture(2, 1)
    assert np.array_equal(rectangle.contains([[1.9, 0.9], [0, 1.1]]), [True, False])
    # A subsequent half-plane-equivalent rectangle clips one half of a disk.
    clipped = SurfaceRow.stop(thickness=2)
    clipped.aperture = RectangularAperture(1, 2)
    clipped.placement = RigidTransform.from_euler_xyz(origin=[1, 0, 0])
    system = OpticalSystem([SurfaceRow.stop(thickness=1, semidiameter=1), clipped], object_z=-10)
    pupil = trace_pupil(
        SequentialTracer(system),
        FieldPoint.angle(),
        sampling=PupilSampling(kind="gauss", azimuth=50),
    )
    assert_allclose(pupil.geometric_throughput, 0.5, atol=1e-14)
    stop = to_branching_surfaces(OpticalSystem([SurfaceRow.stop(semidiameter=1)]))[0]
    assert stop.hit(Ray(np.array([-1.0, 0.5]), np.array([1.0, 0.0]))) is None
    assert stop.hit(Ray(np.array([-1.0, 2.0]), np.array([1.0, 0.0]))) is not None


def test_06_three_dimensional_covariance_and_folded_mirror():
    """Rigid placement preserves the trace; a tilted flat mirror folds by 90°."""

    system = singlet()
    origins, directions = collimated_bundle(np.linspace(-5, 5, 15), start_z=-100)
    original = SequentialTracer(system).trace_batch(origins, directions, keep_paths=True)
    moved = copy.deepcopy(system)
    transform = RigidTransform.from_euler_xyz([10, -20, 30], [12, 23, 34])
    moved.frame = transform
    result = SequentialTracer(moved).trace_batch(
        transform.to_world(origins), transform.direction_to_world(directions), keep_paths=True
    )
    assert np.array_equal(original.status, result.status)
    assert_allclose(result.paths, transform.to_world(original.paths), atol=2e-11)
    assert_allclose(
        result.directions, transform.direction_to_world(original.directions), atol=2e-13
    )
    assert_allclose(result.opl, original.opl, atol=2e-11)
    field = FieldPoint(x=2, y=3)
    before = parabasal_focus(SequentialTracer(system), field)
    after = parabasal_focus(SequentialTracer(moved), field)
    assert_allclose(
        [before.tangential_shift_mm, before.sagittal_shift_mm],
        [after.tangential_shift_mm, after.sagittal_shift_mm],
        atol=2e-7,
        rtol=0,
    )
    assert_allclose(
        distortion_map(SequentialTracer(system), [field]).chief_xy_mm,
        distortion_map(SequentialTracer(moved), [field]).chief_xy_mm,
        atol=2e-11,
    )
    assert_allclose(
        seidel_coefficients(system, FieldPoint(y=3)).sums_mm,
        seidel_coefficients(moved, FieldPoint(y=3)).sums_mm,
        atol=1e-15,
    )
    # Move a complete two-surface element while preserving its internal geometry.
    grouped = copy.deepcopy(system)
    grouped.transform_surfaces([0, 1], transform)
    for i in range(2):
        assert_allclose(
            grouped.surface_frame(i).origin,
            transform.to_world(system.surface_frame(i).origin),
            atol=1e-13,
        )
    with pytest.raises(ValueError, match="centered"):
        ParaxialModel(grouped)
    with pytest.raises(ValueError, match="centered"):
        grouped.require_axial_coordinates()
    mirror = SurfaceRow.mirror(radius=0, thickness=0)
    mirror.placement = RigidTransform.from_euler_xyz(angles_deg=[0, 45, 0])
    detector = RigidTransform.from_euler_xyz(origin=[-10, 0, 0], angles_deg=[0, 90, 0])
    folded = OpticalSystem([mirror], image_placement=detector)
    ray = SequentialTracer(folded).trace([0, 0, -10], [0, 0, 1])
    assert ray.ok
    assert_allclose(ray.image_point, [-10, 0, 0], atol=1e-12)
    assert_allclose(ray.direction, [-1, 0, 0], atol=1e-14)
    assert_allclose(ray.opl, 20, atol=1e-12)

    # A flat mirror tilt alpha sends its centroid to -L*tan(2*alpha).
    plane = OpticalSystem(
        [SurfaceRow.mirror(radius=0, thickness=-100, semidiameter=2, is_stop=True)]
    )
    evaluator = GeometricEvaluator(
        SequentialTracer(plane),
        [FieldPoint.angle()],
        [0.55],
        sampling=PupilSampling(kind="gauss", radial=3, azimuth=12),
    )
    parameter = Perturbation("tilt_y", (0,), label="mirror tilt")
    derivative = sensitivity(evaluator, parameter, 0.001)
    key = "field_0.centroid_x_mm"
    assert_allclose(derivative.derivative[key], -200 * np.pi / 180, rtol=2e-9)
    assert derivative.derivative_step_difference[key] < 1e-8
    assert not derivative.changed_ray_membership
    assert_allclose(plane.rows[0].placement.rotation, np.eye(3), atol=0)
    tolerance = Tolerance(parameter, 0.01)
    draws = monte_carlo(evaluator, [tolerance], samples=12, seed=41)
    expected = -100 * np.tan(2 * np.deg2rad(draws.draws[:, 0]))
    assert_allclose(draws.metrics[key], expected, atol=1e-11)
    repeated = monte_carlo(evaluator, [tolerance], samples=12, seed=41)
    assert np.array_equal(draws.draws, repeated.draws)
    assert draws.successful_fraction == 1
    # Complete beam loss remains a recorded failure, never a zero-RMS success.
    losses = monte_carlo(
        evaluator,
        [Tolerance(Perturbation("decenter_x", (0,)), 100)],
        samples=8,
        seed=41,
    )
    assert losses.failures
    assert np.isnan(losses.metrics[key][list(losses.failures)]).all()
    # A changed last thickness must not silently translate the fixed detector.
    changed = Perturbation("thickness", (0,)).apply(plane, 10)
    assert_allclose(
        evaluator.evaluate(changed).fields[0].rms_radius_mm,
        evaluator.nominal.fields[0].rms_radius_mm,
        atol=1e-12,
    )


def test_07_finite_infinite_aiming_and_cone_symmetry():
    """Every ray reaches its prescribed stop coordinate, not just the chief."""

    system = singlet()
    sample = PupilSampling(kind="gauss", radial=6, azimuth=20)
    for field in (FieldPoint(x=1, y=2), FieldPoint.angle(x_deg=2, y_deg=3)):
        pupil = trace_pupil(SequentialTracer(system), field, sampling=sample, keep_paths=True)
        assert pupil.valid.all()
        assert np.max(pupil.aiming_residual_mm) < 1e-9
        stop_points = system.surface_frame(0).to_local(pupil.batch.paths[:, 1])[:, :2]
        assert_allclose(stop_points, pupil.pupil_uv * 8, atol=1e-9)
    stop = OpticalSystem([SurfaceRow.stop(thickness=10, semidiameter=20)], object_z=-1)
    pupil = trace_pupil(
        SequentialTracer(stop),
        FieldPoint(),
        na_object_sine=0.5,
        slope_model="sine",
        chief_slope=0,
        sampling=PupilSampling(radial=2, azimuth=4),
    )
    assert_allclose(np.linalg.norm(pupil.directions[1:, :2], axis=1), 0.5, atol=2e-15)
    assert_allclose(pupil.directions[1:, 2], np.sqrt(0.75), atol=2e-15)


def test_08_geometric_spherical_aberration_and_best_focus():
    """A spherical mirror has LSA~h²/(4R), transverse error~-h³/(2R²)."""

    radius = 100.0
    system = OpticalSystem([SurfaceRow.mirror(radius=-radius, thickness=-radius / 2)])
    heights = np.array([0.5, 1.0, 2.0, 4.0])
    origins, directions = collimated_bundle(heights)
    batch = SequentialTracer(system).trace_batch(origins, directions)
    assert batch.valid.all()
    intercepts, miss = axial_intercepts(batch.image_points, batch.directions)
    longitudinal = intercepts + radius / 2
    transverse = batch.image_points[:, 1]
    assert_allclose(miss, 0, atol=1e-13)
    assert_allclose(longitudinal[:2] / heights[:2] ** 2, 1 / (4 * radius), rtol=1e-4)
    assert_allclose(transverse[:2] / heights[:2] ** 3, -1 / (2 * radius**2), rtol=3e-4)
    assert_allclose(longitudinal[1:] / longitudinal[:-1], 4, rtol=0.002)
    assert_allclose(transverse[1:] / transverse[:-1], 8, rtol=0.004)
    system.rows[0].semidiameter = 8
    system.rows[0].is_stop = True
    pupil = trace_pupil(SequentialTracer(system), FieldPoint.angle())
    report = geometric_aberrations(pupil, image_frame=system.image_frame)
    assert report.best_focus_rms_mm < report.rms_radius_mm
    assert report.best_focus_shift_mm > 0
    assert report.geometric_throughput == pytest.approx(1)

    # Check analytic refocus against an actual trace to the displaced detector.
    sampling = PupilSampling(kind="gauss", radial=5, azimuth=16)
    evaluator = GeometricEvaluator(
        SequentialTracer(system),
        [FieldPoint.angle()],
        [0.55],
        sampling=sampling,
        focus_policy="refocus",
    )
    result = evaluator.nominal.fields[0]
    focused = copy.deepcopy(system)
    focused.image_placement = RigidTransform.from_euler_xyz(origin=[0, 0, result.focus_shift_mm])
    focused_pupil = trace_pupil(SequentialTracer(focused), FieldPoint.angle(), sampling=sampling)
    measured = geometric_aberrations(focused_pupil, image_frame=focused.image_frame)
    assert_allclose(result.rms_radius_mm, measured.rms_radius_mm, atol=2e-12)
    with pytest.raises(RuntimeError, match="parallel"):
        GeometricEvaluator(
            SequentialTracer(
                OpticalSystem(
                    [SurfaceRow.mirror(radius=0, thickness=-100, semidiameter=2, is_stop=True)]
                )
            ),
            [FieldPoint.angle()],
            [0.55],
            sampling=sampling,
            focus_policy="refocus",
        )

    # Exact Coddington result for a spherical mirror, chief at its vertex:
    # z_T = -R*cos(theta)^2/2, z_S = -R/2.
    for angle in (0.0, 3.0, 10.0):
        field = FieldPoint.angle(y_deg=angle)
        focus = parabasal_focus(SequentialTracer(system), field)
        assert_allclose(
            focus.tangential_shift_mm,
            radius / 2 * np.sin(np.deg2rad(angle)) ** 2,
            atol=3e-9,
            rtol=0,
        )
        assert_allclose(focus.sagittal_shift_mm, 0, atol=3e-9)
        assert np.max(focus.step_difference_mm) < 1e-8
    theta = np.deg2rad(10.0)
    field = FieldPoint.angle(y_deg=10.0)
    rectilinear = distortion_map(SequentialTracer(system), [FieldPoint.angle(), field])
    assert_allclose(rectilinear.deviation_mm, 0, atol=5e-12)
    assert np.isnan(rectilinear.radial_percent[0])
    angular = distortion_map(SequentialTracer(system), [field], reference="f_theta")
    assert_allclose(angular.deviation_mm[0, 1], radius / 2 * (np.tan(theta) - theta), atol=3e-12)
    # Stop obstruction changes physical transmission, not geometric reference.
    obscured = copy.deepcopy(system)
    obscured.rows[0].aperture = CircularAperture(8.0, inner_radius=1.0)
    assert not parabasal_focus(SequentialTracer(obscured), field).chief_transmitted
    assert_allclose(
        distortion_map(SequentialTracer(obscured), [field]).chief_xy_mm,
        rectilinear.chief_xy_mm[1:],
        atol=2e-12,
    )


def test_09_prescription_roundtrip_preserves_the_complete_system(tmp_path):
    """Reopening a project preserves geometry, materials, frames and ray results."""

    system = singlet(material=sellmeier_glass("N-BK7"), wavelength_um=0.55)
    system.rows[0].profile = AsphereProfile.from_radius(50, -0.2, (1e-8, 0, 0, 0, 0, 0, 1e-24))
    system.rows[0].semidiameter = 1.23456789123
    system.rows[0].aperture = CircularAperture(4, 0.2)
    system.rows[1].placement = RigidTransform.from_euler_xyz([0.01, 0.02, 0.03], [0.1, 0.2, 0.3])
    system.frame = RigidTransform.from_euler_xyz([1, 2, 3], [4, 5, 6])
    path = tmp_path / "system.json"
    system.to_prescription(path, fmt="json")
    restored = OpticalSystem.from_prescription(path, fmt="json")
    second = tmp_path / "second.json"
    restored.to_prescription(second, fmt="json")
    assert json.loads(path.read_text()) == json.loads(second.read_text())
    origins, directions = collimated_bundle([0.5, 1, 2], start_z=-100)
    origins = system.frame.to_world(origins)
    directions = system.frame.direction_to_world(directions)
    before = SequentialTracer(system).trace_batch(origins, directions, keep_paths=True)
    after = SequentialTracer(restored).trace_batch(origins, directions, keep_paths=True)
    assert before.valid.all()
    assert_allclose(after.paths, before.paths, atol=1e-12)
    assert_allclose(after.opl, before.opl, atol=1e-12)

    from raytracer.io import read_materials, write_analysis
    from raytracer.optics.materials import IndexOffsetMaterial

    catalog = tmp_path / "materials.csv"
    catalog.write_text(
        "name,model,c1,c2,source,wavelength_min_um,wavelength_max_um\n"
        "Measured,cauchy,1.5,.005,independent calibration,.4,.7\n",
        encoding="utf-8",
    )
    measured = read_materials(catalog)["MEASURED"]
    assert measured.metadata.source == "independent calibration"
    assert measured.metadata.wavelength_range_um == (0.4, 0.7)
    with pytest.raises(ValueError, match="outside supported"):
        measured.index(0.8)
    system.rows[0].material_after = IndexOffsetMaterial("offset", measured, 1e-4)
    system.to_prescription(path, fmt="json")
    restored = OpticalSystem.from_prescription(path, fmt="json")
    assert_allclose(restored.rows[0].material_after.index(0.55), measured.index(0.55) + 1e-4)
    assert restored.rows[0].material_after.base.metadata == measured.metadata
    record = write_analysis(
        tmp_path / "analysis.json",
        system=system,
        settings={"focus_policy": "fixed"},
        results={"not_defined": np.nan, "known": np.array([1.0, 2.0])},
    )
    reopened = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    assert reopened["results"]["not_defined"] is None
    assert reopened["system_sha256"] == record["system_sha256"]
    assert (
        reopened["system"]["surfaces"][0]["material_after"]["parameters"]["base"]["parameters"][
            "metadata"
        ]["source"]
        == "independent calibration"
    )
    oval = OpticalSystem(
        [
            SurfaceRow.cartesian_oval(
                n0=1, z0=-30, ni=1.7, zi=10, thickness=10, material=ConstantIndex("G", 1.7)
            )
        ],
        object_z=-30,
    )
    oval.to_prescription(path, fmt="json")
    restored = OpticalSystem.from_prescription(path, fmt="json")
    assert restored.rows[0].profile == oval.rows[0].profile
    with pytest.raises(ValueError, match="json"):
        oval.to_prescription(tmp_path / "unsupported.csv")
    payload = json.loads(path.read_text())
    payload["version"] = 100
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="version"):
        OpticalSystem.from_prescription(path, fmt="json")


def test_10_independent_rayoptics_cooke_reference():
    """Compare every surface point, final direction and OPL for 45 external rays."""

    prescription = ROOT / "data/optical_systems/photographic/cooke_triplet_prescription.csv"
    reference = json.loads((Path(__file__).parent / "reference/rayoptics_cooke.json").read_text())
    assert reference["version"] == "0.9.8"
    assert (
        hashlib.sha256(prescription.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        == reference["prescription_sha256"]
    )
    assert len(reference["cases"]) == 45
    for case in reference["cases"]:
        system = OpticalSystem.from_prescription(prescription)
        for row, n in zip(system.rows, case["indices_after"]):
            row.material_after = ConstantIndex("reference", n)
        tracer = SequentialTracer(system, clip_apertures=False, allow_virtual_segments=True)
        ray = tracer.trace(case["origin_mm"], case["direction"])
        assert ray.ok, case
        assert_allclose(
            ray.path, case["paths_mm"], atol=1e-9, rtol=0, err_msg=str(case["origin_mm"])
        )
        assert_allclose(ray.direction, case["outgoing_direction"], atol=1e-11, rtol=0)
        assert_allclose(ray.opl, case["opl_mm"], atol=1e-9, rtol=0)

    reference = json.loads((ROOT / "tests/reference/rayoptics_third_order.json").read_text())
    assert reference["version"] == "0.9.8"
    pupil = np.array([[0.0, 0.7], [0.4, 0.2], [0.0, 0.0], [0.5, -0.5]])
    for case in reference["cases"]:
        system, field = reference_third_order_system(case)
        tracer = SequentialTracer(system, clip_apertures=False)
        ledger = seidel_coefficients(system, field)
        assert_allclose(
            ledger.surface_sums_mm,
            case["surface_sums_mm"],
            atol=2e-14,
            rtol=0,
            err_msg=case["name"],
        )
        assert_allclose(ledger.sums_mm, case["sums_mm"], atol=2e-14, rtol=0)
        focus = parabasal_focus(tracer, field)
        assert_allclose(
            [focus.tangential_shift_mm, focus.sagittal_shift_mm],
            case["parabasal_shifts_mm"],
            atol=2e-8,
            rtol=0,
        )
        distortion = distortion_map(tracer, [field])
        assert_allclose(distortion.chief_xy_mm[0], case["chief_image_xy_mm"], atol=2e-10, rtol=0)
        assert_allclose(ledger.gaussian_image_height_mm, case["gaussian_height_mm"], atol=2e-12)
        # Correct cubic ray theory leaves a fifth-order remainder when pupil
        # and field are both scaled. This also exercises the S-V contribution.
        errors = []
        focus_errors = []
        for scale in (0.5, 0.25, 0.125):
            scaled = (
                (FieldPoint.angle(y_deg=np.rad2deg(np.arctan(scale * np.tan(np.deg2rad(field.y))))))
                if field.kind == "angle"
                else FieldPoint(y=field.y * scale)
            )
            origins, directions, residual = aim_stop_targets(
                tracer, scaled, pupil * ledger.pupil_radius_mm * scale
            )
            assert residual.max() < 1e-11
            rays = tracer.trace_batch(origins, directions)
            assert rays.valid.all()
            exact = rays.image_points[:, :2] - [0, scale * ledger.gaussian_image_height_mm]
            errors.append(np.max(abs(exact - ledger.transverse(pupil) * scale**3)))
            measured = parabasal_focus(tracer, scaled, step_mm=5e-4)
            curvatures = ledger.field_curvatures_per_mm
            predicted_focus = (
                0.5
                * np.array([curvatures["tangential"], curvatures["sagittal"]])
                * (scale * ledger.gaussian_image_height_mm) ** 2
            )
            focus_errors.append(
                np.max(
                    abs(
                        np.array([measured.tangential_shift_mm, measured.sagittal_shift_mm])
                        - predicted_focus
                    )
                )
            )
        ratios = np.array(errors[:-1]) / errors[1:]
        assert_allclose(ratios, 32.0, rtol=0.015, err_msg=case["name"])
        assert_allclose(
            np.array(focus_errors[:-1]) / focus_errors[1:], 16.0, rtol=0.05, err_msg=case["name"]
        )
        # Marginal/field normalization and the Lagrange invariant are explicit.
        assert_allclose(ledger.marginal_y_mm[case["stop"]], case["radius"], atol=1e-13)
        assert_allclose(ledger.chief_y_mm[case["stop"]], 0, atol=1e-13)
        invariant = ledger.signed_indices_after * (
            ledger.marginal_y_mm * ledger.chief_slope_after
            - ledger.chief_y_mm * ledger.marginal_slope_after
        )
        assert_allclose(invariant, ledger.invariant_mm, atol=1e-13)


def reference_third_order_system(case):
    """Reconstruct only the explicit inputs of the independent reference."""
    rows = []
    for index, record in enumerate(case["rows"]):
        options = dict(
            radius=record["radius"],
            thickness=record["thickness"],
            conic=record.get("conic", 0.0),
            coefficients=(record.get("a4", 0.0),),
            semidiameter=case["radius"] if index == case["stop"] else 20.0,
            is_stop=index == case["stop"],
        )
        row = (
            SurfaceRow.mirror(**options)
            if record.get("mirror")
            else SurfaceRow.refracting(
                **options, material=ConstantIndex("reference", record["index"])
            )
        )
        rows.append(row)
    system = OpticalSystem(rows, object_z=-np.inf if case["object_z"] is None else case["object_z"])
    system.set_thickness(len(rows) - 1, case["gaussian_z_mm"] - system.vertices[-1])
    field = (
        FieldPoint.angle(y_deg=case["field"])
        if case["object_z"] is None
        else FieldPoint(y=case["field"])
    )
    return system, field
