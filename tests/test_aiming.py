"""Ray aiming and tracing a configuration: every ray crosses the stop where its pupil point says."""

from pathlib import Path

import numpy as np
import pytest

import raytracer as rt
from raytracer.elements import Lens, Sensor, Stop
from raytracer.geometry.surfaces import Sphere
from raytracer.io import read_prescription
from raytracer.materials import ConstantIndex
from raytracer.tracing import FieldPoint, SequentialTracer, aim, chief_ray_slopes, solve_object_plane
from raytracer.tracing.paraxial import Paraxial

DATA = Path(__file__).resolve().parents[1] / "data/optical_systems"
COOKE = DATA / "photographic/cooke_triplet_prescription.csv"
US7557996 = DATA / "lithography/US7557996_Fig3_Table3_prescription.csv"


def cooke(**changes) -> rt.Configuration:
    conditions = rt.Conditions(object=rt.Object.at_infinity(), aperture=rt.SystemAperture.f_number(4.5),
                               fields=rt.Fields.angles([0.0, 10.0, 14.0, 20.0]), spectrum=rt.Spectrum.fdc())
    return read_prescription(COOKE).system.under(conditions.replace(**changes))


def stop_crossings(configuration: rt.Configuration, rays: rt.Rays, wavelength_um: float) -> np.ndarray:
    traced = SequentialTracer(configuration.system, wavelength_um=wavelength_um, clip_apertures=False).trace(rays)
    return traced.trajectory[:, configuration.system.stop_index + 1, :2]


@pytest.mark.parametrize("field", [0, 1, 3])
@pytest.mark.parametrize("wavelength", [0, 1, 2])
def test_real_aiming_crosses_the_stop_where_the_pupil_point_says(field, wavelength):
    triplet = cooke()
    points = rt.pupil.hexapolar(5).points()
    w = triplet.conditions.spectrum.wavelengths[wavelength]
    rays = aim(triplet, field, points, w)
    radius = Paraxial(triplet).stop_radius
    crossed = stop_crossings(triplet, rays, w)
    reached = np.isfinite(crossed[:, 0])
    np.testing.assert_allclose(crossed[reached], points[reached] * radius, atol=1e-9)
    # Only light that cannot get there at all fails: at 20 degrees the bottom of the pupil would pass outside L1's glass.
    assert reached.all() if field < 3 else reached.mean() > 0.85
    # From infinity every ray of a field shares its direction, the field angle.
    angle = np.degrees(np.arctan2(rays.direction[:, 1], rays.direction[:, 2]))
    np.testing.assert_allclose(angle, triplet.conditions.fields.points[field][1], atol=1e-12)


def test_paraxial_aiming_misses_by_the_aberration_of_the_pupil():
    def miss(f_number, field):
        triplet = cooke(aperture=rt.SystemAperture.f_number(f_number), aiming=rt.Aiming.paraxial())
        points, radius = rt.pupil.hexapolar(3).points(), Paraxial(triplet).stop_radius
        w = triplet.conditions.spectrum.primary
        crossed = stop_crossings(triplet, aim(triplet, field, points, w), w)
        return np.nanmax(np.hypot(*(crossed - points * radius).T)) / radius

    # On axis the pupil's spherical aberration grows as the cube of the aperture: it fades as the stop closes.
    assert miss(64.0, 0) < miss(4.5, 0) * (4.5 / 64.0) ** 2 * 1.1
    # Off axis even the chief ray misses the stop's centre, by an amount the aperture does not shrink.
    assert miss(4.5, 3) > 0.1 and miss(64.0, 3) > miss(4.5, 3)


def test_the_chief_ray_agrees_with_the_chief_ray_solver_of_the_finite_conjugate_objective():
    """US 7,557,996: object heights, an object-space NA, a telecentric entrance pupil, a virtual stop."""

    prescription = read_prescription(US7557996)
    tracer = prescription.tracer()
    solve_object_plane(tracer)
    system = prescription.system
    conditions = rt.Conditions(
        object=rt.Object.finite(system.path[0].face.pose.position[2] - tracer.object_z),
        aperture=rt.SystemAperture.object_na(0.3),
        fields=rt.Fields.object_heights([56.0, 62.0, 67.0]),
        spectrum=rt.Spectrum.single(prescription.wavelength_um),
    )
    duv = system.under(conditions)
    assert abs(Paraxial(duv).entrance_pupil.z_mm) > 1e4  # all but telecentric in object space
    for k, height in enumerate([56.0, 62.0, 67.0]):
        chief = aim(duv, k, [[0.0, 0.0]], prescription.wavelength_um)
        slopes = chief.direction[0, :2] / chief.direction[0, 2]
        np.testing.assert_allclose(slopes, chief_ray_slopes(tracer, FieldPoint(y=height)), atol=1e-9)


def test_rays_from_a_virtual_object_count_their_path_from_it():
    lens = Lens("L", surfaces=[Sphere(40.0), Sphere(-40.0)], thicknesses=[6.0], materials=[ConstantIndex("G", 1.5)],
                semi_diameter=12.0)
    system = rt.OpticalSystem([Stop("S", aperture=4.0, z=-2.0), lens.placed(rt.Pose((0.0, 0.0, 0.0))), Sensor("IMG", z=30.0)])
    conditions = rt.Conditions(object=rt.Object.finite(-20.0), aperture=rt.SystemAperture.from_stop(),
                               fields=rt.Fields.object_heights([0.0, 2.0]), spectrum=rt.Spectrum.single(0.55))
    behind = system.under(conditions)
    rays = aim(behind, 1, rt.pupil.hexapolar(2).points(), 0.55)
    virtual = np.array([0.0, 2.0, -2.0 + 20.0])  # 20 mm behind the first vertex, the stop at z = -2
    start, d = rays.position, rays.direction
    assert (start[:, 2] < -2.0).all()  # every ray starts in front of the system
    np.testing.assert_allclose(np.cross(virtual - start, d), 0.0, atol=1e-12)  # on its line back from the object
    np.testing.assert_allclose(rays.opl, -np.linalg.norm(virtual - start, axis=1), rtol=1e-12)


def test_image_heights_are_reached_by_the_real_chief_ray():
    triplet = cooke(fields=rt.Fields.image_heights([0.0, 10.0, 17.0]))
    bundle = rt.trace(triplet, field=[1, 2], pupil=rt.Pupil(), wavelength="primary")
    image = triplet.system.path[-1].face.pose.to_local(bundle.rays.position)
    np.testing.assert_allclose(image[:, 1], [10.0, 17.0], atol=1e-9)
    paraxial = cooke(fields=rt.Fields.image_heights([0.0, 10.0, 17.0], real=False))
    landed = rt.trace(paraxial, field=2, pupil=rt.Pupil(), wavelength="primary").rays.position[0, 1]
    assert landed != pytest.approx(17.0, abs=1e-3)  # paraxially stated, the real ray lands off by the distortion
    assert landed == pytest.approx(17.0, rel=0.02)


def test_tracing_a_configuration_gives_a_bundle():
    triplet = cooke()
    bundle = rt.trace(triplet, pupil=rt.pupil.hexapolar(3))
    per = 37  # 1 + 3 * 3 * 4 points
    assert len(bundle.rays) == 4 * 3 * per and bundle.field.shape == (4 * 3 * per, 2) and bundle.pupil.shape == bundle.field.shape
    assert set(bundle.rays.wavelength_um) == set(triplet.conditions.spectrum.wavelengths)
    np.testing.assert_allclose(np.unique(bundle.field[:, 1]), [0.0, 10.0, 14.0, 20.0])
    for f in (0.0, 20.0):  # each field's rays carry, between them, the whole of its light
        assert bundle.rays.power[bundle.field[:, 1] == f].sum() == pytest.approx(1.0)
    on_axis = bundle.rays[(bundle.field[:, 1] == 0.0)]
    assert on_axis.detected.all()  # nothing vignettes on axis
    one = rt.trace(triplet, field=rt.Field.normalized(hy=0.5), pupil=rt.Pupil(0.0, 1.0), wavelength=1)
    assert len(one.rays) == 1 and one.field[0].tolist() == [0.0, 10.0] and one.rays.wavelength_um[0] == pytest.approx(0.5875618)
