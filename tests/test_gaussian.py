"""First-order optics against closed forms: a thick lens, a concave mirror, Gaussian imaging, the pupils."""

from pathlib import Path

import numpy as np
import pytest

import raytracer as rt
from raytracer.analysis import first_order
from raytracer.elements import Lens, Mirror, Sensor, Stop
from raytracer.geometry.surfaces import Sphere
from raytracer.io import read_prescription
from raytracer.materials import ConstantIndex
from raytracer.system import Gap
from raytracer.tracing import SequentialTracer
from raytracer.tracing.paraxial import Paraxial

COOKE = Path(__file__).resolve().parents[1] / "data/optical_systems/photographic/cooke_triplet_prescription.csv"
N, R1, R2, D = 1.5, 50.0, -50.0, 5.0
# The thick lens in air: focal length, and the principal planes from each vertex.
F = 1.0 / ((N - 1) * (1 / R1 - 1 / R2 + (N - 1) * D / (N * R1 * R2)))
H_FRONT = -F * (N - 1) * D / (N * R2)
H_BACK = -F * (N - 1) * D / (N * R1)


def conditions(**changes) -> rt.Conditions:
    base = rt.Conditions(object=rt.Object.at_infinity(), aperture=rt.SystemAperture.f_number(5.0),
                         fields=rt.Fields.angles([0.0, 5.0]), spectrum=rt.Spectrum.single(0.55))
    return base.replace(**changes)


def thick_lens(stop_z: float = 0.0, lens_z: float = 1.0) -> rt.OpticalSystem:
    lens = Lens("L", surfaces=[Sphere(R1), Sphere(R2)], thicknesses=[D], materials=[ConstantIndex("G", N)],
                semi_diameter=15.0, z=lens_z)
    return rt.OpticalSystem([Stop("S", aperture=5.0, z=stop_z), lens, Sensor("IMG", z=100.0)])


def gaussian_image(object_z: float) -> tuple[float, float]:
    """Image position and magnification from the principal planes: 1/s' - 1/s = 1/f."""

    s = object_z - (1.0 + H_FRONT)
    s_image = 1.0 / (1.0 / F + 1.0 / s)
    return 1.0 + D + H_BACK + s_image, s_image / s


def test_a_thick_lens_in_air():
    fo = first_order(thick_lens().under(conditions()))
    assert fo.efl_mm == pytest.approx(F, rel=1e-12)
    assert fo.bfl_mm == pytest.approx(F + H_BACK, rel=1e-12)  # from the back vertex, past the stop at z = 0
    assert fo.ffl_mm == pytest.approx(-(F - H_FRONT), rel=1e-12)
    np.testing.assert_allclose(fo.principal_planes_z_mm, [1.0 + H_FRONT, 1.0 + D + H_BACK], rtol=1e-12)
    np.testing.assert_allclose(fo.nodal_points_z_mm, fo.principal_planes_z_mm, rtol=1e-12)  # air on both sides
    # The stop stands alone in front: it is the entrance pupil, and the lens images it into the exit pupil.
    assert fo.entrance_pupil.z_mm == pytest.approx(0.0, abs=1e-12)
    assert fo.entrance_pupil.diameter_mm == pytest.approx(F / 5.0, rel=1e-12)
    assert fo.f_number == pytest.approx(5.0, rel=1e-12) and fo.working_f_number == pytest.approx(5.0, rel=1e-12)
    xp_z, xp_m = gaussian_image(0.0)
    assert fo.exit_pupil.z_mm == pytest.approx(xp_z, rel=1e-12)
    assert fo.exit_pupil.diameter_mm == pytest.approx(F / 5.0 * abs(xp_m), rel=1e-12)
    assert fo.paraxial_image_z_mm == pytest.approx(1.0 + D + F + H_BACK, rel=1e-12)
    assert fo.paraxial_image_height_mm == pytest.approx(F * np.tan(np.radians(5.0)), rel=1e-12)
    assert fo.image_defocus_mm == pytest.approx(100.0 - fo.paraxial_image_z_mm, rel=1e-12)


def test_gaussian_imaging_at_finite_conjugates():
    system = thick_lens()
    near = conditions(object=rt.Object.finite(150.0), aperture=rt.SystemAperture.object_na(0.05),
                      fields=rt.Fields.object_heights([0.0, 4.0]))
    fo = first_order(system.under(near))
    image_z, m = gaussian_image(-150.0)
    assert fo.paraxial_image_z_mm == pytest.approx(image_z, rel=1e-12)
    assert fo.magnification == pytest.approx(m, rel=1e-12)
    assert fo.paraxial_image_height_mm == pytest.approx(4.0 * m, rel=1e-12)
    assert fo.object_na == pytest.approx(0.05, rel=1e-12)
    assert fo.image_na == pytest.approx(0.05 / abs(m), rel=1e-12)  # the Lagrange invariant, in air
    by_image = first_order(system.under(near.replace(aperture=rt.SystemAperture.image_na(0.1))))
    assert by_image.image_na == pytest.approx(0.1, rel=1e-12) and by_image.working_f_number == pytest.approx(5.0)
    by_stop = first_order(system.under(near.replace(aperture=rt.SystemAperture.from_stop())))
    assert by_stop.stop_radius_mm == 5.0 and by_stop.entrance_pupil.diameter_mm == pytest.approx(10.0)


def test_a_concave_mirror():
    mirror = Mirror("M", surface=Sphere(-100.0), aperture=20.0)
    system = rt.OpticalSystem([Stop("S", aperture=5.0, z=-20.0), mirror, Sensor("IMG", z=-50.0)],
                              path=["S", "M:reflect", "IMG"])
    fo = first_order(system.under(conditions()))
    assert fo.efl_mm == pytest.approx(50.0, rel=1e-12) and fo.bfl_mm == pytest.approx(50.0, rel=1e-12)
    np.testing.assert_allclose(fo.focal_points_z_mm, [-50.0, -50.0], rtol=1e-12)
    np.testing.assert_allclose(fo.principal_planes_z_mm, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(fo.nodal_points_z_mm, [-100.0, -100.0], rtol=1e-12)  # the centre of curvature
    assert fo.paraxial_image_z_mm == pytest.approx(-50.0, rel=1e-12) and fo.image_defocus_mm == pytest.approx(0.0, abs=1e-12)


def test_the_lagrange_invariant_is_the_same_at_every_step():
    p = Paraxial(read_prescription(COOKE).system.under(conditions(fields=rt.Fields.angles([0.0, 20.0]))))
    m, c = p.marginal, p.chief
    invariant = p.n_out * (c.y * m.u - m.y * c.u)
    np.testing.assert_allclose(invariant, invariant[0], rtol=1e-12)
    assert abs(invariant[0]) == pytest.approx(p.entrance_pupil.diameter_mm / 2 * np.tan(np.radians(20.0)), rel=1e-12)


def test_paraxial_and_exact_tracing_agree_on_the_focal_length():
    """A real ray parallel to the axis, very near it, leaves at the paraxial angle: -h / tan u' -> EFL."""

    cooke = read_prescription(COOKE)
    fo = first_order(cooke.system.under(conditions(spectrum=rt.Spectrum.single(cooke.wavelength_um))))
    h = 1e-4
    ray = SequentialTracer(cooke.system, wavelength_um=cooke.wavelength_um).trace(rt.Rays([0.0, h, -10.0], [0.0, 0.0, 1.0]))
    d = ray.direction[0]
    assert -h / (d[1] / d[2]) == pytest.approx(fo.efl_mm, rel=1e-7)


def test_a_telecentric_entrance_pupil_has_no_diameter_to_state():
    """The stop at the rear focal point: the entrance pupil is at infinity, so an F-number cannot size it."""

    stop_z = 1.0 + D + F + H_BACK
    system = rt.OpticalSystem([thick_lens()["L"], Stop("S", aperture=5.0, z=stop_z), Sensor("IMG", z=stop_z + 30.0)])
    with pytest.raises(ValueError, match="telecentric"):
        first_order(system.under(conditions()))
    near = conditions(object=rt.Object.finite(150.0), aperture=rt.SystemAperture.object_na(0.05),
                      fields=rt.Fields.object_heights([0.0, 4.0]))
    fo = first_order(system.under(near))
    assert np.isinf(fo.entrance_pupil.z_mm) and fo.object_na == pytest.approx(0.05)
    assert np.isnan(fo.f_number)  # an infinite pupil has no F-number
    with pytest.raises(ValueError, match="field angles mean nothing"):
        first_order(system.under(near.replace(fields=rt.Fields.angles([0.0, 5.0]))))
