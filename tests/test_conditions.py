"""Conditions of use and configurations: values, and contradictions rejected where they are built."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import raytracer as rt
from raytracer.elements import Lens, Sensor, Stop
from raytracer.geometry.apertures import Rectangle
from raytracer.geometry.surfaces import Plane, Sphere
from raytracer.io import read_prescription
from raytracer.materials import ConstantIndex
from raytracer.system import Gap

COOKE = Path(__file__).resolve().parents[1] / "data/optical_systems/photographic/cooke_triplet_prescription.csv"
GLASS = ConstantIndex("G", 1.5)


def day(**changes) -> rt.Conditions:
    conditions = rt.Conditions(
        object=rt.Object.at_infinity(),
        aperture=rt.SystemAperture.f_number(4.5),
        fields=rt.Fields.angles([0.0, 10.0, 14.0, 20.0], weights=[1, 1, 1, 0.5]),
        spectrum=rt.Spectrum.fdc(),
    )
    return conditions.replace(**changes)


def lens_system(stop=True) -> rt.OpticalSystem:
    lens = Lens("L", surfaces=[Sphere(50.0), Sphere(-50.0)], thicknesses=[5.0], materials=[GLASS], semi_diameter=12.0)
    parts = [Stop("S", aperture=5.0), Gap(1.0), lens, Gap(48.0), Sensor("IMG")] if stop else [lens, Gap(48.0), Sensor("IMG")]
    return rt.OpticalSystem.compose(*parts)


def test_conditions_and_configurations_are_values():
    assert day() == day() and hash(day()) == hash(day())
    with pytest.raises(dataclasses.FrozenInstanceError):
        day().spectrum = rt.Spectrum.single(0.55)
    system = read_prescription(COOKE).system
    triplet = system.under(day())
    assert triplet == system.under(day()) and hash(triplet) == hash(system.under(day()))
    with pytest.raises(AttributeError):
        triplet.system = system
    narrower = triplet.replace(conditions=day(aperture=rt.SystemAperture.f_number(8.0)))
    assert narrower != triplet and narrower.system is system
    assert not any(hasattr(system, name) for name in ("wavelength_um", "fields", "spectrum", "object_z"))  # law 2


def test_the_spectrum_names_its_primary():
    fdc = rt.Spectrum.fdc()
    assert fdc.primary == pytest.approx(0.5875618) and len(fdc) == 3
    assert fdc.select("all") == fdc.wavelengths and fdc.select("primary") == (fdc.primary,)
    assert fdc.select(0) == (fdc.wavelengths[0],) and fdc.select(0.55) == (0.55,)
    assert rt.Spectrum.single(0.193).primary == 0.193
    with pytest.raises(ValueError, match="primary"):
        rt.Spectrum([0.48, 0.59])
    with pytest.raises(ValueError, match="not one of"):
        rt.Spectrum([0.48, 0.59], primary=0.55)


def test_fields_resolve_indices_and_normalized_points():
    fields = rt.Fields.angles([0.0, (3.0, 4.0), 10.0])
    assert fields.points == ((0.0, 0.0), (3.0, 4.0), (0.0, 10.0)) and fields.unit == "deg"
    assert fields.max_radius == 10.0
    assert fields.absolute(1) == (3.0, 4.0)
    assert fields.absolute(rt.Field(1.0, 2.0)) == (1.0, 2.0)
    assert fields.absolute(rt.Field.normalized(hy=0.5)) == (0.0, 5.0)
    with pytest.raises(ValueError, match="weights"):
        rt.Fields.angles([0.0, 10.0], weights=[1.0])
    with pytest.raises(ValueError, match="90"):
        rt.Fields.angles([95.0])


def test_contradictory_conditions_are_rejected_where_they_are_built():
    with pytest.raises(ValueError, match="object heights.*infinity"):
        day(fields=rt.Fields.object_heights([0.0, 5.0]))
    with pytest.raises(ValueError, match="object-space NA.*infinity"):
        day(aperture=rt.SystemAperture.object_na(0.1))
    with pytest.raises(ValueError, match="aperture stop"):
        lens_system(stop=False).under(day())
    blocked = rt.OpticalSystem.compose(Stop("S", aperture=Rectangle(4.0, 4.0)), Gap(1.0), lens_system(False)["L"], Gap(48.0), Sensor("IMG"))
    with pytest.raises(ValueError, match="circular opening"):
        blocked.under(day(aperture=rt.SystemAperture.from_stop()))

    class Infrared:  # a medium with no index in the visible
        name = "IR"

        def index(self, wavelength_um):
            if wavelength_um < 1.0:
                raise ValueError("outside the fit")
            return 2.4

    ir_lens = Lens("L", surfaces=[Sphere(50.0), Plane()], thicknesses=[5.0], materials=[Infrared()], semi_diameter=10.0)
    ir = rt.OpticalSystem.compose(Stop("S", aperture=5.0), Gap(1.0), ir_lens, Gap(50.0), Sensor("IMG"))
    with pytest.raises(ValueError, match="0.4861327 µm.*'IR'"):
        ir.under(day())
    ir.under(day(spectrum=rt.Spectrum.single(4.0)))  # fine where it has an index


def test_pupil_samplings():
    hexa = rt.pupil.hexapolar(rings=4)
    assert hexa.points().shape == (1 + 3 * 4 * 5, 2)
    assert hexa.weights().sum() == pytest.approx(1.0)
    radii = np.hypot(*hexa.points().T)
    assert radii.max() == pytest.approx(1.0)
    # every point inside the rim stands for the same area
    inner = (radii > 0) & (radii < 0.99)
    assert np.ptp(hexa.weights()[inner]) < 1e-15
    assert rt.pupil.polar(rings=3, arms=8).points().shape == (25, 2)
    fan = rt.pupil.fan("tangential", n=5).points()
    np.testing.assert_allclose(fan, [[0, -1], [0, -0.5], [0, 0], [0, 0.5], [0, 1]])
    grid = rt.pupil.grid(11).points()
    assert np.hypot(*grid.T).max() <= 1.0 + 1e-12
    a, b = rt.pupil.random(50, seed=3), rt.pupil.random(50, seed=3)
    np.testing.assert_array_equal(a.points(), b.points())  # law 10: the same seed, the same points
    assert not np.array_equal(a.points(), rt.pupil.random(50, seed=4).points())
