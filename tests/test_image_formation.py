"""Geometric and diffractive image-formation ground truths."""

import numpy as np
import pytest

from raytracer.elements import Lens, Sensor
from raytracer.geometry.apertures import Rectangle
from raytracer.geometry.surfaces import Sphere
from raytracer.materials import ConstantIndex
from raytracer.sources import ImageSource
from raytracer.system import OpticalSystem
from raytracer.tracing import NonSequentialTracer


def _relay(profile, *, object_height=6.0, rays_per_point=41):
    """A biconvex relay imaging a 1-D object profile at z = -90 onto a sensor at 171.4.

    The old 2-D frame (axis +x, height y) maps to the meridional y-z plane;
    the old full fan aperture of 8 degrees is a 4 degree half-angle.
    """

    lens = Lens(
        "relay", front=Sphere(60.0), back=Sphere(-60.0), thickness=9.0,
        material=ConstantIndex("N1.5168", 1.5168), semi_diameter=16.0,
    )
    screen = Sensor("screen", aperture=Rectangle(40.0, 40.0), z=171.4)
    source = ImageSource(
        np.asarray(profile, dtype=float),
        p0=(0.0, -object_height, -90.0),
        p1=(0.0, object_height, -90.0),
        half_angle_deg=4.0,
        rays_per_point=rays_per_point,
    )
    tree = NonSequentialTracer(OpticalSystem([lens, screen]), split=False).trace(source)
    return tree.hits("screen")


def test_two_point_object_images_inverted_and_magnified():
    profile = np.zeros(101)
    profile[20] = 1.0  # object point at y = -3.6 mm
    profile[80] = 1.0  # object point at y = +3.6 mm
    hits = _relay(profile)

    positions = hits.local_points[:, 1]
    weights = hits.power
    # Split arrivals by sign and locate both image points.
    y_up = np.average(positions[positions > 0], weights=weights[positions > 0])
    y_lo = np.average(positions[positions < 0], weights=weights[positions < 0])

    # Nominal conjugates: object 90 -> image ~171.4 behind the lens.
    # Paraxial magnification ~ -1.83: object at -3.6 -> image at +6.6.
    magnification = y_up / -3.6
    assert magnification == pytest.approx(-1.83, abs=0.08)
    assert y_lo == pytest.approx(-y_up, abs=0.15)  # symmetric pair


def test_image_intensity_scales_with_pixel_value():
    profile = np.zeros(101)
    profile[30] = 1.0
    profile[70] = 0.4
    hits = _relay(profile)
    positions = hits.local_points[:, 1]
    bright = hits.power[positions > 0].sum()  # pixel 30 (lower object) -> upper image
    dim = hits.power[positions < 0].sum()
    assert dim / bright == pytest.approx(0.4, abs=0.03)


def test_binary_mask_from_image_roundtrip(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from raytracer.analysis import BinaryMask

    pattern = np.zeros((64, 64))
    pattern[20:44, 28:36] = 1.0  # vertical bar
    path = tmp_path / "bar.png"
    plt.imsave(path, pattern, cmap="gray")

    mask = BinaryMask.from_image(path, pixel_nm=10.0, size=64, threshold=0.5)
    assert mask.data.shape == (64, 64)
    # The bar survives binarization (flipped vertically by convention).
    assert mask.data.sum() == pytest.approx(pattern.sum(), rel=0.15)
    column_occupancy = mask.data[:, 30].sum()
    assert column_occupancy >= 20


def test_abbe_image_of_mask_resolves_large_features():
    from raytracer.analysis import BinaryMask, abbe_image, pupil_function

    mask = BinaryMask.empty(256, 10.0)
    mask.rect(0.0, 0.0, 400.0, 400.0)  # 400 nm square, far above resolution
    grid = pupil_function(None, na=1.2, wavelength_mm=193.368e-6, size=256,
                          pixel_mm=10e-6)
    aerial = abbe_image(grid, mask, sigma=0.7, source_points=7)
    centre = aerial[128, 128]
    corner = aerial[10, 10]
    assert centre > 0.8
    assert corner < 0.05
