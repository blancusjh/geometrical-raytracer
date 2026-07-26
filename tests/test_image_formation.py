"""Geometric and diffractive image-formation ground truths."""

import numpy as np
import pytest

from raytracer.optics import ImageSource, Lens, Screen
from raytracer.propagation import BranchingTracer, TraceConfig


def _relay(profile, *, object_height=6.0, rays_per_point=41):
    lens = Lens.from_radii(
        r1=60.0, r2=-60.0, thickness=9.0, semidiameter=16.0, n=1.5168,
        vertex=(0.0, 0.0), name="relay",
    )
    source = ImageSource(
        profile=np.asarray(profile, dtype=float),
        p0=np.array([-90.0, -object_height]),
        p1=np.array([-90.0, object_height]),
        axis_direction=np.array([1.0, 0.0]),
        aperture=np.deg2rad(8.0),
        rays_per_point=rays_per_point,
    )
    screen = Screen([171.4, -20.0], [171.4, 20.0])
    tracer = BranchingTracer([*lens.surfaces(), screen], TraceConfig(max_generations=4))
    tracer.trace([source])
    return screen


def test_two_point_object_images_inverted_and_magnified():
    profile = np.zeros(101)
    profile[20] = 1.0  # object point at y = -3.6 mm
    profile[80] = 1.0  # object point at y = +3.6 mm
    screen = _relay(profile)

    positions = screen.coordinates() - screen.length / 2.0
    weights = screen.intensities()
    # Split arrivals by sign and locate both image points.
    upper = positions[positions > 0]
    lower = positions[positions < 0]
    w_up = weights[positions > 0]
    w_lo = weights[positions < 0]
    y_up = np.average(upper, weights=w_up)
    y_lo = np.average(lower, weights=w_lo)

    # Nominal conjugates: object 90 -> image ~171.4 behind the lens.
    # Paraxial magnification ~ -1.83: object at -3.6 -> image at +6.6.
    magnification = y_up / -3.6
    assert magnification == pytest.approx(-1.83, abs=0.08)
    assert y_lo == pytest.approx(-y_up, abs=0.15)  # symmetric pair


def test_image_intensity_scales_with_pixel_value():
    profile = np.zeros(101)
    profile[30] = 1.0
    profile[70] = 0.4
    screen = _relay(profile)
    positions = screen.coordinates() - screen.length / 2.0
    weights = screen.intensities()
    bright = weights[positions > 0].sum()  # pixel 30 (lower object) -> upper image
    dim = weights[positions < 0].sum()
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
