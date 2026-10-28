"""Diffraction regressions with analytic translations and direct Fourier sums."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from raytracer.analysis.imaging.diffraction import (
    BinaryMask,
    abbe_image,
    contrast_curve,
    pupil_function,
    scalar_psf,
)


def test_tilt_translation_and_coherent_impulse_share_image_coordinates():
    # W/lambda = 2u-v gives (dx,dy) = (-2,+1)*lambda/NA.
    # lambda/NA = 5 pixels here, independently of the FFT grid size.
    for size in (127, 128):
        settings = dict(na=0.5, wavelength_mm=0.0005, size=size, pixel_mm=0.0002)
        perfect = pupil_function(None, **settings)
        tilted = pupil_function(lambda u, v: 2 * u - v, wavefront_unit="waves", **settings)
        psf = scalar_psf(tilted)
        expected = np.roll(scalar_psf(perfect), (5, -10), axis=(0, 1))
        assert_allclose(psf, expected, atol=2e-14, rtol=0)
        assert np.unravel_index(psf.argmax(), psf.shape) == (size // 2 + 5, size // 2 - 10)

        impulse = BinaryMask.empty(size, pixel_nm=200)
        impulse.data[size // 2, size // 2] = 1
        assert_allclose(abbe_image(tilted, impulse, sigma=0, source_points=1), psf, atol=2e-14)
        # Raw PSF retains the squared coherent-sum scale; Abbe uses NumPy's
        # normalized inverse transform, contributing N^-4 to intensity.
        raw = abbe_image(tilted, impulse, sigma=0, source_points=1, normalize=False)
        assert_allclose(raw * size**4, scalar_psf(tilted, normalize=False), atol=2e-10)

        impulse.data[:] = 0
        impulse.data[size // 2 + 3, size // 2 - 4] = 1
        assert_allclose(
            abbe_image(tilted, impulse, sigma=0, source_points=1),
            np.roll(psf, (3, -4), axis=(0, 1)),
            atol=2e-14,
        )


def test_asymmetric_coma_matches_direct_physical_fourier_sum():
    size = 33
    grid = pupil_function(
        lambda u, v: 0.8 * u * (3 * (u * u + v * v) - 2) + 0.3 * v,
        na=0.5,
        wavelength_mm=0.0005,
        size=size,
        pixel_mm=0.0002,
        wavefront_unit="waves",
    )
    # Direct sum exp(+2 pi i f x) on physical centred coordinates, without FFT.
    x_mm = (np.arange(size) - size // 2) * grid.pixel_mm
    kernel = np.exp(2j * np.pi * np.outer(x_mm, grid.frequency))
    intensity = abs(kernel @ grid.pupil @ kernel.T) ** 2
    expected = intensity / intensity.max()
    assert_allclose(scalar_psf(grid), expected, atol=2e-14, rtol=0)
    impulse = np.zeros((size, size))
    impulse[size // 2, size // 2] = 1
    assert_allclose(abbe_image(grid, impulse, sigma=0, source_points=1), expected, atol=2e-14)
    assert np.max(abs(expected - expected[::-1, ::-1])) > 0.1


def test_wavefront_is_evaluated_only_inside_transmitting_aperture():
    settings = dict(na=0.5, wavelength_mm=0.0005, size=64, pixel_mm=0.0002)
    calls = []

    def disk_only(u, v):
        calls.append(u.size)
        assert np.all(u * u + v * v <= 1)
        # Undefined outside the disk, as for an aperture-limited interpolant.
        return np.where(u * u + v * v <= 1, 0.0, np.nan)

    grid = pupil_function(disk_only, **settings)
    inside = grid.u * grid.u + grid.v * grid.v <= 1
    assert calls == [np.count_nonzero(inside)]
    assert np.isfinite(grid.pupil).all()
    assert np.all(grid.pupil[~inside] == 0)
    assert_allclose(scalar_psf(grid), scalar_psf(pupil_function(None, **settings)))
    scalar = pupil_function(lambda u, v: 0.25, wavefront_unit="waves", **settings)
    assert_allclose(scalar_psf(scalar), scalar_psf(grid), atol=2e-14)
    mm = pupil_function(lambda u, v: settings["wavelength_mm"] * u, **settings)
    waves = pupil_function(lambda u, v: u, wavefront_unit="waves", **settings)
    assert_allclose(mm.pupil, waves.pupil, atol=2e-15)
    for invalid in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite inside"):
            pupil_function(lambda u, v: invalid, **settings)
        with pytest.raises(ValueError, match="finite inside"):
            contrast_curve(
                lambda u, v: invalid,
                [800],
                na=0.5,
                wavelength_mm=0.0005,
                size=64,
                pixel_nm=200,
                sigma=0,
                source_points=1,
            )


def test_source_sampling_and_small_grid_contrast():
    grid = pupil_function(None, na=0.5, wavelength_mm=0.0005, size=64, pixel_mm=0.0002)
    constant = np.ones((64, 64))
    assert_allclose(abbe_image(grid, constant, source_points=1, normalize=False), 1)
    settings = dict(na=0.5, wavelength_mm=0.0005, size=64, pixel_nm=200, source_points=1)
    on_axis = contrast_curve(None, [800], sigma=0, **settings)
    assert_allclose(contrast_curve(None, [800], sigma=0.7, **settings), on_axis)
    # A periodic grating provides the same 1-D contrast as its repeated 2-D
    # coherent image. All 64 pixels are used by the small-grid window.
    x_nm = (np.arange(64) - 32) * 200
    grating = (np.mod(x_nm, 1600) < 800).astype(float)
    mask = np.broadcast_to(grating, (64, 64))
    image = abbe_image(grid, mask, sigma=0, source_points=1)
    expected = (image.max() - image.min()) / (image.max() + image.min())
    assert_allclose(on_axis, [expected], atol=2e-14)
    assert np.isfinite(on_axis).all()

    for count in (0, -1, 1.5, True, 2):
        with pytest.raises(ValueError, match="source_points"):
            abbe_image(grid, constant, source_points=count)
        with pytest.raises(ValueError, match="source_points"):
            contrast_curve(None, [800], **(settings | {"source_points": count}))
    for sigma in (-0.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="sigma"):
            abbe_image(grid, constant, sigma=sigma)
        with pytest.raises(ValueError, match="sigma"):
            contrast_curve(None, [800], sigma=sigma, **settings)


def test_public_sampling_errors_are_explicit():
    settings = dict(na=0.5, wavelength_mm=0.0005, size=64, pixel_mm=0.0002)
    for name in ("na", "wavelength_mm", "pixel_mm"):
        for invalid in (0, -1, np.nan, np.inf):
            with pytest.raises(ValueError, match=name):
                pupil_function(None, **(settings | {name: invalid}))
    for invalid in (0, 1, 2.5, True):
        with pytest.raises(ValueError, match="size"):
            pupil_function(None, **(settings | {"size": invalid}))
    with pytest.raises(ValueError, match="wavefront_unit"):
        pupil_function(None, wavefront_unit="nm", **settings)

    grid = pupil_function(None, **settings)
    with pytest.raises(ValueError, match="same shape"):
        abbe_image(grid, np.zeros((8, 8)))
    with pytest.raises(ValueError, match="finite"):
        abbe_image(grid, np.full((64, 64), np.nan))
    with pytest.raises(ValueError, match="does not match"):
        abbe_image(grid, BinaryMask.empty(64, pixel_nm=100))
    with pytest.raises(ValueError, match="finite and positive"):
        abbe_image(grid, BinaryMask.empty(64, pixel_nm=np.nan))
    for pitches in ([0], [-1], [np.nan], [np.inf], [[800]]):
        with pytest.raises(ValueError, match="half_pitches_nm"):
            contrast_curve(None, pitches, na=0.5, wavelength_mm=0.0005)
