"""Scalar Fourier imaging: the pupil function and the point-spread function.

The pupil grid maps FFT spatial frequencies to normalized pupil coordinates
``u = f_x * lambda / NA``. Wavefronts can come from a fitted
:class:`~raytracer.analysis.aberrations.zernike.ZernikeExpansion` (mm) or interpolated
:class:`~raytracer.analysis.aberrations.wavefront.WavefrontSamples` (waves).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np


@dataclass
class PupilGrid:
    """Complex pupil function sampled on an FFT frequency grid."""

    pupil: np.ndarray  # (N, N) complex
    frequency: np.ndarray  # (N,) cycles/mm along each axis
    u: np.ndarray  # (N, N) normalized pupil coordinate
    v: np.ndarray
    na: float
    wavelength_mm: float
    pixel_mm: float

    @property
    def size(self) -> int:
        return self.pupil.shape[0]


def pupil_function(
    wavefront: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    *,
    na: float,
    wavelength_mm: float,
    size: int = 512,
    pixel_mm: float = 10e-6,
    wavefront_unit: Literal["mm", "waves"] = "mm",
) -> PupilGrid:
    """Build the complex pupil ``P = 1_disk * exp(i 2 pi W / lambda)``.

    *wavefront* is any callable ``W(u, v)`` (``None`` for a perfect pupil);
    pass ``wavefront_unit="waves"`` when W is already in waves.
    """

    frequency = np.fft.fftfreq(size, pixel_mm)
    fx, fy = np.meshgrid(frequency, frequency)
    u = fx * wavelength_mm / na
    v = fy * wavelength_mm / na
    rho2 = u * u + v * v
    disk = rho2 <= 1.0
    if wavefront is None:
        phase = np.zeros_like(u)
    else:
        w = np.asarray(wavefront(u, v), dtype=float)
        phase = 2.0 * np.pi * (w if wavefront_unit == "waves" else w / wavelength_mm)
    pupil = disk * np.exp(1j * phase)
    return PupilGrid(
        pupil=pupil,
        frequency=frequency,
        u=u,
        v=v,
        na=na,
        wavelength_mm=wavelength_mm,
        pixel_mm=pixel_mm,
    )


def scalar_psf(grid: PupilGrid, *, normalize: bool = True) -> np.ndarray:
    """Intensity PSF (centred) of the pupil function."""

    amplitude = np.fft.fftshift(np.fft.fft2(grid.pupil))
    psf = np.abs(amplitude) ** 2
    if normalize and psf.max() > 0:
        psf /= psf.max()
    return psf


def airy_radius_mm(na: float, wavelength_mm: float) -> float:
    return 0.61 * wavelength_mm / na


__all__ = ["PupilGrid", "pupil_function", "scalar_psf", "airy_radius_mm"]
