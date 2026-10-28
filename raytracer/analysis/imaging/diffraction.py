"""Scalar Fourier imaging: pupil function, PSF, and Abbe partially-coherent images.

The pupil grid maps FFT spatial frequencies to normalized pupil coordinates
``u = f_x * lambda / NA``. Image columns increase with x and rows with y;
image positions are ``(index - N // 2) * pixel_mm``. The coherent image field
uses the inverse Fourier transform of ``P = exp(+i 2 pi W / lambda)``:
a positive wavefront slope displaces the image towards negative coordinates,
consistent with ``grad W = -NA * transverse_aberration``.
Wavefronts can come from a fitted
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


def _positive_finite(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _sample_count(name: str, value: int, *, minimum: int = 1) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _source_samples(sigma: float, source_points: int) -> list[tuple[float, float]]:
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    _sample_count("source_points", source_points)
    # A one-point quadrature represents on-axis coherent illumination.
    axis = np.array([0.0]) if source_points == 1 else np.linspace(-sigma, sigma, source_points)
    sources = [(sx, sy) for sx in axis for sy in axis if sx * sx + sy * sy <= sigma * sigma]
    if not sources:
        raise ValueError("source_points does not sample the source disk; use 1 or at least 3")
    return sources


def _pupil_values(wavefront, u, v, inside, *, scale: float) -> np.ndarray:
    pupil = np.zeros(u.shape, dtype=complex)
    if not inside.any():
        return pupil
    if wavefront is None:
        pupil[inside] = 1.0
    else:
        # Wavefronts need only be defined on the transmitting aperture.
        values = np.asarray(wavefront(u[inside], v[inside]), dtype=float)
        phase = values * scale
        if not np.all(np.isfinite(phase)):
            raise ValueError("Wavefront must be finite inside the pupil aperture")
        pupil[inside] = np.exp(1j * phase)
    return pupil


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
    pass ``wavefront_unit="waves"`` when W is already in waves. The callable
    receives only coordinates inside the unit disk and may return a scalar.
    Values within the aperture must be finite; values outside are never used.
    """

    _sample_count("size", size, minimum=2)
    _positive_finite("na", na)
    _positive_finite("wavelength_mm", wavelength_mm)
    _positive_finite("pixel_mm", pixel_mm)
    if wavefront_unit not in ("mm", "waves"):
        raise ValueError("wavefront_unit must be 'mm' or 'waves'")
    frequency = np.fft.fftfreq(size, pixel_mm)
    fx, fy = np.meshgrid(frequency, frequency)
    u = fx * wavelength_mm / na
    v = fy * wavelength_mm / na
    rho2 = u * u + v * v
    disk = rho2 <= 1.0
    scale = 2.0 * np.pi * (1.0 if wavefront_unit == "waves" else 1.0 / wavelength_mm)
    pupil = _pupil_values(wavefront, u, v, disk, scale=scale)
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
    """Intensity PSF, centred on the same x/y axes as an Abbe image.

    Without normalization the intensity is the squared modulus of the discrete
    coherent pupil sum (no pixel-area or radiometric scaling).
    """

    amplitude = np.fft.fftshift(np.fft.ifft2(grid.pupil, norm="forward"))
    psf = np.abs(amplitude) ** 2
    if normalize and psf.max() > 0:
        psf /= psf.max()
    return psf


def airy_radius_mm(na: float, wavelength_mm: float) -> float:
    return 0.61 * wavelength_mm / na


@dataclass
class BinaryMask:
    """Binary amplitude mask on a square pixel grid (dimensions in nm)."""

    data: np.ndarray  # (N, N) float in [0, 1]
    pixel_nm: float

    @classmethod
    def empty(cls, size: int, pixel_nm: float) -> "BinaryMask":
        return cls(data=np.zeros((size, size)), pixel_nm=pixel_nm)

    def _grids(self) -> tuple[np.ndarray, np.ndarray]:
        size = self.data.shape[0]
        axis = (np.arange(size) - size // 2) * self.pixel_nm
        return np.meshgrid(axis, axis)

    def rect(self, x0: float, y0: float, width: float, height: float) -> "BinaryMask":
        """Add a filled rectangle (centre and size in nm)."""

        xx, yy = self._grids()
        self.data[(np.abs(xx - x0) <= width / 2) & (np.abs(yy - y0) <= height / 2)] = 1.0
        return self

    @classmethod
    def lithography_target(
        cls, *, size: int = 512, pixel_nm: float = 10.0, feature_nm: float = 70.0
    ) -> "BinaryMask":
        """Synthetic test target: L/S arrays, elbows, isolated line, contacts.

        The default parameters reproduce the DUV reference mask; scale
        ``pixel_nm``/``feature_nm`` down for EUV-class systems.
        """

        mask = cls.empty(size, pixel_nm)
        f = feature_nm / 70.0  # scale factor relative to the reference target
        for k in range(-5, 6):
            mask.rect(f * (-1450 + k * 140), f * 1300, f * 70, f * 1150)
            mask.rect(f * 1450, f * (1300 + k * 140), f * 1150, f * 70)
        for k in range(5):
            mask.rect(f * (-1950 + k * 210), f * -1400, f * 100, f * 1050)
            mask.rect(f * -1530, f * (-1950 + k * 210), f * 950, f * 100)
        mask.rect(f * 250, f * -1350, f * 70, f * 1200)
        for kx in range(5):
            for ky in range(5):
                mask.rect(f * (1100 + kx * 260), f * (-1950 + ky * 260), f * 120, f * 120)
        return mask

    @classmethod
    def from_image(
        cls, path, *, pixel_nm: float, size: int = 512, threshold: float = 0.5, invert: bool = False
    ) -> "BinaryMask":
        """Load an image, convert to grayscale, resample to *size*, binarize."""

        import matplotlib.image as mpimg
        from scipy import ndimage

        raw = mpimg.imread(path)
        if raw.ndim == 3:
            raw = raw[..., :3].mean(axis=2)
        raw = np.asarray(raw, dtype=float)
        if raw.max() > 1.0:
            raw = raw / 255.0
        zoom = (size / raw.shape[0], size / raw.shape[1])
        resampled = ndimage.zoom(raw, zoom, order=1)[:size, :size]
        padded = np.zeros((size, size))
        padded[: resampled.shape[0], : resampled.shape[1]] = resampled
        binary = (padded < threshold) if invert else (padded > threshold)
        return cls(data=np.flipud(binary.astype(float)), pixel_nm=pixel_nm)


def abbe_image(
    grid: PupilGrid,
    mask: BinaryMask | np.ndarray,
    *,
    sigma: float = 0.7,
    source_points: int = 11,
    normalize: bool = True,
) -> np.ndarray:
    """Partially coherent Abbe image of *mask* through the pupil.

    A conventional disc source of coherence factor *sigma* is integrated by
    shifting the pupil in the frequency domain (integer-pixel roll). A single
    source point is on-axis; two points per axis cannot sample a nonzero disk.
    Unnormalized intensity is the sum over source samples. Illumination samples
    have equal weights; frequency shifts are quantized to the pupil grid.
    """

    sources = _source_samples(sigma, source_points)
    data = np.asarray(mask.data if isinstance(mask, BinaryMask) else mask, dtype=float)
    if not np.all(np.isfinite(data)):
        raise ValueError("Mask values must be finite")
    if data.shape != grid.pupil.shape:
        raise ValueError("Mask and pupil grids must have the same shape")
    if isinstance(mask, BinaryMask):
        _positive_finite("mask.pixel_nm", mask.pixel_nm)
        expected_pixel_nm = grid.pixel_mm * 1e6
        if abs(mask.pixel_nm - expected_pixel_nm) > 1e-9:
            raise ValueError(
                f"Mask pixel ({mask.pixel_nm} nm) does not match the pupil grid "
                f"({expected_pixel_nm} nm)"
            )

    spectrum = np.fft.fft2(data)
    aerial = np.zeros_like(data)
    df = grid.frequency[1] - grid.frequency[0]
    for sx, sy in sources:
        shift_x = int(round(sx * grid.na / grid.wavelength_mm / df))
        shift_y = int(round(sy * grid.na / grid.wavelength_mm / df))
        shifted = np.roll(np.roll(grid.pupil, shift_y, axis=0), shift_x, axis=1)
        aerial += np.abs(np.fft.ifft2(spectrum * shifted)) ** 2
    if normalize and aerial.max() > 0:
        aerial /= aerial.max()
    return aerial


def contrast_curve(
    wavefront_waves: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    half_pitches_nm,
    *,
    na: float,
    wavelength_mm: float,
    sigma: float = 0.7,
    size: int = 4096,
    pixel_nm: float = 0.5,
    source_points: int = 13,
) -> np.ndarray:
    """Image contrast of 1-D equal line/space gratings vs half-pitch (nm)."""

    _sample_count("size", size, minimum=2)
    _positive_finite("na", na)
    _positive_finite("wavelength_mm", wavelength_mm)
    _positive_finite("pixel_nm", pixel_nm)
    sources = _source_samples(sigma, source_points)
    half_pitches = np.asarray(half_pitches_nm, dtype=float)
    if half_pitches.ndim != 1 or not np.all(np.isfinite(half_pitches) & (half_pitches > 0)):
        raise ValueError(
            "half_pitches_nm must be a one-dimensional array of finite positive values"
        )
    pixel_mm = pixel_nm * 1e-6
    frequency = np.fft.fftfreq(size, pixel_mm)
    x_nm = (np.arange(size) - size // 2) * pixel_nm
    contrasts = []
    for hp in half_pitches:
        grating = (np.mod(x_nm, 2 * hp) < hp).astype(float)
        spectrum = np.fft.fft(grating)
        intensity = np.zeros(size)
        for sx, sy in sources:
            u = frequency * wavelength_mm / na + sx
            v = np.full_like(u, sy)
            inside = (u * u + v * v) <= 1.0
            pupil_line = _pupil_values(wavefront_waves, u, v, inside, scale=2 * np.pi)
            intensity += np.abs(np.fft.ifft(spectrum * pupil_line)) ** 2
        half_window = min(800, size // 2)
        window = slice(size // 2 - half_window, size // 2 + half_window + size % 2)
        peak, valley = intensity[window].max(), intensity[window].min()
        if peak + valley == 0:
            raise ValueError("No transmitted intensity; contrast is undefined")
        contrasts.append((peak - valley) / (peak + valley))
    return np.asarray(contrasts)


def coherent_cutoff_half_pitch_nm(na: float, wavelength_mm: float, sigma: float) -> float:
    """Partially-coherent resolution limit lambda / [2 NA (1 + sigma)] in nm."""

    return wavelength_mm * 1e6 / (2.0 * na * (1.0 + sigma))


__all__ = [
    "PupilGrid",
    "pupil_function",
    "scalar_psf",
    "airy_radius_mm",
    "BinaryMask",
    "abbe_image",
    "contrast_curve",
    "coherent_cutoff_half_pitch_nm",
]
