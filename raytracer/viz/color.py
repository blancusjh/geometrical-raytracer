"""Colorimetry: wavelength to linear sRGB (CIE 1931 fits), and hex colours to RGBA."""

from __future__ import annotations

import numpy as np

_XYZ_TO_LINEAR_SRGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])


def wavelength_to_rgb(wavelength_nm, *, normalize: bool = True) -> np.ndarray:
    """Linear-sRGB colour ``(..., 3)`` of a monochromatic wavelength (nm), out-of-gamut parts clipped.

    With *normalize* the brightest channel is 1: the fully saturated spectral colour.
    """

    rgb = np.clip(wavelength_to_xyz(wavelength_nm) @ _XYZ_TO_LINEAR_SRGB.T, 0.0, None)
    if normalize:
        peak = rgb.max(axis=-1, keepdims=True)
        rgb = np.where(peak > 0, rgb / np.where(peak == 0, 1.0, peak), rgb)
    return rgb[0] if np.isscalar(wavelength_nm) else rgb


def spectral_rgb(wavelength_nm: float) -> np.ndarray:
    """Displayable colour of a wavelength: its spectral colour, or deep violet outside the visible (193 nm DUV).

    The test is on the unnormalized response: beyond the visible the fits leave residue that normalizing
    would blow up to a wrong hue.
    """

    if float(wavelength_to_rgb(float(wavelength_nm), normalize=False).max()) < 1e-3:
        return np.array([0.45, 0.18, 1.0])
    return np.asarray(wavelength_to_rgb(float(wavelength_nm)), dtype=float)


def wavelength_to_xyz(wavelength_nm) -> np.ndarray:
    """CIE 1931 colour-matching functions via the Wyman/Sloan/Shirley fits."""

    wl = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))

    def g(alpha, mu, s1, s2):
        return alpha * np.exp(-0.5 * ((wl - mu) / np.where(wl < mu, s1, s2)) ** 2)

    x = g(1.056, 599.8, 37.9, 31.0) + g(0.362, 442.0, 16.0, 26.7) + g(-0.065, 501.1, 20.4, 26.2)
    y = g(0.821, 568.8, 46.9, 40.5) + g(0.286, 530.9, 16.3, 31.1)
    z = g(1.217, 437.0, 11.8, 36.0) + g(0.681, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=-1)


def hex_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0, float(alpha))


__all__ = ["wavelength_to_rgb", "spectral_rgb", "wavelength_to_xyz", "hex_rgba"]
