"""Colorimetry: wavelength to linear-sRGB conversion (CIE 1931 fits)."""

from __future__ import annotations

import numpy as np


def _gaussian(x: np.ndarray, alpha: float, mu: float, s1: float, s2: float) -> np.ndarray:
    sigma = np.where(x < mu, s1, s2)
    return alpha * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def wavelength_to_xyz(wavelength_nm) -> np.ndarray:
    """CIE 1931 color-matching functions via the Wyman/Sloan/Shirley fits."""

    wl = np.atleast_1d(np.asarray(wavelength_nm, dtype=float))
    x = (
        _gaussian(wl, 1.056, 599.8, 37.9, 31.0)
        + _gaussian(wl, 0.362, 442.0, 16.0, 26.7)
        + _gaussian(wl, -0.065, 501.1, 20.4, 26.2)
    )
    y = _gaussian(wl, 0.821, 568.8, 46.9, 40.5) + _gaussian(wl, 0.286, 530.9, 16.3, 31.1)
    z = _gaussian(wl, 1.217, 437.0, 11.8, 36.0) + _gaussian(wl, 0.681, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=-1)


_XYZ_TO_LINEAR_SRGB = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ]
)


def wavelength_to_rgb(wavelength_nm, *, normalize: bool = True) -> np.ndarray:
    """Linear-sRGB color of a monochromatic wavelength (nm).

    Out-of-gamut components are clipped; with *normalize* the result is
    scaled so max(channel) == 1 (fully saturated spectral color). Returns
    shape (..., 3).
    """

    xyz = wavelength_to_xyz(wavelength_nm)
    rgb = xyz @ _XYZ_TO_LINEAR_SRGB.T
    rgb = np.clip(rgb, 0.0, None)
    if normalize:
        peak = rgb.max(axis=-1, keepdims=True)
        rgb = np.where(peak > 0, rgb / np.where(peak == 0, 1.0, peak), rgb)
    if np.isscalar(wavelength_nm):
        return rgb[0]
    return rgb


__all__ = ["wavelength_to_xyz", "wavelength_to_rgb"]
