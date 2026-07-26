"""Image formation: spot diagrams, PSF/aerial image, contrast."""

from .diffraction import (
    BinaryMask,
    PupilGrid,
    abbe_image,
    airy_radius_mm,
    coherent_cutoff_half_pitch_nm,
    contrast_curve,
    pupil_function,
    scalar_psf,
)
from .spots import SpotData, spot_data, spot_data_from_points, weighted_mean, weighted_quantile

__all__ = [
    "SpotData",
    "spot_data",
    "spot_data_from_points",
    "weighted_mean",
    "weighted_quantile",
    "PupilGrid",
    "pupil_function",
    "scalar_psf",
    "airy_radius_mm",
    "BinaryMask",
    "abbe_image",
    "contrast_curve",
    "coherent_cutoff_half_pitch_nm",
]
