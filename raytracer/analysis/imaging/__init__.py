"""Image formation: spot diagrams, point-spread functions, lithographic aerial images."""

from .diffraction import PupilGrid, airy_radius_mm, pupil_function, scalar_psf
from .lithography import BinaryMask, abbe_image, coherent_cutoff_half_pitch_nm, contrast_curve
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
