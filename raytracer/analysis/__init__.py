"""Optical analysis: spots, fans, aberration metrics, Zernike, imaging."""

from .fans import FanData, ray_fans
from .imaging import (
    BinaryMask,
    PupilGrid,
    abbe_image,
    airy_radius_mm,
    coherent_cutoff_half_pitch_nm,
    contrast_curve,
    pupil_function,
    scalar_psf,
)
from .metrics import export_metrics_csv, field_metrics
from .spots import SpotData, spot_data, spot_data_from_points
from .stigmatism import (
    StigmatismReport,
    point_line_distances,
    rays_by_generation,
    stigmatism_report,
)
from .wavefront import WavefrontSamples, exit_pupil_wavefront
from .zernike import (
    Mode,
    ZernikeExpansion,
    fit_opd,
    fit_transverse,
    zernike,
    zernike_modes,
)

__all__ = [
    "SpotData",
    "spot_data",
    "spot_data_from_points",
    "StigmatismReport",
    "stigmatism_report",
    "point_line_distances",
    "rays_by_generation",
    "FanData",
    "ray_fans",
    "field_metrics",
    "export_metrics_csv",
    "Mode",
    "zernike",
    "zernike_modes",
    "ZernikeExpansion",
    "fit_transverse",
    "fit_opd",
    "WavefrontSamples",
    "exit_pupil_wavefront",
    "PupilGrid",
    "pupil_function",
    "scalar_psf",
    "airy_radius_mm",
    "BinaryMask",
    "abbe_image",
    "contrast_curve",
    "coherent_cutoff_half_pitch_nm",
]
