"""Exit-pupil wavefront from accumulated optical path length (eikonal).

Each ray's OPL is referenced to a sphere of radius ``reference_radius``
centred on the chief-ray image point (EUV notebook section 8): the OPD is

    W = (L - s) - (L_chief - R)

where ``s`` is the distance from the ray's image point back to the
reference sphere along the ray.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...propagation.fields import PupilTrace


@dataclass
class WavefrontSamples:
    """Scattered exit-pupil wavefront samples in waves."""

    u: np.ndarray
    v: np.ndarray
    opd_waves: np.ndarray
    wavelength_mm: float

    @property
    def rms_waves(self) -> float:
        return float(np.std(self.opd_waves))

    @property
    def pv_waves(self) -> float:
        return float(np.ptp(self.opd_waves))

    def to_grid(self, u_grid: np.ndarray, v_grid: np.ndarray, *, method: str = "cubic") -> np.ndarray:
        """Interpolate the scattered samples onto a grid (NaN-safe, zero fill)."""

        from scipy.interpolate import griddata

        values = griddata(
            np.column_stack([self.u, self.v]),
            self.opd_waves,
            (u_grid, v_grid),
            method=method,
            fill_value=0.0,
        )
        return np.nan_to_num(values)

    def __call__(self, u, v) -> np.ndarray:
        """Evaluate by interpolation at arbitrary pupil coordinates."""

        return self.to_grid(np.asarray(u, dtype=float), np.asarray(v, dtype=float))


def exit_pupil_wavefront(
    pupil: PupilTrace,
    *,
    na_image: float,
    wavelength_mm: float,
    n_image: float = 1.0,
    reference_radius: float = 500.0,
    remove_mean: bool = True,
) -> WavefrontSamples:
    """Compute the exit-pupil OPD of a traced bundle (in waves).

    Pupil coordinates are arrival directions relative to the chief ray,
    normalized by the image-space NA. Requires OPL tracking, which
    :class:`~raytracer.propagation.sequential.SequentialTracer` always performs.
    """

    chief = pupil.chief
    chief_opl = chief.opl
    chief_point = chief.image_point
    chief_direction = chief.direction

    directions = pupil.directions
    points = pupil.image_points
    opl = pupil.opl

    u = n_image * (directions[:, 0] - chief_direction[0]) / na_image
    v = n_image * (directions[:, 1] - chief_direction[1]) / na_image

    w = points - chief_point
    b = np.einsum("ij,ij->i", w, directions)
    radicand = np.maximum(b * b - (np.einsum("ij,ij->i", w, w) - reference_radius**2), 0.0)
    s = b + np.sqrt(radicand)

    opd = (opl - n_image * s) - (chief_opl - n_image * reference_radius)
    opd_waves = opd / wavelength_mm
    if remove_mean:
        opd_waves = opd_waves - opd_waves.mean()

    return WavefrontSamples(u=u, v=v, opd_waves=opd_waves, wavelength_mm=wavelength_mm)


__all__ = ["WavefrontSamples", "exit_pupil_wavefront"]
