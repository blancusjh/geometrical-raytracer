"""ANSI/OSA Zernike polynomials and wavefront fits.

Two fitting routes:

- :func:`fit_transverse` reconstructs the wavefront from transverse ray
  aberrations through Hamilton's relation grad_q W = -NA * dr' (the DUV
  reference method, robust through folded paths).
- :func:`fit_opd` fits sampled OPD values directly (for eikonal wavefronts).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ...sequential.fields import PupilTrace

ZERNIKE_NAMES = {
    (0, 0): "Piston",
    (1, -1): "Y tilt",
    (1, 1): "X tilt",
    (2, -2): "Oblique astigmatism",
    (2, 0): "Defocus",
    (2, 2): "Vertical astigmatism",
    (3, -3): "Oblique trefoil",
    (3, -1): "Y coma",
    (3, 1): "X coma",
    (3, 3): "Vertical trefoil",
    (4, -4): "Oblique quadrafoil",
    (4, -2): "Secondary oblique astigmatism",
    (4, 0): "Primary spherical",
    (4, 2): "Secondary vertical astigmatism",
    (4, 4): "Vertical quadrafoil",
    (5, -1): "Secondary Y coma",
    (5, 1): "Secondary X coma",
    (6, 0): "Secondary spherical",
}


class Mode(NamedTuple):
    ansi: int
    n: int
    m: int
    name: str


def zernike_modes(max_order: int = 6, *, include_piston: bool = False) -> list[Mode]:
    modes = []
    start = 0 if include_piston else 1  # piston is unobservable from ray slopes
    for n in range(start, max_order + 1):
        for m in range(-n, n + 1, 2):
            ansi = (n * (n + 2) + m) // 2
            modes.append(Mode(ansi, n, m, ZERNIKE_NAMES.get((n, m), f"Z({n},{m})")))
    return modes


def _zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    m = abs(m)
    result = np.zeros_like(np.asarray(rho, dtype=float))
    for k in range((n - m) // 2 + 1):
        coefficient = (
            (-1) ** k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + m) // 2 - k)
                * math.factorial((n - m) // 2 - k)
            )
        )
        result += coefficient * np.asarray(rho) ** (n - 2 * k)
    return result


def zernike(n: int, m: int, u, v) -> np.ndarray:
    """ANSI/OSA RMS-normalized Zernike polynomial on the unit disk."""

    rho = np.hypot(u, v)
    theta = np.arctan2(v, u)
    normalization = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
    angular = np.sin(abs(m) * theta) if m < 0 else np.cos(m * theta)
    return normalization * _zernike_radial(n, m, rho) * angular


@dataclass
class ZernikeExpansion:
    """Fitted Zernike coefficients (in the length unit of the fit, mm)."""

    modes: list[Mode]
    coefficients_mm: np.ndarray
    wavelength_mm: float | None = None
    u: np.ndarray | None = None
    v: np.ndarray | None = None
    weights: np.ndarray | None = None
    gradient_fit_residual_um: float | None = None

    def _mask(self, *, exclude_tilt: bool, exclude_defocus: bool, exclude_piston: bool = True):
        keep = []
        for mode in self.modes:
            if exclude_piston and mode.n == 0:
                keep.append(False)
            elif exclude_tilt and mode.n == 1:
                keep.append(False)
            elif exclude_defocus and (mode.n, mode.m) == (2, 0):
                keep.append(False)
            else:
                keep.append(True)
        return np.asarray(keep)

    def rms_nm(self, *, exclude_tilt: bool = True, exclude_defocus: bool = False) -> float:
        keep = self._mask(exclude_tilt=exclude_tilt, exclude_defocus=exclude_defocus)
        return float(np.sqrt(np.sum(self.coefficients_mm[keep] ** 2)) * 1e6)

    @property
    def rms_no_tilt_nm(self) -> float:
        return self.rms_nm(exclude_tilt=True, exclude_defocus=False)

    @property
    def rms_refocused_nm(self) -> float:
        return self.rms_nm(exclude_tilt=True, exclude_defocus=True)

    def wavefront(self, u, v, *, remove_tilt: bool = True, refocus: bool = False) -> np.ndarray:
        """Evaluate the fitted wavefront (mm) on pupil coordinates (u, v)."""

        keep = self._mask(exclude_tilt=remove_tilt, exclude_defocus=refocus)
        result = np.zeros_like(np.asarray(u, dtype=float))
        for mode, coefficient, kept in zip(self.modes, self.coefficients_mm, keep):
            if not kept:
                continue
            result += coefficient * zernike(mode.n, mode.m, u, v)
        return result

    @property
    def records(self) -> list[dict]:
        rows = []
        for mode, coefficient in zip(self.modes, self.coefficients_mm):
            rows.append(
                {
                    "ansi_index": mode.ansi,
                    "radial_order_n": mode.n,
                    "azimuthal_frequency_m": mode.m,
                    "name": mode.name,
                    "coefficient_nm_rms": coefficient * 1e6,
                    "coefficient_waves_rms": (
                        coefficient / self.wavelength_mm if self.wavelength_mm else np.nan
                    ),
                }
            )
        return rows

    def export_csv(self, path) -> None:
        import csv

        rows = self.records
        with open(path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def fit_transverse(
    pupil: PupilTrace,
    *,
    na_image: float,
    n_image: float,
    wavelength_mm: float | None = None,
    max_order: int = 6,
    keep_radius: float = 1.01,
) -> ZernikeExpansion:
    """Fit Zernike gradients to transverse ray aberrations.

    Pupil coordinates are normalized image-space direction cosines,
    ``u = n_image (d - d_chief) / NA``; the fit solves
    ``grad_q W = -NA * (r' - centroid)`` by weighted least squares.
    """

    points = pupil.image_points[:, :2]
    directions = pupil.directions
    weights = pupil.weights.copy()
    chief_direction = pupil.chief.direction

    u = n_image * (directions[:, 0] - chief_direction[0]) / na_image
    v = n_image * (directions[:, 1] - chief_direction[1]) / na_image
    keep = np.hypot(u, v) <= keep_radius
    u, v, points, weights = u[keep], v[keep], points[keep], weights[keep]
    weights /= weights.sum()
    centroid = np.sum(points * weights[:, None], axis=0)
    transverse = points - centroid

    modes = zernike_modes(max_order)
    step = 1e-5
    derivative_u = np.column_stack(
        [
            (zernike(mode.n, mode.m, u + step, v) - zernike(mode.n, mode.m, u - step, v))
            / (2 * step)
            for mode in modes
        ]
    )
    derivative_v = np.column_stack(
        [
            (zernike(mode.n, mode.m, u, v + step) - zernike(mode.n, mode.m, u, v - step))
            / (2 * step)
            for mode in modes
        ]
    )
    matrix = np.vstack([derivative_u, derivative_v])
    target = np.r_[-na_image * transverse[:, 0], -na_image * transverse[:, 1]]
    fit_weights = np.r_[weights, weights]
    coefficients = np.linalg.lstsq(
        matrix * np.sqrt(fit_weights[:, None]), target * np.sqrt(fit_weights), rcond=None
    )[0]
    gradient_residual = target - matrix @ coefficients

    return ZernikeExpansion(
        modes=modes,
        coefficients_mm=coefficients,
        wavelength_mm=wavelength_mm,
        u=u,
        v=v,
        weights=weights,
        gradient_fit_residual_um=float(
            np.sqrt(np.average(gradient_residual**2, weights=fit_weights)) * 1e3
        ),
    )


def fit_opd(
    u: np.ndarray,
    v: np.ndarray,
    opd: np.ndarray,
    *,
    max_order: int = 6,
    weights: np.ndarray | None = None,
    include_piston: bool = True,
    wavelength_mm: float | None = None,
) -> ZernikeExpansion:
    """Directly fit sampled OPD values (same length unit as *opd*)."""

    modes = zernike_modes(max_order, include_piston=include_piston)
    matrix = np.column_stack([zernike(mode.n, mode.m, u, v) for mode in modes])
    if weights is None:
        weights = np.ones_like(np.asarray(opd, dtype=float))
    weights = weights / weights.sum()
    coefficients = np.linalg.lstsq(
        matrix * np.sqrt(weights[:, None]), np.asarray(opd) * np.sqrt(weights), rcond=None
    )[0]
    return ZernikeExpansion(
        modes=modes,
        coefficients_mm=coefficients,
        wavelength_mm=wavelength_mm,
        u=np.asarray(u),
        v=np.asarray(v),
        weights=weights,
    )


__all__ = [
    "Mode",
    "ZERNIKE_NAMES",
    "zernike_modes",
    "zernike",
    "ZernikeExpansion",
    "fit_transverse",
    "fit_opd",
]
