"""Quantitative stigmatism verification for a traced pupil bundle.

Packages the two existing convergence metrics — geometric spot size
(:func:`raytracer.analysis.spots.spot_data`) and wavefront OPD
(:func:`raytracer.analysis.wavefront.exit_pupil_wavefront`) — into a single
printable/assertable verdict, so an example can *confirm* "these rays
converge to a point" instead of only asserting it in a comment or a print
banner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sequential.fields import PupilTrace
from .spots import SpotData, spot_data
from .wavefront import WavefrontSamples, exit_pupil_wavefront


@dataclass
class StigmatismReport:
    """RMS spot size, and (if a wavelength/NA were given) peak OPD, for one field."""

    spot: SpotData
    rms_radius_um: float
    wavefront: WavefrontSamples | None = None
    max_opd_waves: float | None = None

    def is_stigmatic(self, *, tol_um: float = 0.05, tol_waves: float = 0.05) -> bool:
        """True if the RMS spot radius (and OPD, when available) are within tolerance."""

        ok = self.rms_radius_um < tol_um
        if self.max_opd_waves is not None:
            ok = ok and self.max_opd_waves < tol_waves
        return ok

    def summary(self, *, tol_um: float = 0.05, tol_waves: float = 0.05) -> str:
        lines = [f"RMS spot radius: {self.rms_radius_um:.4g} um"]
        if self.max_opd_waves is not None:
            lines.append(f"Max |OPD|: {self.max_opd_waves:.4g} waves")
        verdict = "stigmatic" if self.is_stigmatic(tol_um=tol_um, tol_waves=tol_waves) else "NOT stigmatic"
        lines.append(f"-> {verdict} (tol: {tol_um:.3g} um" + (
            f", {tol_waves:.3g} waves)" if self.max_opd_waves is not None else ")"
        ))
        return "\n".join(lines)


def point_line_distances(origins: np.ndarray, directions: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Perpendicular distance from *target* to each ray's infinite 2-D line.

    Verifies convergence to a point without needing a physical screen
    intercept — useful for the 2-D non-sequential engine, where a screen
    placed near the convergence point can occlude direct (unreflected/
    unrefracted) rays before they even reach the stigmatic surface.
    """

    origins = np.atleast_2d(np.asarray(origins, dtype=float))
    directions = np.atleast_2d(np.asarray(directions, dtype=float))
    target = np.asarray(target, dtype=float)
    to_target = target - origins
    cross = to_target[:, 0] * directions[:, 1] - to_target[:, 1] * directions[:, 0]
    norm = np.linalg.norm(directions, axis=1)
    return np.abs(cross) / norm


def rays_by_generation(tree, generation: int) -> tuple[np.ndarray, np.ndarray]:
    """Origins and directions of every node at *generation* in a 2-D ``RayTree``."""

    origins, directions = [], []
    for node in tree.nodes():
        if node.generation == generation:
            origins.append(node.ray.origin)
            directions.append(node.ray.direction)
    return np.asarray(origins, dtype=float), np.asarray(directions, dtype=float)


def stigmatism_report(
    pupil: PupilTrace,
    *,
    wavelength_mm: float | None = None,
    na_image: float | None = None,
    reference_radius: float = 500.0,
) -> StigmatismReport:
    """Compute the spot-size (and, given a wavelength/NA, OPD) convergence metrics.

    Pass ``wavelength_mm`` and ``na_image`` to also compute the wavefront OPD
    (reusing :func:`exit_pupil_wavefront`); otherwise only the geometric spot
    RMS is reported.
    """

    spot = spot_data(pupil)
    wavefront = None
    max_opd_waves = None
    if wavelength_mm is not None and na_image is not None:
        wavefront = exit_pupil_wavefront(
            pupil, na_image=na_image, wavelength_mm=wavelength_mm,
            reference_radius=reference_radius,
        )
        max_opd_waves = float(np.max(np.abs(wavefront.opd_waves)))
    return StigmatismReport(
        spot=spot,
        rms_radius_um=spot.rms_radius_um,
        wavefront=wavefront,
        max_opd_waves=max_opd_waves,
    )


__all__ = [
    "StigmatismReport",
    "stigmatism_report",
    "point_line_distances",
    "rays_by_generation",
]
