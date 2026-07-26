"""Snell's law, reflection, and Fresnel coefficients: the physical laws of
light hitting an interface.

All functions are dimension-agnostic: directions and normals may be 2-D or
3-D unit vectors. This is the physics "crown" of the package — everything
else (geometry, the sequential/non-sequential engines, aberration analysis)
exists to set up an interface for these laws to act on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..math.vectors import normalize


@dataclass(frozen=True)
class SnellResult:
    """Summary of Snell's law for a single ray/surface interaction."""

    eta: float
    cos_incident: float
    cos_transmitted: Optional[float]

    @property
    def total_internal_reflection(self) -> bool:
        return self.cos_transmitted is None


def snell(direction: np.ndarray, normal: np.ndarray, n1: float, n2: float) -> SnellResult:
    """Return Snell's-law parameters for an interface between ``n1`` and ``n2``."""

    d = normalize(direction)
    n = normalize(normal)
    eta = float(n1) / float(n2)
    cos_i = -np.dot(d, n)
    k = 1.0 - eta**2 * (1.0 - cos_i**2)
    if k < 0.0:
        return SnellResult(eta=eta, cos_incident=cos_i, cos_transmitted=None)
    return SnellResult(eta=eta, cos_incident=cos_i, cos_transmitted=float(np.sqrt(k)))


def reflect(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Return the reflected unit direction for an incident ray."""

    d = normalize(direction)
    n = normalize(normal)
    return normalize(d - 2.0 * np.dot(d, n) * n)


def refract(direction: np.ndarray, normal: np.ndarray, n1: float, n2: float) -> np.ndarray | None:
    """Return the refracted unit direction, or ``None`` if TIR occurs."""

    result = snell(direction, normal, n1, n2)
    if result.total_internal_reflection:
        return None
    d = normalize(direction)
    n = normalize(normal)
    cos_i = result.cos_incident
    cos_t = result.cos_transmitted if result.cos_transmitted is not None else 0.0
    eta = result.eta
    return normalize(eta * d + (eta * cos_i - cos_t) * n)


@dataclass(frozen=True)
class FresnelCoefficients:
    """Power reflectance/transmittance for s- and p-polarisation."""

    r_s: float
    r_p: float
    t_s: float
    t_p: float

    @property
    def reflectance(self) -> float:
        """Unpolarised power reflectance."""

        return 0.5 * (self.r_s + self.r_p)

    @property
    def transmittance(self) -> float:
        """Unpolarised power transmittance."""

        return 0.5 * (self.t_s + self.t_p)


def fresnel_coefficients(
    direction: np.ndarray, normal: np.ndarray, n1: float, n2: float
) -> FresnelCoefficients:
    """Return Fresnel power coefficients for an interface between ``n1`` and ``n2``.

    On total internal reflection the reflectances are 1 and transmittances 0.
    """

    result = snell(direction, normal, n1, n2)
    cos_i = abs(result.cos_incident)
    if result.total_internal_reflection:
        return FresnelCoefficients(r_s=1.0, r_p=1.0, t_s=0.0, t_p=0.0)
    cos_t = result.cos_transmitted or 0.0

    rs_amp = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    rp_amp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    r_s = rs_amp**2
    r_p = rp_amp**2
    return FresnelCoefficients(r_s=r_s, r_p=r_p, t_s=1.0 - r_s, t_p=1.0 - r_p)


__all__ = [
    "SnellResult",
    "FresnelCoefficients",
    "snell",
    "reflect",
    "refract",
    "fresnel_coefficients",
]
