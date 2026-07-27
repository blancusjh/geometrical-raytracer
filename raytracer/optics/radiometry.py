"""Fresnel coefficients: the energetic (power) laws at an interface.

Distinct in nature from the direction laws in :mod:`raytracer.optics.laws`
(reflect/refract answer "which way does the ray go"; this answers "how much
power does it carry there").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .laws import _refraction_cosines


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

    _, cos_incident, cos_t = _refraction_cosines(direction, normal, n1, n2)
    cos_i = abs(cos_incident)
    if cos_t is None:
        return FresnelCoefficients(r_s=1.0, r_p=1.0, t_s=0.0, t_p=0.0)

    rs_amp = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    rp_amp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    r_s = rs_amp**2
    r_p = rp_amp**2
    return FresnelCoefficients(r_s=r_s, r_p=r_p, t_s=1.0 - r_s, t_p=1.0 - r_p)


__all__ = ["FresnelCoefficients", "fresnel_coefficients"]
