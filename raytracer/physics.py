"""Optical laws for ray-surface interactions (dimension-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .rays import normalize


@dataclass(frozen=True)
class SnellResult:
    """Container describing the outcome of applying Snell's law."""

    eta: float
    cos_incident: float
    cos_transmitted: Optional[float]

    @property
    def total_internal_reflection(self) -> bool:
        return self.cos_transmitted is None


def snell(direction: np.ndarray, normal: np.ndarray, n1: float, n2: float) -> SnellResult:
    """Return cosines for incidence/transmission and the relative index."""

    d = normalize(direction)
    n = normalize(normal)
    eta = float(n1) / float(n2)
    cos_i = -np.dot(d, n)
    k = 1.0 - eta ** 2 * (1.0 - cos_i ** 2)
    if k < 0.0:
        return SnellResult(eta=eta, cos_incident=cos_i, cos_transmitted=None)
    cos_t = np.sqrt(k)
    return SnellResult(eta=eta, cos_incident=cos_i, cos_transmitted=cos_t)


def reflect(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Return the reflected unit vector for an incident ray."""

    d = normalize(direction)
    n = normalize(normal)
    return normalize(d - 2.0 * np.dot(d, n) * n)


def refract(direction: np.ndarray, normal: np.ndarray, n1: float, n2: float) -> np.ndarray | None:
    """Return refracted unit vector or ``None`` on total internal reflection."""

    result = snell(direction, normal, n1, n2)
    if result.total_internal_reflection:
        return None
    d = normalize(direction)
    n = normalize(normal)
    cos_i = result.cos_incident
    cos_t = result.cos_transmitted if result.cos_transmitted is not None else 0.0
    eta = result.eta
    return normalize(eta * d + (eta * cos_i - cos_t) * n)


def fresnel_coefficients(result: SnellResult, n1: float, n2: float) -> tuple[float, float]:
    """Return the Fresnel reflection coefficients (Rs, Rp).

    The coefficients are returned for s- and p-polarised light respectively.
    When total internal reflection occurs both values are ``1.0``.
    """

    if result.total_internal_reflection:
        return 1.0, 1.0

    cos_i = result.cos_incident
    cos_t = result.cos_transmitted if result.cos_transmitted is not None else 0.0
    n1 = float(n1)
    n2 = float(n2)

    rs_num = n1 * cos_i - n2 * cos_t
    rs_den = n1 * cos_i + n2 * cos_t
    rp_num = n2 * cos_i - n1 * cos_t
    rp_den = n2 * cos_i + n1 * cos_t

    rs = (rs_num / rs_den) ** 2 if rs_den != 0.0 else 1.0
    rp = (rp_num / rp_den) ** 2 if rp_den != 0.0 else 1.0
    return rs, rp


def fresnel_unpolarised(result: SnellResult, n1: float, n2: float) -> float:
    """Return the unpolarised Fresnel reflectance."""

    rs, rp = fresnel_coefficients(result, n1, n2)
    return 0.5 * (rs + rp)
