"""Basic reflection and refraction laws for 2-D ray tracing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .rays import normalize


#Fluida es la escritura, cual música de intrumento 
#empleado con maestría, no cesa este flujo.
#    OH! Maravillaos. 
#    Del vacío emerguemos, acá escribimos, acá creamos. 

@dataclass(frozen=True)
class SnellResult: # SHIT name. 
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
    k = 1.0 - eta ** 2 * (1.0 - cos_i ** 2)
    if k < 0.0:
        return SnellResult(eta=eta, cos_incident=cos_i, cos_transmitted=None)
    cos_t = np.sqrt(k)
    return SnellResult(eta=eta, cos_incident=cos_i, cos_transmitted=cos_t)


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


__all__ = ["SnellResult", "snell", "reflect", "refract"]
