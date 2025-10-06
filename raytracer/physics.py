"""Optical laws for ray-surface interactions."""

from __future__ import annotations

import numpy as np

from .rays import normalize


def reflect(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Return the reflected unit vector for an incident ray."""
    d = normalize(direction)
    n = normalize(normal)
    return normalize(d - 2.0 * np.dot(d, n) * n)


def refract(direction: np.ndarray, normal: np.ndarray, n1: float, n2: float) -> np.ndarray | None:
    """Snell's law refraction; returns None on total internal reflection."""
    d = normalize(direction)
    n = normalize(normal)
    eta = n1 / n2
    cos_i = -np.dot(d, n)
    k = 1.0 - eta ** 2 * (1.0 - cos_i ** 2)
    if k < 0.0:
        return None
    return normalize(eta * d + (eta * cos_i - np.sqrt(k)) * n)
