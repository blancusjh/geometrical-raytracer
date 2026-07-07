"""Dimension-agnostic vector helpers shared by both engines."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

EPS = 1e-9


def as_vector(vec: ArrayLike, *, dim: int | None = None) -> np.ndarray:
    """Return *vec* coerced to a 1-D float array, optionally checking its length."""

    arr = np.asarray(vec, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1-D vector, got shape {arr.shape}")
    if dim is not None and arr.shape[0] != dim:
        raise ValueError(f"Expected vector of length {dim}, got shape {arr.shape}")
    return arr


def normalize(vec: ArrayLike, eps: float = EPS) -> np.ndarray:
    """Return the unit direction associated with *vec*."""

    arr = np.asarray(vec, dtype=float)
    mag = np.linalg.norm(arr)
    if mag <= eps:
        raise ValueError("Cannot normalize a zero-length vector.")
    return arr / mag


def normalize_rows(vecs: ArrayLike, eps: float = EPS) -> np.ndarray:
    """Normalize each row of an (N, dim) array of vectors."""

    arr = np.asarray(vecs, dtype=float)
    mags = np.linalg.norm(arr, axis=-1, keepdims=True)
    if np.any(mags <= eps):
        raise ValueError("Cannot normalize zero-length vectors.")
    return arr / mags


def direction_from_angle(theta: float) -> np.ndarray:
    """Return a 2-D unit direction given a polar angle in radians."""

    theta = float(theta)
    return np.array([np.cos(theta), np.sin(theta)], dtype=float)


def perpendicular_2d(vec: ArrayLike) -> np.ndarray:
    """Return *vec* rotated by +90 degrees."""

    arr = as_vector(vec, dim=2)
    return np.array([-arr[1], arr[0]], dtype=float)


def rotation_2d(angle: float) -> np.ndarray:
    """Return the 2x2 rotation matrix for *angle* radians."""

    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)


__all__ = [
    "EPS",
    "as_vector",
    "normalize",
    "normalize_rows",
    "direction_from_angle",
    "perpendicular_2d",
    "rotation_2d",
]
