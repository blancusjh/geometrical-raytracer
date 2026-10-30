"""Vector helpers shared by every layer above."""

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


def direction_from_slopes(sx: float, sy: float) -> np.ndarray:
    """Unit direction from transverse slopes ``(dx/dz, dy/dz)``."""

    return np.array([sx, sy, 1.0]) / np.sqrt(1.0 + sx * sx + sy * sy)


def direction_from_angles(x_deg: float = 0.0, y_deg: float = 0.0) -> np.ndarray:
    """Unit direction tilted from +z by field angles (degrees) in the x-z and y-z planes.

    The angles are those of the projections: ``tan(x_deg) = dx/dz`` and
    ``tan(y_deg) = dy/dz`` — the usual definition of a field angle.
    """

    return direction_from_slopes(np.tan(np.radians(x_deg)), np.tan(np.radians(y_deg)))


__all__ = [
    "EPS",
    "as_vector",
    "normalize",
    "normalize_rows",
    "direction_from_slopes",
    "direction_from_angles",
]
