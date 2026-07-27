"""The laws of geometrical optics: reflection and refraction.

Given an incident direction, a surface normal, and the indices on either
side, these give the direction the light leaves in. They are the crown of
the package — surfaces, sources, and propagation all exist to set up an
interface for these laws to act on.

All functions are dimension-agnostic: directions and normals may be 2-D or
3-D unit vectors. How much *power* goes each way is a different question,
answered by :mod:`raytracer.optics.radiometry`.
"""

from __future__ import annotations

import numpy as np

from ..math.vectors import normalize


def _refraction_cosines(
    direction: np.ndarray, normal: np.ndarray, n1: float, n2: float
) -> tuple[float, float, float | None]:
    """Return ``(eta, cos_incident, cos_transmitted)`` for a ray crossing an
    interface from index *n1* to *n2*; ``cos_transmitted`` is ``None`` under
    total internal reflection. Shared by :func:`refract` and
    :func:`~raytracer.optics.radiometry.fresnel_coefficients`, which both
    need the same angle relationship at the interface."""

    d = normalize(direction)
    n = normalize(normal)
    eta = float(n1) / float(n2)
    cos_i = -np.dot(d, n)
    k = 1.0 - eta**2 * (1.0 - cos_i**2)
    if k < 0.0:
        return eta, cos_i, None
    return eta, cos_i, float(np.sqrt(k))


def reflect(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Return the reflected unit direction for an incident ray."""

    d = normalize(direction)
    n = normalize(normal)
    return normalize(d - 2.0 * np.dot(d, n) * n)


def refract(direction: np.ndarray, normal: np.ndarray, n1: float, n2: float) -> np.ndarray | None:
    """Return the refracted unit direction, or ``None`` if TIR occurs."""

    eta, cos_i, cos_t = _refraction_cosines(direction, normal, n1, n2)
    if cos_t is None:
        return None
    d = normalize(direction)
    n = normalize(normal)
    return normalize(eta * d + (eta * cos_i - cos_t) * n)


def reflect_batch(directions: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Vectorized :func:`reflect` over ``(N, dim)`` arrays of unit directions/normals."""

    cos_i = -np.einsum("ij,ij->i", directions, normals)
    out = directions + 2.0 * cos_i[:, None] * normals
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


def refract_batch(
    directions: np.ndarray, normals: np.ndarray, n1: np.ndarray | float, n2: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized :func:`refract` over ``(N, dim)`` arrays; *n1*/*n2* are
    scalars or ``(N,)`` arrays. Returns ``(directions_out, tir)`` where
    ``tir`` is an ``(N,)`` boolean mask of rays that totally internally
    reflect (``directions_out`` is meaningless there; the caller decides
    what a TIR ray does next)."""

    n_rays = directions.shape[0]
    cos_i = -np.einsum("ij,ij->i", directions, normals)
    eta = np.broadcast_to(np.asarray(n1, dtype=float) / np.asarray(n2, dtype=float), (n_rays,))
    radicand = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    tir = radicand < 0.0
    root = np.sqrt(np.maximum(radicand, 0.0))
    out = eta[:, None] * directions + (eta * cos_i - root)[:, None] * normals
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    out /= np.where(norm == 0.0, 1.0, norm)
    return out, tir


__all__ = ["reflect", "refract", "reflect_batch", "refract_batch"]
