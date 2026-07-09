"""Shared surface-sampling helpers for sequential-system 2-D drawing.

Both the matplotlib layout figure (:mod:`raytracer.viz.plots`) and the
OpenGL meridional viewers under ``examples/lithography`` need to sample a
``SurfaceRow``'s sag profile into an open ``(z, h)`` curve, and to build a
lens element's closed outline / triangulated fill mesh from a front/back
surface pair. This was independently reimplemented in three places; these
functions are the single source so a fix (sampling density, semidiameter
fallback) only needs to happen once.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SEMIDIAMETER = 50.0


def surface_semidiameter(row, fallback: float = DEFAULT_SEMIDIAMETER) -> float:
    """The row's clear-aperture semidiameter, or *fallback* if unset."""

    return row.semidiameter if row.semidiameter is not None else fallback


def sample_profile_curve(
    system, index: int, *, semidiameter: float | None = None, samples: int = 240
) -> tuple[np.ndarray, np.ndarray]:
    """Open meridional ``(z, h)`` curve of ``system.rows[index]``, ``h`` in ``[-semi, semi]``."""

    row = system.rows[index]
    semi = surface_semidiameter(row) if semidiameter is None else semidiameter
    h = np.linspace(-semi, semi, samples)
    z = system.vertices[index] + row.profile.sag(np.abs(h))
    return z, h


def lens_outline(system, i: int, j: int, *, samples: int = 220) -> np.ndarray:
    """Closed ``(z, h)`` polygon of the element spanning front surface *i* and back surface *j*."""

    semi = max(surface_semidiameter(system.rows[i]), surface_semidiameter(system.rows[j]))
    zf, h = sample_profile_curve(system, i, semidiameter=semi, samples=samples)
    zb, _ = sample_profile_curve(system, j, semidiameter=semi, samples=samples)
    front = np.column_stack([zf, h])
    back = np.column_stack([zb, h])[::-1]
    return np.vstack([front, back, front[:1]])  # closed loop


def lens_fill_mesh(system, i: int, j: int, *, samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Triangulated strip ``(verts, faces)`` of the element body between faces *i* and *j*."""

    semi = max(surface_semidiameter(system.rows[i]), surface_semidiameter(system.rows[j]))
    zf, h = sample_profile_curve(system, i, semidiameter=semi, samples=samples)
    zb, _ = sample_profile_curve(system, j, semidiameter=semi, samples=samples)
    verts = np.empty((2 * samples, 2))
    verts[0::2] = np.column_stack([zf, h])  # front-face vertices (even)
    verts[1::2] = np.column_stack([zb, h])  # back-face vertices (odd)
    k = np.arange(samples - 1)
    faces = np.empty((2 * (samples - 1), 3), dtype=np.uint32)
    faces[0::2] = np.column_stack([2 * k, 2 * k + 1, 2 * k + 2])
    faces[1::2] = np.column_stack([2 * k + 1, 2 * k + 3, 2 * k + 2])
    return verts, faces


__all__ = ["surface_semidiameter", "sample_profile_curve", "lens_outline", "lens_fill_mesh"]
