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


def mirror_arcs_from_paths(
    system,
    paths,
    *,
    pad: float = 0.06,
    samples: int = 200,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Per-mirror meridional arcs spanning the heights the rays actually hit.

    For off-axis systems (ring fields, folded paths) a mirror's *used*
    sub-aperture is displaced from the axis, so drawing the parent surface
    as a symmetric on-axis cap puts the drawn curve away from the real
    reflection points. This derives each mirror's arc from the traced
    *paths* themselves (``keep_paths`` output, shape ``(N, n_surfaces+2, 3)``;
    column ``i+1`` is the hit point on surface ``i``), extended by *pad*
    fractionally beyond the hit envelope.

    Returns ``[(mirror_number, z, y), ...]`` ready for ``ax.plot(z, y)``.
    """

    paths = np.asarray(paths, dtype=float)
    arcs: list[tuple[int, np.ndarray, np.ndarray]] = []
    for count, i in enumerate(system.mirror_indices, 1):
        y_hits = paths[:, i + 1, 1]
        y_hits = y_hits[np.isfinite(y_hits)]
        if y_hits.size == 0:
            continue
        y_lo, y_hi = float(y_hits.min()), float(y_hits.max())
        margin = pad * max(y_hi - y_lo, 1.0)
        h = np.linspace(y_lo - margin, y_hi + margin, samples)
        z = system.vertices[i] + system.rows[i].profile.sag(np.abs(h))
        arcs.append((count, z, h))
    return arcs


def beam_foci_from_paths(
    paths,
    *,
    pinch_ratio: float = 0.2,
    min_rays: int = 3,
) -> list[tuple[float, float, float, int, bool]]:
    """Where a traced beam actually focuses, segment by segment.

    Between consecutive surfaces every ray is a straight line; the beam's
    focus on that leg is the least-squares convergence point of those lines
    (the parabola ``var(a)·z² + 2·cov(a,b)·z + var(b)`` in the meridional
    slopes/intercepts has its minimum at the pinch). An *intermediate* leg
    counts as a focus only when the pinch is genuine — residual blur below
    ``pinch_ratio`` of the beam's spread at both ends, a constriction of
    5× or better at the default — so mere narrowing is not decorated. An
    aberrated intermediate image reports where the traced beam *actually*
    pinches, which need not be the design's nominal plane. The *final* leg (to the image plane) is where the
    system claims to focus, and is always reported when the beam converges
    to a real point on or beyond the last surface; a collimated exit
    (afocal system) or virtual focus reports nothing.

    Returns ``[(z, y, blur_rms, leg_index, is_final), ...]`` with *paths*
    of shape ``(rays, points, 3)`` from ``keep_path`` traces.
    """

    paths = np.asarray(paths, dtype=float)
    foci: list[tuple[float, float, float, int, bool]] = []
    n_legs = paths.shape[1] - 1
    for leg in range(1, n_legs):
        p0, p1 = paths[:, leg, :], paths[:, leg + 1, :]
        dz = p1[:, 2] - p0[:, 2]
        keep = np.isfinite(dz) & (np.abs(dz) > 1e-12)
        keep &= np.isfinite(p0[:, 1]) & np.isfinite(p1[:, 1])
        if keep.sum() < min_rays:
            continue
        slope = (p1[keep, 1] - p0[keep, 1]) / dz[keep]
        intercept = p0[keep, 1] - slope * p0[keep, 2]
        var_a = float(np.var(slope))
        if var_a < 1e-18:
            continue  # collimated leg: no focus
        z_star = -float(np.cov(slope, intercept, bias=True)[0, 1]) / var_a
        y_star = float(np.mean(slope * z_star + intercept))
        blur = float(np.std(slope * z_star + intercept))
        is_final = leg == n_legs - 1
        z_lo = float(np.minimum(p0[keep, 2], p1[keep, 2]).min())
        z_hi = float(np.maximum(p0[keep, 2], p1[keep, 2]).max())
        if is_final:
            if z_star < z_lo:  # virtual focus behind the last surface
                continue
        else:
            span = z_hi - z_lo
            inside = z_lo + 0.01 * span <= z_star <= z_hi - 0.01 * span
            spread = min(float(np.std(p0[keep, 1])), float(np.std(p1[keep, 1])))
            if not inside or blur > pinch_ratio * max(spread, 1e-12):
                continue
        foci.append((z_star, y_star, blur, leg, is_final))
    return foci


__all__ = [
    "surface_semidiameter",
    "sample_profile_curve",
    "lens_outline",
    "lens_fill_mesh",
    "mirror_arcs_from_paths",
    "beam_foci_from_paths",
]
