"""Where a ray meets a surface: one equation, solved for many rays at once.

A ray is the set of points ``o + t d``. A surface written as the graph
``z = s(x, y)`` in its local frame is the zero set of
``f(p) = z - s(x, y)``, whose gradient ``(-∂s/∂x, -∂s/∂y, 1)`` is at once
the Newton derivative and the surface normal. Two ways to solve
``f(o + t d) = 0`` live here:

- :func:`newton_graph` — Newton from the crossing of the vertex plane
  ``z = 0``. Works for every graph surface.
- :func:`solve_quadric` — closed form for a surface whose ``f`` is a
  quadric (the conics), with the root chosen on the sheet that holds the
  vertex.

Both take a mode. ``"nearest"`` keeps the root closest to the vertex-plane
crossing, whatever its sign: the sequential convention, which lets a
prescription place a surface *behind* the previous one (a virtual stop, a
negative thickness). ``"forward"`` keeps the first root strictly ahead of
the ray: the physical convention of non-sequential tracing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from .surfaces.base import Surface


class SurfaceHits(NamedTuple):
    """Ray parameters of the hits, and why the others failed.

    ``t`` is meaningful only where no failure flag is set. ``parallel``: the
    ray runs inside the vertex plane, so Newton has no seed. ``diverged``:
    the solve did not converge. ``missed``: the ray does not meet the surface
    (no real root on the vertex sheet, or none ahead of the ray in
    ``"forward"`` mode).
    """

    t: np.ndarray
    parallel: np.ndarray
    diverged: np.ndarray
    missed: np.ndarray

    @property
    def ok(self) -> np.ndarray:
        return ~(self.parallel | self.diverged | self.missed)


def newton_graph(
    surface: "Surface",
    origins: np.ndarray,
    directions: np.ndarray,
    alive: np.ndarray,
    *,
    mode: str = "nearest",
    t_min: float = 1e-9,
    max_newton: int = 15,
    newton_tol: float = 2e-11,
) -> SurfaceHits:
    """Newton on ``f(o + t d) = 0`` from the vertex-plane crossing, all rays at once."""

    p, d = origins, directions
    n_rays = p.shape[0]
    dz = d[:, 2]
    parallel = alive & (np.abs(dz) < 1e-14)
    solving = alive & ~parallel
    diverged = np.zeros(n_rays, dtype=bool)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        t = np.where(solving, -p[:, 2] / np.where(dz == 0.0, 1.0, dz), 0.0)
        step = np.zeros(n_rays)
        for _ in range(max_newton):
            q = p + t[:, None] * d
            sag, dsdx, dsdy = surface.sag_gradient(q[:, 0], q[:, 1])
            residual = q[:, 2] - sag
            derivative = dz - dsdx * d[:, 0] - dsdy * d[:, 1]
            stuck = solving & (np.abs(derivative) < 1e-14)
            if stuck.any():
                diverged |= stuck
                solving &= ~stuck
            step = np.where(solving, residual / np.where(derivative == 0.0, 1.0, derivative), 0.0)
            t = t - step
            if not np.any(np.abs(step[solving]) >= newton_tol):
                break

    diverged |= solving & (~np.isfinite(t) | (np.abs(step) > 1e-6))
    missed = np.zeros(n_rays, dtype=bool)
    if mode == "forward":
        missed = alive & ~parallel & ~diverged & (t <= t_min)
    return SurfaceHits(t, parallel, diverged, missed)


def solve_quadric(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    on_sheet,
    origins: np.ndarray,
    directions: np.ndarray,
    alive: np.ndarray,
    *,
    mode: str = "nearest",
    t_min: float = 1e-9,
) -> SurfaceHits:
    """Roots of ``a t² + b t + c = 0``, the right one chosen per ray.

    ``on_sheet(points) -> bool`` says whether a point lies on the sheet of
    the quadric that holds the vertex (a conic's other sheet, or the far
    half of a sphere, is not the optical surface).
    """

    n_rays = origins.shape[0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        disc = b * b - 4.0 * a * c
        linear = np.abs(a) < 1e-15 * np.maximum(np.abs(b), 1e-300)
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        # Numerically stable pair: q = -(b + sign(b) sqrt(disc)) / 2.
        q = -0.5 * (b + np.copysign(sqrt_disc, b))
        t1 = np.where(linear, -c / np.where(b == 0.0, 1.0, b), q / np.where(a == 0.0, 1.0, a))
        t2 = np.where(linear, np.inf, c / np.where(q == 0.0, 1.0, q))
    real = alive & (linear | (disc >= 0.0))
    roots = np.stack([t1, t2], axis=1)
    valid = np.stack(
        [
            real & np.isfinite(t1) & on_sheet(origins + t1[:, None] * directions),
            real & np.isfinite(t2) & on_sheet(origins + t2[:, None] * directions),
        ],
        axis=1,
    )
    if mode == "forward":
        valid &= roots > t_min
        key = np.where(valid, roots, np.inf)
    else:
        dz = directions[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            seed = np.where(np.abs(dz) < 1e-300, 0.0, -origins[:, 2] / np.where(dz == 0.0, 1.0, dz))
        key = np.where(valid, np.abs(roots - seed[:, None]), np.inf)
    choice = np.argmin(key, axis=1)
    t = roots[np.arange(n_rays), choice]
    missed = alive & ~valid[np.arange(n_rays), choice]
    t = np.where(missed, np.nan, t)
    zeros = np.zeros(n_rays, dtype=bool)
    return SurfaceHits(t, zeros, zeros.copy(), missed)


__all__ = ["SurfaceHits", "newton_graph", "solve_quadric"]
