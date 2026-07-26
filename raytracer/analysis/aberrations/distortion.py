"""Distortion grid: trace a square object-space grid through the system.

This is the classic optical-design-software distortion display — draw a
regular grid of field points, trace each one's chief ray, and see the grid
warp at the image plane (barrel/pincushion) instead of reading a single
1-D distortion-vs-field curve. It uses the general 2-D chief-ray solver
(:func:`raytracer.sequential.fields.chief_ray_slopes`), not the
rotationally-symmetric 1-D shortcut, so it is correct for a genuinely
off-axis (x, y) field point in any system this tracer can trace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...sequential.fields import FieldPoint, chief_ray_slopes, trace_from_object
from ...sequential.trace import SequentialTracer


@dataclass
class DistortionGrid:
    """Object- and image-space square grids, plus the ideal linear map.

    ``object_points``, ``ideal_points``, ``actual_points`` are ``(n, n, 2)``
    arrays; ``valid`` is the matching ``(n, n)`` boolean mask. Invalid
    (vignetted) grid points hold ``NaN`` in ``actual_points`` rather than
    being dropped, so the array shape stays a regular grid that a plot can
    still draw lines across (skipping only the missing points/edges).
    """

    object_points: np.ndarray
    ideal_points: np.ndarray
    actual_points: np.ndarray
    valid: np.ndarray
    magnification: float

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.mean())


def distortion_grid(
    tracer: SequentialTracer,
    *,
    magnification: float,
    half_field: float | tuple[float, float],
    center: tuple[float, float] = (0.0, 0.0),
    n: int = 11,
    stop_index: int | None = None,
) -> DistortionGrid:
    """Trace an ``n x n`` object-space grid spanning ``center +/- half_field``.

    ``half_field`` is either a single half-extent used for both axes, or an
    ``(half_x, half_y)`` pair. ``center`` shifts the whole grid in object
    space — the default ``(0, 0)`` suits an on-axis field, but a ring-field
    system (e.g. an annular-field EUV objective, whose usable field is an
    off-axis band, not a disk around the origin) needs a grid centered on
    its actual working field instead; the ideal map stays the single linear
    ``magnification * object_point`` regardless of where the grid sits,
    since that map always passes through the origin.

    Each grid point's chief ray is solved with :func:`chief_ray_slopes`,
    warm-started from an already-solved neighbor (left, then above) so the
    solve stays well-conditioned near the edge of the field instead of
    restarting from the paraxial guess every time. Points whose chief ray
    vignettes (at the stop or anywhere downstream) are left invalid rather
    than raising, since a real field of view is rarely a perfect rectangle
    — check ``valid``/``valid_fraction`` and report how many points were
    dropped rather than assuming full coverage.
    """

    if isinstance(half_field, (int, float)):
        half_x = half_y = float(half_field)
    else:
        half_x, half_y = half_field
    center_x, center_y = center

    xs = center_x + np.linspace(-half_x, half_x, n)
    ys = center_y + np.linspace(-half_y, half_y, n)
    xx, yy = np.meshgrid(xs, ys)

    object_points = np.stack([xx, yy], axis=-1)
    ideal_points = object_points * magnification
    actual_points = np.full((n, n, 2), np.nan)
    valid = np.zeros((n, n), dtype=bool)

    guesses: dict[tuple[int, int], tuple[float, float]] = {}
    for i in range(n):
        for j in range(n):
            field = FieldPoint(x=float(xx[i, j]), y=float(yy[i, j]))
            # The point's own paraxial estimate is tried first — a neighbor's
            # *solved* slope looks like a good warm start, but for a system
            # where the chief-ray direction is this sensitive to the field
            # point (a huge finite-object stand-in for infinity threading a
            # tiny front aperture), even one grid step away it can already
            # miss the aperture outright, which a per-point paraxial estimate
            # does not. Neighbor solutions remain useful *fallbacks* for
            # systems where the paraxial line isn't as good an estimate.
            candidates = [None, guesses.get((i, j - 1)), guesses.get((i - 1, j))]
            for guess in candidates:
                try:
                    sx, sy = chief_ray_slopes(
                        tracer, field, stop_index=stop_index, initial_guess=guess
                    )
                    result = trace_from_object(tracer, (field.x, field.y), (sx, sy))
                    if not result.ok:
                        raise RuntimeError("chief ray vignetted downstream of the stop")
                except RuntimeError:
                    continue
                actual_points[i, j] = result.image_point[:2]
                valid[i, j] = True
                guesses[(i, j)] = (sx, sy)
                break

    return DistortionGrid(
        object_points=object_points,
        ideal_points=ideal_points,
        actual_points=actual_points,
        valid=valid,
        magnification=magnification,
    )


__all__ = ["DistortionGrid", "distortion_grid"]
