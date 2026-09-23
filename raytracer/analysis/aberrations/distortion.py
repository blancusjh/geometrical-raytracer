"""Chief-ray distortion with an explicit reference map.

The scalar sweep supports finite heights and true angular fields; its
reference is computed by field_geometry unless magnification is supplied.
The older square-grid function uses finite object heights and a specified
linear magnification. Both require centered systems; arbitrary placed-system
mapping needs a separately defined ideal map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...propagation.fields import FieldPoint, chief_ray_slopes, trace_from_object
from ...propagation.sequential import SequentialTracer


def _solve_chief_ray(
    tracer: SequentialTracer,
    field: FieldPoint,
    *,
    stop_index: int | None,
    fallback_guesses: tuple = (),
) -> tuple[float, float] | None:
    """Solve *field*'s chief ray, or return ``None`` if it vignettes.

    The point's own paraxial estimate is always tried first. A neighbouring
    field point's *solved* slope looks like the better warm start, but for a
    system whose chief-ray direction is very sensitive to field height — a
    huge finite-object stand-in for infinity threading a small front
    aperture — even one step away it can miss the aperture outright, where
    the per-point paraxial estimate lands close every time. Neighbour
    solutions stay useful as fallbacks for systems where the paraxial line
    is the poorer guess.
    """

    for guess in (None, *fallback_guesses):
        try:
            slopes = chief_ray_slopes(tracer, field, stop_index=stop_index, initial_guess=guess)
        except RuntimeError:
            continue
        if trace_from_object(tracer, (field.x, field.y), slopes).ok:
            return slopes
    return None


def chief_ray_distortion(
    tracer: SequentialTracer,
    fields,
    *,
    magnification: float | None = None,
    field_unit: str = "mm",
    stop_index: int | None = None,
    reference: str = "paraxial",
) -> list[dict]:
    """One-dimensional chief distortion with explicit finite/angular fields.

    Degrees mean a true parallel incident bundle, not a remote finite object.
    The default ideal map comes from ABCD at the current detector. A supplied
    magnification overrides that map for finite heights only. On-axis relative
    distortion is undefined (NaN). See distortion_map for centroid comparison.
    """
    from .field_geometry import distortion_map

    if field_unit not in ("mm", "deg"):
        raise ValueError("field_unit must be mm or deg")
    if magnification is not None and (field_unit != "mm" or not np.isfinite(magnification)):
        raise ValueError("magnification overrides require finite height fields")
    values = np.asarray(fields, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("fields must be a nonempty finite vector")
    points = [
        FieldPoint.angle(y_deg=value) if field_unit == "deg" else FieldPoint(y=value)
        for value in values
    ]
    report = distortion_map(tracer, points, stop_index=stop_index, reference=reference)
    rows = []
    for index, value in enumerate(values):
        ideal = report.ideal_xy_mm[index, 1] if magnification is None else magnification * value
        actual = report.chief_xy_mm[index, 1]
        rows.append(
            {
                "field": float(value),
                "field_unit": field_unit,
                "object_height_mm": float(value) if field_unit == "mm" else None,
                "paraxial_image_height_mm": float(ideal),
                "chief_image_height_mm": float(actual),
                "chief_ray_distortion_um": float((actual - ideal) * 1000),
                "chief_ray_relative_distortion_ppm": float((actual / ideal - 1) * 1e6)
                if ideal
                else np.nan,
                "chief_transmitted": bool(report.chief_transmitted[index]),
                "reference": "specified_magnification" if magnification is not None else reference,
            }
        )
    return rows


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

    @property
    def deviation_mm(self) -> np.ndarray:
        """``(n, n, 2)`` displacement of each traced point from its ideal one.

        ``NaN`` at vignetted points, inherited from ``actual_points``.
        """

        return self.actual_points - self.ideal_points

    @property
    def max_distortion_um(self) -> float:
        """Largest departure from the ideal map, over the valid points."""

        radius = np.linalg.norm(self.deviation_mm[self.valid], axis=-1)
        return float(radius.max() * 1e3) if radius.size else float("nan")

    @property
    def max_relative_distortion_percent(self) -> float:
        """The same departure as a fraction of the ideal image height.

        The reference height is measured from the axis, where the ideal map
        sends the axial object point, so an off-axis (ring-field) grid is
        referred to the same origin as an on-axis one. The axial point
        itself has zero ideal height and is excluded.
        """

        ideal_height = np.linalg.norm(self.ideal_points, axis=-1)
        usable = self.valid & (ideal_height > 0.0)
        if not usable.any():
            return float("nan")
        radius = np.linalg.norm(self.deviation_mm[usable], axis=-1)
        return float((radius / ideal_height[usable]).max() * 100.0)


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

    tracer.system.require_axial_coordinates()

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
            neighbours = tuple(
                guess
                for guess in (guesses.get((i, j - 1)), guesses.get((i - 1, j)))
                if guess is not None
            )
            slopes = _solve_chief_ray(
                tracer, field, stop_index=stop_index, fallback_guesses=neighbours
            )
            if slopes is None:
                continue
            actual_points[i, j] = trace_from_object(tracer, (field.x, field.y), slopes).image_point[
                :2
            ]
            valid[i, j] = True
            guesses[(i, j)] = slopes

    return DistortionGrid(
        object_points=object_points,
        ideal_points=ideal_points,
        actual_points=actual_points,
        valid=valid,
        magnification=magnification,
    )


__all__ = ["DistortionGrid", "chief_ray_distortion", "distortion_grid"]
