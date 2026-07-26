"""Chief-ray distortion: a 1-D field sweep, and the classic warped grid.

Distortion is the departure of the real object-to-image map from the ideal
linear one, ``image = magnification * object``. Both routines here measure
it with the *chief ray* (the ray through the centre of the aperture stop),
the industry-standard reference: being a single ray it is immune to the
coma and vignetting that pull a flux-weighted spot centroid around. See
:func:`~raytracer.analysis.aberrations.metrics.field_metrics` for the
centroid-based figure alongside the full aberration summary.

- :func:`chief_ray_distortion` sweeps field height and returns the
  familiar distortion-vs-field curve.
- :func:`distortion_grid` traces a square grid of field points and returns
  it warped at the image plane — the classic optical-design-software
  display, where barrel/pincushion is visible as a shape rather than read
  off a curve.

Both use the general 2-D chief-ray solver
(:func:`raytracer.sequential.fields.chief_ray_slopes`), not the
rotationally-symmetric 1-D shortcut, so they are correct for a genuinely
off-axis (x, y) field point in any system this tracer can trace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...sequential.fields import FieldPoint, chief_ray_slopes, trace_from_object
from ...sequential.trace import SequentialTracer


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
            slopes = chief_ray_slopes(
                tracer, field, stop_index=stop_index, initial_guess=guess
            )
        except RuntimeError:
            continue
        if trace_from_object(tracer, (field.x, field.y), slopes).ok:
            return slopes
    return None


def chief_ray_distortion(
    tracer: SequentialTracer,
    fields,
    *,
    magnification: float,
    field_unit: str = "mm",
    stop_index: int | None = None,
) -> list[dict]:
    """Chief-ray distortion swept over a 1-D field range.

    ``fields`` are object heights in mm by default; pass
    ``field_unit="deg"`` to give them as field *angles* instead, which is
    the natural parametrization when the object sits at (or stands in for)
    infinity — they are converted with the system's own object distance.

    Returns one dict per field point, with the same key names
    :func:`~raytracer.analysis.aberrations.metrics.field_metrics` uses for
    the corresponding quantities, so the two can be read interchangeably.
    This is the cheap way to get just distortion: one chief ray per field,
    where ``field_metrics`` traces a whole pupil bundle to also give spot
    size, focus shifts, and astigmatism.
    """

    if field_unit not in ("mm", "deg"):
        raise ValueError(f"field_unit must be 'mm' or 'deg', got {field_unit!r}")
    object_z = tracer.system.object_z
    if object_z is None:
        raise ValueError(
            "system.object_z is unset; call sequential.solve_object_plane(tracer) "
            "or assign it explicitly"
        )

    rows = []
    previous = None
    for value in np.asarray(fields, dtype=float):
        height = (
            abs(object_z) * np.tan(np.deg2rad(value)) if field_unit == "deg" else value
        )
        field = FieldPoint(y=float(height))
        fallbacks = (previous,) if previous is not None else ()
        slopes = _solve_chief_ray(
            tracer, field, stop_index=stop_index, fallback_guesses=fallbacks
        )
        if slopes is None:
            raise RuntimeError(
                f"No unvignetted chief ray at field {value:g} {field_unit} "
                f"(object height {height:.4g} mm); trim the field range"
            )
        previous = slopes
        ideal = height * magnification
        actual = float(
            trace_from_object(tracer, (0.0, field.y), slopes).image_point[1]
        )
        rows.append(
            {
                "field": float(value),
                "object_height_mm": float(height),
                "paraxial_image_height_mm": float(ideal),
                "chief_image_height_mm": actual,
                "chief_ray_distortion_um": (actual - ideal) * 1e3,
                "chief_ray_relative_distortion_ppm": (
                    (actual / ideal - 1.0) * 1e6 if ideal != 0.0 else 0.0
                ),
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
            actual_points[i, j] = trace_from_object(
                tracer, (field.x, field.y), slopes
            ).image_point[:2]
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
