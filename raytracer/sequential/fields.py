"""Field points, chief-ray solving, and pupil sampling."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Literal

import numpy as np
from scipy import optimize

from .paraxial import direction_from_slopes
from .trace import BatchTraceResult, SequentialTracer, TraceResult, TraceStatus


@dataclass(frozen=True)
class FieldPoint:
    """Object-plane transverse coordinates (mm)."""

    x: float = 0.0
    y: float = 0.0


def _object_z(tracer: SequentialTracer) -> float:
    z = tracer.system.object_z
    if z is None:
        raise ValueError(
            "system.object_z is unset; call sequential.solve_object_plane(tracer) "
            "or assign it explicitly"
        )
    return z


def trace_from_object(
    tracer: SequentialTracer,
    xy: tuple[float, float],
    slopes: tuple[float, float],
    *,
    keep_path: bool = False,
    keep_aoi: bool = False,
) -> TraceResult:
    """Trace a single ray from the object plane given transverse slopes."""

    origin = np.array([xy[0], xy[1], _object_z(tracer)])
    return tracer.trace(
        origin,
        direction_from_slopes(slopes[0], slopes[1]),
        keep_path=keep_path,
        keep_aoi=keep_aoi,
    )


def chief_ray_slope(
    tracer: SequentialTracer,
    field: FieldPoint,
    *,
    stop_index: int | None = None,
    bracket: tuple[float, float] = (-0.08, 0.08),
    scan: int = 65,
    xtol: float = 1e-14,
) -> float:
    """Slope s_y whose ray crosses the axis at the stop surface.

    The bracket is scanned first because rays with extreme slopes vignette;
    ``brentq`` then refines the first sign change among surviving samples.
    """

    system = tracer.system
    if stop_index is None:
        stop_index = system.stop_index
    if stop_index is None:
        raise ValueError("System has no aperture stop; pass stop_index explicitly")
    object_z = _object_z(tracer)

    samples = np.linspace(bracket[0], bracket[1], scan)
    origins = np.repeat([[field.x, field.y, object_z]], scan, axis=0)
    directions = np.array([direction_from_slopes(0.0, s) for s in samples])
    batch = tracer.trace_batch(origins, directions, keep_paths=True)

    def height_at_stop(slope: float) -> float:
        result = trace_from_object(tracer, (field.x, field.y), (0.0, float(slope)), keep_path=True)
        if not result.ok and (
            result.failed_surface is not None and result.failed_surface <= stop_index
        ):
            raise RuntimeError("Ray vignetted before the stop")
        return float(result.path[stop_index + 1, 1])

    valid: list[tuple[float, float]] = []
    for j in range(scan):
        # A ray is usable if it survived at least up to the stop surface.
        if batch.status[j] == TraceStatus.OK or batch.failed_surface[j] > stop_index:
            y_stop = batch.paths[j, stop_index + 1, 1]
            if np.isfinite(y_stop):
                valid.append((float(samples[j]), float(y_stop)))

    for (a, fa), (b, fb) in zip(valid, valid[1:]):
        if fa == 0.0:
            return a
        if fa * fb < 0.0:
            return float(optimize.brentq(height_at_stop, a, b, xtol=xtol))
    raise RuntimeError(f"No unvignetted chief ray found for field {field}")


def chief_ray_slopes(
    tracer: SequentialTracer,
    field: FieldPoint,
    *,
    stop_index: int | None = None,
    initial_guess: tuple[float, float] | None = None,
    xtol: float = 1e-9,
    max_iterations: int = 30,
) -> tuple[float, float]:
    """General ``(s_x, s_y)`` whose ray crosses the axis at the stop surface.

    Unlike :func:`chief_ray_slope`, this does not fix ``s_x = 0`` or lean on
    rotational symmetry to reduce the problem to one dimension: it solves
    for both transverse slopes jointly, by damped Newton iteration (a
    numerical 2x2 Jacobian, backtracked whenever a trial step vignettes or
    makes the residual worse) on the ray's transverse position at the stop
    surface. That makes it correct for a field point with ``x`` and ``y``
    both nonzero in *any* system this tracer can trace — including a future
    decentered or tilted one where the meridional-plane shortcut wouldn't
    apply — not only the rotationally symmetric systems in this repo today.

    The default initial guess is the paraxial straight line from the object
    point to the stop center; pass ``initial_guess`` (e.g. the converged
    solution for a neighboring field point) to warm-start a scan over many
    field points, which converges faster and is more robust near the edge
    of the field than restarting from the paraxial guess each time.
    """

    system = tracer.system
    if stop_index is None:
        stop_index = system.stop_index
    if stop_index is None:
        raise ValueError("System has no aperture stop; pass stop_index explicitly")
    object_z = _object_z(tracer)

    if initial_guess is None:
        z_stop = float(system.vertices[stop_index])
        denom = z_stop - object_z
        initial_guess = (-field.x / denom, -field.y / denom)

    def stop_position(slopes: np.ndarray) -> np.ndarray:
        origin = np.array([[field.x, field.y, object_z]])
        direction = direction_from_slopes(float(slopes[0]), float(slopes[1]))[None, :]
        batch = tracer.trace_batch(origin, direction, keep_paths=True)
        ok = batch.status[0] == TraceStatus.OK or batch.failed_surface[0] > stop_index
        position = batch.paths[0, stop_index + 1, :2]
        if not ok or not np.all(np.isfinite(position)):
            raise RuntimeError("Ray vignetted before the stop")
        return position

    try:
        s = np.asarray(initial_guess, dtype=float)
        residual = stop_position(s)
        for _ in range(max_iterations):
            if np.hypot(*residual) < xtol:
                break
            eps = 1e-6
            jacobian = np.empty((2, 2))
            for k in range(2):
                step = s.copy()
                step[k] += eps
                jacobian[:, k] = (stop_position(step) - residual) / eps
            delta = np.linalg.solve(jacobian, -residual)
            damping = 1.0
            trial, trial_residual = s, residual
            while damping > 1e-3:
                candidate = s + damping * delta
                try:
                    candidate_residual = stop_position(candidate)
                except RuntimeError:
                    damping *= 0.5
                    continue
                if np.hypot(*candidate_residual) < np.hypot(*residual):
                    trial, trial_residual = candidate, candidate_residual
                    break
                damping *= 0.5
            s, residual = trial, trial_residual
        else:
            if np.hypot(*residual) >= xtol:
                raise RuntimeError("Chief-ray solve did not converge")
    except RuntimeError as exc:
        raise RuntimeError(f"No unvignetted chief ray found for field {field}") from exc

    return float(s[0]), float(s[1])


@dataclass(frozen=True)
class PupilSampling:
    """Normalized pupil sample layout.

    ``rings``: concentric rings (reference layout — the axial sample plus
    ``azimuth`` points per ring). ``grid``: Cartesian grid clipped to the
    unit disk. ``fan_t``/``fan_s``: 1-D meridional/sagittal fans.
    """

    kind: Literal["rings", "grid", "fan_t", "fan_s"] = "rings"
    radial: int = 10
    azimuth: int = 48
    n: int = 81
    rmax: float = 1.0
    adaptive: bool = False  # azimuth count proportional to ring radius (EUV layout)

    def points(self) -> np.ndarray:
        if self.kind == "rings":
            pts = []
            for r in np.linspace(0.0, self.rmax, self.radial):
                if self.adaptive:
                    count = max(1, int(round(self.azimuth * r)))
                else:
                    count = 1 if r == 0.0 else self.azimuth
                for az in np.linspace(0.0, 2.0 * np.pi, count, endpoint=False):
                    pts.append((r * np.cos(az), r * np.sin(az)))
            return np.asarray(pts)
        if self.kind == "grid":
            axis = np.linspace(-1.0, 1.0, self.n)
            xx, yy = np.meshgrid(axis, axis)
            keep = xx**2 + yy**2 <= 1.0
            return np.column_stack([xx[keep], yy[keep]])
        if self.kind == "fan_t":
            axis = np.linspace(-1.0, 1.0, self.n)
            return np.column_stack([np.zeros_like(axis), axis])
        if self.kind == "fan_s":
            axis = np.linspace(-1.0, 1.0, self.n)
            return np.column_stack([axis, np.zeros_like(axis)])
        raise ValueError(f"Unknown sampling kind {self.kind!r}")

    def area_weights(self, points: np.ndarray | None = None) -> np.ndarray:
        """Per-sample pupil-area weights (normalized to unit sum)."""

        points = self.points() if points is None else points
        if self.kind == "rings" and not self.adaptive:
            r = np.hypot(points[:, 0], points[:, 1])
            weights = np.maximum(r, 0.5 / (self.radial - 1))
        else:
            # Adaptive rings sample the disk near-uniformly; grids are uniform.
            weights = np.ones(points.shape[0])
        return weights / weights.sum()


@dataclass
class PupilTrace:
    """Traced pupil bundle for one field point — the central analysis object."""

    field: FieldPoint
    pupil_uv: np.ndarray  # (N, 2) normalized coordinates
    weights: np.ndarray  # (N,) pupil-area weights over the *valid* subset
    batch: BatchTraceResult
    chief: TraceResult
    na_object_sine: float
    sampling: PupilSampling = dataclass_field(default_factory=PupilSampling)

    @property
    def valid(self) -> np.ndarray:
        return self.batch.valid

    @property
    def image_points(self) -> np.ndarray:
        """(M, 3) image-plane intersections of the surviving rays."""

        return self.batch.image_points[self.valid]

    @property
    def directions(self) -> np.ndarray:
        return self.batch.directions[self.valid]

    @property
    def opl(self) -> np.ndarray:
        return self.batch.opl[self.valid]

    @property
    def uv(self) -> np.ndarray:
        return self.pupil_uv[self.valid]


def trace_pupil(
    tracer: SequentialTracer,
    field: FieldPoint,
    *,
    na_object_sine: float,
    sampling: PupilSampling | None = None,
    slope_model: Literal["tangent", "sine"] = "tangent",
    chief_slope: float | None = None,
    stop_index: int | None = None,
    keep_paths: bool = False,
) -> PupilTrace:
    """Trace a pupil-filling bundle for *field*.

    ``slope_model="tangent"`` maps the normalized pupil radius to transverse
    slopes via s = p * sine / sqrt(1 - sine^2) added around the chief slope
    (the DUV reference convention). ``"sine"`` offsets direction cosines
    directly (the EUV notebook convention).
    """

    sampling = sampling or PupilSampling()
    if chief_slope is None:
        chief_slope = chief_ray_slope(tracer, field, stop_index=stop_index)

    points = sampling.points()
    object_z = _object_z(tracer)
    origins = np.repeat([[field.x, field.y, object_z]], points.shape[0], axis=0)

    if slope_model == "tangent":
        max_slope = na_object_sine / np.sqrt(1.0 - na_object_sine**2)
        sx = points[:, 0] * max_slope
        sy = chief_slope + points[:, 1] * max_slope
        norm = np.sqrt(1.0 + sx**2 + sy**2)
        directions = np.column_stack([sx / norm, sy / norm, 1.0 / norm])
    elif slope_model == "sine":
        theta = np.arctan(chief_slope)
        dx = points[:, 0] * na_object_sine
        a = theta + points[:, 1] * na_object_sine
        dz2 = 1.0 - dx**2 - np.sin(a) ** 2
        directions = np.column_stack([dx, np.sin(a), np.sqrt(np.maximum(dz2, 0.0))])
    else:
        raise ValueError(f"Unknown slope model {slope_model!r}")

    batch = tracer.trace_batch(origins, directions, keep_paths=keep_paths)
    chief = trace_from_object(tracer, (field.x, field.y), (0.0, chief_slope), keep_path=True)

    weights_all = sampling.area_weights(points)
    weights = weights_all[batch.valid]
    total = weights.sum()
    if total > 0.0:
        weights = weights / total

    return PupilTrace(
        field=field,
        pupil_uv=points,
        weights=weights,
        batch=batch,
        chief=chief,
        na_object_sine=na_object_sine,
        sampling=sampling,
    )


__all__ = [
    "FieldPoint",
    "trace_from_object",
    "chief_ray_slope",
    "chief_ray_slopes",
    "PupilSampling",
    "PupilTrace",
    "trace_pupil",
]
