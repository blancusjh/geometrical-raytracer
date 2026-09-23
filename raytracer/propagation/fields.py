"""Field points, chief-ray solving, and pupil sampling."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Literal

import numpy as np
from scipy import optimize

from .paraxial import direction_from_slopes
from .sequential import BatchTraceResult, SequentialTracer, TraceResult, TraceStatus


@dataclass(frozen=True)
class FieldPoint:
    """Finite object coordinates (mm), or component field angles (degrees)."""

    x: float = 0.0
    y: float = 0.0
    kind: Literal["height", "angle"] = "height"

    def __post_init__(self):
        if self.kind not in ("height", "angle") or not np.all(np.isfinite([self.x, self.y])):
            raise ValueError("field coordinates must be finite and kind must be height or angle")
        if self.kind == "angle" and max(abs(self.x), abs(self.y)) >= 90:
            raise ValueError("field angles must lie strictly between -90 and 90 degrees")

    @classmethod
    def angle(cls, x_deg=0.0, y_deg=0.0):
        """Collimated field; component angles in the system frame, in degrees."""
        if max(abs(x_deg), abs(y_deg)) >= 90:
            raise ValueError("field angles must lie strictly between -90 and 90 degrees")
        return cls(float(x_deg), float(y_deg), "angle")


def _object_z(tracer: SequentialTracer) -> float:
    z = tracer.system.object_z
    if z is None or not np.isfinite(z):
        raise ValueError(
            "system.object_z must be finite; call paraxial.solve_object_plane(tracer) "
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
        tracer.system.frame.to_world(origin),
        tracer.system.frame.direction_to_world(direction_from_slopes(slopes[0], slopes[1])),
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
    batch = tracer.trace_batch(
        system.frame.to_world(origins), system.frame.direction_to_world(directions), keep_paths=True
    )

    def height_at_stop(slope: float) -> float:
        result = trace_from_object(tracer, (field.x, field.y), (0.0, float(slope)), keep_path=True)
        if not result.ok and (
            result.failed_surface is not None and result.failed_surface <= stop_index
        ):
            raise RuntimeError("Ray vignetted before the stop")
        return float(system.surface_frame(stop_index).to_local(result.path[stop_index + 1])[1])

    valid: list[tuple[float, float]] = []
    for j in range(scan):
        # A ray is usable if it survived at least up to the stop surface.
        if batch.status[j] == TraceStatus.OK or batch.failed_surface[j] > stop_index:
            y_stop = system.surface_frame(stop_index).to_local(batch.paths[j, stop_index + 1])[1]
            if np.isfinite(y_stop):
                valid.append((float(samples[j]), float(y_stop)))

    for (a, fa), (b, fb) in zip(valid, valid[1:]):
        if fa == 0.0:
            return a
        if fa * fb < 0.0:
            return float(optimize.brentq(height_at_stop, a, b, xtol=xtol))
    raise RuntimeError(f"No unvignetted chief ray found for field {field}")


class _RayVignetted(RuntimeError):
    """A probe ray failed to reach the stop — internal to the 2-D solve.

    Distinct from the ``RuntimeError`` the solver raises to its caller, so
    the backtracking loop can treat "that step was too long" separately
    from "this field point has no chief ray at all".
    """


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
        batch = tracer.trace_batch(
            system.frame.to_world(origin),
            system.frame.direction_to_world(direction),
            keep_paths=True,
        )
        ok = batch.status[0] == TraceStatus.OK or batch.failed_surface[0] > stop_index
        position = system.surface_frame(stop_index).to_local(batch.paths[0, stop_index + 1])[:2]
        if not ok or not np.all(np.isfinite(position)):
            raise _RayVignetted
        return position

    def probe(slopes: np.ndarray) -> np.ndarray:
        """``stop_position``, reporting a vignette against *field* by name."""

        try:
            return stop_position(slopes)
        except _RayVignetted as exc:
            raise RuntimeError(f"No unvignetted chief ray found for field {field}") from exc

    s = np.asarray(initial_guess, dtype=float)
    residual = probe(s)
    for _ in range(max_iterations):
        if np.hypot(*residual) < xtol:
            break
        eps = 1e-6
        jacobian = np.empty((2, 2))
        for k in range(2):
            step = s.copy()
            step[k] += eps
            jacobian[:, k] = (probe(step) - residual) / eps
        # lstsq, not solve: a singular Jacobian (a ray marching parallel to
        # the stop, a degenerate afocal space) makes solve raise LinAlgError,
        # which is a ValueError — it would sail straight past every caller's
        # `except RuntimeError` and abort a whole field scan over one bad
        # point. lstsq gives the Newton step whenever the Jacobian is well
        # conditioned, and a sane minimum-norm step when it isn't.
        delta = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        # Backtrack until the step both survives the system and actually
        # reduces the residual; a vignetting trial is simply too long a step.
        damping = 1.0
        trial, trial_residual = s, residual
        while damping > 1e-3:
            candidate = s + damping * delta
            try:
                candidate_residual = stop_position(candidate)
            except _RayVignetted:
                damping *= 0.5
                continue
            if np.hypot(*candidate_residual) < np.hypot(*residual):
                trial, trial_residual = candidate, candidate_residual
                break
            damping *= 0.5
        s, residual = trial, trial_residual
    else:
        if np.hypot(*residual) >= xtol:
            raise RuntimeError(
                f"Chief-ray solve for field {field} did not converge in "
                f"{max_iterations} iterations (residual {np.hypot(*residual):.3g} mm "
                f"at the stop, tolerance {xtol:.3g})"
            )

    return float(s[0]), float(s[1])


@dataclass(frozen=True)
class PupilSampling:
    """Normalized pupil sample layout.

    ``gauss``: Gauss-Legendre quadrature in squared radius, uniform azimuth.
    ``rings``: concentric rings (reference layout — the axial sample plus
    ``azimuth`` points per ring). ``grid``: Cartesian grid clipped to the
    unit disk. ``fan_t``/``fan_s``: 1-D meridional/sagittal fans.
    """

    kind: Literal["rings", "gauss", "grid", "fan_t", "fan_s"] = "rings"
    radial: int = 10
    azimuth: int = 48
    n: int = 81
    rmax: float = 1.0
    adaptive: bool = False  # azimuth count proportional to ring radius (EUV layout)

    def __post_init__(self):
        if self.radial < 2 or self.azimuth < 4 or self.n < 2:
            raise ValueError("sampling requires radial >= 2, azimuth >= 4 and n >= 2")
        if not np.isfinite(self.rmax) or not 0 < self.rmax <= 1:
            raise ValueError("rmax must lie in (0, 1]")

    def points(self) -> np.ndarray:
        if self.kind == "gauss":
            nodes, _ = np.polynomial.legendre.leggauss(self.radial)
            radii = self.rmax * np.sqrt((nodes + 1) / 2)
            angles = np.arange(self.azimuth) * (2 * np.pi / self.azimuth)
            return np.column_stack(
                [
                    (radii[:, None] * np.cos(angles)).ravel(),
                    (radii[:, None] * np.sin(angles)).ravel(),
                ]
            )
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
            axis = np.linspace(-self.rmax, self.rmax, self.n)
            xx, yy = np.meshgrid(axis, axis)
            keep = xx**2 + yy**2 <= self.rmax**2
            return np.column_stack([xx[keep], yy[keep]])
        if self.kind == "fan_t":
            axis = np.linspace(-self.rmax, self.rmax, self.n)
            return np.column_stack([np.zeros_like(axis), axis])
        if self.kind == "fan_s":
            axis = np.linspace(-self.rmax, self.rmax, self.n)
            return np.column_stack([axis, np.zeros_like(axis)])
        raise ValueError(f"Unknown sampling kind {self.kind!r}")

    def area_weights(self, points: np.ndarray | None = None) -> np.ndarray:
        """Per-sample pupil-area weights (normalized to unit sum)."""

        points = self.points() if points is None else points
        if self.kind == "gauss":
            nodes, weights = np.polynomial.legendre.leggauss(self.radial)
            radii_squared = self.rmax**2 * (nodes + 1) / 2
            ring = np.argmin(abs(np.sum(points**2, axis=1)[:, None] - radii_squared), axis=1)
            weights = weights[ring] / self.azimuth
        elif self.kind == "rings":
            # Integrate in u = rho²: dA = (1/2) du dphi. Divide each ring's
            # trapezoidal weight among its own samples, including the center.
            u = np.linspace(0.0, self.rmax, self.radial) ** 2
            intervals = np.diff(u)
            ring_weights = np.r_[intervals[0], intervals[:-1] + intervals[1:], intervals[-1]] / 2
            ring = np.rint(np.linalg.norm(points, axis=1) * (self.radial - 1) / self.rmax).astype(
                int
            )
            counts = np.bincount(ring, minlength=self.radial)
            weights = ring_weights[ring] / counts[ring]
        else:
            weights = np.ones(len(points))
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
    sample_weights: np.ndarray | None = None
    aiming_residual_mm: np.ndarray | None = None
    launch_origins: np.ndarray | None = None
    launch_directions: np.ndarray | None = None

    @property
    def geometric_throughput(self) -> float:
        """Transmitted fraction of the specified sampling measure, without Fresnel."""
        weights = self.sample_weights
        if weights is None:
            weights = self.sampling.area_weights(self.pupil_uv)
        return float(np.sum(weights[self.valid]))

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
    na_object_sine: float | None = None,
    sampling: PupilSampling | None = None,
    slope_model: Literal["tangent", "sine"] = "tangent",
    chief_slope: float | tuple[float, float] | None = None,
    stop_index: int | None = None,
    keep_paths: bool = False,
) -> PupilTrace:
    """Trace a pupil-filling bundle for *field*.

    ``slope_model="tangent"`` maps the normalized pupil radius to transverse
    slopes via s = p * sine / sqrt(1 - sine^2) added around the chief slope
    (the legacy DUV convention). ``"sine"`` samples equal transverse direction
    cosine offsets in an orthonormal frame about the chief ray.

    ``chief_slope`` accepts either a meridional slope ``s_y`` or a full
    ``(s_x, s_y)`` pair, so a chief ray solved by :func:`chief_ray_slopes`
    for a genuinely off-axis ``(x, y)`` field point can be used as-is.
    Omitted, it is solved with :func:`chief_ray_slopes` — the general 2-D
    solver rather than the 1-D :func:`chief_ray_slope`, which agrees with
    it to machine precision where both converge but scans a fixed, fairly
    coarse slope bracket that comes up empty whenever the true chief ray
    occupies a needle-thin slice of it (any large finite-object stand-in
    for infinity, such as the Cooke triplet and double-Gauss examples).

    The ``"sine"`` model samples a circular cone about the chief direction,
    using an orthonormal transverse frame. With ``na_object_sine=None``, rays
    are instead aimed at the physical stop, for finite or angular fields.
    """

    if na_object_sine is None:
        from .aiming import trace_stop_pupil

        return trace_stop_pupil(
            tracer, field, sampling=sampling, stop_index=stop_index, keep_paths=keep_paths
        )
    if field.kind != "height":
        raise ValueError("angular fields use stop aiming; omit na_object_sine")
    if not 0 < na_object_sine < 1:
        raise ValueError("na_object_sine is sin(angle), and must lie in (0, 1)")
    sampling = sampling or PupilSampling()
    if chief_slope is None:
        chief_slope = chief_ray_slopes(tracer, field, stop_index=stop_index)
    chief_sx, chief_sy = (
        (0.0, float(chief_slope))
        if np.isscalar(chief_slope)
        else (float(chief_slope[0]), float(chief_slope[1]))
    )

    points = sampling.points()
    object_z = _object_z(tracer)
    origins = np.repeat([[field.x, field.y, object_z]], points.shape[0], axis=0)

    if slope_model == "tangent":
        max_slope = na_object_sine / np.sqrt(1.0 - na_object_sine**2)
        sx = chief_sx + points[:, 0] * max_slope
        sy = chief_sy + points[:, 1] * max_slope
        norm = np.sqrt(1.0 + sx**2 + sy**2)
        directions = np.column_stack([sx / norm, sy / norm, 1.0 / norm])
    elif slope_model == "sine":
        axis = direction_from_slopes(chief_sx, chief_sy)
        transverse_x = np.array([1.0, 0.0, 0.0])
        transverse_x -= (transverse_x @ axis) * axis
        transverse_x /= np.linalg.norm(transverse_x)
        transverse_y = np.cross(axis, transverse_x)
        transverse = points * na_object_sine
        axial = np.sqrt(1 - np.sum(transverse**2, axis=1))
        directions = (
            transverse[:, :1] * transverse_x
            + transverse[:, 1:] * transverse_y
            + axial[:, None] * axis
        )
    else:
        raise ValueError(f"Unknown slope model {slope_model!r}")

    batch = tracer.trace_batch(
        tracer.system.frame.to_world(origins),
        tracer.system.frame.direction_to_world(directions),
        keep_paths=keep_paths,
    )
    chief = trace_from_object(tracer, (field.x, field.y), (chief_sx, chief_sy), keep_path=True)

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
        sample_weights=weights_all,
        launch_origins=tracer.system.frame.to_world(origins),
        launch_directions=tracer.system.frame.direction_to_world(directions),
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
