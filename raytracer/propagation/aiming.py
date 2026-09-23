"""Batch ray aiming at the physical aperture stop, for finite or infinite objects."""

import copy

import numpy as np

from ..surfaces.apertures import CircularAperture
from .fields import FieldPoint, PupilSampling, PupilTrace
from .sequential import SequentialTracer, TraceStatus


def trace_stop_pupil(
    tracer: SequentialTracer,
    field: FieldPoint,
    *,
    sampling: PupilSampling | None = None,
    stop_index: int | None = None,
    keep_paths: bool = False,
    tolerance_mm: float = 1e-9,
    max_iterations: int = 30,
) -> PupilTrace:
    """Aim samples to a circular/annular stop in its local frame.

    Finite fields vary launch slopes; angular fields vary launch positions of
    parallel rays. The solve ignores apertures, then the final physical trace
    applies every aperture. Failed targets remain invalid in the returned
    bundle; their weights are included in the geometric throughput denominator.
    An annular pupil's chief ray is a geometrical reference through its center.
    """

    system = tracer.system
    system._rebuild()
    stop = system.stop_index if stop_index is None else stop_index
    if stop is None or not 0 <= stop < len(system.rows):
        raise ValueError("ray aiming requires a valid stop surface")
    aperture = system.rows[stop].clear_aperture
    if not isinstance(aperture, CircularAperture):
        raise ValueError("stop-pupil sampling currently requires a circular or annular aperture")
    if field.kind not in ("height", "angle"):
        raise ValueError("field kind must be height or angle")

    sampling = sampling or PupilSampling(kind="gauss")
    uv = sampling.points()
    targets = uv * aperture.radius
    if aperture.inner_radius:
        if sampling.kind != "gauss":
            raise ValueError("annular stop aiming requires Gauss sampling")
        radius = np.linalg.norm(uv, axis=1)
        physical_radius = np.sqrt(
            aperture.inner_radius**2 + radius**2 * (aperture.radius**2 - aperture.inner_radius**2)
        )
        targets = uv * (physical_radius / radius)[:, None]
    targets = np.vstack([np.zeros(2), targets])
    count = len(targets)
    stop_frame = system.surface_frame(stop)
    target_sag = system.rows[stop].profile.sag(np.linalg.norm(targets, axis=1))
    target_world = stop_frame.to_world(np.column_stack([targets, target_sag]))
    target_system = system.frame.to_local(target_world)

    if field.kind == "angle":
        if max(abs(field.x), abs(field.y)) >= 90:
            raise ValueError("angular fields must point into positive system z")
        slopes = np.tan(np.deg2rad([field.x, field.y]))
        direction = np.r_[slopes, 1.0]
        direction /= np.linalg.norm(direction)
        launch_z = min(system.vertices) - 2 * max(aperture.radius, 1.0)
        initial = target_system[:, :2] - (target_system[:, 2] - launch_z)[:, None] * slopes

        def launch(parameters):
            origins = np.column_stack([parameters, np.full(count, launch_z)])
            directions = np.tile(direction, (count, 1))
            return system.frame.to_world(origins), system.frame.direction_to_world(directions)

    else:
        if system.object_z is None or not np.isfinite(system.object_z):
            raise ValueError("finite field aiming requires a finite object_z")
        origin = np.array([field.x, field.y, system.object_z])
        reach = target_system[:, 2] - origin[2]
        if np.any(np.abs(reach) < 1e-12):
            raise ValueError("object and target stop cannot share a launch plane")
        initial = (target_system[:, :2] - origin[:2]) / reach[:, None]

        def launch(parameters):
            origins = np.tile(origin, (count, 1))
            directions = np.column_stack([parameters, np.ones(count)])
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            return system.frame.to_world(origins), system.frame.direction_to_world(directions)

    aiming_tracer = copy.copy(tracer)
    aiming_tracer.clip_apertures = False

    def residual(parameters):
        origins, directions = launch(parameters)
        batch = aiming_tracer.trace_batch(origins, directions, stop_at=stop)
        return stop_frame.to_local(batch.image_points)[:, :2] - targets

    parameters = initial.copy()
    error = residual(parameters)
    for _ in range(max_iterations):
        norms = np.linalg.norm(error, axis=1)
        active = np.isfinite(norms) & (norms > tolerance_mm)
        if not active.any():
            break
        jacobian = np.empty((count, 2, 2))
        for axis in range(2):
            step = 1e-6 * np.maximum(1.0, np.abs(parameters[:, axis]))
            plus, minus = parameters.copy(), parameters.copy()
            plus[:, axis] += step
            minus[:, axis] -= step
            jacobian[:, :, axis] = (residual(plus) - residual(minus)) / (2 * step[:, None])
        usable = active & np.all(np.isfinite(jacobian), axis=(1, 2))
        delta = np.zeros_like(parameters)
        if usable.any():
            delta[usable] = np.einsum(
                "nij,nj->ni", np.linalg.pinv(jacobian[usable]), -error[usable]
            )
        pending = usable.copy()
        for damping in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            candidate = parameters + damping * delta
            trial = residual(candidate)
            improved = pending & (np.linalg.norm(trial, axis=1) < norms)
            parameters[improved] = candidate[improved]
            error[improved] = trial[improved]
            pending[improved] = False
            if not pending.any():
                break

    origins, directions = launch(parameters)
    errors = np.linalg.norm(error, axis=1)
    if not np.isfinite(errors[0]) or errors[0] > tolerance_mm:
        raise RuntimeError("chief-ray aiming did not converge to the stop center")
    chief = aiming_tracer.trace(origins[0], directions[0], keep_path=True)
    batch = tracer.trace_batch(origins[1:], directions[1:], keep_paths=keep_paths)
    failed = ~np.isfinite(errors[1:]) | (errors[1:] > tolerance_mm)
    batch.status[failed] = TraceStatus.DIVERGED
    batch.failed_surface[failed] = stop
    batch.image_points[failed] = np.nan
    if batch.paths is not None:
        batch.paths[failed] = np.nan
    all_weights = sampling.area_weights(uv)
    weights = all_weights[batch.valid]
    if weights.size:
        weights = weights / weights.sum()
    return PupilTrace(
        field,
        uv,
        weights,
        batch,
        chief,
        np.nan,
        sampling,
        sample_weights=all_weights,
        aiming_residual_mm=errors[1:],
        launch_origins=origins[1:].copy(),
        launch_directions=directions[1:].copy(),
    )
