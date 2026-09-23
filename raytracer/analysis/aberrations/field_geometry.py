"""Parabasal field focus and explicit ideal image maps for centered systems."""

from copy import copy
from dataclasses import dataclass

import numpy as np

from ...propagation.aiming import aim_stop_targets
from ...propagation.fields import FieldPoint
from ...propagation.paraxial import ParaxialModel
from ...propagation.sequential import SequentialTracer, TraceStatus


def ideal_image_xy(system, field: FieldPoint, *, stop_index=None, reference="paraxial"):
    """Linear chief-ray map at the actual detector, in its local coordinates.

    For angular fields 'paraxial' is rectilinear: the ABCD scale multiplies
    component tangents. 'f_theta' instead uses polar field angle, with that
    same on-axis scale. It requires angular fields. At Gaussian focus in
    air this is the usual f*tan(theta) versus f*theta distinction.
    """
    if reference not in ("paraxial", "f_theta"):
        raise ValueError("reference must be paraxial or f_theta")
    model = ParaxialModel(system)
    points = []
    for coordinate in (field.x, field.y):
        meridional = FieldPoint(y=coordinate, kind=field.kind)
        state = model.launch_state(meridional, stop_index=stop_index)
        points.append((model.matrix_first_vertex_to_image @ state)[0])
    points = np.array(points)
    if reference == "f_theta":
        if field.kind != "angle":
            raise ValueError("f_theta requires angular fields")
        tangent = np.linalg.norm(np.tan(np.deg2rad([field.x, field.y])))
        if tangent:
            points *= np.arctan(tangent) / tangent
    return points


@dataclass
class ParabasalFocus:
    field: FieldPoint
    chief_image_xy_mm: np.ndarray
    chief_transmitted: bool
    tangential_shift_mm: float
    sagittal_shift_mm: float
    step_mm: float
    step_difference_mm: np.ndarray  # tangential, sagittal
    aiming_residual_mm: float

    @property
    def astigmatic_separation_mm(self):
        return self.tangential_shift_mm - self.sagittal_shift_mm


def parabasal_focus(tracer: SequentialTracer, field: FieldPoint, *, step_mm=1e-3, stop_index=None):
    """Differential foci around the chief ray, from ±h and ±h/2 at the stop.

    This is not a finite-aperture RMS focus. Both focus shifts are along
    detector z, including any nominal defocus. Apertures are ignored for
    geometric probes; chief_transmitted reports the physical chief status.
    An obscured chief remains a mathematical reference, not transmitted light.
    """
    ParaxialModel(tracer.system)  # reject decentered/tilted local configurations
    if not np.isfinite(step_mm) or step_mm <= 0:
        raise ValueError("parabasal step must be finite and positive")
    direction = np.array([field.x, field.y])
    if field.kind == "angle":
        direction = np.tan(np.deg2rad(direction))
    norm = np.linalg.norm(direction)
    tangential = direction / norm if norm else np.array([0.0, 1.0])
    sagittal = np.array([tangential[1], -tangential[0]])
    axes = np.array([tangential, sagittal])
    targets = np.vstack(
        [
            np.zeros(2),
            *[
                amount * axis
                for axis in axes
                for amount in (-step_mm, step_mm, -step_mm / 2, step_mm / 2)
            ],
        ]
    )
    origins, directions, residual = aim_stop_targets(tracer, field, targets, stop_index=stop_index)
    if not np.all(np.isfinite(residual)) or residual.max() > 1e-11:
        raise RuntimeError("parabasal stop aiming did not converge")
    geometric = copy(tracer)
    geometric.clip_apertures = False
    batch = geometric.trace_batch(origins, directions)
    if not batch.valid.all():
        raise RuntimeError("parabasal reference rays did not reach the detector")
    frame = tracer.system.image_frame
    points = frame.to_local(batch.image_points)[:, :2]
    outgoing = frame.direction_to_local(batch.directions)
    slopes = outgoing[:, :2] / outgoing[:, 2:]
    focuses, differences = [], []
    for index, axis in enumerate(axes):
        start = 1 + 4 * index
        p = points[start : start + 4] @ axis
        q = slopes[start : start + 4] @ axis
        denominator = np.array([q[1] - q[0], q[3] - q[2]])
        if np.any(np.abs(denominator) < 1e-14 * step_mm):
            raise ValueError("parabasal focus is undefined for parallel differential rays")
        focus = -np.array([p[1] - p[0], p[3] - p[2]]) / denominator
        focuses.append(float(focus[1]))
        differences.append(float(abs(focus[1] - focus[0])))
    physical = copy(tracer)
    physical.clip_apertures = True
    transmitted = physical.trace(origins[0], directions[0]).ok
    return ParabasalFocus(
        field,
        points[0],
        transmitted,
        *focuses,
        step_mm,
        np.array(differences),
        float(residual.max()),
    )


@dataclass
class DistortionMap:
    fields: tuple[FieldPoint, ...]
    ideal_xy_mm: np.ndarray
    chief_xy_mm: np.ndarray
    chief_transmitted: np.ndarray
    reference: str
    centroid_xy_mm: np.ndarray | None = None
    geometric_throughput: np.ndarray | None = None

    @property
    def deviation_mm(self):
        return self.chief_xy_mm - self.ideal_xy_mm

    @property
    def radial_percent(self):
        radius_squared = np.sum(self.ideal_xy_mm**2, axis=1)
        result = np.full(len(radius_squared), np.nan)
        usable = radius_squared > 0
        result[usable] = (
            100
            * np.sum(self.deviation_mm[usable] * self.ideal_xy_mm[usable], axis=1)
            / radius_squared[usable]
        )
        return result


def distortion_map(
    tracer, fields, *, reference="paraxial", stop_index=None, centroid_sampling=None
):
    """Chief distortion for finite heights or true infinite-object angles.

    Relative distortion on axis is undefined (NaN), not an arbitrary zero.
    Optional area-weighted centroids are kept separately: coma and clipping
    can shift them. Failed chief aiming or geometry raises explicitly.
    """
    from ...propagation.fields import trace_pupil

    fields = tuple(fields)
    if not fields:
        raise ValueError("distortion requires at least one field")
    ideals, chiefs, transmitted, centroids, throughputs = [], [], [], [], []
    geometric, physical = copy(tracer), copy(tracer)
    geometric.clip_apertures = False
    physical.clip_apertures = True
    for field in fields:
        ideals.append(
            ideal_image_xy(tracer.system, field, reference=reference, stop_index=stop_index)
        )
        origins, directions, residual = aim_stop_targets(
            tracer, field, [[0, 0]], stop_index=stop_index
        )
        if not np.all(np.isfinite(residual)) or residual[0] > 1e-11:
            raise RuntimeError("distortion chief aiming did not converge")
        chief = geometric.trace(origins[0], directions[0])
        if not chief.ok:
            raise RuntimeError("distortion chief did not reach the detector")
        chiefs.append(tracer.system.image_frame.to_local(chief.image_point)[:2])
        transmitted.append(physical.trace(origins[0], directions[0]).ok)
        if centroid_sampling is not None:
            pupil = trace_pupil(physical, field, sampling=centroid_sampling, stop_index=stop_index)
            if not np.isin(
                pupil.batch.status, [TraceStatus.OK, TraceStatus.VIGNETTED, TraceStatus.TIR]
            ).all():
                raise RuntimeError("centroid pupil has an aiming or geometric trace failure")
            throughputs.append(pupil.geometric_throughput)
            points = tracer.system.image_frame.to_local(pupil.image_points)[:, :2]
            centroids.append(pupil.weights @ points if len(points) else [np.nan, np.nan])
    return DistortionMap(
        fields,
        np.array(ideals),
        np.array(chiefs),
        np.array(transmitted),
        reference,
        np.array(centroids) if centroid_sampling is not None else None,
        np.array(throughputs) if centroid_sampling is not None else None,
    )


__all__ = ["ParabasalFocus", "parabasal_focus", "ideal_image_xy", "DistortionMap", "distortion_map"]
