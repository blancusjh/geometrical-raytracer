"""Transverse ray-aberration fans (tangential and sagittal)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...propagation.fields import FieldPoint, PupilSampling, trace_pupil
from ...propagation.sequential import SequentialTracer


@dataclass
class FanData:
    """1-D transverse-aberration fans for one field point."""

    field_y: float
    tangential: np.ndarray  # (Mt, 2): normalized pupil coord, delta-y' (um)
    sagittal: np.ndarray  # (Ms, 2): normalized pupil coord, delta-x' (um)


def ray_fans(
    tracer: SequentialTracer,
    field: FieldPoint,
    *,
    na_object_sine: float,
    magnification: float,
    n: int = 81,
    chief_slope: float | None = None,
) -> FanData:
    """Trace meridional and sagittal fans and return transverse aberrations.

    Tangential aberration is measured against the ideal image height
    ``field.y * magnification``; sagittal against x' = 0 (reference
    convention for a meridional field point).
    """

    ideal_y = field.y * magnification

    tangential_trace = trace_pupil(
        tracer,
        field,
        na_object_sine=na_object_sine,
        sampling=PupilSampling(kind="fan_t", n=n),
        chief_slope=chief_slope,
    )
    keep = tangential_trace.valid
    tangential = np.column_stack(
        [
            tangential_trace.pupil_uv[keep, 1],
            (tangential_trace.image_points[:, 1] - ideal_y) * 1e3,
        ]
    )

    sagittal_trace = trace_pupil(
        tracer,
        field,
        na_object_sine=na_object_sine,
        sampling=PupilSampling(kind="fan_s", n=n),
        chief_slope=chief_slope if chief_slope is not None else None,
    )
    keep = sagittal_trace.valid
    sagittal = np.column_stack(
        [
            sagittal_trace.pupil_uv[keep, 0],
            sagittal_trace.image_points[:, 0] * 1e3,
        ]
    )

    return FanData(field_y=field.y, tangential=tangential, sagittal=sagittal)


__all__ = ["FanData", "ray_fans"]
