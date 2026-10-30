"""Tracing: the algorithms that carry light through an optical system.

- :mod:`.sequential` — every ray along the system's path, vectorized.
- :mod:`.nonsequential` — every ray wherever the physics sends it.
- :mod:`.paraxial` — conjugates and ABCD matrices of a coaxial path.
- :mod:`.fields` — field points, chief rays, pupil sampling.
"""

from .fields import (
    FieldPoint,
    PupilSampling,
    PupilTrace,
    chief_ray_slope,
    chief_ray_slopes,
    trace_from_object,
    trace_pupil,
)
from .nonsequential import DetectorHits, NonSequentialTracer, RayTree
from .paraxial import ConjugateSolution, ParaxialModel, differential_conjugates, solve_object_plane
from .sequential import BatchTraceResult, SequentialTracer, TraceResult, TraceStatus

__all__ = [
    "SequentialTracer",
    "NonSequentialTracer",
    "RayTree",
    "DetectorHits",
    "TraceResult",
    "BatchTraceResult",
    "TraceStatus",
    "ConjugateSolution",
    "ParaxialModel",
    "differential_conjugates",
    "solve_object_plane",
    "FieldPoint",
    "PupilSampling",
    "PupilTrace",
    "chief_ray_slope",
    "chief_ray_slopes",
    "trace_from_object",
    "trace_pupil",
]
