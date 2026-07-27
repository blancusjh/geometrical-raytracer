"""Emit rays, prolong them to a collision, decide the new ray, repeat.

The only place that governs what a ray does after it hits a surface:

- ``sequential`` -- every ray through every surface in fixed order, one
  deterministic path each, vectorized. ``paraxial`` and ``fields``
  (chief-ray/pupil solving, conjugate recovery) build on it, which is why
  they live here rather than with the data model in :mod:`raytracer.design`.
- ``branching`` -- a breadth-first tree taking both the reflected and the
  refracted child at every collision.

Both call :mod:`raytracer.math.intersections` for the root-finding and
:mod:`raytracer.optics.laws` for the physics; neither re-derives either.
"""

from .branching import BranchingTracer, RayLabeler, RayNode, RayTree, TraceConfig
from .fields import (
    FieldPoint,
    PupilSampling,
    PupilTrace,
    chief_ray_slope,
    chief_ray_slopes,
    trace_from_object,
    trace_pupil,
)
from .paraxial import (
    ConjugateSolution,
    ParaxialModel,
    differential_conjugates,
    direction_from_slopes,
    solve_object_plane,
)
from .sequential import BatchTraceResult, SequentialTracer, TraceResult, TraceStatus

__all__ = [
    "BranchingTracer",
    "TraceConfig",
    "RayNode",
    "RayTree",
    "RayLabeler",
    "SequentialTracer",
    "TraceResult",
    "BatchTraceResult",
    "TraceStatus",
    "ConjugateSolution",
    "ParaxialModel",
    "differential_conjugates",
    "solve_object_plane",
    "direction_from_slopes",
    "FieldPoint",
    "PupilSampling",
    "PupilTrace",
    "chief_ray_slope",
    "chief_ray_slopes",
    "trace_from_object",
    "trace_pupil",
]
