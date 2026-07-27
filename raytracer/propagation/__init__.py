"""Propagation algorithms: emit rays, prolong them to a collision, decide
the new ray(s), repeat.

This is the *only* place that governs what a ray does after it hits a
surface. Two independent strategies are provided:

- :mod:`raytracer.propagation.sequential` — every ray through every surface
  of a :class:`~raytracer.design.system.OpticalSystem` in a fixed order,
  one deterministic path each, vectorized over many rays. ``paraxial`` and
  ``fields`` build on it (chief-ray/pupil solving, conjugate recovery) --
  they are sequential-specific, which is why they live here rather than
  alongside the system's data model in :mod:`raytracer.design`.
- :mod:`raytracer.propagation.branching` — a breadth-first tree exploring
  both the reflected and refracted child at every collision.

Both tracers call into :mod:`raytracer.math.intersections` for the actual
root-finding and :mod:`raytracer.optics.laws` for the reflection/refraction/
Fresnel laws; neither re-derives that math itself.
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
