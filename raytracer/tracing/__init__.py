"""Tracing: the algorithms that carry light through an optical system.

Both tracers take :class:`~raytracer.rays.Rays` and return the same rays,
carried further; they differ only in how each ray's next boundary is
chosen, and in whether light divides.

- :mod:`.sequential` — every ray along the system's path, vectorized.
- :mod:`.nonsequential` — every ray wherever the physics sends it.
- :mod:`.arrival` — what both do where rays meet a boundary.
- :mod:`.paraxial` — conjugates and ABCD matrices of a coaxial path.
- :mod:`.fields` — field points, chief rays, pupil sampling.
"""

from .fields import (
    FieldPoint,
    PupilSampling,
    PupilTrace,
    chief_ray_slopes,
    trace_from_object,
    trace_pupil,
)
from .nonsequential import NonSequentialTracer
from .paraxial import ConjugateSolution, ParaxialModel, differential_conjugates, solve_object_plane
from .sequential import SequentialTracer

__all__ = [
    "SequentialTracer",
    "NonSequentialTracer",
    "ConjugateSolution",
    "ParaxialModel",
    "differential_conjugates",
    "solve_object_plane",
    "FieldPoint",
    "PupilSampling",
    "PupilTrace",
    "chief_ray_slopes",
    "trace_from_object",
    "trace_pupil",
]
