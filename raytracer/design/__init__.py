"""The data model of an engineered system: ``SurfaceRow`` and ``OpticalSystem``.

Describes *what* a system is, holding no propagation logic -- tracing it or
solving its chief rays needs :mod:`raytracer.propagation`, and a design must
not depend on an engine to be a design. Reading it from a file is a
different nature again: :mod:`raytracer.io`.
"""

from .bridge import to_branching_surfaces
from .rows import SurfaceKind, SurfaceRow
from .system import OpticalSystem

__all__ = [
    "SurfaceKind",
    "SurfaceRow",
    "OpticalSystem",
    "to_branching_surfaces",
]
