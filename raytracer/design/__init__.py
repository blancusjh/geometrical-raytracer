"""Optical-system engineering: the data model of a designed system.

``SurfaceRow``/``SurfaceKind`` and ``OpticalSystem`` describe *what* a
system is (an ordered stack of surfaces along an axis) -- they hold no
propagation logic themselves. Actually tracing a system, solving for its
chief rays, or recovering its conjugate planes needs the propagation
algorithms in :mod:`raytracer.propagation` (which is why those live there,
not here, despite characterizing a design): a design must not depend on an
engine to be a design. Reading/writing a design from a file is a different
nature again -- see :mod:`raytracer.io`.
"""

from .bridge import to_branching_surfaces
from .surfaces import SurfaceKind, SurfaceRow
from .system import OpticalSystem

__all__ = [
    "SurfaceKind",
    "SurfaceRow",
    "OpticalSystem",
    "to_branching_surfaces",
]
