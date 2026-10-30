"""Optical systems: elements and assemblies placed in an ambient medium.

The derived facts — which medium lies on each side of each boundary, and
the sequential path — live in :mod:`.faces`; the concise ways of writing a
system (composition, prescription tables) in :mod:`.table`.
"""

from .assembly import Assembly
from .faces import DETECT, REFLECT, STOP, TRANSMIT, PhysicalFace, Step
from .optical_system import OpticalSystem
from .table import Gap, TableRow

__all__ = [
    "OpticalSystem",
    "Assembly",
    "Gap",
    "TableRow",
    "PhysicalFace",
    "Step",
    "TRANSMIT",
    "REFLECT",
    "STOP",
    "DETECT",
]
