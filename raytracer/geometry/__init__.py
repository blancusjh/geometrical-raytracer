"""Geometry: where things are and what shape they have — no optics.

- :mod:`~raytracer.geometry.transforms` — :class:`Rotation`, :class:`Pose`.
- :mod:`~raytracer.geometry.surfaces` — smooth boundaries ``z = s(x, y)``.
- :mod:`~raytracer.geometry.apertures` — regions of a surface's (x, y) plane.
- :mod:`~raytracer.geometry.solids` — closed regions of space.
- :mod:`~raytracer.geometry.intersections` — ray/surface solvers.
"""

from . import apertures, solids, surfaces
from .transforms import Pose, Rotation
from .vectors import direction_from_angles, direction_from_slopes, normalize

__all__ = [
    "apertures",
    "solids",
    "surfaces",
    "Pose",
    "Rotation",
    "normalize",
    "direction_from_slopes",
    "direction_from_angles",
]
