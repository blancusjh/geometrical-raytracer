"""Optical surfaces: smooth boundaries, as geometry only.

Hierarchy (a subclass is a subset of its parent's family)::

    Surface
    ├── SurfaceOfRevolution
    │   ├── Conic
    │   │   ├── Plane            (c = 0: every family's flat member)
    │   │   ├── Ellipsoid        (k > -1)
    │   │   │   └── Sphere       (k = 0)
    │   │   ├── Paraboloid       (k = -1)
    │   │   └── Hyperboloid      (k < -1)
    │   ├── CartesianSurface     (Descartes' stigmatic surface)
    │   └── Asphere              (conic base + departure)
    │       ├── EvenAsphere
    │       └── QTypeAsphere
    ├── ToroidalSurface
    │   └── CylindricalSurface   (infinite radius of rotation)
    └── FreeformSurface
"""

from .aspheres import Asphere, EvenAsphere, QTypeAsphere
from .base import Surface, SurfaceOfRevolution
from .cartesian import CartesianSurface
from .conics import Conic, Ellipsoid, Hyperboloid, Paraboloid, Plane, Sphere
from .freeform import FreeformSurface
from .toroidal import CylindricalSurface, ToroidalSurface

__all__ = [
    "Surface",
    "SurfaceOfRevolution",
    "Conic",
    "Plane",
    "Ellipsoid",
    "Sphere",
    "Paraboloid",
    "Hyperboloid",
    "CartesianSurface",
    "Asphere",
    "EvenAsphere",
    "QTypeAsphere",
    "ToroidalSurface",
    "CylindricalSurface",
    "FreeformSurface",
]
