"""Optical ray tracing on one physical model.

An optical system is **elements placed in an ambient medium**. Each piece of
the package answers one question, and imports only from the pieces above
it in this list:

- :mod:`~raytracer.geometry` — shape and place, no optics: rotations and
  poses, surfaces, apertures, solids, ray/surface intersection.
- :mod:`~raytracer.materials` — the media that fill space.
- :mod:`~raytracer.coatings` — how much light a boundary reflects and transmits.
- :mod:`~raytracer.interfaces` — a surface as an optical boundary, and the
  laws of reflection and refraction.
- :mod:`~raytracer.elements` — lenses, mirrors, beam splitters, prisms,
  stops, sensors.
- :mod:`~raytracer.system` — :class:`OpticalSystem` and :class:`Assembly`;
  the media on each side of each boundary and the sequential path follow
  from the placed elements.
- :mod:`~raytracer.tracing` — the engines: sequential (along the path) and
  non-sequential (wherever light goes), plus paraxial optics and pupils.
- :mod:`~raytracer.sources` — emitters for non-sequential tracing.
- :mod:`~raytracer.families` — generators of systems (stigmatic trains).
- :mod:`~raytracer.io` — reading and writing designs and catalogues.
- :mod:`~raytracer.analysis` — what is measured on a traced system.
- :mod:`~raytracer.optimize` — moving a family's parameters.
- :mod:`~raytracer.viz` — figures and the OpenGL viewer.

The namespaces are the API (``from raytracer import elements as el``);
the top level holds only the system and the most used names.
"""

from . import coatings, elements, geometry, interfaces, materials, system
from .geometry import Pose, Rotation
from .interfaces import Interface
from .materials import AIR, VACUUM
from .system import Assembly, Gap, OpticalSystem, TableRow

__version__ = "0.4.0"

__all__ = [
    "OpticalSystem",
    "Assembly",
    "Gap",
    "TableRow",
    "Interface",
    "Pose",
    "Rotation",
    "AIR",
    "VACUUM",
    "geometry",
    "materials",
    "coatings",
    "interfaces",
    "elements",
    "system",
]
