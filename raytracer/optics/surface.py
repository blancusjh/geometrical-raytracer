"""The boundary between two media: a hit-testable surface.

A ``Surface`` is the interface a :class:`~raytracer.optics.ray.Ray` collides
with. Hit-testing (:meth:`Surface.hit`) delegates its actual numerics to
:mod:`raytracer.math.intersections` and the shape definitions in
:mod:`raytracer.shapes` — a ``Surface`` owns the medium/aperture bookkeeping
and the coordinate frame, not the root-finding itself. What happens after a
hit (reflect, refract, stop, spawn a child ray) is the propagation
algorithm's decision, not the surface's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from numpy.typing import ArrayLike

from ..math.vectors import as_vector, normalize
from .ray import Ray


class Surface:
    """Abstract base class for surfaces with optional medium indices.

    ``absorbing`` surfaces terminate rays: the tracer records the hit but
    spawns no children (used by instruments, baffles, and lens rims).
    """

    def __init__(
        self,
        *,
        surface_id: str = "surface",
        n_exterior: float = 1.0,
        n_interior: float = 1.0,
        absorbing: bool = False,
    ) -> None:
        self.surface_id = surface_id
        self.n_exterior = float(n_exterior)
        self.n_interior = float(n_interior)
        self.absorbing = bool(absorbing)

    def polyline(self, samples: int = 512) -> np.ndarray:
        """Return vertices approximating the surface in world space."""

        raise NotImplementedError

    def hit(self, ray: Ray) -> Optional["Intersection"]:
        """Return the first intersection between *ray* and the surface."""

        raise NotImplementedError


@dataclass
class Intersection:
    """Hit record storing the intersection between a ray and a surface."""

    point: ArrayLike
    normal: ArrayLike
    distance: float
    surface_id: str
    parameters: Optional[np.ndarray] = None
    surface: object | None = None
    meta: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        self.point = as_vector(self.point)
        self.normal = normalize(self.normal)
        if self.parameters is not None:
            self.parameters = np.asarray(self.parameters, dtype=float)
        if self.meta is None:
            self.meta = {}


__all__ = ["Surface", "Intersection"]
