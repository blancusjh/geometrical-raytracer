"""2-D surface abstraction used by the non-sequential tracer."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .rays import Intersection2D, Ray2D


class Surface2D:
    """Abstract base class for 2-D surfaces with optional medium indices.

    ``absorbing`` surfaces terminate rays: the tracer records the hit but
    spawns no children (used by detectors, baffles, and lens rims).
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

    def intersect(self, ray: Ray2D) -> Optional[Intersection2D]:
        """Return the first intersection between *ray* and the surface."""

        raise NotImplementedError


__all__ = ["Surface2D"]
