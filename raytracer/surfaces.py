"""2-D surface abstractions used by the simplified tracer."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .rays import Intersection2D, Ray2D, normalize


class Surface2D:
    """Abstract base class for 2-D surfaces with optional medium indices."""

    def __init__(
        self,
        *,
        surface_id: str = "surface",
        n_exterior: float = 1.0,
        n_interior: float = 1.0,
    ) -> None:
        self.surface_id = surface_id
        self.n_exterior = float(n_exterior)
        self.n_interior = float(n_interior)

    def polyline(self, samples: int = 512) -> np.ndarray:
        """Return vertices approximating the surface in world space."""

        raise NotImplementedError

    def intersect(self, ray: Ray2D) -> Optional[Intersection2D]:
        """Return the first intersection between *ray* and the surface."""

        raise NotImplementedError


class LocalFrame:
    """Rigid transform that maps between local and world coordinates."""

    def __init__(self, origin: np.ndarray, rotation: np.ndarray) -> None:
        self.origin = np.asarray(origin, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        if rotation.shape != (2, 2):
            raise ValueError("rotation must be a 2x2 matrix")
        self.rotation = rotation
        self.inv_rotation = rotation.T

    def to_world(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float)
        return self.rotation @ point + self.origin

    def to_local(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float)
        return self.inv_rotation @ (point - self.origin)

    def direction_to_world(self, direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction, dtype=float)
        return self.rotation @ direction

    def direction_to_local(self, direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction, dtype=float)
        return self.inv_rotation @ direction


__all__ = ["Surface2D", "LocalFrame", "Intersection2D", "Ray2D", "normalize"]
