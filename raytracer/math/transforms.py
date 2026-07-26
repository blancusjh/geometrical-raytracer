"""Rigid transforms between local and world coordinates (2-D or 3-D)."""

from __future__ import annotations

import numpy as np

from .vectors import rotation_2d


class RigidTransform:
    """Rigid transform mapping local coordinates to world coordinates.

    Works for any dimension: *rotation* must be a square orthonormal matrix
    matching the length of *origin*.
    """

    def __init__(self, origin: np.ndarray, rotation: np.ndarray) -> None:
        self.origin = np.asarray(origin, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        dim = self.origin.shape[0]
        if rotation.shape != (dim, dim):
            raise ValueError(f"rotation must be a {dim}x{dim} matrix")
        self.rotation = rotation
        self.inv_rotation = rotation.T

    @classmethod
    def from_angle_2d(cls, origin: np.ndarray, angle: float) -> "RigidTransform":
        return cls(origin=origin, rotation=rotation_2d(angle))

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


__all__ = ["RigidTransform"]
