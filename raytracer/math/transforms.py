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
        self.origin = np.array(origin, dtype=float, copy=True)
        rotation = np.array(rotation, dtype=float, copy=True)
        if self.origin.ndim != 1 or not np.all(np.isfinite(self.origin)):
            raise ValueError("origin must be a finite coordinate vector")
        dim = self.origin.shape[0]
        if rotation.shape != (dim, dim):
            raise ValueError(f"rotation must be a {dim}x{dim} matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(dim), atol=1e-12, rtol=0):
            raise ValueError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12, rtol=0):
            raise ValueError("rotation must preserve orientation")
        self.rotation = rotation
        self.inv_rotation = rotation.T

    @classmethod
    def identity(cls, dimension: int = 3) -> "RigidTransform":
        return cls(np.zeros(dimension), np.eye(dimension))

    @classmethod
    def from_euler_xyz(cls, origin=(0.0, 0.0, 0.0), angles_deg=(0.0, 0.0, 0.0)):
        """Active fixed-axis rotations: x first, then y, then z; degrees."""

        x, y, z = np.deg2rad(angles_deg)
        cx, cy, cz = np.cos([x, y, z])
        sx, sy, sz = np.sin([x, y, z])
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return cls(origin, rz @ ry @ rx)

    def compose(self, local: "RigidTransform") -> "RigidTransform":
        """Place a local frame inside this frame."""

        return RigidTransform(self.to_world(local.origin), self.rotation @ local.rotation)

    @classmethod
    def from_angle_2d(cls, origin: np.ndarray, angle: float) -> "RigidTransform":
        return cls(origin=origin, rotation=rotation_2d(angle))

    def to_world(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float)
        return point @ self.rotation.T + self.origin

    def to_local(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float)
        return (point - self.origin) @ self.rotation

    def direction_to_world(self, direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction, dtype=float)
        return direction @ self.rotation.T

    def direction_to_local(self, direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction, dtype=float)
        return direction @ self.rotation


__all__ = ["RigidTransform"]
