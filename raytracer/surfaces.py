"""Dimension-agnostic surface abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .rays import IntersectionND, RayND, normalize


@dataclass
class SurfaceIntersection:
    """Intermediate payload returned by surface intersection solvers."""

    distance: float
    parameters: np.ndarray
    point: Optional[np.ndarray] = None
    normal: Optional[np.ndarray] = None
    meta: Optional[Dict[str, float]] = None


class SurfaceND:
    """Abstract surface embedded in *dimension* dimensional space."""

    def __init__(self, *, dimension: int, parameter_dimension: Optional[int] = None, surface_id: str = "surface") -> None:
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self.dimension = int(dimension)
        self.parameter_dimension = int(parameter_dimension) if parameter_dimension is not None else self.dimension
        self.surface_id = surface_id

    # -- Hooks -----------------------------------------------------------------
    def point_from_parameters(self, parameters: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def normal_from_parameters(self, parameters: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def solve_intersection(self, ray: RayND) -> SurfaceIntersection | None:
        raise NotImplementedError

    # -- Public API -------------------------------------------------------------
    def first_intersection(self, ray: RayND) -> IntersectionND | None:
        if ray.dimension != self.dimension:
            raise ValueError(
                f"Ray dimensionality {ray.dimension} does not match surface dimension {self.dimension}."
            )

        hit = self.solve_intersection(ray)
        if hit is None:
            return None

        params = np.asarray(hit.parameters, dtype=float)
        if params.ndim != 1 or params.shape[0] != self.parameter_dimension:
            raise ValueError(
                f"Expected parameter vector of length {self.parameter_dimension}, got {params.shape}"
            )

        point = hit.point if hit.point is not None else self.point_from_parameters(params)
        normal = hit.normal if hit.normal is not None else self.normal_from_parameters(params)

        # Ensure outputs respect dimensional expectations.
        point = np.asarray(point, dtype=float)
        normal = normalize(normal)
        if point.shape != (self.dimension,):
            raise ValueError(f"point_from_parameters returned shape {point.shape}, expected {(self.dimension,)}")
        if normal.shape != (self.dimension,):
            raise ValueError(f"normal_from_parameters returned shape {normal.shape}, expected {(self.dimension,)}")

        meta = hit.meta or {}
        return IntersectionND(
            point=point,
            normal=normal,
            distance=float(hit.distance),
            surface_id=self.surface_id,
            parameters=params,
            meta=meta,
            surface=self,
        )


class LocalFrame:
    """Affine transform that maps between local and world coordinates."""

    def __init__(self, origin: np.ndarray, rotation: np.ndarray) -> None:
        self.origin = np.asarray(origin, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
            raise ValueError("rotation must be a square matrix")
        self.rotation = rotation
        self.inv_rotation = rotation.T

    def to_world(self, point: np.ndarray) -> np.ndarray:
        return self.rotation @ np.asarray(point, dtype=float) + self.origin

    def to_local(self, point: np.ndarray) -> np.ndarray:
        return self.inv_rotation @ (np.asarray(point, dtype=float) - self.origin)

    def direction_to_world(self, direction: np.ndarray) -> np.ndarray:
        return self.rotation @ np.asarray(direction, dtype=float)

    def direction_to_local(self, direction: np.ndarray) -> np.ndarray:
        return self.inv_rotation @ np.asarray(direction, dtype=float)
