"""Core ray data structures and genealogy utilities (dimension agnostic)."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Dict, Iterable, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .surfaces import SurfaceND  # pragma: no cover - imported lazily

EPS = 1e-9


def _as_vector(vec: ArrayLike, *, dim: int | None = None) -> np.ndarray:
    """Return *vec* coerced to a 1-D float array, optionally enforcing dimension."""

    arr = np.asarray(vec, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected a 1-D vector.")
    if dim is not None and arr.shape[0] != dim:
        raise ValueError(f"Expected vector of length {dim}, got {arr.shape[0]}")
    return arr


def normalize(vec: ArrayLike, eps: float = EPS) -> np.ndarray:
    """Return the unit version of *vec*, raising for near-zero vectors."""

    arr = _as_vector(vec)
    mag = np.linalg.norm(arr)
    if mag <= eps:
        raise ValueError("Cannot normalize zero-length vector.")
    return arr / mag


@dataclass
class RayND:
    """Half-line defined by origin and direction in arbitrary dimension."""

    origin: ArrayLike
    direction: ArrayLike

    def __post_init__(self) -> None:
        origin = _as_vector(self.origin)
        direction = normalize(self.direction)
        if origin.shape != direction.shape:
            raise ValueError("Origin and direction must share the same dimensionality.")
        self.origin = origin
        self.direction = direction

    @property
    def dimension(self) -> int:
        return self.origin.shape[0]

    def point_at(self, lam: float) -> np.ndarray:
        """Return the position along the ray at parameter *lam*."""

        return self.origin + lam * self.direction


class Ray2D(RayND):
    """2-D specialisation of :class:`RayND`."""

    def __init__(self, origin: ArrayLike, direction: ArrayLike) -> None:
        super().__init__(origin=origin, direction=direction)
        if self.dimension != 2:
            raise ValueError("Ray2D requires vectors of length 2.")


class Ray3D(RayND):
    """3-D specialisation of :class:`RayND`."""

    def __init__(self, origin: ArrayLike, direction: ArrayLike) -> None:
        super().__init__(origin=origin, direction=direction)
        if self.dimension != 3:
            raise ValueError("Ray3D requires vectors of length 3.")


def direction_from_angles(theta: float, phi: float | None = None) -> np.ndarray:
    """Return a unit direction vector for polar/spherical coordinates."""

    theta = float(theta)
    if phi is None:
        return np.array([np.cos(theta), np.sin(theta)], dtype=float)
    phi = float(phi)
    sin_theta = np.sin(theta)
    return np.array(
        [
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            np.cos(theta),
        ],
        dtype=float,
    )


def ray_from_angles(origin: ArrayLike, theta: float, phi: float | None = None) -> RayND:
    """Convenience constructor that builds a ray from angular coordinates."""

    if phi is None:
        return Ray2D(origin=origin, direction=direction_from_angles(theta))
    return Ray3D(origin=origin, direction=direction_from_angles(theta, phi))


@dataclass
class IntersectionND:
    """Hit record storing the location, normal and distance along the ray."""

    point: ArrayLike
    normal: ArrayLike
    distance: float
    surface_id: str
    parameters: np.ndarray | None = None
    meta: Dict[str, float] = field(default_factory=dict)
    surface: "SurfaceND | None" = None

    def __post_init__(self) -> None:
        point = _as_vector(self.point)
        normal = normalize(self.normal)
        if point.shape != normal.shape:
            raise ValueError("Point and normal must share the same dimensionality.")
        if self.parameters is not None:
            self.parameters = _as_vector(self.parameters)
        self.point = point
        self.normal = normal

    @property
    def dimension(self) -> int:
        return self.point.shape[0]


Intersection2D = IntersectionND
Intersection3D = IntersectionND


@dataclass
class RayNode:
    """Node inside a ray genealogy tree."""

    label: str
    ray: RayND
    parent_label: Optional[str]
    generation: int
    intersection: Optional[IntersectionND] = None
    children: list[str] = field(default_factory=list)
    medium_n: float = 1.0


class RayTree:
    """Container mapping ray labels to nodes while preserving ancestry."""

    def __init__(self) -> None:
        self._nodes: Dict[str, RayNode] = {}

    def add(self, node: RayNode) -> None:
        if node.label in self._nodes:
            raise ValueError(f"Duplicate ray label {node.label}")
        self._nodes[node.label] = node
        if node.parent_label:
            parent = self._nodes.get(node.parent_label)
            if parent is None:
                raise KeyError(f"Parent label {node.parent_label} missing for {node.label}")
            parent.children.append(node.label)

    def __getitem__(self, label: str) -> RayNode:
        return self._nodes[label]

    def nodes(self) -> Iterable[RayNode]:
        return self._nodes.values()


class RayLabeler:
    """Utility to assign stable unique labels to rays."""

    def __init__(self) -> None:
        self._counter = count(1)

    def _next_label(self) -> str:
        return f"ray_{next(self._counter)}"

    def new_primary(self) -> str:
        return self._next_label()

    def child(self, parent_label: str, branch_index: int) -> str:
        return self._next_label()


# Backwards compatibility re-exports for existing 2-D focused modules.
Ray = Ray2D
