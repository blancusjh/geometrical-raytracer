"""2-D ray primitives and genealogy helpers for the non-sequential engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Dict, Iterable, Optional

import numpy as np
from numpy.typing import ArrayLike

from ..core.vectors import as_vector, direction_from_angle, normalize


@dataclass
class Ray2D:
    """Half-line defined by a 2-D origin and unit direction."""

    origin: ArrayLike
    direction: ArrayLike

    def __post_init__(self) -> None:
        self.origin = as_vector(self.origin, dim=2)
        self.direction = normalize(self.direction)

    def point_at(self, lam: float) -> np.ndarray:
        return self.origin + float(lam) * self.direction


@dataclass
class Intersection2D:
    """Hit record storing the intersection between a ray and a surface."""

    point: ArrayLike
    normal: ArrayLike
    distance: float
    surface_id: str
    parameters: Optional[np.ndarray] = None
    surface: object | None = None
    meta: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        self.point = as_vector(self.point, dim=2)
        self.normal = normalize(self.normal)
        if self.parameters is not None:
            self.parameters = np.asarray(self.parameters, dtype=float)
        if self.meta is None:
            self.meta = {}


@dataclass
class RayNode:
    """Node inside a ray genealogy tree.

    ``opl`` is the optical path length accumulated from the tree root up to
    this node's origin; ``intensity`` is the radiometric weight carried by
    the ray (unit for sources unless configured otherwise).
    """

    label: str
    ray: Ray2D
    parent_label: Optional[str]
    generation: int
    medium_n: float
    intersection: Optional[Intersection2D] = None
    children: list[str] = field(default_factory=list)
    opl: float = 0.0
    intensity: float = 1.0
    wavelength_um: Optional[float] = None
    kind: str = "primary"  # "primary" | "reflected" | "refracted"


class RayTree:
    """Container that stores traced rays by label."""

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

    def __len__(self) -> int:
        return len(self._nodes)

    def nodes(self) -> Iterable[RayNode]:
        return self._nodes.values()


class RayLabeler:
    """Utility producing readable, unique ray labels."""

    def __init__(self) -> None:
        self._counter = count(1)

    def new_primary(self) -> str:
        return f"ray_{next(self._counter)}"

    def child(self, parent_label: str) -> str:
        return f"{parent_label}.{next(self._counter)}"


def ray_from_angle(origin: ArrayLike, theta: float) -> Ray2D:
    """Convenience helper that spawns a ray from an origin and angle."""

    return Ray2D(origin=origin, direction=direction_from_angle(theta))


__all__ = [
    "Ray2D",
    "Intersection2D",
    "RayNode",
    "RayTree",
    "RayLabeler",
    "ray_from_angle",
]
