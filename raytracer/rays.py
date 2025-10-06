"""Core ray data structures and genealogy utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Dict, Iterable, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .geometry import Surface2D

EPS = 1e-9


def normalize(vec: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Return the unit version of *vec*, raising for near-zero vectors."""
    vec = np.asarray(vec, dtype=float)
    mag = np.linalg.norm(vec)
    if mag <= eps:
        raise ValueError("Cannot normalize zero vector.")
    return vec / mag


@dataclass
class Ray:
    """A half-line defined by origin and direction."""

    origin: np.ndarray
    direction: np.ndarray

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=float)
        self.direction = normalize(self.direction)


@dataclass
class Intersection2D:
    """Hit record storing the location, normal and distance along the ray."""

    point: np.ndarray
    normal: np.ndarray
    distance: float
    surface_id: str
    meta: Dict[str, float] = field(default_factory=dict)
    surface: "Surface2D | None" = None


@dataclass
class RayNode:
    """Node inside a ray genealogy tree."""

    label: str
    ray: Ray
    parent_label: Optional[str]
    generation: int
    intersection: Optional[Intersection2D] = None
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
