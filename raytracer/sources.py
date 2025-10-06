"""Light source primitives that emit batches of rays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        raise ValueError("Direction vector cannot be zero.")
    return vec / norm


@dataclass
class RaySeed:
    """Pack origin/direction parameters before the Ray object is created."""

    origin: np.ndarray
    direction: np.ndarray
    params: Dict[str, Any] | None = None


class Source2D:
    """Base class for 2D sources."""

    def emit(self) -> List[RaySeed]:
        raise NotImplementedError


@dataclass
class PointSource2D(Source2D):
    """Point source emitting rays within an angular aperture."""

    origin: np.ndarray
    axis_direction: np.ndarray
    aperture: float
    samples: int

    def emit(self) -> List[RaySeed]:
        axis = _unit(self.axis_direction)
        alpha0 = np.arctan2(axis[1], axis[0])
        half = self.aperture / 2.0
        thetas = np.linspace(-half, half, self.samples)
        seeds: List[RaySeed] = []
        base_origin = np.asarray(self.origin, dtype=float)
        for theta in thetas:
            angle = alpha0 + theta
            seeds.append(
                RaySeed(
                    origin=base_origin.copy(),
                    direction=np.array([np.cos(angle), np.sin(angle)]),
                    params={"theta": float(angle)},
                )
            )
        return seeds


@dataclass
class ParallelSource2D(Source2D):
    """Generator for a bundle of parallel rays across a finite width."""

    origin: np.ndarray
    direction: np.ndarray
    width: float
    samples: int

    def emit(self) -> List[RaySeed]:
        direction = _unit(self.direction)
        ortho = np.array([-direction[1], direction[0]])
        ortho = _unit(ortho)
        offsets = np.linspace(-self.width / 2.0, self.width / 2.0, self.samples)
        seeds: List[RaySeed] = []
        base_origin = np.asarray(self.origin, dtype=float)
        for offset in offsets:
            seeds.append(
                RaySeed(
                    origin=base_origin + offset * ortho,
                    direction=direction.copy(),
                    params={"offset": float(offset)},
                )
            )
        return seeds
