"""Simple ray sources for 2-D scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class RaySeed:
    """Origin/direction pair emitted by a light source."""

    origin: np.ndarray
    direction: np.ndarray
    params: Optional[dict] = None


class Source2D:
    """Base class for 2-D ray sources."""

    samples: int

    def emit(self) -> List[RaySeed]:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class PointSource2D(Source2D):
    """Point source emitting rays across an angular aperture."""

    origin: np.ndarray
    axis_direction: np.ndarray
    aperture: float
    samples: int

    def emit(self) -> List[RaySeed]:
        axis = _unit(self.axis_direction)
        base_angle = float(np.arctan2(axis[1], axis[0]))
        half = float(self.aperture) / 2.0
        thetas = np.linspace(-half, half, self.samples)
        seeds: List[RaySeed] = []
        base_origin = np.asarray(self.origin, dtype=float)
        for offset in thetas:
            angle = base_angle + offset
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
    """Bundle of parallel rays sampled across a finite width."""

    origin: np.ndarray
    direction: np.ndarray
    width: float
    samples: int

    def emit(self) -> List[RaySeed]:
        direction = _unit(self.direction)
        ortho = np.array([-direction[1], direction[0]])
        offsets = np.linspace(-self.width / 2.0, self.width / 2.0, self.samples)
        base_origin = np.asarray(self.origin, dtype=float)
        seeds: List[RaySeed] = []
        for offset in offsets:
            seeds.append(
                RaySeed(
                    origin=base_origin + offset * ortho,
                    direction=direction.copy(),
                    params={"offset": float(offset)},
                )
            )
        return seeds


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        raise ValueError("Direction vector cannot be zero.")
    return vec / norm


__all__ = ["RaySeed", "Source2D", "PointSource2D", "ParallelSource2D"]
