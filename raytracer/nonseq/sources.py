"""Ray sources for 2-D non-sequential scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..core.vectors import normalize, perpendicular_2d


@dataclass
class RaySeed:
    """Origin/direction pair emitted by a light source."""

    origin: np.ndarray
    direction: np.ndarray
    intensity: float = 1.0
    wavelength_um: Optional[float] = None
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
    intensity: float = 1.0
    wavelength_um: Optional[float] = None

    def emit(self) -> List[RaySeed]:
        axis = normalize(self.axis_direction)
        base_angle = float(np.arctan2(axis[1], axis[0]))
        half = float(self.aperture) / 2.0
        thetas = np.linspace(-half, half, self.samples)
        base_origin = np.asarray(self.origin, dtype=float)
        seeds: List[RaySeed] = []
        for offset in thetas:
            angle = base_angle + offset
            seeds.append(
                RaySeed(
                    origin=base_origin.copy(),
                    direction=np.array([np.cos(angle), np.sin(angle)]),
                    intensity=self.intensity,
                    wavelength_um=self.wavelength_um,
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
    intensity: float = 1.0
    wavelength_um: Optional[float] = None

    def emit(self) -> List[RaySeed]:
        direction = normalize(self.direction)
        ortho = perpendicular_2d(direction)
        offsets = np.linspace(-self.width / 2.0, self.width / 2.0, self.samples)
        base_origin = np.asarray(self.origin, dtype=float)
        seeds: List[RaySeed] = []
        for offset in offsets:
            seeds.append(
                RaySeed(
                    origin=base_origin + offset * ortho,
                    direction=direction.copy(),
                    intensity=self.intensity,
                    wavelength_um=self.wavelength_um,
                    params={"offset": float(offset)},
                )
            )
        return seeds


__all__ = ["RaySeed", "Source2D", "PointSource2D", "ParallelSource2D"]
