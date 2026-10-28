"""Clear apertures in a surface's local transverse coordinates, in mm."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Aperture(Protocol):
    def contains(self, xy: np.ndarray, slack: float = 0.0) -> np.ndarray: ...


@dataclass(frozen=True)
class CircularAperture:
    radius: float
    inner_radius: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.radius) or not 0 <= self.inner_radius < self.radius:
            raise ValueError("aperture radii must satisfy 0 <= inner < outer < infinity")

    def contains(self, xy, slack=0.0):
        radius = np.linalg.norm(np.asarray(xy), axis=-1)
        return (radius <= self.radius + slack) & (radius >= self.inner_radius - slack)


@dataclass(frozen=True)
class RectangularAperture:
    half_width: float
    half_height: float

    def __post_init__(self):
        dimensions = np.array([self.half_width, self.half_height])
        if not np.all(np.isfinite(dimensions) & (dimensions > 0)):
            raise ValueError("aperture half dimensions must be finite and positive")

    def contains(self, xy, slack=0.0):
        return np.all(np.abs(xy) <= np.array([self.half_width, self.half_height]) + slack, axis=-1)
