"""Observation screens for the non-sequential 2-D engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .rays import Intersection2D, Ray2D
from .segments2d import LineSegment2D


@dataclass
class ScreenHit:
    """One ray arrival on a screen."""

    s: float  # position along the screen in [0, length]
    point: np.ndarray
    direction: np.ndarray
    intensity: float
    opl: float
    generation: int
    label: str
    wavelength_um: Optional[float] = None


class Screen2D(LineSegment2D):
    """Absorbing line-segment detector that records every ray arrival.

    The tracer notifies the screen through :meth:`on_hit`; accumulated
    arrivals expose position histograms (irradiance) and raw hit records
    for spot-style analysis.
    """

    def __init__(self, p0, p1, *, surface_id: str = "screen") -> None:
        super().__init__(
            p0=np.asarray(p0, dtype=float),
            p1=np.asarray(p1, dtype=float),
            surface_id=surface_id,
            absorbing=True,
        )
        self.hits: list[ScreenHit] = []

    def on_hit(self, node, hit: Intersection2D) -> None:
        s_norm = float(hit.parameters[0]) if hit.parameters is not None else 0.0
        self.hits.append(
            ScreenHit(
                s=s_norm * self.length,
                point=np.asarray(hit.point, dtype=float),
                direction=np.asarray(node.ray.direction, dtype=float),
                intensity=float(getattr(node, "intensity", 1.0)),
                opl=float(getattr(node, "opl", 0.0))
                + float(getattr(node, "medium_n", 1.0)) * float(hit.distance),
                generation=int(node.generation),
                label=node.label,
                wavelength_um=getattr(node, "wavelength_um", None),
            )
        )

    def clear(self) -> None:
        self.hits.clear()

    def coordinates(self) -> np.ndarray:
        """Arrival positions along the screen (mm from p0)."""

        return np.array([hit.s for hit in self.hits])

    def intensities(self) -> np.ndarray:
        return np.array([hit.intensity for hit in self.hits])

    def irradiance(self, bins: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """Intensity-weighted histogram along the screen.

        Returns (bin_edges, values); values integrate to the collected power.
        """

        edges = np.linspace(0.0, self.length, bins + 1)
        if not self.hits:
            return edges, np.zeros(bins)
        values, _ = np.histogram(
            self.coordinates(), bins=edges, weights=self.intensities()
        )
        return edges, values


__all__ = ["ScreenHit", "Screen2D"]
