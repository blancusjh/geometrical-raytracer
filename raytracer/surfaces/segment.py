"""The straight segment: baffles, lens rims, and the body of a screen.

A segment through ``p0`` and ``p1`` with normal ``n`` has the implicit
description ``f_Sigma(x) = n . (x - p0)``, a degenerate (linear) quadratic —
so the solver reaches it in closed form, and its gradient is the constant
normal. Its finite extent is the clear aperture: ``0 <= s <= 1`` along the
segment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..math.vectors import perpendicular_2d
from .surface import Surface


@dataclass
class LineSegment(Surface):
    """Straight segment between two points (baffles, rims, screens)."""

    p0: np.ndarray = field(default_factory=lambda: np.zeros(2))
    p1: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    surface_id: str = "segment"
    n_exterior: float = 1.0
    n_interior: float = 1.0
    absorbing: bool = False

    def __post_init__(self) -> None:
        Surface.__init__(
            self,
            surface_id=self.surface_id,
            n_exterior=self.n_exterior,
            n_interior=self.n_interior,
            absorbing=self.absorbing,
        )
        self.p0 = np.asarray(self.p0, dtype=float)
        self.p1 = np.asarray(self.p1, dtype=float)
        normal = perpendicular_2d(self.p1 - self.p0)
        # f_Sigma = n . (x - p0) = n_x x + n_y y - n . p0
        self.quadratic_form = (
            0.0, 0.0, 0.0,
            float(normal[0]), float(normal[1]), float(-np.dot(normal, self.p0)),
        )
        self._normal = normal

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    # -- implicit description ----------------------------------------------

    def implicit(self, point: np.ndarray) -> float:
        return float(np.dot(self._normal, np.asarray(point, dtype=float) - self.p0))

    def implicit_gradient(self, point: np.ndarray) -> np.ndarray:
        return self._normal

    def fraction_along(self, point: np.ndarray) -> float:
        """Position of *point* along the segment, ``0`` at ``p0`` and ``1`` at ``p1``."""

        edge = self.p1 - self.p0
        return float(np.dot(np.asarray(point, dtype=float) - self.p0, edge) / np.dot(edge, edge))

    def within_aperture(self, point: np.ndarray) -> bool:
        return 0.0 <= self.fraction_along(point) <= 1.0

    def local_parameters(self, point: np.ndarray) -> np.ndarray:
        return np.array([self.fraction_along(point)])

    # -- drawing ------------------------------------------------------------

    def polyline(self, samples: int = 2) -> np.ndarray:
        return np.array([self.p0, self.p1])


__all__ = ["LineSegment"]
