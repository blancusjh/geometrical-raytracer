"""The light primitive: a ray is an origin and a direction, nothing more.

A ``Ray`` only knows how to extend itself through space (:meth:`Ray.point_at`).
It has no notion of hitting anything — detecting where a ray meets a
surface is the surface's job (:mod:`raytracer.surfaces.surface`), and
deciding what a source emits and what a hit produces next is the
propagation algorithm's job (:mod:`raytracer.propagation`). Keeping those
three responsibilities apart is deliberate: conflating "what a ray is"
with "how it collides" is exactly the mistake this module exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..math.vectors import as_vector, direction_from_angle, normalize


@dataclass
class Ray:
    """Half-line defined by an origin and a unit direction (any dimension)."""

    origin: ArrayLike
    direction: ArrayLike

    def __post_init__(self) -> None:
        self.origin = as_vector(self.origin)
        self.direction = normalize(self.direction)

    def point_at(self, lam: float) -> np.ndarray:
        return self.origin + float(lam) * self.direction


def ray_from_angle(origin: ArrayLike, theta: float) -> Ray:
    """Convenience helper that spawns a 2-D ray from an origin and angle."""

    return Ray(origin=origin, direction=direction_from_angle(theta))


__all__ = ["Ray", "ray_from_angle"]
