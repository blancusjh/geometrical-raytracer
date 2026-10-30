"""What a surface is: a smooth boundary given as a graph in its own frame.

Every optical surface here is described in a **local frame** with its
vertex at the origin and its axis along +z, as the graph ``z = s(x, y)``.
Where it sits in the world is not the surface's business — an element's
pose places it. What light does there is not the surface's business
either — the media on its two sides and its coating decide that. A
surface is geometry, and only geometry.

Sign convention: a positive radius puts the centre of curvature on the +z
side of the vertex, so the sag grows toward +z.

Inheritance follows mathematical specialization. A subclass is a subset of
its parent's family: every :class:`Sphere` is an :class:`Ellipsoid`, every
:class:`Ellipsoid` a :class:`Conic`, every conic a
:class:`SurfaceOfRevolution`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields

import numpy as np
from numpy.typing import ArrayLike

from ..intersections import SurfaceHits, newton_graph


class Surface(ABC):
    """A smooth surface ``z = s(x, y)`` in its local frame."""

    # -- what every surface states --------------------------------------

    @abstractmethod
    def sag_gradient(self, x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(s, ∂s/∂x, ∂s/∂y)`` at the points ``(x, y)``."""

    # -- what follows from it -------------------------------------------

    def sag(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        return self.sag_gradient(x, y)[0]

    def implicit(self, points: ArrayLike) -> np.ndarray:
        """``f(p) = z - s(x, y)``: zero on the surface, positive on its +z side."""

        p = np.asarray(points, dtype=float)
        return p[..., 2] - self.sag(p[..., 0], p[..., 1])

    def normal(self, points: ArrayLike) -> np.ndarray:
        """Unit normals ``∇f / |∇f|``, pointing to the surface's +z side."""

        p = np.asarray(points, dtype=float)
        _, dsdx, dsdy = self.sag_gradient(p[..., 0], p[..., 1])
        n = np.stack([-dsdx, -dsdy, np.ones_like(dsdx)], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def intersect(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        alive: np.ndarray | None = None,
        *,
        mode: str = "nearest",
        t_min: float = 1e-9,
    ) -> SurfaceHits:
        """Solve ``f(o + t d) = 0`` for many rays given in the local frame."""

        if alive is None:
            alive = np.ones(origins.shape[0], dtype=bool)
        return newton_graph(self, origins, directions, alive, mode=mode, t_min=t_min)

    @property
    def vertex_curvature(self) -> tuple[float, float]:
        """``(∂²s/∂x², ∂²s/∂y²)`` at the vertex: what paraxial optics sees."""

        h = 1e-4
        x = np.array([-h, 0.0, h])
        zero = np.zeros(3)
        sx = self.sag(x, zero)
        sy = self.sag(zero, x)
        return (float((sx[0] - 2 * sx[1] + sx[2]) / h**2), float((sy[0] - 2 * sy[1] + sy[2]) / h**2))

    @property
    def max_radius(self) -> float:
        """Largest distance from the axis where the sag is defined (``inf`` if everywhere)."""

        return np.inf

    @property
    def is_plane(self) -> bool:
        return False

    def __repr__(self) -> str:
        args = ", ".join(f"{f.name}={getattr(self, f.name)!r}" for f in fields(self) if f.init)
        return f"{type(self).__name__}({args})"


class SurfaceOfRevolution(Surface):
    """A surface symmetric about its local z axis: ``s(x, y) = s(r)``."""

    @abstractmethod
    def sag_and_slope(self, r: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """``(s(r), ds/dr)`` at radial distance(s) *r* ≥ 0."""

    def sag_gradient(self, x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        r = np.hypot(x, y)
        s, slope = self.sag_and_slope(r)
        safe_r = np.where(r == 0.0, 1.0, r)
        return s, np.where(r > 0.0, slope * x / safe_r, 0.0), np.where(r > 0.0, slope * y / safe_r, 0.0)

    def profile(self, r: ArrayLike) -> np.ndarray:
        """The meridional curve ``s(r)``."""

        return self.sag_and_slope(np.abs(np.asarray(r, dtype=float)))[0]

    @property
    def vertex_curvature(self) -> tuple[float, float]:
        c = self.curvature
        return (c, c)

    @property
    def curvature(self) -> float:
        """Vertex curvature ``s''(0)``."""

        h = 1e-4
        s = self.sag_and_slope(np.array([0.0, h]))[0]
        return float(2.0 * (s[1] - s[0]) / h**2)

    @property
    def radius(self) -> float:
        """Vertex radius of curvature; ``inf`` when flat at the vertex."""

        c = self.curvature
        return np.inf if c == 0.0 else 1.0 / c


__all__ = ["Surface", "SurfaceOfRevolution"]
