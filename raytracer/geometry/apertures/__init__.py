"""Apertures: regions of the local (x, y) plane where a surface exists.

An aperture is two-dimensional. It bounds a surface by projection along
the surface's local z axis: a point of the surface belongs to the bounded
piece when its ``(x, y)`` lies in the region. Each region is named by its
natural size — a circle by its radius, a rectangle by its width and
height — and may be decentred with ``center=``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Real
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike


class Aperture(ABC):
    """A closed region of the local (x, y) plane."""

    center: tuple[float, float] = (0.0, 0.0)

    @abstractmethod
    def contains(self, x: ArrayLike, y: ArrayLike, *, slack: float = 0.0) -> np.ndarray:
        """Whether each point lies in the region (grown by *slack* mm)."""

    @property
    @abstractmethod
    def extent(self) -> float:
        """Distance from the local axis to the farthest point of the region."""

    @abstractmethod
    def boundary(self, samples: int = 128) -> np.ndarray:
        """Points ``(M, 2)`` along the outline, for drawing."""

    @staticmethod
    def coerce(spec: "Aperture | float | None") -> "Aperture | None":
        """A bare number is a circle of that radius."""

        if spec is None or isinstance(spec, Aperture):
            return spec
        if isinstance(spec, Real):
            return Circle(float(spec))
        raise TypeError(f"cannot read {spec!r} as an aperture")

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self._key() == other._key()  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._key()))

    def _key(self) -> tuple:
        return tuple(vars(self).values())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in vars(self).items())})"


def _center(center: Sequence[float]) -> tuple[float, float]:
    cx, cy = (float(v) for v in center)
    return (cx, cy)


class Circle(Aperture):
    def __init__(self, radius: float, center: Sequence[float] = (0.0, 0.0)) -> None:
        if not float(radius) > 0.0:
            raise ValueError("a circle has a positive radius")
        self.radius = float(radius)
        self.center = _center(center)

    @property
    def extent(self) -> float:
        return float(np.hypot(*self.center) + self.radius)

    def contains(self, x, y, *, slack=0.0):
        return np.hypot(np.asarray(x) - self.center[0], np.asarray(y) - self.center[1]) <= self.radius + slack

    def boundary(self, samples=128):
        a = np.linspace(0.0, 2 * np.pi, samples)
        return np.column_stack([self.center[0] + self.radius * np.cos(a), self.center[1] + self.radius * np.sin(a)])

    def __repr__(self) -> str:
        c = "" if self.center == (0.0, 0.0) else f", center={self.center}"
        return f"Circle({self.radius}{c})"


class Annulus(Aperture):
    def __init__(self, inner: float, outer: float, center: Sequence[float] = (0.0, 0.0)) -> None:
        if not 0.0 <= float(inner) < float(outer):
            raise ValueError("an annulus has 0 <= inner < outer")
        self.inner, self.outer = float(inner), float(outer)
        self.center = _center(center)

    @property
    def extent(self) -> float:
        return float(np.hypot(*self.center) + self.outer)

    def contains(self, x, y, *, slack=0.0):
        r = np.hypot(np.asarray(x) - self.center[0], np.asarray(y) - self.center[1])
        return (r >= self.inner - slack) & (r <= self.outer + slack)

    def boundary(self, samples=128):
        a = np.linspace(0.0, 2 * np.pi, samples)
        return np.column_stack([self.center[0] + self.outer * np.cos(a), self.center[1] + self.outer * np.sin(a)])


class Rectangle(Aperture):
    def __init__(self, width: float, height: float, center: Sequence[float] = (0.0, 0.0)) -> None:
        if not (float(width) > 0.0 and float(height) > 0.0):
            raise ValueError("a rectangle has positive width and height")
        self.width, self.height = float(width), float(height)
        self.center = _center(center)

    @property
    def extent(self) -> float:
        return float(np.hypot(abs(self.center[0]) + self.width / 2, abs(self.center[1]) + self.height / 2))

    def contains(self, x, y, *, slack=0.0):
        return (np.abs(np.asarray(x) - self.center[0]) <= self.width / 2 + slack) & (
            np.abs(np.asarray(y) - self.center[1]) <= self.height / 2 + slack
        )

    def boundary(self, samples=128):
        w, h = self.width / 2, self.height / 2
        cx, cy = self.center
        return np.array([[cx - w, cy - h], [cx + w, cy - h], [cx + w, cy + h], [cx - w, cy + h], [cx - w, cy - h]])


class Polygon(Aperture):
    """A simple polygon given by its vertices in order (either orientation)."""

    def __init__(self, vertices: ArrayLike) -> None:
        v = np.asarray(vertices, dtype=float)
        if v.ndim != 2 or v.shape[1] != 2 or v.shape[0] < 3:
            raise ValueError("a polygon has at least three (x, y) vertices")
        self.vertices = tuple(map(tuple, v))
        self.center = (0.0, 0.0)

    @property
    def extent(self) -> float:
        return float(np.max(np.hypot(*np.asarray(self.vertices).T)))

    def contains(self, x, y, *, slack=0.0):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        v = np.asarray(self.vertices)
        inside = np.zeros(np.broadcast(x, y).shape, dtype=bool)
        for (x0, y0), (x1, y1) in zip(v, np.roll(v, -1, axis=0)):
            crosses = (y0 > y) != (y1 > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            inside ^= crosses & (x < x_cross)
        if slack > 0.0:
            inside |= self._edge_distance(x, y) <= slack
        return inside

    def _edge_distance(self, x, y):
        v = np.asarray(self.vertices)
        best = np.full(np.broadcast(x, y).shape, np.inf)
        for a, b in zip(v, np.roll(v, -1, axis=0)):
            ab = b - a
            t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / (ab @ ab), 0.0, 1.0)
            best = np.minimum(best, np.hypot(x - a[0] - t * ab[0], y - a[1] - t * ab[1]))
        return best

    def boundary(self, samples=128):
        v = np.asarray(self.vertices)
        return np.vstack([v, v[:1]])


__all__ = ["Aperture", "Circle", "Annulus", "Rectangle", "Polygon"]
