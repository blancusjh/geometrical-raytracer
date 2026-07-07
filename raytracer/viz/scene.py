"""Backend-neutral 2-D scene model.

A :class:`Scene` is pure data: items describing ray segments, surface
polylines, filled lens bodies, screens, and markers. Both the matplotlib
backend (:mod:`raytracer.viz.mpl`) and the OpenGL backend consume the same
scene, making the two visualizations interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np


@dataclass
class RaySegmentsItem:
    segments: np.ndarray  # (N, 2, 2) world endpoints
    colors: np.ndarray  # (N, 4) rgba; alpha = per-ray weight
    intensities: np.ndarray  # (N,)
    escaping: np.ndarray  # (N,) bool — extend to view edge
    directions: np.ndarray  # (N, 2) unit directions (for escaping rays)


@dataclass
class SurfaceItem:
    polyline: np.ndarray  # (M, 2)
    color: tuple = (1.0, 1.0, 1.0, 1.0)
    width: float = 2.0


@dataclass
class LensBodyItem:
    polygon: np.ndarray  # (M, 2) closed outline
    facecolor: tuple = (0.74, 0.88, 1.0, 0.55)
    edgecolor: tuple = (0.16, 0.47, 0.71, 1.0)


@dataclass
class ScreenItem:
    p0: np.ndarray
    p1: np.ndarray
    color: tuple = (1.0, 0.85, 0.3, 1.0)
    width: float = 3.0
    screen: object | None = None  # optional Screen2D for hit overlays


@dataclass
class MarkerItem:
    points: np.ndarray  # (M, 2)
    color: tuple = (0.86, 0.08, 0.24, 1.0)
    size: float = 6.0


class Scene:
    """Container of drawable items plus optional view limits."""

    def __init__(
        self,
        x_lims: tuple[float, float] | None = None,
        y_lims: tuple[float, float] | None = None,
    ) -> None:
        self.items: list = []
        self.x_lims = x_lims
        self.y_lims = y_lims

    def add(self, item) -> "Scene":
        self.items.append(item)
        return self

    def add_tree(
        self,
        tree,
        *,
        color_resolver: Optional[Callable] = None,
        intensity_resolver: Optional[Callable] = None,
        show_misses: bool = True,
        markers: bool = False,
        default_color: tuple = (1.0, 1.0, 1.0, 1.0),
    ) -> "Scene":
        """Convert a traced :class:`RayTree` into a segments item."""

        segments, colors, intensities, escaping, directions, hit_points = (
            [], [], [], [], [], []
        )
        for node in tree.nodes():
            origin = np.asarray(node.ray.origin, dtype=float)
            direction = np.asarray(node.ray.direction, dtype=float)
            hit = node.intersection
            if hit is not None:
                end = np.asarray(hit.point, dtype=float)
                is_escaping = False
                hit_points.append(end)
            else:
                if not show_misses:
                    continue
                end = origin + direction
                is_escaping = True
            segments.append([origin, end])
            colors.append(
                tuple(color_resolver(node)) if color_resolver else default_color
            )
            intensities.append(
                float(intensity_resolver(node))
                if intensity_resolver
                else float(getattr(node, "intensity", 1.0))
            )
            escaping.append(is_escaping)
            directions.append(direction)

        if segments:
            self.add(
                RaySegmentsItem(
                    segments=np.asarray(segments),
                    colors=np.asarray(colors, dtype=float),
                    intensities=np.asarray(intensities, dtype=float),
                    escaping=np.asarray(escaping, dtype=bool),
                    directions=np.asarray(directions, dtype=float),
                )
            )
        if markers and hit_points:
            self.add(MarkerItem(points=np.asarray(hit_points)))
        return self

    def add_elements(self, elements: Iterable) -> "Scene":
        """Add lenses (filled), mirrors, screens, or bare surfaces."""

        from ..nonseq.detectors import Screen2D
        from ..nonseq.elements import Lens2D, Mirror2D

        for element in elements:
            if isinstance(element, Lens2D):
                self.add(LensBodyItem(polygon=element.body_polygon()))
            elif isinstance(element, Mirror2D):
                self.add(
                    SurfaceItem(
                        polyline=element.face.polyline(),
                        color=(0.13, 0.13, 0.13, 1.0),
                        width=3.0,
                    )
                )
            elif isinstance(element, Screen2D):
                self.add(ScreenItem(p0=element.p0, p1=element.p1, screen=element))
            else:
                polyline = element.polyline()
                self.add(SurfaceItem(polyline=np.asarray(polyline)))
        return self

    def add_screen(self, screen) -> "Scene":
        self.add(ScreenItem(p0=screen.p0, p1=screen.p1, screen=screen))
        return self

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """(x_lims, y_lims): explicit limits or data bounds with margin."""

        if self.x_lims is not None and self.y_lims is not None:
            return self.x_lims, self.y_lims
        points = []
        for item in self.items:
            if isinstance(item, RaySegmentsItem):
                points.append(item.segments[~item.escaping].reshape(-1, 2))
                points.append(item.segments[:, 0, :])
            elif isinstance(item, SurfaceItem):
                points.append(item.polyline)
            elif isinstance(item, LensBodyItem):
                points.append(item.polygon)
            elif isinstance(item, ScreenItem):
                points.append(np.array([item.p0, item.p1]))
            elif isinstance(item, MarkerItem):
                points.append(item.points)
        if not points:
            return (-10.0, 10.0), (-10.0, 10.0)
        allpts = np.vstack([p for p in points if p.size])
        finite = allpts[np.isfinite(allpts).all(axis=1)]
        x0, y0 = finite.min(axis=0)
        x1, y1 = finite.max(axis=0)
        mx = 0.05 * max(x1 - x0, 1e-6)
        my = 0.05 * max(y1 - y0, 1e-6)
        x_lims = self.x_lims or (x0 - mx, x1 + mx)
        y_lims = self.y_lims or (y0 - my, y1 + my)
        return x_lims, y_lims


__all__ = [
    "Scene",
    "RaySegmentsItem",
    "SurfaceItem",
    "LensBodyItem",
    "ScreenItem",
    "MarkerItem",
]
