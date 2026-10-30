"""Matplotlib backend for the neutral 2-D drawing: publication diagrams with few rays."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from .drawing import Drawing, LensBodyItem, MarkerItem, RaySegmentsItem, SensorItem, SurfaceItem


class MplBackend:
    """Renders a :class:`~raytracer.viz.drawing.Drawing` into a matplotlib figure.

    Ray intensity maps to line alpha, normalized to the brightest ray, so a few hundred rays stay readable;
    dense caustic fields belong to the OpenGL backend.
    """

    def __init__(self, *, figsize=(11, 7), background: str = "white", show_sensor_hits: bool = True) -> None:
        self.figsize = figsize
        self.background = background
        self.show_sensor_hits = show_sensor_hits
        self.fig = None
        self.ax = None

    def render(self, drawing: Drawing):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.fig.patch.set_facecolor(self.background)
        self.ax.set_facecolor(self.background)
        self.update(drawing)
        return self.fig

    def update(self, drawing: Drawing):
        ax = self.ax
        ax.clear()
        x_lims, y_lims = drawing.bounds()
        reach = 2.0 * max(x_lims[1] - x_lims[0], y_lims[1] - y_lims[0])
        light = self.background in ("white", "#ffffff")

        for item in drawing.items:
            if isinstance(item, RaySegmentsItem):
                segments = item.segments.copy()
                idx = np.flatnonzero(item.escaping)
                segments[idx, 1, :] = segments[idx, 0, :] + item.directions[idx] * reach
                weights = item.intensities * item.colors[:, 3]
                peak = weights.max() if weights.size else 1.0
                rgba = item.colors.copy()
                rgba[:, 3] = np.clip(weights / peak, 0.02, 1.0) if peak > 0 else weights
                if light and np.allclose(rgba[:, :3], 1.0):  # white rays vanish on white paper
                    rgba[:, :3] = 0.12
                ax.add_collection(LineCollection(segments, colors=rgba, linewidths=0.7, zorder=3))
            elif isinstance(item, (SurfaceItem, SensorItem)):
                color = item.color
                if light and tuple(color[:3]) == (1.0, 1.0, 1.0):
                    color = (0.1, 0.1, 0.1, color[3] if len(color) > 3 else 1.0)
                sensor = isinstance(item, SensorItem)
                ax.plot(item.polyline[:, 0], item.polyline[:, 1], color=color, lw=item.width, zorder=4 if sensor else 2)
                if sensor and self.show_sensor_hits and item.hits is not None and len(item.hits):
                    ax.scatter(item.hits[:, 0], item.hits[:, 1], s=6, color="#d62728", zorder=5)
            elif isinstance(item, LensBodyItem):
                ax.add_patch(Polygon(item.polygon, closed=True, facecolor=item.facecolor, edgecolor=item.edgecolor, lw=1.0, zorder=1))
            elif isinstance(item, MarkerItem):
                ax.scatter(item.points[:, 0], item.points[:, 1], s=item.size**2, color=item.color, zorder=5)

        ax.set_xlim(*x_lims)
        ax.set_ylim(*y_lims)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("y (mm)")
        return self.fig

    def show(self) -> None:  # pragma: no cover - interactive
        plt.show()

    def save(self, path, *, dpi: int = 180) -> None:
        self.fig.savefig(path, dpi=dpi, facecolor=self.fig.get_facecolor())


__all__ = ["MplBackend"]
