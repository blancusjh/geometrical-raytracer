"""Matplotlib backend for the neutral 2-D scene (publication-style)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from .scene import (
    LensBodyItem,
    MarkerItem,
    RaySegmentsItem,
    Scene,
    ScreenItem,
    SurfaceItem,
)


class MplBackend:
    """Renders a :class:`Scene` into a matplotlib figure.

    Ray intensity maps to line alpha (normalized to the brightest ray), so a
    few hundred rays stay readable; for dense caustic fields use the OpenGL
    backend instead.
    """

    def __init__(
        self,
        *,
        figsize: tuple[float, float] = (11, 7),
        background: str = "white",
        show_screen_hits: bool = True,
    ) -> None:
        self.figsize = figsize
        self.background = background
        self.show_screen_hits = show_screen_hits
        self.fig = None
        self.ax = None

    def render(self, scene: Scene):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.fig.patch.set_facecolor(self.background)
        self.ax.set_facecolor(self.background)
        self.update(scene)
        return self.fig

    def update(self, scene: Scene):
        ax = self.ax
        ax.clear()
        x_lims, y_lims = scene.bounds()
        reach = 2.0 * max(
            x_lims[1] - x_lims[0], y_lims[1] - y_lims[0]
        )

        for item in scene.items:
            if isinstance(item, RaySegmentsItem):
                segments = item.segments.copy()
                if item.escaping.any():
                    idx = np.flatnonzero(item.escaping)
                    segments[idx, 1, :] = (
                        segments[idx, 0, :] + item.directions[idx] * reach
                    )
                weights = item.intensities * item.colors[:, 3]
                peak = weights.max() if weights.size else 1.0
                alphas = np.clip(weights / peak, 0.02, 1.0) if peak > 0 else weights
                rgba = item.colors.copy()
                rgba[:, 3] = alphas
                # Dark rays on light backgrounds: flip pure-white defaults.
                if self.background in ("white", "#ffffff") and np.allclose(
                    rgba[:, :3], 1.0
                ):
                    rgba[:, :3] = 0.12
                ax.add_collection(
                    LineCollection(segments, colors=rgba, linewidths=0.7, zorder=3)
                )
            elif isinstance(item, SurfaceItem):
                ax.plot(
                    item.polyline[:, 0],
                    item.polyline[:, 1],
                    color=self._vis_color(item.color),
                    lw=item.width,
                    zorder=2,
                )
            elif isinstance(item, LensBodyItem):
                ax.add_patch(
                    Polygon(
                        item.polygon,
                        closed=True,
                        facecolor=item.facecolor,
                        edgecolor=item.edgecolor,
                        lw=1.0,
                        zorder=1,
                    )
                )
            elif isinstance(item, ScreenItem):
                ax.plot(
                    [item.p0[0], item.p1[0]],
                    [item.p0[1], item.p1[1]],
                    color=self._vis_color(item.color),
                    lw=item.width,
                    zorder=4,
                )
                if self.show_screen_hits and item.screen is not None and item.screen.hits:
                    points = np.array([hit.point for hit in item.screen.hits])
                    ax.scatter(
                        points[:, 0], points[:, 1], s=6, color="#d62728", zorder=5
                    )
            elif isinstance(item, MarkerItem):
                ax.scatter(
                    item.points[:, 0],
                    item.points[:, 1],
                    s=item.size**2,
                    color=item.color,
                    zorder=5,
                )

        ax.set_xlim(*x_lims)
        ax.set_ylim(*y_lims)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        return self.fig

    def _vis_color(self, color):
        # White overlays are invisible on the default white background.
        if self.background in ("white", "#ffffff") and tuple(color[:3]) == (1.0, 1.0, 1.0):
            return (0.1, 0.1, 0.1, color[3] if len(color) > 3 else 1.0)
        return color

    def show(self) -> None:  # pragma: no cover - interactive
        plt.show()

    def save(self, path, *, dpi: int = 180) -> None:
        self.fig.savefig(path, dpi=dpi, facecolor=self.fig.get_facecolor())


__all__ = ["MplBackend"]
