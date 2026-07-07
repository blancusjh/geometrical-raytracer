"""Pure 2-D camera math for the OpenGL viewer (no GL dependency).

Screen coordinates are pixels with the origin at the bottom-left (GL
convention, y up). The viewer converts window events (top-left origin) at
the event boundary, so every function here uses one single convention.
The aspect ratio is locked: world scale is isotropic by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Camera:
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    half_height: float = 5.0
    viewport: tuple[float, float] = (800.0, 600.0)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)
        self.half_height = float(self.half_height)

    @classmethod
    def from_limits(
        cls,
        x_lims: tuple[float, float],
        y_lims: tuple[float, float],
        viewport: tuple[float, float],
    ) -> "Camera":
        """Fit the rectangle into the viewport (contain), aspect locked."""

        cx = 0.5 * (x_lims[0] + x_lims[1])
        cy = 0.5 * (y_lims[0] + y_lims[1])
        half_w = 0.5 * abs(x_lims[1] - x_lims[0])
        half_h = 0.5 * abs(y_lims[1] - y_lims[0])
        aspect = viewport[0] / viewport[1]
        needed_half_h = max(half_h, half_w / aspect)
        return cls(center=np.array([cx, cy]), half_height=needed_half_h, viewport=viewport)

    @property
    def pixels_per_world(self) -> float:
        return self.viewport[1] / (2.0 * self.half_height)

    @property
    def half_width(self) -> float:
        return self.half_height * self.viewport[0] / self.viewport[1]

    @property
    def view_rect(self) -> tuple[float, float, float, float]:
        """(x0, x1, y0, y1) world extents currently visible."""

        return (
            self.center[0] - self.half_width,
            self.center[0] + self.half_width,
            self.center[1] - self.half_height,
            self.center[1] + self.half_height,
        )

    @property
    def diagonal(self) -> float:
        return 2.0 * float(np.hypot(self.half_width, self.half_height))

    def world_to_screen(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        half = np.asarray(self.viewport, dtype=float) / 2.0
        return (points - self.center) * self.pixels_per_world + half

    def screen_to_world(self, pixels) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=float)
        half = np.asarray(self.viewport, dtype=float) / 2.0
        return (pixels - half) / self.pixels_per_world + self.center

    def pan_pixels(self, dx: float, dy: float) -> None:
        """Shift the view by a screen-space delta (drag content with cursor)."""

        self.center = self.center - np.array([dx, dy]) / self.pixels_per_world

    def zoom_about(self, factor: float, screen_pos) -> None:
        """Zoom by *factor* (>1 zooms in) keeping *screen_pos* anchored."""

        anchor = self.screen_to_world(screen_pos)
        self.half_height /= factor
        # Re-place the center so the anchor stays under the cursor.
        half = np.asarray(self.viewport, dtype=float) / 2.0
        offset = (np.asarray(screen_pos, dtype=float) - half) / self.pixels_per_world
        self.center = anchor - offset

    def resize(self, viewport: tuple[float, float]) -> None:
        """Change the viewport keeping world-units-per-pixel constant."""

        old_ppw = self.pixels_per_world
        self.viewport = (float(viewport[0]), float(viewport[1]))
        self.half_height = self.viewport[1] / (2.0 * old_ppw)


__all__ = ["Camera"]
