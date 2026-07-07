"""Interactive OpenGL viewer with HDR ray accumulation.

Public API is backward compatible with the previous ``visualization_opengl``
module: ``RenderConfig``, ``OpenGLViewer.draw_surfaces / draw_rays /
draw_markers / update_visual_params_only / run / close``.

Rendering pipeline per frame:

1. ray quads blend additively (ONE, ONE) into a float32 framebuffer in
   linear radiometric units — overlap never saturates;
2. a fullscreen tone-mapping pass (exponential by default, auto-exposed at
   the 99th percentile) compresses the energy for display;
3. surfaces and markers draw on top as plain overlays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from vispy import app
from vispy import gloo
from vispy.color import Color

from .camera import Camera
from .renderers import (
    AccumulationTarget,
    FilledPolygonRenderer,
    MarkerRenderer,
    PolylineRenderer,
    RayRenderer,
    TonemapPass,
    auto_exposure,
)

logger = logging.getLogger("raytracer.viz.gl")


@dataclass
class RenderConfig:
    """Rendering configuration (superset of the legacy fields)."""

    ray_width: float = 0.5  # world units
    sigma_factor: float = 0.5  # sigma = width_px * sigma_factor
    accumulation_mode: str = "squared"  # legacy alias: mapped onto tone_map
    default_intensity: float = 1.0
    weight_scale: float = 1.0
    min_pixels: float = 1.0
    use_solid_rays: bool = False
    background: object = "black"
    # New HDR controls:
    exposure: float = 1.0
    tone_map: str = "exponential"  # "exponential" | "reinhard" | "linear"
    auto_exposure: bool = True

    def resolved_tone_map(self) -> str:
        # Legacy modes map onto the new pipeline: "squared" was an additive
        # sqrt hack -> exponential; "alpha" approximated normal blending ->
        # linear clamp.
        if self.accumulation_mode == "alpha":
            return "linear"
        if self.accumulation_mode in ("squared", "none", ""):
            return self.tone_map
        return self.tone_map


class OpenGLViewer(app.Canvas):
    """2-D scene viewer: pan with left drag, zoom with the scroll wheel."""

    def __init__(
        self,
        x_lims: tuple[float, float] = (-10.0, 10.0),
        y_lims: tuple[float, float] = (-10.0, 10.0),
        *,
        size: tuple[int, int] = (900, 700),
        bgcolor: object = None,
        render_config: RenderConfig | None = None,
        title: str = "raytracer",
        show: bool = False,
    ) -> None:
        app.Canvas.__init__(self, title=title, size=size, keys="interactive", show=show)
        self.config = render_config or RenderConfig()
        if bgcolor is not None:
            self.config.background = bgcolor

        self.camera = Camera.from_limits(x_lims, y_lims, self._viewport_px())

        self._scene_center: np.ndarray | None = None
        self._ray_renderer = RayRenderer()
        self._segments: np.ndarray | None = None  # scene-relative
        self._escaping: np.ndarray | None = None  # bool per segment
        self._directions: np.ndarray | None = None  # unit dir per segment
        self._surface_renderers: list[PolylineRenderer] = []
        self._fill_renderers: list[FilledPolygonRenderer] = []
        self._marker_renderers: list[MarkerRenderer] = []
        self._accumulation = AccumulationTarget()
        self._tonemap = TonemapPass()
        self._exposure = self.config.exposure
        self._exposure_dirty = True
        self._drag_start: np.ndarray | None = None

    # -- coordinate helpers ------------------------------------------------

    def _viewport_px(self) -> tuple[float, float]:
        w, h = self.physical_size
        return (float(w), float(h))

    def _mouse_to_gl(self, pos) -> np.ndarray:
        """Window position (top-left origin, logical px) -> GL px (bottom-left)."""

        w, h = self.size
        scale = self.physical_size[1] / h if h else 1.0
        x = pos[0] * (self.physical_size[0] / w if w else 1.0)
        y = (h - pos[1]) * scale
        return np.array([x, y])

    def _ensure_scene_center(self, sample: np.ndarray) -> None:
        if self._scene_center is None:
            self._scene_center = np.asarray(sample, dtype=float)

    @property
    def _view_center_rel(self) -> np.ndarray:
        center = self._scene_center if self._scene_center is not None else np.zeros(2)
        return self.camera.center - center

    # -- public drawing API --------------------------------------------------

    def draw_surfaces(
        self,
        surfaces: Sequence,
        *,
        color: object = "white",
        width: float = 2.0,
        samples: int = 800,
    ) -> None:
        rgba = Color(color).rgba
        for surface in surfaces:
            try:
                polyline = np.asarray(surface.polyline(samples), dtype=float)
            except Exception:
                logger.warning(
                    "Surface %s does not provide a polyline; skipped",
                    getattr(surface, "surface_id", surface),
                )
                continue
            if polyline.shape[0] < 2:
                logger.warning(
                    "Surface %s produced an empty polyline",
                    getattr(surface, "surface_id", surface),
                )
                continue
            self._ensure_scene_center(np.nanmean(polyline, axis=0))
            self._surface_renderers.append(
                PolylineRenderer(polyline - self._scene_center, rgba, width)
            )
        self.update()

    def draw_rays(
        self,
        tree,
        *,
        tail_length: float = 12.0,  # kept for API compat; escapes now reach the view edge
        color_resolver: Optional[Callable] = None,
        intensity_resolver: Optional[Callable] = None,
        show_misses: bool = True,
        marker_color: object = None,
        marker_size: float = 6.0,
        leaf_extension: Optional[Callable] = None,
    ) -> None:
        del tail_length  # legacy parameter; escaping rays are view-extended
        starts, ends, colors, intensities = [], [], [], []
        escaping, directions, markers = [], [], []

        for node in tree.nodes():
            origin = np.asarray(node.ray.origin, dtype=float)
            direction = np.asarray(node.ray.direction, dtype=float)
            hit = node.intersection
            if hit is not None:
                end = np.asarray(hit.point, dtype=float)
                is_escaping = False
                markers.append(end)
            else:
                if not show_misses:
                    continue
                if leaf_extension is not None:
                    end = np.asarray(leaf_extension(node), dtype=float)
                    is_escaping = False
                else:
                    end = origin + direction  # placeholder; extended per view
                    is_escaping = True

            if color_resolver is not None:
                rgba = tuple(color_resolver(node))
            else:
                rgba = (1.0, 1.0, 1.0, 1.0)
            if intensity_resolver is not None:
                intensity = float(intensity_resolver(node))
            else:
                intensity = self.config.default_intensity * float(
                    getattr(node, "intensity", 1.0)
                )

            starts.append(origin)
            ends.append(end)
            colors.append(rgba)
            intensities.append(intensity)
            escaping.append(is_escaping)
            directions.append(direction)

        if not starts:
            return
        starts = np.asarray(starts)
        ends = np.asarray(ends)
        self._ensure_scene_center(starts.mean(axis=0))
        center = self._scene_center

        self._segments = np.column_stack([starts - center, ends - center]).astype(
            np.float32
        )
        self._escaping = np.asarray(escaping, dtype=bool)
        self._directions = np.asarray(directions, dtype=np.float32)
        self._extend_escaping()
        self._ray_renderer.set_segments(
            self._segments, np.asarray(colors, dtype=np.float32),
            np.asarray(intensities, dtype=np.float32),
        )
        if marker_color is not None and markers:
            self.draw_markers(np.asarray(markers), color=marker_color, size=marker_size)
        self._exposure_dirty = True
        self.update()

    def draw_ray_segments(
        self,
        segments: np.ndarray,  # (N, 2, 2) world endpoints
        *,
        colors: np.ndarray | None = None,  # (N, 4)
        intensities: np.ndarray | None = None,  # (N,)
        escaping: np.ndarray | None = None,  # (N,) bool
        directions: np.ndarray | None = None,  # (N, 2), required with escaping
    ) -> None:
        """Scene-level entry point: draw raw ray segments (no RayTree needed)."""

        segments = np.asarray(segments, dtype=float)
        n = segments.shape[0]
        if n == 0:
            return
        colors = (
            np.tile([1.0, 1.0, 1.0, 1.0], (n, 1)) if colors is None else np.asarray(colors)
        )
        intensities = (
            np.ones(n) if intensities is None else np.asarray(intensities, dtype=float)
        )
        escaping_arr = (
            np.zeros(n, dtype=bool) if escaping is None else np.asarray(escaping, dtype=bool)
        )
        if directions is None:
            deltas = segments[:, 1, :] - segments[:, 0, :]
            norms = np.linalg.norm(deltas, axis=1, keepdims=True)
            directions = deltas / np.where(norms == 0.0, 1.0, norms)

        self._ensure_scene_center(segments[:, 0, :].mean(axis=0))
        center = self._scene_center
        self._segments = np.column_stack(
            [segments[:, 0, :] - center, segments[:, 1, :] - center]
        ).astype(np.float32)
        self._escaping = escaping_arr
        self._directions = np.asarray(directions, dtype=np.float32)
        self._extend_escaping()
        self._ray_renderer.set_segments(
            self._segments,
            np.asarray(colors, dtype=np.float32),
            intensities.astype(np.float32),
        )
        self._exposure_dirty = True
        self.update()

    def draw_polyline(self, polyline, *, color="white", width: float = 2.0) -> None:
        """Draw a raw polyline overlay (surfaces, screens, outlines)."""

        polyline = np.asarray(polyline, dtype=float)
        if polyline.shape[0] < 2:
            return
        self._ensure_scene_center(np.nanmean(polyline, axis=0))
        rgba = Color(color).rgba if not isinstance(color, (tuple, list, np.ndarray)) else tuple(color)
        self._surface_renderers.append(
            PolylineRenderer(polyline - self._scene_center, rgba, width)
        )
        self.update()

    def draw_filled_polygon(self, vertices, faces, *, color) -> None:
        """Draw a translucent filled triangle mesh (e.g. a lens body by material).

        vertices: (M, 2) world; faces: (K, 3) int indices. Rendered as an overlay
        under the surface outlines and markers.
        """
        vertices = np.asarray(vertices, dtype=float).reshape(-1, 2)
        faces = np.asarray(faces).reshape(-1, 3)
        if vertices.shape[0] < 3 or faces.shape[0] == 0:
            return
        self._ensure_scene_center(vertices.mean(axis=0))
        rgba = Color(color).rgba if not isinstance(color, (tuple, list, np.ndarray)) else tuple(color)
        self._fill_renderers.append(
            FilledPolygonRenderer(vertices - self._scene_center, faces, rgba)
        )
        self.update()

    def draw_markers(self, points, *, color="crimson", size: float = 6.0) -> None:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if points.shape[0] == 0:
            return
        self._ensure_scene_center(points.mean(axis=0))
        self._marker_renderers.append(
            MarkerRenderer(points - self._scene_center, Color(color).rgba, size)
        )
        self.update()

    def clear_markers(self) -> None:
        self._marker_renderers.clear()
        self.update()

    def update_visual_params_only(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning("Unknown render parameter %r ignored", key)
        self._exposure_dirty = True
        self.update()

    def run(self) -> None:
        self.show()
        app.run()

    def close(self) -> None:  # pragma: no cover - passthrough
        app.Canvas.close(self)

    def read_accumulation(self) -> np.ndarray:
        """Read back the HDR accumulation buffer (activates the context)."""

        self.set_current()
        return self._accumulation.read()

    def snapshot(self) -> np.ndarray:
        """Render the current scene offscreen and return a uint8 RGBA image.

        Works without an exposed window, so it is also the reliable way to
        save screenshots programmatically.
        """

        self.set_current()
        viewport = self._viewport_px()
        h, w = int(viewport[1]), int(viewport[0])
        color = gloo.Texture2D(shape=(h, w, 4))
        fbo = gloo.FrameBuffer(color=color)
        with fbo:
            self._render_scene(viewport)
            image = gloo.read_pixels(alpha=True)
        return image

    # -- escaping rays ------------------------------------------------------

    def _extend_escaping(self) -> None:
        """Extend miss rays past the current view edge (cheap, subset-only)."""

        if self._segments is None or self._escaping is None or not self._escaping.any():
            return
        idx = np.flatnonzero(self._escaping)
        center_rel = self._view_center_rel
        starts = self._segments[idx, 0:2]
        reach = (
            np.linalg.norm(starts - center_rel, axis=1) + 2.0 * self.camera.diagonal
        )
        self._segments[idx, 2:4] = starts + self._directions[idx] * reach[:, None]

    # -- rendering ----------------------------------------------------------

    def _render_pixels(self) -> tuple[float, float]:
        viewport = self._viewport_px()
        # Camera viewport tracks logical size; rendering uses physical px.
        return viewport

    def on_draw(self, event) -> None:
        self._render_scene(self._viewport_px())

    def _render_scene(self, viewport: tuple[float, float]) -> None:
        self.camera.viewport = viewport
        width_px = max(
            self.config.ray_width * self.camera.pixels_per_world,
            self.config.min_pixels,
        )
        sigma_px = width_px * self.config.sigma_factor

        h, w = int(viewport[1]), int(viewport[0])
        self._accumulation.ensure((h, w))
        with self._accumulation.fbo:
            gloo.set_viewport(0, 0, w, h)
            gloo.clear(color=(0.0, 0.0, 0.0, 0.0))
            self._ray_renderer.draw(
                viewport=viewport,
                view_center=self._view_center_rel,
                ppw=self.camera.pixels_per_world,
                width_px=width_px,
                sigma_px=sigma_px,
                weight_scale=self.config.weight_scale,
                use_solid=self.config.use_solid_rays,
            )

        if self.config.auto_exposure and self._exposure_dirty:
            try:
                self._exposure = self.config.exposure * auto_exposure(
                    self._accumulation.read()
                )
            except Exception as error:  # pragma: no cover - driver dependent
                logger.warning("Auto-exposure readback failed: %s", error)
                self._exposure = self.config.exposure
            self._exposure_dirty = False
        elif not self.config.auto_exposure:
            self._exposure = self.config.exposure

        gloo.set_viewport(0, 0, w, h)
        background = Color(self.config.background).rgb
        gloo.clear(color=Color(self.config.background))
        self._tonemap.draw(
            self._accumulation.texture,
            exposure=self._exposure,
            mode=self.config.resolved_tone_map(),
            background=tuple(background) if self.config.background != "black" else (0.0, 0.0, 0.0),
        )
        for renderer in self._fill_renderers:  # material bodies, under the outlines
            renderer.draw(
                viewport=viewport,
                view_center=self._view_center_rel,
                ppw=self.camera.pixels_per_world,
            )
        for renderer in self._surface_renderers:
            renderer.draw(
                viewport=viewport,
                view_center=self._view_center_rel,
                ppw=self.camera.pixels_per_world,
            )
        for renderer in self._marker_renderers:
            renderer.draw(
                viewport=viewport,
                view_center=self._view_center_rel,
                ppw=self.camera.pixels_per_world,
            )

    # -- interaction ----------------------------------------------------------

    def on_resize(self, event) -> None:
        self.camera.resize(self._viewport_px())
        self.update()

    def on_mouse_press(self, event) -> None:
        if event.button == 1:
            self._drag_start = self._mouse_to_gl(event.pos)

    def on_mouse_release(self, event) -> None:
        self._drag_start = None

    def on_mouse_move(self, event) -> None:
        if self._drag_start is None or not event.is_dragging:
            return
        pos = self._mouse_to_gl(event.pos)
        delta = pos - self._drag_start
        self._drag_start = pos
        self.camera.pan_pixels(delta[0], delta[1])
        self._extend_escaping()
        if self._segments is not None and self._escaping is not None and self._escaping.any():
            self._ray_renderer.update_endpoints(self._segments)
        self.update()

    def on_mouse_wheel(self, event) -> None:
        factor = 1.1 ** event.delta[1]
        self.camera.zoom_about(factor, self._mouse_to_gl(event.pos))
        self._extend_escaping()
        if self._segments is not None and self._escaping is not None and self._escaping.any():
            self._ray_renderer.update_endpoints(self._segments)
        self.update()


__all__ = ["RenderConfig", "OpenGLViewer"]
