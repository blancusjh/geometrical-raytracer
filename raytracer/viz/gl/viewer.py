"""Interactive OpenGL viewer with HDR ray accumulation.

The viewer draws the meridional drawing plane ``(z, y)`` of the 3-D model
(:mod:`raytracer.viz.sag_drawing`): ``draw_system`` / ``draw_elements``
for the optics, ``draw_rays`` for a non-sequential
:class:`~raytracer.tracing.RayTree`, ``draw_ray_segments`` /
``draw_polyline`` / ``draw_filled_polygon`` / ``draw_markers`` for raw
drawing-plane data, and the neutral :class:`~raytracer.viz.scene.Drawing`
through :class:`~raytracer.viz.protocol.GLBackend`.

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
    SolidRayRenderer,
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
        self._solid_ray_renderer = SolidRayRenderer()
        # Ray state accumulates across draw_rays/draw_ray_segments calls, so
        # a scene can add several bundles (one call each) without later calls
        # silently replacing earlier ones. clear_rays() resets for re-draws.
        self._segments: np.ndarray | None = None  # scene-relative
        self._escaping: np.ndarray | None = None  # bool per segment
        self._directions: np.ndarray | None = None  # unit dir per segment
        self._ray_colors: np.ndarray | None = None  # (N, 4)
        self._ray_intensities: np.ndarray | None = None  # (N,)
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

    def draw_system(
        self,
        system,
        *,
        wavelength_um: float | None = None,
        material_colors: dict | None = None,
        tree=None,
    ) -> None:
        """Draw every element of an :class:`~raytracer.system.OpticalSystem`.

        The meridional section of each element (see
        :mod:`raytracer.viz.sag_drawing`): lens bodies filled by index,
        mirrors, stops, sensors — with what each sensor recorded when the
        traced *tree* is given.
        """

        from ..protocol import populate_viewer
        from ..scene import Drawing

        drawing = Drawing().add_system(system, wavelength_um=wavelength_um, material_colors=material_colors, tree=tree)
        populate_viewer(self, drawing)

    def draw_elements(self, elements, *, wavelength_um: float | None = None) -> None:
        """Draw loose elements at their own poses (or ``(element, pose)`` pairs)."""

        from ..protocol import populate_viewer
        from ..scene import Drawing

        populate_viewer(self, Drawing().add_elements(elements, wavelength_um=wavelength_um))

    def draw_rays(
        self,
        tree,
        *,
        color_resolver=None,
        intensity_resolver=None,
        show_misses: bool = True,
        marker_color: object = None,
        marker_size: float = 6.0,
        escape_length: float | None = None,
        select: np.ndarray | None = None,
    ) -> None:
        """Draw a non-sequential :class:`~raytracer.tracing.RayTree`, projected on ``(z, y)``.

        ``color_resolver`` is an ``(M, 4)`` array or a callable
        ``tree -> (M, 4)`` (default white); ``intensity_resolver`` likewise
        ``(M,)`` (default: ``config.default_intensity`` times each segment's
        power). Segments that left the system extend to the edge of the
        view, or end ``escape_length`` mm from their start when given;
        ``show_misses=False`` drops them. ``marker_color`` marks where the
        other segments end on a boundary. ``select`` is an optional boolean
        mask of the segments to draw.
        """

        n = len(tree)
        if n == 0:
            return
        starts = np.asarray(tree.start, dtype=float)[:, [2, 1]]
        ends = np.asarray(tree.end, dtype=float)[:, [2, 1]]
        directions = np.asarray(tree.direction, dtype=float)[:, [2, 1]]
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.where(norms == 0.0, 1.0, norms)
        escaping = np.asarray(tree.escaped, dtype=bool).copy()

        colors = color_resolver(tree) if callable(color_resolver) else color_resolver
        colors = np.tile([1.0, 1.0, 1.0, 1.0], (n, 1)) if colors is None else np.asarray(colors, dtype=float)
        intensities = intensity_resolver(tree) if callable(intensity_resolver) else intensity_resolver
        if intensities is None:
            intensities = self.config.default_intensity * np.asarray(tree.power, dtype=float)
        intensities = np.broadcast_to(np.asarray(intensities, dtype=float), (n,))

        keep = np.ones(n, dtype=bool) if select is None else np.asarray(select, dtype=bool).copy()
        if not show_misses:
            keep &= ~escaping
        if escape_length is not None:
            ends = np.where(escaping[:, None], starts + float(escape_length) * directions, ends)
            escaping[:] = False
        if not keep.any():
            return
        segments = np.stack([starts, ends], axis=1)[keep]
        self.draw_ray_segments(
            segments,
            colors=colors[keep],
            intensities=intensities[keep],
            escaping=escaping[keep],
            directions=directions[keep],
        )
        hit = keep & ~np.asarray(tree.escaped, dtype=bool)
        if marker_color is not None and hit.any():
            self.draw_markers(ends[hit], color=marker_color, size=marker_size)

    def draw_ray_segments(
        self,
        segments: np.ndarray,  # (N, 2, 2) drawing-plane endpoints (z, y)
        *,
        colors: np.ndarray | None = None,  # (N, 4)
        intensities: np.ndarray | None = None,  # (N,)
        escaping: np.ndarray | None = None,  # (N,) bool
        directions: np.ndarray | None = None,  # (N, 2), required with escaping
    ) -> None:
        """Draw raw drawing-plane ray segments (no RayTree needed); the Drawing entry point."""

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
        segments_rel = np.column_stack(
            [segments[:, 0, :] - center, segments[:, 1, :] - center]
        ).astype(np.float32)
        self._append_ray_batch(
            segments_rel,
            np.asarray(colors, dtype=np.float32),
            intensities.astype(np.float32),
            escaping_arr,
            np.asarray(directions, dtype=np.float32),
        )
        self._exposure_dirty = True
        self.update()

    def _append_ray_batch(
        self,
        segments_rel: np.ndarray,
        colors: np.ndarray,
        intensities: np.ndarray,
        escaping: np.ndarray,
        directions: np.ndarray,
    ) -> None:
        """Append a bundle to the accumulated ray state and re-upload it whole."""

        if self._segments is None:
            self._segments = segments_rel
            self._ray_colors = colors
            self._ray_intensities = intensities
            self._escaping = escaping
            self._directions = directions
        else:
            self._segments = np.vstack([self._segments, segments_rel])
            self._ray_colors = np.vstack([self._ray_colors, colors])
            self._ray_intensities = np.concatenate([self._ray_intensities, intensities])
            self._escaping = np.concatenate([self._escaping, escaping])
            self._directions = np.vstack([self._directions, directions])
        self._extend_escaping()
        self._ray_renderer.set_segments(self._segments, self._ray_colors, self._ray_intensities)
        self._solid_ray_renderer.set_segments(self._segments, self._ray_colors, self._ray_intensities)

    def clear_rays(self) -> None:
        """Drop every accumulated ray bundle (for interactive re-draws)."""

        self._segments = None
        self._ray_colors = None
        self._ray_intensities = None
        self._escaping = None
        self._directions = None
        self._ray_renderer.set_segments(
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
        self._solid_ray_renderer.set_segments(
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
        self.update()

    def draw_polyline(self, polyline, *, color="white", width: float = 2.0) -> None:
        """Draw a raw drawing-plane polyline overlay (surfaces, sensors, outlines)."""

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
        from .._window import announce_window

        announce_window(self, "[raytracer GL]")
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
        use_solid = self.config.use_solid_rays

        h, w = int(viewport[1]), int(viewport[0])

        if not use_solid:
            # "Beam" mode: additive HDR accumulation, tone-mapped for display —
            # overlapping rays glow, good for dense bundles standing in for a
            # continuous beam of light (see examples/duv_objective_2d.py).
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
        if not use_solid:
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
        if use_solid:
            # "Geometric" mode: crisp, non-additive lines drawn straight onto
            # the visible framebuffer — perfectly sharp physical rays, no
            # HDR glow, no aliasing blur (see examples/telescopes/*.py).
            self._solid_ray_renderer.draw(
                viewport=viewport,
                view_center=self._view_center_rel,
                ppw=self.camera.pixels_per_world,
                width_px=width_px,
                weight_scale=self.config.weight_scale,
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
            self._solid_ray_renderer.update_endpoints(self._segments)
        self.update()

    def on_mouse_wheel(self, event) -> None:
        factor = 1.1 ** event.delta[1]
        self.camera.zoom_about(factor, self._mouse_to_gl(event.pos))
        self._extend_escaping()
        if self._segments is not None and self._escaping is not None and self._escaping.any():
            self._ray_renderer.update_endpoints(self._segments)
            self._solid_ray_renderer.update_endpoints(self._segments)
        self.update()


__all__ = ["RenderConfig", "OpenGLViewer"]
