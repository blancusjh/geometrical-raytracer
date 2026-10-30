"""Interactive OpenGL viewer of the drawing plane ``(z, y)``, with HDR ray accumulation.

It draws a neutral :class:`~raytracer.viz.scene.Drawing` (:meth:`OpenGLViewer.draw`), or builds one from a
system, loose elements or a non-sequential :class:`~raytracer.tracing.RayTree`. Each frame:

1. ray quads blend additively (ONE, ONE) into a float32 framebuffer in linear radiometric units, so overlap
   never saturates;
2. a fullscreen pass tone-maps that energy for display (exponential by default, auto-exposed at the 99th
   percentile);
3. fills, surfaces and markers draw on top as plain overlays.

With ``RenderConfig(use_solid_rays=True)`` rays are instead crisp lines drawn straight onto the display.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from vispy import app, gloo
from vispy.color import Color

from ..scene import Drawing, LensBodyItem, MarkerItem, RaySegmentsItem, SensorItem, SurfaceItem
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
    ray_width: float = 0.5  # world units
    sigma_factor: float = 0.5  # Gaussian sigma = width in pixels × sigma_factor
    default_intensity: float = 1.0
    weight_scale: float = 1.0
    min_pixels: float = 1.0
    use_solid_rays: bool = False
    background: object = "black"
    exposure: float = 1.0
    tone_map: str = "exponential"  # "exponential" | "reinhard" | "linear"
    auto_exposure: bool = True


class OpenGLViewer(app.Canvas):
    """2-D scene viewer: pan with left drag, zoom with the scroll wheel."""

    def __init__(self, x_lims=(-10.0, 10.0), y_lims=(-10.0, 10.0), *, size=(900, 700), bgcolor=None,
                 render_config: RenderConfig | None = None, title: str = "raytracer", show: bool = False) -> None:
        app.Canvas.__init__(self, title=title, size=size, keys="interactive", show=show)
        self.config = render_config or RenderConfig()
        if bgcolor is not None:
            self.config.background = bgcolor
        self.camera = Camera.from_limits(x_lims, y_lims, self._viewport_px())
        # Geometry is stored relative to a scene centre (the first thing drawn) to keep float32 precision.
        self._scene_center: np.ndarray | None = None
        self._ray_renderers = (RayRenderer(), RayRenderer(solid=True))
        self._rays: dict | None = None  # every bundle drawn so far: segments, colors, intensities, escaping, directions
        self._surface_renderers: list[PolylineRenderer] = []
        self._fill_renderers: list[FilledPolygonRenderer] = []
        self._marker_renderers: list[MarkerRenderer] = []
        self._accumulation = AccumulationTarget()
        self._tonemap = TonemapPass()
        self._exposure = self.config.exposure
        self._exposure_dirty = True
        self._drag_start: np.ndarray | None = None

    # -- drawing ----------------------------------------------------------------

    def draw(self, drawing: Drawing) -> None:
        """Draw every item of a neutral drawing."""

        for item in drawing.items:
            if isinstance(item, RaySegmentsItem):
                self.draw_ray_segments(item.segments, colors=item.colors, intensities=item.intensities,
                                       escaping=item.escaping, directions=item.directions)
            elif isinstance(item, (SurfaceItem, SensorItem)):
                self.draw_polyline(item.polyline, color=item.color, width=item.width)
                if isinstance(item, SensorItem) and item.hits is not None and len(item.hits):
                    self.draw_markers(item.hits, color=(0.86, 0.08, 0.24, 1.0), size=4.0)
            elif isinstance(item, LensBodyItem):
                verts, faces = item.triangles()
                self.draw_filled_polygon(verts, faces, color=item.facecolor)
                outline = np.asarray(item.polygon, dtype=float)
                if not np.allclose(outline[0], outline[-1]):
                    outline = np.vstack([outline, outline[:1]])
                self.draw_polyline(outline, color=item.edgecolor, width=1.5)
            elif isinstance(item, MarkerItem):
                self.draw_markers(item.points, color=item.color, size=item.size)

    def draw_system(self, system, *, wavelength_um=None, material_colors=None, tree=None) -> None:
        """Every element of an :class:`~raytracer.system.OpticalSystem`, with what each sensor recorded in *tree*."""

        self.draw(Drawing().add_system(system, wavelength_um=wavelength_um, material_colors=material_colors, tree=tree))

    def draw_elements(self, elements, *, wavelength_um=None) -> None:
        """Loose elements at their own poses, or ``(element, pose)`` pairs."""

        self.draw(Drawing().add_elements(elements, wavelength_um=wavelength_um))

    def draw_rays(self, tree, *, intensities=None, **options) -> None:
        """A non-sequential :class:`~raytracer.tracing.RayTree`, with the options of :meth:`Drawing.add_tree`;
        intensities default to ``config.default_intensity`` times each segment's power."""

        if intensities is None:
            intensities = self.config.default_intensity * np.asarray(tree.power, dtype=float)
        self.draw(Drawing().add_tree(tree, intensities=intensities, **options))

    def draw_ray_segments(self, segments, *, colors=None, intensities=None, escaping=None, directions=None) -> None:
        """Add a bundle of ``(N, 2, 2)`` drawing-plane segments; bundles accumulate until :meth:`clear_rays`.

        ``escaping`` rays extend past the view's edge along ``directions``.
        """

        segments = np.asarray(segments, dtype=float)
        n = segments.shape[0]
        if n == 0:
            return
        if directions is None:
            deltas = segments[:, 1, :] - segments[:, 0, :]
            norms = np.linalg.norm(deltas, axis=1, keepdims=True)
            directions = deltas / np.where(norms == 0.0, 1.0, norms)
        self._ensure_scene_center(segments[:, 0, :].mean(axis=0))
        bundle = dict(
            segments=np.column_stack([segments[:, 0, :] - self._scene_center, segments[:, 1, :] - self._scene_center]).astype(np.float32),
            colors=np.asarray(np.tile([1.0, 1.0, 1.0, 1.0], (n, 1)) if colors is None else colors, dtype=np.float32),
            intensities=(np.ones(n) if intensities is None else np.asarray(intensities, dtype=float)).astype(np.float32),
            escaping=np.zeros(n, dtype=bool) if escaping is None else np.asarray(escaping, dtype=bool),
            directions=np.asarray(directions, dtype=np.float32),
        )
        if self._rays is not None:
            bundle = {k: np.concatenate([self._rays[k], v]) for k, v in bundle.items()}
        self._rays = bundle
        self._extend_escaping()
        for renderer in self._ray_renderers:
            renderer.set_segments(bundle["segments"], bundle["colors"], bundle["intensities"])
        self._exposure_dirty = True
        self.update()

    def draw_polyline(self, polyline, *, color="white", width: float = 2.0) -> None:
        """A drawing-plane polyline overlay (surfaces, sensors, outlines)."""

        polyline = np.asarray(polyline, dtype=float)
        if polyline.shape[0] >= 2:
            self._ensure_scene_center(np.nanmean(polyline, axis=0))
            self._surface_renderers.append(PolylineRenderer(polyline - self._scene_center, _rgba(color), width))
            self.update()

    def draw_filled_polygon(self, vertices, faces, *, color) -> None:
        """A translucent filled triangle mesh — ``(M, 2)`` vertices, ``(K, 3)`` faces — under the outlines."""

        vertices = np.asarray(vertices, dtype=float).reshape(-1, 2)
        faces = np.asarray(faces).reshape(-1, 3)
        if vertices.shape[0] >= 3 and faces.shape[0]:
            self._ensure_scene_center(vertices.mean(axis=0))
            self._fill_renderers.append(FilledPolygonRenderer(vertices - self._scene_center, faces, _rgba(color)))
            self.update()

    def draw_markers(self, points, *, color="crimson", size: float = 6.0) -> None:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if points.shape[0]:
            self._ensure_scene_center(points.mean(axis=0))
            self._marker_renderers.append(MarkerRenderer(points - self._scene_center, Color(color).rgba, size))
            self.update()

    def clear_rays(self) -> None:
        """Drop every ray bundle drawn so far."""

        self._rays = None
        for renderer in self._ray_renderers:
            renderer.set_segments(np.empty((0, 4), np.float32), np.empty((0, 4), np.float32), np.empty(0, np.float32))
        self.update()

    def clear_markers(self) -> None:
        self._marker_renderers.clear()
        self.update()

    def clear(self) -> None:
        """Drop everything drawn: rays, fills, surfaces and markers."""

        self._surface_renderers.clear()
        self._fill_renderers.clear()
        self.clear_markers()
        self.clear_rays()

    def _ensure_scene_center(self, sample) -> None:
        if self._scene_center is None:
            self._scene_center = np.asarray(sample, dtype=float)

    def _extend_escaping(self) -> None:
        """Extend the rays that left the system past the current view's edge."""

        if self._rays is None or not self._rays["escaping"].any():
            return
        idx = np.flatnonzero(self._rays["escaping"])
        starts = self._rays["segments"][idx, 0:2]
        reach = np.linalg.norm(starts - self._view_center_rel, axis=1) + 2.0 * self.camera.diagonal
        self._rays["segments"][idx, 2:4] = starts + self._rays["directions"][idx] * reach[:, None]
        for renderer in self._ray_renderers:
            renderer.update_endpoints(self._rays["segments"])

    # -- rendering ------------------------------------------------------------------

    def run(self) -> None:  # pragma: no cover - interactive
        self.show()
        from .._window import announce_window

        announce_window(self, "[raytracer GL]")
        app.run()

    def read_accumulation(self) -> np.ndarray:
        """The HDR accumulation buffer (activates the context)."""

        self.set_current()
        return self._accumulation.read()

    def snapshot(self) -> np.ndarray:
        """Render offscreen into a uint8 RGBA image; works without an exposed window."""

        self.set_current()
        w, h = (int(v) for v in self._viewport_px())
        with gloo.FrameBuffer(color=gloo.Texture2D(shape=(h, w, 4))):
            self._render_scene(self._viewport_px())
            return gloo.read_pixels(alpha=True)

    def on_draw(self, event) -> None:
        self._render_scene(self._viewport_px())

    def _render_scene(self, viewport) -> None:
        self.camera.viewport = viewport
        config = self.config
        view = dict(viewport=viewport, view_center=self._view_center_rel, ppw=self.camera.pixels_per_world)
        width_px = max(config.ray_width * self.camera.pixels_per_world, config.min_pixels)
        rays = dict(view, width_px=width_px, sigma_px=width_px * config.sigma_factor, weight_scale=config.weight_scale)
        beam, solid = self._ray_renderers
        w, h = int(viewport[0]), int(viewport[1])

        if not config.use_solid_rays:
            self._accumulation.ensure((h, w))
            with self._accumulation.fbo:
                gloo.set_viewport(0, 0, w, h)
                gloo.clear(color=(0.0, 0.0, 0.0, 0.0))
                beam.draw(**rays)
            if not config.auto_exposure:
                self._exposure = config.exposure
            elif self._exposure_dirty:
                try:
                    self._exposure = config.exposure * auto_exposure(self._accumulation.read())
                except Exception as error:  # pragma: no cover - driver dependent
                    logger.warning("Auto-exposure readback failed: %s", error)
                    self._exposure = config.exposure
                self._exposure_dirty = False

        gloo.set_viewport(0, 0, w, h)
        gloo.clear(color=Color(config.background))
        if not config.use_solid_rays:
            background = tuple(Color(config.background).rgb) if config.background != "black" else (0.0, 0.0, 0.0)
            self._tonemap.draw(self._accumulation.texture, exposure=self._exposure, mode=config.tone_map, background=background)
        for renderer in self._fill_renderers + self._surface_renderers:
            renderer.draw(**view)
        if config.use_solid_rays:
            solid.draw(**rays)
        for renderer in self._marker_renderers:
            renderer.draw(**view)

    # -- interaction -------------------------------------------------------------------

    def _viewport_px(self) -> tuple[float, float]:
        w, h = self.physical_size
        return float(w), float(h)

    @property
    def _view_center_rel(self) -> np.ndarray:
        return self.camera.center - (self._scene_center if self._scene_center is not None else np.zeros(2))

    def _mouse_to_gl(self, pos) -> np.ndarray:
        """Window position (top-left origin, logical px) to GL pixels (bottom-left origin)."""

        (w, h), (pw, ph) = self.size, self.physical_size
        return np.array([pos[0] * (pw / w if w else 1.0), (h - pos[1]) * (ph / h if h else 1.0)])

    def on_resize(self, event) -> None:
        if hasattr(self, "camera"):  # some backends (glfw) resize during Canvas.__init__
            self.camera.resize(self._viewport_px())
            self.update()

    def on_mouse_press(self, event) -> None:
        if event.button == 1:
            self._drag_start = self._mouse_to_gl(event.pos)

    def on_mouse_release(self, event) -> None:
        self._drag_start = None

    def on_mouse_move(self, event) -> None:
        if self._drag_start is not None and event.is_dragging:
            pos = self._mouse_to_gl(event.pos)
            self.camera.pan_pixels(*(pos - self._drag_start))
            self._drag_start = pos
            self._extend_escaping()
            self.update()

    def on_mouse_wheel(self, event) -> None:
        self.camera.zoom_about(1.1 ** event.delta[1], self._mouse_to_gl(event.pos))
        self._extend_escaping()
        self.update()


def _rgba(color) -> tuple:
    return tuple(color) if isinstance(color, (tuple, list, np.ndarray)) else Color(color).rgba


__all__ = ["RenderConfig", "OpenGLViewer"]
