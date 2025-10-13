"""Lightweight 2-D OpenGL viewer with quadratic accumulation for ray intensity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
from vispy import app, gloo
from vispy.color import Color

AccumulationMode = Literal["squared", "alpha"]


@dataclass
class RenderConfig:
    """Configuration parameters for ray rendering."""

    ray_width: float = 0.5
    sigma_factor: float = 0.5
    accumulation_mode: AccumulationMode = "squared"
    default_intensity: float = 1.0
    weight_scale: float = 1.0
    min_pixels: float = 1.0

RAY_VERT = "\n".join(
    [
        "#ifdef GL_ES",
        "precision highp float;",
        "precision mediump int;",
        "#endif",
        "",
        "attribute vec2 a_corner;",
        "attribute vec2 a_start;",
        "attribute vec2 a_end;",
        "attribute float a_width;",
        "attribute float a_sigma;",
        "attribute float a_intensity;",
        "attribute vec4 a_color;",
        "",
        "uniform vec2 u_viewport;",
        "uniform vec2 u_screen_offset;",
        "uniform vec2 u_screen_scale;",
        "",
        "varying vec2 v_start;",
        "varying vec2 v_end;",
        "varying float v_sigma;",
        "varying float v_intensity;",
        "varying vec4 v_color;",
        "varying vec2 v_pos;",
        "",
        "void main() {",
        "    vec2 start_world = a_start;",
        "    vec2 end_world = a_end;",
        "    vec2 start_screen = (start_world - u_screen_offset) * u_screen_scale;",
        "    vec2 end_screen = (end_world - u_screen_offset) * u_screen_scale;",
        "",
        "    vec2 dir = end_screen - start_screen;",
        "    float len = length(dir);",
        "    if (len < 1e-6) {",
        "        dir = vec2(1.0, 0.0);",
        "        len = 1.0;",
        "    } else {",
        "        dir = dir / len;",
        "    }",
        "    vec2 perp = vec2(-dir.y, dir.x);",
        "",
        "    float t = 0.5 * (a_corner.x + 1.0);",
        "    vec2 along_screen = mix(start_screen, end_screen, t);",
        "    float geom = max(a_width, 0.5);",
        "    vec2 pos_screen = along_screen + perp * a_corner.y * geom * 3.0;",
        "    vec2 pos_ndc = (pos_screen / u_viewport) * 2.0 - 1.0;",
        "    gl_Position = vec4(pos_ndc, 0.0, 1.0);",
        "",
        "    v_start = start_screen;",
        "    v_end = end_screen;",
        "    v_sigma = max(a_sigma, 1e-6);",
        "    v_intensity = a_intensity;",
        "    v_color = a_color;",
        "    v_pos = pos_screen;",
        "}",
    ]
) + "\n"

RAY_FRAG = "\n".join(
    [
        "#ifdef GL_ES",
        "precision highp float;",
        "precision mediump int;",
        "#endif",
        "varying vec2 v_start;",
        "varying vec2 v_end;",
        "varying float v_sigma;",
        "varying float v_intensity;",
        "varying vec4 v_color;",
        "varying vec2 v_pos;",
        "uniform int u_accum_mode;",
        "uniform float u_weight_scale;",
        "float point_to_segment_dist(vec2 p, vec2 a, vec2 b) {",
        "    vec2 pa = p - a;",
        "    vec2 ba = b - a;",
        "    float denom = dot(ba, ba);",
        "    if (denom < 1e-12) {",
        "        return length(pa);",
        "    }",
        "    float h = clamp(dot(pa, ba) / denom, 0.0, 1.0);",
        "    return length(pa - ba * h);",
        "}",
        "void main() {",
        "    float dist = point_to_segment_dist(v_pos, v_start, v_end);",
        "    float sigma = v_sigma;",
        "    float gaussian = exp(-(dist * dist) / (sigma * sigma));",
        "    if (gaussian < 1e-4) {",
        "        discard;",
        "    }",
        "    float weight = clamp(v_color.a * u_weight_scale, 0.0, 10.0);",
        "    float I = weight * v_intensity * gaussian;",
        "    if (u_accum_mode == 0) {",
        "        float rootI = sqrt(I);",
        "        gl_FragColor = vec4(v_color.rgb * rootI, rootI);",
        "    } else {",
        "        float alpha = clamp(I, 0.0, 1.0);",
        "        gl_FragColor = vec4(v_color.rgb * I, alpha);",
        "    }",
        "}",
    ]
) + "\n"

LINE_VERT = "\n".join(
    [
        "#ifdef GL_ES",
        "precision highp float;",
        "precision mediump int;",
        "#endif",
        "attribute vec2 a_pos;",
        "uniform vec2 u_viewport;",
        "uniform vec2 u_screen_offset;",
        "uniform vec2 u_screen_scale;",
        "void main() {",
        "    vec2 screen = (a_pos - u_screen_offset) * u_screen_scale;",
        "    vec2 ndc = (screen / u_viewport) * 2.0 - 1.0;",
        "    gl_Position = vec4(ndc, 0.0, 1.0);",
        "}",
    ]
) + "\n"

LINE_FRAG = "#ifdef GL_ES\nprecision highp float;\nprecision mediump int;\n#endif\nuniform vec4 u_color;\nvoid main() {\n    gl_FragColor = u_color;\n}\n"

MARKER_VERT = "\n".join(
    [
        "#ifdef GL_ES",
        "precision highp float;",
        "precision mediump int;",
        "#endif",
        "attribute vec2 a_pos;",
        "uniform vec2 u_viewport;",
        "uniform vec2 u_screen_offset;",
        "uniform vec2 u_screen_scale;",
        "uniform float u_point_size;",
        "void main() {",
        "    vec2 screen = (a_pos - u_screen_offset) * u_screen_scale;",
        "    vec2 ndc = (screen / u_viewport) * 2.0 - 1.0;",
        "    gl_Position = vec4(ndc, 0.0, 1.0);",
        "    gl_PointSize = u_point_size;",
        "}",
    ]
) + "\n"

MARKER_FRAG = "\n".join(
    [
        "#ifdef GL_ES",
        "precision highp float;",
        "precision mediump int;",
        "#endif",
        "uniform vec4 u_color;",
        "void main() {",
        "    vec2 uv = gl_PointCoord * 2.0 - 1.0;",
        "    float r2 = dot(uv, uv);",
        "    if (r2 > 1.0) {",
        "        discard;",
        "    }",
        "    float alpha = smoothstep(1.0, 0.6, r2);",
        "    gl_FragColor = vec4(u_color.rgb, u_color.a * alpha);",
        "}",
    ]
) + "\n"

class RayRenderer:
    def __init__(self, config: RenderConfig):
        self.program = gloo.Program(RAY_VERT, RAY_FRAG)
        self.vbo = gloo.VertexBuffer()
        self.ibo = gloo.IndexBuffer()
        self._n_indices = 0
        self.config = config

    def set_data(self, vertices: np.ndarray, indices: np.ndarray) -> None:
        self.vbo.set_data(vertices)
        self.ibo.set_data(indices.astype(np.uint32))
        self._n_indices = int(indices.size)

    def draw(
        self,
        viewport: Tuple[int, int],
        screen_offset: np.ndarray,
        screen_scale: np.ndarray,
        accumulation_mode: AccumulationMode,
        weight_scale: float,
    ) -> None:
        if self._n_indices == 0:
            return
        self.program.bind(self.vbo)
        self.program["u_viewport"] = np.array(viewport, dtype=np.float32)
        self.program["u_screen_offset"] = screen_offset.astype(np.float32)
        self.program["u_screen_scale"] = screen_scale.astype(np.float32)
        mode_int = 0 if accumulation_mode == "squared" else 1
        self.program["u_accum_mode"] = int(mode_int)
        self.program["u_weight_scale"] = float(weight_scale)
        gloo.set_state(depth_test=False, blend=True)
        if mode_int == 0:
            gloo.set_blend_func('one', 'one')
        else:
            gloo.set_blend_func('src_alpha', 'one_minus_src_alpha')
        self.program.draw('triangles', self.ibo)


class PolylineRenderer:
    def __init__(self):
        self.program = gloo.Program(LINE_VERT, LINE_FRAG)
        self.vbo = gloo.VertexBuffer()
        self._n_vertices = 0
        self.color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.width = 1.0

    def set_data(self, points: np.ndarray, color: Sequence[float], width: float) -> None:
        structured = np.zeros(points.shape[0], dtype=[('a_pos', np.float32, 2)])
        structured['a_pos'] = points.astype(np.float32)
        self.vbo.set_data(structured)
        self._n_vertices = int(points.shape[0])
        self.color = np.array(color, dtype=np.float32)
        self.width = float(width)

    def draw(
        self,
        viewport: Tuple[int, int],
        screen_offset: np.ndarray,
        screen_scale: np.ndarray,
    ) -> None:
        if self._n_vertices == 0:
            return
        self.program.bind(self.vbo)
        self.program["u_viewport"] = np.array(viewport, dtype=np.float32)
        self.program["u_screen_offset"] = screen_offset.astype(np.float32)
        self.program["u_screen_scale"] = screen_scale.astype(np.float32)
        self.program["u_color"] = self.color
        gloo.set_state(depth_test=False, blend=False, line_width=self.width)
        self.program.draw('line_strip')


class MarkerRenderer:
    def __init__(self):
        self.program = gloo.Program(MARKER_VERT, MARKER_FRAG)
        self.vbo = gloo.VertexBuffer()
        self._n_points = 0
        self.color = np.array([0.8, 0.1, 0.1, 1.0], dtype=np.float32)
        self.size = 6.0

    def set_data(self, points: np.ndarray, color: Sequence[float], size: float) -> None:
        structured = np.zeros(points.shape[0], dtype=[('a_pos', np.float32, 2)])
        structured['a_pos'] = points.astype(np.float32)
        self.vbo.set_data(structured)
        self._n_points = int(points.shape[0])
        self.color = np.array(color, dtype=np.float32)
        self.size = float(size)

    def draw(
        self,
        viewport: Tuple[int, int],
        screen_offset: np.ndarray,
        screen_scale: np.ndarray,
    ) -> None:
        if self._n_points == 0:
            return
        self.program.bind(self.vbo)
        self.program["u_viewport"] = np.array(viewport, dtype=np.float32)
        self.program["u_screen_offset"] = screen_offset.astype(np.float32)
        self.program["u_screen_scale"] = screen_scale.astype(np.float32)
        self.program["u_color"] = self.color
        self.program["u_point_size"] = float(self.size)
        gloo.set_state(depth_test=False, blend=True)
        gloo.set_blend_func('src_alpha', 'one_minus_src_alpha')
        self.program.draw('points')

class OpenGLViewer:
    """Minimal 2-D OpenGL viewer with quadratic accumulation."""

    def __init__(
        self,
        *,
        x_lims: Tuple[float, float] = (-6.0, 6.0),
        y_lims: Tuple[float, float] = (-6.0, 6.0),
        show: bool = True,
        size: Tuple[int, int] = (900, 700),
        bgcolor: str | Sequence[float] = "black",
        render_config: Optional[RenderConfig] = None,
    ) -> None:
        self.render_config = render_config or RenderConfig()
        self.canvas = app.Canvas(keys="interactive", size=size, show=show)
        self.canvas.events.draw.connect(self._on_draw)
        self.canvas.events.resize.connect(self._on_resize)
        self.canvas.events.mouse_press.connect(self._on_mouse_press)
        self.canvas.events.mouse_release.connect(self._on_mouse_release)
        self.canvas.events.mouse_move.connect(self._on_mouse_move)
        self.canvas.events.mouse_wheel.connect(self._on_mouse_wheel)

        self._bgcolor = np.array(Color(bgcolor).rgba, dtype=np.float32)
        self._viewport = (int(size[0]), int(size[1]))
        self._view_rect = [float(x_lims[0]), float(x_lims[1]), float(y_lims[0]), float(y_lims[1])]
        self._pan_active = False
        self._last_mouse_pos: Optional[Tuple[float, float]] = None

        self._ray_renderer = RayRenderer(self.render_config)
        self._surface_renderers: List[PolylineRenderer] = []
        self._marker_renderer = MarkerRenderer()

        self._ray_records: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []
        self._surface_records: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self._marker_points_world = np.zeros((0, 2), dtype=np.float32)

        if show:
            self.canvas.show()

    def _on_resize(self, event) -> None:
        size = event.size
        self._viewport = (int(size[0]), int(size[1]))
        gloo.set_viewport(0, 0, *self._viewport)
        self._update_all_geometry()
        self.canvas.update()

    def _on_draw(self, event) -> None:
        gloo.clear(color=self._bgcolor)
        viewport = self._viewport
        screen_offset, screen_scale = self._screen_transform()
        for renderer in self._surface_renderers:
            renderer.draw(viewport, screen_offset, screen_scale)
        self._ray_renderer.draw(viewport, screen_offset, screen_scale, self.render_config.accumulation_mode, self.render_config.weight_scale)
        self._marker_renderer.draw(viewport, screen_offset, screen_scale)

    def _on_mouse_press(self, event) -> None:
        if event.button == 1:
            self._pan_active = True
            self._last_mouse_pos = tuple(event.pos)

    def _on_mouse_release(self, event) -> None:
        if event.button == 1:
            self._pan_active = False
            self._last_mouse_pos = None

    def _on_mouse_move(self, event) -> None:
        if not self._pan_active or self._last_mouse_pos is None:
            return
        cur = event.pos
        prev = self._last_mouse_pos
        dx = cur[0] - prev[0]
        dy = cur[1] - prev[1]
        self._last_mouse_pos = tuple(cur)
        self._pan(dx, dy)

    def _on_mouse_wheel(self, event) -> None:
        scale = 1.1 ** (-event.delta[1])
        self._zoom(scale, event.pos)

    def draw_surfaces(self, surfaces: Iterable, color: str | Sequence[float] = "white", width: float = 2.0) -> None:
        self._surface_records.clear()
        rgba = np.array(Color(color).rgba, dtype=np.float32)
        for surface in surfaces:
            if hasattr(surface, "polyline_segments"):
                segments = surface.polyline_segments()
            else:
                segments = [surface.polyline()]
            for seg in segments:
                pts = np.asarray(seg, dtype=np.float32)
                if pts.ndim != 2 or pts.shape[0] < 2:
                    continue
                if pts.shape[1] == 3:
                    pts = pts[:, :2]
                self._surface_records.append((pts, rgba, float(width)))
        self._rebuild_surface_renderers()
        self.canvas.update()

    def draw_rays(
        self,
        tree,
        *,
        tail_length: float = 12.0,
        color_resolver: Optional[Callable] = None,
        intensity_resolver: Optional[Callable] = None,
        show_misses: bool = True,
        marker_color: str | Sequence[float] = "crimson",
        marker_size: float = 6.0,
    ) -> None:
        self._ray_records.clear()
        marker_list: List[np.ndarray] = []

        default_hit = np.array(Color("white").rgba, dtype=np.float32)
        default_miss = np.array(Color("orange").rgba, dtype=np.float32)

        for node in tree.nodes():
            start = np.asarray(node.ray.origin[:2], dtype=np.float32)
            direction = np.asarray(node.ray.direction[:2], dtype=np.float32)

            if node.intersection is None:
                if not show_misses:
                    continue
                end = start + direction * float(tail_length)
                color = default_miss
            else:
                end = np.asarray(node.intersection.point[:2], dtype=np.float32)
                color = default_hit
                marker_list.append(end)

            if color_resolver is not None:
                custom = np.asarray(color_resolver(node), dtype=np.float32)
                if custom.shape[0] == 3:
                    custom = np.concatenate([custom, np.array([1.0], dtype=np.float32)])
                color = custom

            base_intensity = float(self.render_config.default_intensity)
            intensity = float(intensity_resolver(node)) if intensity_resolver else base_intensity

            self._ray_records.append((start, end, intensity, color))

        self._update_ray_geometry()

        if marker_list:
            self._marker_points_world = np.vstack(marker_list).astype(np.float32)
        else:
            self._marker_points_world = np.zeros((0, 2), dtype=np.float32)

        self._update_marker_geometry(color=marker_color, size=marker_size)

        self.canvas.update()

    def update_visual_params_only(self) -> None:
        self._update_ray_geometry()
        self.canvas.update()

    def run(self) -> None:
        app.run()

    def close(self) -> None:
        self.canvas.close()

    def clear_markers(self) -> None:
        self._marker_points_world = np.zeros((0, 2), dtype=np.float32)
        self._marker_renderer.set_data(self._marker_points_world, np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), 1.0)
        self.canvas.update()

    def _rebuild_surface_renderers(self) -> None:
        self._surface_renderers = []
        if not self._surface_records:
            return
        for pts_world, color, width in self._surface_records:
            renderer = PolylineRenderer()
            renderer.set_data(pts_world, color, width)
            self._surface_renderers.append(renderer)

    def _update_all_geometry(self) -> None:
        if self._surface_records:
            self._rebuild_surface_renderers()
        if self._ray_records:
            self._update_ray_geometry()
        self._update_marker_geometry()

    def _update_ray_geometry(self) -> None:
        if not self._ray_records:
            self._ray_renderer.set_data(np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.uint32))
            return

        n = len(self._ray_records)
        dtype = np.dtype([
            ("a_corner", np.float32, 2),
            ("a_start", np.float32, 2),
            ("a_end", np.float32, 2),
            ("a_width", np.float32, 1),
            ("a_sigma", np.float32, 1),
            ("a_intensity", np.float32, 1),
            ("a_color", np.float32, 4),
        ])

        starts = np.array([rec[0] for rec in self._ray_records], dtype=np.float32)
        ends = np.array([rec[1] for rec in self._ray_records], dtype=np.float32)
        intensities = np.array([rec[2] for rec in self._ray_records], dtype=np.float32)
        colors = np.array([rec[3] for rec in self._ray_records], dtype=np.float32)

        direction = ends - starts
        norms = np.linalg.norm(direction, axis=1)
        safe_norms = np.where(norms < 1e-6, 1.0, norms)
        direction /= safe_norms[:, None]
        direction[norms < 1e-6] = np.array([1.0, 0.0], dtype=np.float32)

        perp = np.stack([-direction[:, 1], direction[:, 0]], axis=1)
        offset_world = perp * float(self.render_config.ray_width)

        x0, x1, y0, y1 = self._view_rect
        sx = self._viewport[0] / max(x1 - x0, 1e-6)
        sy = self._viewport[1] / max(y1 - y0, 1e-6)
        offset_screen = np.stack([offset_world[:, 0] * sx, offset_world[:, 1] * sy], axis=1)
        width_pixels = np.maximum(np.linalg.norm(offset_screen, axis=1), self.render_config.min_pixels)
        sigma_pixels = np.maximum(width_pixels * self.render_config.sigma_factor, 1e-3)

        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)
        vertices = np.zeros(n * 4, dtype=dtype)
        vertices['a_corner'] = np.tile(corners, (n, 1))
        vertices['a_start'] = np.repeat(starts, 4, axis=0)
        vertices['a_end'] = np.repeat(ends, 4, axis=0)
        vertices['a_width'] = np.repeat(width_pixels[:, None], 4, axis=0)
        vertices['a_sigma'] = np.repeat(sigma_pixels[:, None], 4, axis=0)
        vertices['a_intensity'] = np.repeat(intensities[:, None], 4, axis=0)
        vertices['a_color'] = np.repeat(colors, 4, axis=0)

        base = (np.arange(n, dtype=np.uint32) * 4).reshape(-1, 1)
        pattern = np.array([[0, 1, 2, 0, 2, 3]], dtype=np.uint32)
        indices = (base + pattern).reshape(-1)

        self._ray_renderer.set_data(vertices, indices)

    def _update_marker_geometry(self, *, color: str | Sequence[float] | None = None, size: float | None = None) -> None:
        rgba = np.array(Color(color).rgba, dtype=np.float32) if color is not None else self._marker_renderer.color
        radius = float(size) if size is not None else self._marker_renderer.size

        pts = self._marker_points_world if self._marker_points_world.size else np.zeros((0, 2), dtype=np.float32)
        self._marker_renderer.set_data(pts, rgba, radius)

    def _world_to_screen(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        x0, x1, y0, y1 = self._view_rect
        w, h = self._viewport
        sx = w / max(x1 - x0, 1e-6)
        sy = h / max(y1 - y0, 1e-6)
        x = (points[..., 0] - x0) * sx
        y = (points[..., 1] - y0) * sy
        return np.stack([x, y], axis=-1)

    def _screen_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        x0, x1, y0, y1 = self._view_rect
        sx = self._viewport[0] / max(x1 - x0, 1e-6)
        sy = self._viewport[1] / max(y1 - y0, 1e-6)
        offset = np.array([x0, y0], dtype=np.float32)
        scale = np.array([sx, sy], dtype=np.float32)
        return offset, scale

    def _ray_width_sigma_pixels(self, start: np.ndarray, end: np.ndarray) -> Tuple[float, float]:
        x0, x1, y0, y1 = self._view_rect
        w, h = self._viewport
        sx = w / (x1 - x0)
        sy = h / (y1 - y0)
        direction = end - start
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float32)
            norm = 1.0
        else:
            direction = direction / norm
        perp_world = np.array([-direction[1], direction[0]], dtype=np.float32)
        offset_world = perp_world * float(self.render_config.ray_width)
        offset_screen = np.array([offset_world[0] * sx, offset_world[1] * sy], dtype=np.float32)
        width_pixels = float(max(np.linalg.norm(offset_screen), self.render_config.min_pixels))
        sigma_pixels = float(max(width_pixels * self.render_config.sigma_factor, 1e-3))
        return width_pixels, sigma_pixels

    def _pan(self, dx_pixels: float, dy_pixels: float) -> None:
        w, h = self._viewport
        x0, x1, y0, y1 = self._view_rect
        sx = (x1 - x0) / w
        sy = (y1 - y0) / h
        self._view_rect[0] -= dx_pixels * sx
        self._view_rect[1] -= dx_pixels * sx
        self._view_rect[2] -= dy_pixels * sy
        self._view_rect[3] -= dy_pixels * sy
        self._update_all_geometry()
        self.canvas.update()

    def _zoom(self, scale: float, cursor_pos: Tuple[float, float]) -> None:
        w, h = self._viewport
        if w <= 0 or h <= 0:
            return
        x0, x1, y0, y1 = self._view_rect
        cx, cy = self._screen_to_world(cursor_pos)
        width = (x1 - x0) * scale
        height = (y1 - y0) * scale
        alpha_x = (cx - x0) / (x1 - x0)
        alpha_y = (cy - y0) / (y1 - y0)
        self._view_rect[0] = cx - width * alpha_x
        self._view_rect[1] = self._view_rect[0] + width
        self._view_rect[2] = cy - height * alpha_y
        self._view_rect[3] = self._view_rect[2] + height
        self._update_all_geometry()
        self.canvas.update()

    def _screen_to_world(self, pos: Tuple[float, float]) -> Tuple[float, float]:
        x0, x1, y0, y1 = self._view_rect
        w, h = self._viewport
        x = x0 + (pos[0] / w) * (x1 - x0)
        y = y0 + (pos[1] / h) * (y1 - y0)
        return float(x), float(y)


__all__ = ["OpenGLViewer", "RenderConfig"]
