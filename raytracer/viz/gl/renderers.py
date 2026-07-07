"""GPU renderers for the OpenGL viewer: rays, polylines, markers, tonemap."""

from __future__ import annotations

import logging

import numpy as np
from vispy import gloo
from vispy.gloo import gl

from . import shaders

logger = logging.getLogger("raytracer.viz.gl")

TONE_MODES = {"exponential": 0, "reinhard": 1, "linear": 2}


def segment_quads(segments: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (starts, ends, corners) per-vertex arrays for N segments.

    Every segment becomes 4 vertices with corner coordinates
    (t, side) in {0,1} x {-1,+1} and 6 indices (two triangles).
    """

    n = segments.shape[0]
    starts = np.repeat(segments[:, 0:2], 4, axis=0).astype(np.float32)
    ends = np.repeat(segments[:, 2:4], 4, axis=0).astype(np.float32)
    corners = np.tile(
        np.array([[0.0, -1.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]], dtype=np.float32),
        (n, 1),
    )
    return starts, ends, corners


def quad_indices(n_segments: int) -> np.ndarray:
    base = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    offsets = (np.arange(n_segments, dtype=np.uint32) * 4)[:, None]
    return (base[None, :] + offsets).ravel()


class AccumulationTarget:
    """Float framebuffer receiving linear additive ray energy."""

    def __init__(self) -> None:
        self._shape: tuple[int, int] | None = None
        self.texture: gloo.Texture2D | None = None
        self.fbo: gloo.FrameBuffer | None = None

    def ensure(self, shape_hw: tuple[int, int]) -> None:
        if self._shape == shape_hw and self.fbo is not None:
            return
        self._shape = shape_hw
        h, w = shape_hw
        self.texture = gloo.Texture2D(
            shape=(h, w, 4), internalformat="rgba32f", interpolation="nearest"
        )
        self.fbo = gloo.FrameBuffer(color=self.texture)

    def read(self) -> np.ndarray:
        """Read back the float accumulation buffer (h, w, 4)."""

        with self.fbo:
            return gloo.read_pixels(alpha=True, out_type="float")


class RayRenderer:
    """Draws ray segments as Gaussian-profile quads with additive blending."""

    def __init__(self) -> None:
        self.program = gloo.Program(shaders.RAY_VERT, shaders.RAY_FRAG)
        self.index_buffer: gloo.IndexBuffer | None = None
        self.n_segments = 0

    def set_segments(
        self,
        segments: np.ndarray,  # (N, 4): x0, y0, x1, y1 (scene-relative world)
        colors: np.ndarray,  # (N, 4) rgb + weight
        intensities: np.ndarray,  # (N,)
    ) -> None:
        self.n_segments = segments.shape[0]
        if self.n_segments == 0:
            return
        starts, ends, corners = segment_quads(segments)
        self.program["a_start"] = starts
        self.program["a_end"] = ends
        self.program["a_corner"] = corners
        self.program["a_color"] = np.repeat(colors, 4, axis=0).astype(np.float32)
        self.program["a_intensity"] = np.repeat(intensities, 4).astype(np.float32)
        self.index_buffer = gloo.IndexBuffer(quad_indices(self.n_segments))

    def update_endpoints(self, segments: np.ndarray) -> None:
        """Re-upload only endpoints (used for escaping-ray extension)."""

        if self.n_segments == 0:
            return
        starts, ends, _ = segment_quads(segments)
        self.program["a_start"] = starts
        self.program["a_end"] = ends

    def draw(
        self,
        *,
        viewport: tuple[float, float],
        view_center: np.ndarray,
        ppw: float,
        width_px: float,
        sigma_px: float,
        weight_scale: float,
        use_solid: bool,
    ) -> None:
        if self.n_segments == 0:
            return
        self.program["u_viewport"] = viewport
        self.program["u_view_center"] = tuple(view_center)
        self.program["u_ppw"] = ppw
        self.program["u_width_px"] = width_px
        self.program["u_sigma_px"] = sigma_px
        self.program["u_weight_scale"] = weight_scale
        self.program["u_use_solid"] = 1 if use_solid else 0
        gloo.set_state(blend=True, depth_test=False)
        gloo.set_blend_func("one", "one")  # linear additive accumulation
        self.program.draw("triangles", self.index_buffer)


class PolylineRenderer:
    """Solid polylines rendered as per-segment quads (line_width-independent)."""

    def __init__(self, polyline: np.ndarray, color, width_px: float) -> None:
        self.program = gloo.Program(shaders.LINE_VERT, shaders.LINE_FRAG)
        self.color = color
        self.width_px = float(width_px)
        points = np.asarray(polyline, dtype=np.float32)
        valid = ~np.isnan(points).any(axis=1)
        points = points[valid]
        if points.shape[0] < 2:
            self.n_segments = 0
            return
        segments = np.column_stack([points[:-1], points[1:]])
        # Drop segments that jump across the whole scene (polar wrap
        # artifacts in conic sampling).
        lengths = np.hypot(
            segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1]
        )
        if lengths.size:
            typical = np.median(lengths[lengths > 0]) if np.any(lengths > 0) else 0.0
            if typical > 0:
                segments = segments[lengths < 50.0 * typical]
        self.n_segments = segments.shape[0]
        if self.n_segments == 0:
            return
        starts, ends, corners = segment_quads(segments)
        self.program["a_start"] = starts
        self.program["a_end"] = ends
        self.program["a_corner"] = corners
        self.index_buffer = gloo.IndexBuffer(quad_indices(self.n_segments))

    def draw(self, *, viewport, view_center, ppw) -> None:
        if self.n_segments == 0:
            return
        self.program["u_viewport"] = viewport
        self.program["u_view_center"] = tuple(view_center)
        self.program["u_ppw"] = ppw
        self.program["u_width_px"] = self.width_px
        self.program["u_color"] = self.color
        gloo.set_state(blend=True, depth_test=False)
        gloo.set_blend_func("src_alpha", "one_minus_src_alpha")
        self.program.draw("triangles", self.index_buffer)


class FilledPolygonRenderer:
    """Translucent filled triangle meshes drawn as flat-color overlays.

    Vertices are scene-relative world (M, 2); faces are (K, 3) uint32 indices.
    Used for lens bodies coloured by material (shows the enclosed medium).
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, color) -> None:
        self.program = gloo.Program(shaders.FILL_VERT, shaders.FILL_FRAG)
        vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 2)
        faces = np.asarray(faces, dtype=np.uint32).reshape(-1, 3)
        self.n_faces = faces.shape[0]
        if self.n_faces == 0 or vertices.shape[0] < 3:
            self.n_faces = 0
            return
        self.program["a_position"] = vertices
        self.index_buffer = gloo.IndexBuffer(faces.ravel())
        self.color = color

    def draw(self, *, viewport, view_center, ppw) -> None:
        if self.n_faces == 0:
            return
        self.program["u_viewport"] = viewport
        self.program["u_view_center"] = tuple(view_center)
        self.program["u_ppw"] = ppw
        self.program["u_color"] = self.color
        gloo.set_state(blend=True, depth_test=False)
        gloo.set_blend_func("src_alpha", "one_minus_src_alpha")
        self.program.draw("triangles", self.index_buffer)


class MarkerRenderer:
    """Round point-sprite markers."""

    def __init__(self, points: np.ndarray, color, size_px: float) -> None:
        self.program = gloo.Program(shaders.MARKER_VERT, shaders.MARKER_FRAG)
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        self.n_points = points.shape[0]
        if self.n_points == 0:
            return
        self.program["a_position"] = points
        self.color = color
        self.size_px = float(size_px)

    def draw(self, *, viewport, view_center, ppw) -> None:
        if self.n_points == 0:
            return
        self.program["u_viewport"] = viewport
        self.program["u_view_center"] = tuple(view_center)
        self.program["u_ppw"] = ppw
        self.program["u_size_px"] = self.size_px
        self.program["u_color"] = self.color
        gloo.set_state(blend=True, depth_test=False)
        gloo.set_blend_func("src_alpha", "one_minus_src_alpha")
        self.program.draw("points")


class TonemapPass:
    """Fullscreen pass compressing the HDR accumulation for display."""

    def __init__(self) -> None:
        self.program = gloo.Program(shaders.TONEMAP_VERT, shaders.TONEMAP_FRAG)
        # Fullscreen triangle (covers the viewport with 3 vertices).
        self.program["a_position"] = np.array(
            [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]], dtype=np.float32
        )

    def draw(
        self,
        texture: gloo.Texture2D,
        *,
        exposure: float,
        mode: str,
        background: tuple[float, float, float],
    ) -> None:
        self.program["u_accum"] = texture
        self.program["u_exposure"] = exposure
        self.program["u_mode"] = TONE_MODES.get(mode, 0)
        self.program["u_background"] = background
        gloo.set_state(blend=False, depth_test=False)
        self.program.draw("triangles")


def auto_exposure(accum_rgba: np.ndarray, *, percentile: float = 99.0) -> float:
    """Exposure that maps the bright end of the accumulation near white.

    With exponential tone mapping, mapping the *percentile* energy E_p to
    0.98 requires exposure = -ln(0.02) / E_p.
    """

    energy = accum_rgba[..., :3].max(axis=-1)
    lit = energy[energy > 0.0]
    if lit.size == 0:
        return 1.0
    reference = float(np.percentile(lit, percentile))
    if reference <= 0.0:
        return 1.0
    return float(-np.log(0.02) / reference)


__all__ = [
    "AccumulationTarget",
    "RayRenderer",
    "PolylineRenderer",
    "FilledPolygonRenderer",
    "MarkerRenderer",
    "TonemapPass",
    "auto_exposure",
    "segment_quads",
    "quad_indices",
    "TONE_MODES",
]
