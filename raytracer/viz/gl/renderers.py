"""GPU renderers of the 2-D viewer: rays, polylines, fills, markers, and the tone-mapping pass."""

from __future__ import annotations

import numpy as np
from vispy import gloo

from . import shaders

TONE_MODES = {"exponential": 0, "reinhard": 1, "linear": 2}

_FULLSCREEN_TRIANGLE = np.array([[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]], dtype=np.float32)


def segment_quads(segments: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(starts, ends, corners)`` per vertex: each of N segments ``(x0, y0, x1, y1)`` becomes 4 vertices with
    corners ``(t, side)`` in ``{0, 1} × {−1, +1}``."""

    starts = np.repeat(segments[:, 0:2], 4, axis=0).astype(np.float32)
    ends = np.repeat(segments[:, 2:4], 4, axis=0).astype(np.float32)
    corners = np.tile(np.array([[0.0, -1.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]], dtype=np.float32), (segments.shape[0], 1))
    return starts, ends, corners


def quad_indices(n_segments: int) -> np.ndarray:
    """Two triangles per quad of :func:`segment_quads`."""

    offsets = (np.arange(n_segments, dtype=np.uint32) * 4)[:, None]
    return (np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)[None, :] + offsets).ravel()


def auto_exposure(accum_rgba: np.ndarray, *, percentile: float = 99.0) -> float:
    """Exposure that maps the *percentile* energy ``E_p`` of the accumulation to 0.98: ``−ln(0.02) / E_p``."""

    energy = accum_rgba[..., :3].max(axis=-1)
    lit = energy[energy > 0.0]
    reference = float(np.percentile(lit, percentile)) if lit.size else 0.0
    return float(-np.log(0.02) / reference) if reference > 0.0 else 1.0


class AccumulationTarget:
    """Float framebuffer receiving linear additive ray energy."""

    def __init__(self) -> None:
        self._shape: tuple[int, int] | None = None
        self.texture = None
        self.fbo = None

    def ensure(self, shape_hw: tuple[int, int]) -> None:
        if self._shape != shape_hw or self.fbo is None:
            self._shape = shape_hw
            self.texture = gloo.Texture2D(shape=(*shape_hw, 4), internalformat="rgba32f", interpolation="nearest")
            self.fbo = gloo.FrameBuffer(color=self.texture)

    def read(self) -> np.ndarray:
        """The float accumulation ``(h, w, 4)``."""

        with self.fbo:
            return gloo.read_pixels(alpha=True, out_type="float")


class RayRenderer:
    """Ray segments as quads: Gaussian profiles summed additively into the HDR target (beam mode), or with
    ``solid`` crisp lines alpha-blended straight onto the display — each ray one sharp line, for few-ray diagrams."""

    def __init__(self, *, solid: bool = False) -> None:
        self.program = gloo.Program(shaders.RAY_VERT, shaders.RAY_SOLID_FRAG if solid else shaders.RAY_FRAG)
        self.blend = ("src_alpha", "one_minus_src_alpha") if solid else ("one", "one")
        self.solid = solid
        self.index_buffer = None
        self.n_segments = 0

    def set_segments(self, segments: np.ndarray, colors: np.ndarray, intensities: np.ndarray) -> None:
        """Upload ``(N, 4)`` scene-relative segments, ``(N, 4)`` colours (alpha: weight) and ``(N,)`` intensities."""

        self.n_segments = segments.shape[0]
        if self.n_segments == 0:
            return
        self.update_endpoints(segments)
        self.program["a_corner"] = segment_quads(segments)[2]
        self.program["a_color"] = np.repeat(colors, 4, axis=0).astype(np.float32)
        self.program["a_intensity"] = np.repeat(intensities, 4).astype(np.float32)
        self.index_buffer = gloo.IndexBuffer(quad_indices(self.n_segments))

    def update_endpoints(self, segments: np.ndarray) -> None:
        if self.n_segments:
            self.program["a_start"], self.program["a_end"], _ = segment_quads(segments)

    def draw(self, *, viewport, view_center, ppw, width_px, sigma_px, weight_scale) -> None:
        if self.n_segments == 0:
            return
        uniforms = dict(u_viewport=viewport, u_view_center=tuple(view_center), u_ppw=ppw, u_width_px=width_px,
                        u_sigma_px=0.0 if self.solid else sigma_px, u_weight_scale=weight_scale)
        for name, value in uniforms.items():
            self.program[name] = value
        gloo.set_state(blend=True, depth_test=False)
        gloo.set_blend_func(*self.blend)
        self.program.draw("triangles", self.index_buffer)


class _Overlay:
    """A plain alpha-blended overlay drawn in the view: polyline, filled triangles or markers."""

    mode = "triangles"
    count = 0
    index_buffer = None

    def __init__(self, vert, frag, color, **uniforms) -> None:
        self.program = gloo.Program(vert, frag)
        self.uniforms = dict(u_color=color, **uniforms)

    def draw(self, *, viewport, view_center, ppw) -> None:
        if self.count == 0:
            return
        for name, value in dict(u_viewport=viewport, u_view_center=tuple(view_center), u_ppw=ppw, **self.uniforms).items():
            self.program[name] = value
        gloo.set_state(blend=True, depth_test=False)
        gloo.set_blend_func("src_alpha", "one_minus_src_alpha")
        self.program.draw(self.mode, self.index_buffer)


class PolylineRenderer(_Overlay):
    """A solid polyline as per-segment quads, independent of the driver's line width."""

    def __init__(self, polyline: np.ndarray, color, width_px: float) -> None:
        super().__init__(shaders.LINE_VERT, shaders.LINE_FRAG, color, u_width_px=float(width_px))
        points = np.asarray(polyline, dtype=np.float32)
        points = points[~np.isnan(points).any(axis=1)]
        if points.shape[0] < 2:
            return
        segments = np.column_stack([points[:-1], points[1:]])
        lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
        if np.any(lengths > 0):  # drop jumps across the scene (wrap-around artefacts of sampling)
            segments = segments[lengths < 50.0 * np.median(lengths[lengths > 0])]
        self.count = segments.shape[0]
        if self.count:
            self.program["a_start"], self.program["a_end"], self.program["a_corner"] = segment_quads(segments)
            self.index_buffer = gloo.IndexBuffer(quad_indices(self.count))


class FilledPolygonRenderer(_Overlay):
    """A translucent filled triangle mesh: vertices ``(M, 2)`` scene-relative, faces ``(K, 3)``."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, color) -> None:
        super().__init__(shaders.FILL_VERT, shaders.FILL_FRAG, color)
        vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 2)
        faces = np.asarray(faces, dtype=np.uint32).reshape(-1, 3)
        if faces.shape[0] and vertices.shape[0] >= 3:
            self.count = faces.shape[0]
            self.program["a_position"] = vertices
            self.index_buffer = gloo.IndexBuffer(faces.ravel())


class MarkerRenderer(_Overlay):
    """Round point-sprite markers."""

    mode = "points"

    def __init__(self, points: np.ndarray, color, size_px: float) -> None:
        super().__init__(shaders.MARKER_VERT, shaders.MARKER_FRAG, color, u_size_px=float(size_px))
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        self.count = points.shape[0]
        if self.count:
            self.program["a_position"] = points


class TonemapPass:
    """Fullscreen pass compressing the HDR accumulation for display."""

    def __init__(self) -> None:
        self.program = gloo.Program(shaders.TONEMAP_VERT, shaders.TONEMAP_FRAG)
        self.program["a_position"] = _FULLSCREEN_TRIANGLE

    def draw(self, texture, *, exposure: float, mode: str, background) -> None:
        self.program["u_accum"] = texture
        self.program["u_exposure"] = exposure
        self.program["u_mode"] = TONE_MODES.get(mode, 0)
        self.program["u_background"] = background
        gloo.set_state(blend=False, depth_test=False)
        self.program.draw("triangles")


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
