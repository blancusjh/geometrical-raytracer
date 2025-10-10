"""OpenGL-based visualization with intensity accumulation and gaussian ray rendering.

- Portable index buffers (uint16) for ES2/ANGLE.
- Correct VisPy transform hookup.
- Dynamic GL blend state per accumulation mode.
- Alpha acts as a per-ray weight in squared accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Literal, List, Tuple, Optional

import numpy as np
from vispy import app, scene, gloo
from vispy.scene.visuals import create_visual_node
from vispy.visuals import Visual
from vispy.color import Color

# --------------------------------------------------------------------------------------
# Shaders
# --------------------------------------------------------------------------------------

RAY_TUBE_VERT = """
// Per-vertex attributes (quad corners: -1,-1 ; 1,-1 ; 1,1 ; -1,1)
attribute vec2 a_corner;

// Per-vertex (per-quad) attributes (duplicated for now; could be instanced later)
attribute vec2 a_ray_start;
attribute vec2 a_ray_end;
attribute float a_intensity;
attribute vec4 a_color;

// Uniforms
uniform float u_ray_width;

// Varyings
varying vec2 v_ray_start;
varying vec2 v_ray_end;
varying float v_intensity;
varying vec4 v_color;
varying vec2 v_pos;

void main() {
    // Compute ray direction and perpendicular in *visual* coordinates
    vec2 dir = a_ray_end - a_ray_start;
    float len = length(dir);
    // Guard against degenerate segments to avoid NaNs
    if (len < 1e-6) {
        dir = vec2(1.0, 0.0);
        len = 1.0;
    } else {
        dir = dir / len;
    }
    vec2 perp = vec2(-dir.y, dir.x);

    // Along-ray interpolation via corner.x in [-1, 1] -> t in [0, 1]
    float t = 0.5 * (a_corner.x + 1.0);
    vec2 along = mix(a_ray_start, a_ray_end, t);

    // Expand quad across-ray via corner.y in [-1, 1]
    // 3x width for Gaussian tail coverage.
    vec2 position = along + perp * a_corner.y * u_ray_width * 3.0;

    v_ray_start = a_ray_start;
    v_ray_end = a_ray_end;
    v_intensity = a_intensity;
    v_color = a_color;
    v_pos = position;

    gl_Position = $transform(vec4(position, 0.0, 1.0));
}
"""

RAY_TUBE_FRAG = """
varying vec2 v_ray_start;
varying vec2 v_ray_end;
varying float v_intensity;
varying vec4 v_color;
varying vec2 v_pos;

uniform float u_sigma_factor;
uniform float u_ray_width;
uniform int   u_accumulation_mode;   // 0: squared(additive), 1: alpha(linear)
uniform float u_weight_scale;         // Extra scale for per-ray weight if desired

// Distance from point to segment
float point_to_segment_dist(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float denom = dot(ba, ba);
    // Handle degenerate segments
    if (denom < 1e-12) {
        return length(pa);
    }
    float h = clamp(dot(pa, ba) / denom, 0.0, 1.0);
    return length(pa - ba * h);
}

void main() {
    float dist = point_to_segment_dist(v_pos, v_ray_start, v_ray_end);

    // Gaussian falloff with σ = u_sigma_factor * u_ray_width; guard σ>0
    float sigma = max(1e-6, u_ray_width * u_sigma_factor);
    float gaussian = exp(- (dist * dist) / (sigma * sigma));

    // Quick reject (optional micro-optimization)
    if (gaussian < 1e-3) {
        discard;
    }

    // Per-ray weight: use color alpha (user-controlled) times a global scale
    float weight = clamp(v_color.a * u_weight_scale, 0.0, 10.0);

    // Physical intensity at this fragment
    float I = weight * v_intensity * gaussian;

    if (u_accumulation_mode == 0) {
        // Squared accumulation (radiance-like): add sqrt(I) into framebuffer
        vec3 out_rgb = v_color.rgb * sqrt(I);
        float out_a  = sqrt(I); // alpha channel sums too with (one, one); not used visually
        gl_FragColor = vec4(out_rgb, out_a);
    } else {
        // Linear alpha blending: non-premultiplied composition
        float a = clamp(I, 0.0, 1.0);
        vec3  c = v_color.rgb * I;
        gl_FragColor = vec4(c, a);
    }
}
"""

# --------------------------------------------------------------------------------------
# Public configuration
# --------------------------------------------------------------------------------------

AccumulationMode = Literal["squared", "alpha"]

@dataclass
class RenderConfig:
    """Configuration for ray rendering."""
    ray_width: float = 0.5
    sigma_factor: float = 0.5
    accumulation_mode: AccumulationMode = "squared"
    default_intensity: float = 1.0
    # Alpha channel in color acts as per-ray weight in squared mode
    weight_scale: float = 1.0

# --------------------------------------------------------------------------------------
# Visual
# --------------------------------------------------------------------------------------

class RayTubeVisual(Visual):
    """Render 2D ray segments as Gaussian tubes with additive or alpha blending."""

    def __init__(self, config: Optional[RenderConfig] = None):
        super().__init__(vcode=RAY_TUBE_VERT, fcode=RAY_TUBE_FRAG)
        self.config: RenderConfig = config or RenderConfig()

        self._vbo = gloo.VertexBuffer()
        self._ibo = gloo.IndexBuffer()
        self._n_indices: int = 0

        self._draw_mode = 'triangles'
        # Initial GL state (will be refined each frame in _prepare_draw)
        self.set_gl_state(depth_test=False, blend=True)

        # Program uniforms with safe defaults
        self.shared_program['u_ray_width'] = float(self.config.ray_width)
        self.shared_program['u_sigma_factor'] = float(self.config.sigma_factor)
        self.shared_program['u_accumulation_mode'] = np.int32(
            0 if self.config.accumulation_mode == "squared" else 1
        )
        self.shared_program['u_weight_scale'] = float(self.config.weight_scale)

    # -------- Public API -----------------------------------------------------

    def set_config(self, config: RenderConfig) -> None:
        """Update render configuration (cheap)."""
        self.config = config
        # No geometry update needed; uniforms/state set in _prepare_draw
        self.update()

    def set_data(self, rays: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]) -> None:
        """Set ray data: list of (start[2], end[2], intensity, color[4])."""
        if not rays:
            self._n_indices = 0
            self._vbo.set_data(np.zeros(0, dtype=np.float32))
            self._ibo.set_data(np.zeros(0, dtype=np.uint16))
            self.update()
            return

        n_rays = len(rays)

        # Structured vertex buffer (4 verts per ray)
        dtype = np.dtype([
            ('a_corner',     np.float32, 2),
            ('a_ray_start',  np.float32, 2),
            ('a_ray_end',    np.float32, 2),
            ('a_intensity',  np.float32, 1),
            ('a_color',      np.float32, 4),
        ])

        verts = np.zeros(n_rays * 4, dtype=dtype)
        idx   = np.zeros(n_rays * 6, dtype=np.uint16)  # portable!

        corners = np.array([[-1, -1], [ 1, -1], [ 1,  1], [-1,  1]], dtype=np.float32)

        for i, (start, end, intensity, color) in enumerate(rays):
            base_v = 4 * i
            base_i = 6 * i

            # sanitize/shape
            start = np.asarray(start, dtype=np.float32).reshape(2)
            end   = np.asarray(end,   dtype=np.float32).reshape(2)
            color = np.asarray(color, dtype=np.float32).reshape(4)
            intensity = float(intensity)

            verts['a_corner'][base_v:base_v+4]    = corners
            verts['a_ray_start'][base_v:base_v+4] = start
            verts['a_ray_end'][base_v:base_v+4]   = end
            verts['a_intensity'][base_v:base_v+4] = intensity
            verts['a_color'][base_v:base_v+4]     = color

            # two triangles
            idx[base_i:base_i+6] = [base_v, base_v+1, base_v+2,
                                    base_v, base_v+2, base_v+3]

        self._vbo.set_data(verts)
        self._ibo.set_data(idx)
        self._n_indices = int(idx.size)
        self.update()

    # -------- VisPy hooks ----------------------------------------------------

    def _prepare_transforms(self, view):
        tr = view.transforms.get_transform('visual', 'render')
        self.shared_program.vert['transform'] = tr

    def _compute_bounds(self, axis, view):
        # No automatic bounds; keep None unless you want auto-zoom behavior
        return None

    def _prepare_draw(self, view):
        if self._n_indices == 0:
            return False

        # Bind VBO attributes
        self.shared_program.bind(self._vbo)

        # Update uniforms
        self.shared_program['u_ray_width'] = float(self.config.ray_width)
        self.shared_program['u_sigma_factor'] = float(self.config.sigma_factor)
        self.shared_program['u_weight_scale'] = float(self.config.weight_scale)
        mode_int = np.int32(0 if self.config.accumulation_mode == "squared" else 1)
        self.shared_program['u_accumulation_mode'] = mode_int

        # Set GL blending *per mode* (do not mix presets + custom funcs)
        if mode_int == 0:
            # Additive accumulation across color channels
            self.set_gl_state(depth_test=False, blend=True,
                              blend_func=('one', 'one'))
        else:
            # Standard alpha blend for linear mode (non-premultiplied)
            self.set_gl_state(depth_test=False, blend=True,
                              blend_func=('src_alpha', 'one_minus_src_alpha'))
        return True

    def _draw(self, view):
        self.shared_program.draw(self._draw_mode, self._ibo)

# VisualNode wrapper for scenegraph integration
RayTube = create_visual_node(RayTubeVisual)

# --------------------------------------------------------------------------------------
# Viewer
# --------------------------------------------------------------------------------------

class OpenGLViewer:
    """Unified OpenGL viewer for 2D and 3D ray tracing with intensity accumulation."""

    def __init__(
        self,
        mode: Literal["2d", "3d"] = "2d",
        x_lims: Tuple[float, float] = (-6.0, 6.0),
        y_lims: Tuple[float, float] = (-6.0, 6.0),
        z_lims: Optional[Tuple[float, float]] = None,
        show_axis: bool = True,
        show: bool = True,
        size: Tuple[int, int] = (900, 700),
        bgcolor: str | Tuple[float, float, float, float] = "black",
        render_config: Optional[RenderConfig] = None,
    ):
        self.mode = mode
        self.render_config = render_config or RenderConfig()

        self.canvas = scene.SceneCanvas(keys="interactive", show=show, bgcolor=bgcolor, size=size)
        self.view = self.canvas.central_widget.add_view()

        if show_axis:
            scene.visuals.XYZAxis(parent=self.view.scene)

        if mode == "2d":
            self.camera = scene.cameras.PanZoomCamera()
            self.camera.set_range(x=x_lims, y=y_lims)
            self.view.camera = self.camera
        else:
            self.turntable_cam = scene.cameras.TurntableCamera(fov=45.0, azimuth=35.0, elevation=25.0)
            self.fly_cam = scene.cameras.FlyCamera(fov=45.0)
            self.view.camera = self.turntable_cam

            @self.canvas.events.key_press.connect
            def on_key(event):
                if event.key == 'c':
                    if self.view.camera is self.turntable_cam:
                        self.view.camera = self.fly_cam
                        print("Switched to Fly Camera")
                    else:
                        self.view.camera = self.turntable_cam
                        print("Switched to Turntable Camera")

        # Visual storage
        self._surface_visuals: List[scene.VisualNode] = []
        self._ray_visual: Optional[RayTube] = None
        self._marker_visuals: List[scene.VisualNode] = []

        self._cached_ray_data: Optional[list] = None

    def _clear_visuals(self, attr: str) -> None:
        visuals = getattr(self, attr, [])
        for v in visuals:
            v.parent = None
        setattr(self, attr, [])

    def draw_surfaces(self, surfaces: list, color: str = "white", width: float = 2.0) -> None:
        self._clear_visuals("_surface_visuals")

        for surface in surfaces:
            if hasattr(surface, "polyline_segments"):
                segments = surface.polyline_segments()
            else:
                segments = [surface.polyline()]

            for seg in segments:
                seg = np.asarray(seg, dtype=np.float32)
                if seg.ndim != 2 or seg.shape[0] < 2:
                    continue
                if self.mode == "2d" and seg.shape[1] == 2:
                    seg = np.column_stack([seg, np.zeros(len(seg), dtype=np.float32)])
                line = scene.visuals.Line(pos=seg, color=color, width=width, parent=self.view.scene)
                self._surface_visuals.append(line)

    def draw_rays(
        self,
        tree,
        tail_length: float = 12.0,
        width: Optional[float] = None,
        show_misses: bool = True,
        color_resolver: Optional[Callable] = None,
        hit_color: Color | str | Sequence[float] = "yellow",
        miss_color: Color | str | Sequence[float] = "orange",
        intensity_resolver: Optional[Callable] = None,
        update_markers: bool = True,
    ) -> None:
        if width is not None:
            self.render_config.ray_width = float(width)

        ray_data = []
        hit_positions = []

        default_miss = np.array(Color(miss_color).rgba, dtype=np.float32)
        default_hit = np.array(Color(hit_color).rgba, dtype=np.float32)

        # Deterministic order is helpful for debugging
        for node in sorted(tree.nodes(), key=lambda n: n.label):
            start = np.asarray(node.ray.origin, dtype=np.float32)
            direction = np.asarray(node.ray.direction, dtype=np.float32)

            # Intensity per ray (can depend on generation)
            intensity = float(intensity_resolver(node)) if intensity_resolver else float(self.render_config.default_intensity)

            # Color (RGBA); alpha will act as a *weight* under squared accumulation
            if color_resolver:
                color = np.array(color_resolver(node), dtype=np.float32)
                if color.size == 3:
                    color = np.append(color, 1.0)
            elif node.intersection is None:
                if not show_misses:
                    continue
                color = default_miss.copy()
            else:
                color = default_hit.copy()

            # End point
            if node.intersection is None:
                end = start + direction * float(tail_length)
            else:
                end = np.asarray(node.intersection.point, dtype=np.float32)
                if update_markers:
                    hit_positions.append(end)

            # 2D reduction if needed
            if self.mode == "2d":
                start = start[:2]
                end = end[:2]

            ray_data.append((start, end, intensity, color))

        # Create/reuse visual node
        if self._ray_visual is None:
            self._ray_visual = RayTube(config=self.render_config, parent=self.view.scene)
        else:
            self._ray_visual.set_config(self.render_config)

        self._ray_visual.set_data(ray_data)
        self._cached_ray_data = ray_data

        # Markers
        if update_markers:
            self._clear_visuals("_marker_visuals")
            if hit_positions:
                if self.mode == "2d":
                    hit_positions = [p[:2] if len(p) > 2 else p for p in hit_positions]
                markers = scene.visuals.Markers(
                    pos=np.asarray(hit_positions, dtype=np.float32),
                    size=5,
                    face_color=Color("crimson").rgba,
                    parent=self.view.scene,
                )
                markers.set_gl_state(depth_test=False, blend=True)
                markers.order = 1000
                self._marker_visuals.append(markers)

    def update_visual_params_only(self) -> None:
        """Fast path: change width/sigma/weights without recomputing geometry."""
        if self._ray_visual is not None and self._cached_ray_data is not None:
            self._ray_visual.set_config(self.render_config)
            self._ray_visual.update()
            self.canvas.update()

    def run(self) -> None:
        app.run()

    def close(self) -> None:
        self.canvas.close()
