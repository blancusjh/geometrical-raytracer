"""3-D visualization of optical systems and traces (vispy.scene).

Everything is drawn where the system places it: each element's faces are
meshed in the element's local frame and carried to the world by its pose
(``system.walk()`` / ``system.pose_of(name)``), so tilted, decentred and
folded systems render as built. World axes are the package's: the optical
axis is z, the meridional plane y-z, and the turntable camera keeps +y up.

- Lenses are closed solids (front face, back face, cylindrical rim) lathed
  from their surfaces' sag ``s(x, y)`` — any surface, revolution-symmetric
  or not — out to the semi-diameter, each face clamped to its
  single-valued branch and the faces joined where they cross (the same
  extents the 2-D outlines use, :func:`raytracer.viz.bodies.element_outline`;
  the inter-lens cuts of the 2-D drawing are not applied in 3-D). With
  ``section="half"`` each element is a **cutaway** (half solid with flat
  cut faces on the y = 0 plane) so the interior and the traced rays show
  without translucent overlap.
- Mirrors, beam splitters, diaphragm blades and sensors are their faces
  over their apertures; prisms are their facets.
- Rays come from sequential pupil bundles (:meth:`Viewer3D.add_field_bundles`),
  polylines (:meth:`Viewer3D.add_paths`) or a non-sequential
  :class:`~raytracer.tracing.RayTree` (:meth:`Viewer3D.add_tree`).

Lighting is a **headlight** (the light direction follows the camera, so no
side stays fixed in shadow while orbiting). Press ``w`` to toggle a
wireframe overlay. Glass is coloured by material name, else by index on the
celestial ramp of :func:`raytracer.viz.bodies.index_color`.

The mesh generators (:func:`lathe`, :func:`lathe_solid`) are pure numpy and
testable headless.
"""

from __future__ import annotations

import numpy as np
from vispy.color import Color

from ..elements import BeamSplitter, Lens, Mirror, Prism, Sensor, Stop
from ..geometry.apertures import Annulus, Aperture, Circle, Rectangle
from ..geometry.surfaces import Plane
from ..geometry.transforms import Pose
from .bodies import element_outline, index_color, medium_index
from .plots import DEFAULT_MATERIAL_COLORS

TWO_PI = 2.0 * np.pi
MIRROR_COLOR = (0.62, 0.66, 0.72, 1.0)
STOP_COLOR = (0.18, 0.18, 0.20, 1.0)
SENSOR_COLOR = (0.85, 0.72, 0.30, 1.0)
# Glass uses standard alpha translucency: stacked surfaces get DENSER (as real
# glass does), never brighter. Safe now that the depth-mask clear bug is fixed.
# Light (the beam modes) is what adds; glass is not.
GLASS_ALPHA = 0.30
RAY_COLORS = [
    (0.95, 0.35, 0.25, 0.55),
    (0.30, 0.95, 0.45, 0.55),
    (0.35, 0.65, 1.00, 0.55),
    (0.80, 0.55, 0.95, 0.55),
]

# Draw order: opaque mirrors first, then rays (depth-tested), then translucent
# glass blended over them — a ray behind glass is tinted by it, a ray in front
# stays crisp, and anything behind a mirror is properly occluded.
ORDER_OPAQUE, ORDER_RAYS, ORDER_GLASS = 0, 1, 2


# --- HDR pass: the whole scene renders into a float32 FBO (beams accumulate
# linearly without 8-bit quantization or clipping), then a soft-knee curve maps
# to the display: identity below KNEE (lenses and beam body keep their exact
# linear look -> per-pixel brightness stays proportional to ray count), smooth
# compression above (foci glow without flattening).
HDR_TONEMAP_VERT = """
attribute vec2 a_position;
varying vec2 v_uv;
void main() {
    v_uv = 0.5 * (a_position + 1.0);
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

HDR_TONEMAP_FRAG = """
uniform sampler2D u_tex;
varying vec2 v_uv;
const float K = 0.9;
void main() {
    vec3 c = texture2D(u_tex, v_uv).rgb;
    vec3 over = max(c - K, 0.0);
    vec3 soft = K + (1.0 - K) * (1.0 - exp(-over / (1.0 - K)));
    vec3 mapped = mix(c, soft, step(K, c));   // c < K ? c : soft-knee
    gl_FragColor = vec4(mapped, 1.0);
}
"""


def spectral_rgb(wavelength_nm: float) -> np.ndarray:
    """Linear-sRGB of a wavelength via the CIE 1931 fits (diffractsim-style).

    Wavelengths outside the visible gamut (e.g. 193 nm DUV) map to a deep
    violet so the beam stays representable on screen.
    """
    from .color import wavelength_to_rgb

    # Check the *unnormalized* response: outside the visible range the CIE fits
    # return numerical residue that normalize=True would blow up to a wrong hue.
    raw = np.asarray(wavelength_to_rgb(float(wavelength_nm), normalize=False), dtype=float)
    if float(raw.max()) < 1e-3:  # UV / IR: no visible response
        return np.array([0.45, 0.18, 1.0])
    return np.asarray(wavelength_to_rgb(float(wavelength_nm)), dtype=float)


def _angles(angle_range, n):
    a0, a1 = angle_range
    closed = (a1 - a0) >= TWO_PI - 1e-6
    return np.linspace(a0, a1, n, endpoint=not closed), closed


def _ring_grid(surface, radii, angles, z_offset=0.0):
    """Local points of *surface* on a polar grid: rings of *radii* (outer index) by *angles*."""

    rr, aa = np.meshgrid(radii, angles, indexing="ij")
    x, y = rr * np.cos(aa), rr * np.sin(aa)
    z = z_offset + surface.sag(x, y)
    return np.column_stack([x.ravel(), y.ravel(), np.asarray(z, dtype=float).ravel()])


def _surface_faces(offset, R, A, closed):
    faces = []
    jmax = A if closed else A - 1
    for i in range(R - 1):
        for j in range(jmax):
            jn = (j + 1) % A if closed else j + 1
            a = offset + i * A + j
            b = offset + i * A + jn
            c = offset + (i + 1) * A + j
            d = offset + (i + 1) * A + jn
            faces += [[a, b, c], [b, d, c]]
    return faces


def _to_world(vertices: np.ndarray, pose: Pose | None) -> np.ndarray:
    return vertices if pose is None else pose.to_world(vertices)


def lathe(surface, semidiameter, *, pose: Pose | None = None, radial_samples=24,
          angular_samples=96, inner_radius=0.0, angle_range=(0.0, TWO_PI),
          aperture: Aperture | None = None):
    """Mesh one face over a polar grid out to *semidiameter* (an open sheet).

    The grid lives in the face's local frame (``z = s(x, y)``, any surface)
    and is carried to the world by *pose*. ``inner_radius`` leaves a central
    hole (an annular mirror, a diaphragm blade); an *aperture* other than a
    centred circle drops the triangles reaching outside it.
    """

    radii = np.linspace(inner_radius, semidiameter, radial_samples)
    angles, closed = _angles(angle_range, angular_samples)
    vertices = _ring_grid(surface, radii, angles)
    faces = np.asarray(_surface_faces(0, radial_samples, len(angles), closed), dtype=np.int32)
    if aperture is not None and not (isinstance(aperture, Circle) and aperture.center == (0.0, 0.0)):
        inside = aperture.contains(vertices[:, 0], vertices[:, 1], slack=1e-9)
        faces = faces[inside[faces].all(axis=1)]
    return _to_world(vertices, pose).astype(np.float32), faces.astype(np.int32)


def lathe_solid(front, back, semidiameter, *, thickness, pose: Pose | None = None,
                back_semidiameter=None, radial_samples=22, angular_samples=96,
                angle_range=(0.0, TWO_PI)):
    """Closed lens body: front face, back face at ``z = thickness``, and the rim.

    Each face runs to its own extent (``semidiameter`` for the front,
    ``back_semidiameter`` — default the same — for the back); the rim joins
    their outer rings. If ``angle_range`` < 2π, flat cut faces close the two
    end meridians → a watertight cutaway half-solid. Built in the lens frame
    and carried to the world by *pose*.
    """

    R = radial_samples
    angles, closed = _angles(angle_range, angular_samples)
    A = len(angles)
    back_semi = semidiameter if back_semidiameter is None else back_semidiameter

    front_v = _ring_grid(front, np.linspace(0.0, semidiameter, R), angles)
    back_v = _ring_grid(back, np.linspace(0.0, back_semi, R), angles, z_offset=thickness)
    vertices = np.vstack([front_v, back_v])
    back_off = R * A

    faces = _surface_faces(0, R, A, closed) + _surface_faces(back_off, R, A, closed)
    # rim connecting the outer rings
    fo, bo = (R - 1) * A, back_off + (R - 1) * A
    jmax = A if closed else A - 1
    for j in range(jmax):
        jn = (j + 1) % A if closed else j + 1
        faces += [[fo + j, fo + jn, bo + j], [fo + jn, bo + jn, bo + j]]
    # cut caps at the two end meridians (front↔back strip) when not closed
    if not closed:
        for c in (0, A - 1):
            for i in range(R - 1):
                f0, f1 = i * A + c, (i + 1) * A + c
                b0, b1 = back_off + i * A + c, back_off + (i + 1) * A + c
                faces += [[f0, f1, b0], [f1, b1, b0]]
    return _to_world(vertices, pose).astype(np.float32), np.asarray(faces, dtype=np.int32)


def _rectangle_sheet(surface, rect: Rectangle, pose: Pose | None, samples: int = 12):
    """A face over a rectangular aperture, as a grid of quads."""

    cx, cy = rect.center
    x = np.linspace(cx - rect.width / 2, cx + rect.width / 2, samples)
    y = np.linspace(cy - rect.height / 2, cy + rect.height / 2, samples)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    vertices = np.column_stack([xx.ravel(), yy.ravel(), np.asarray(surface.sag(xx, yy)).ravel()])
    faces = np.asarray(_surface_faces(0, samples, samples, False), dtype=np.int32)
    return _to_world(vertices, pose).astype(np.float32), faces


def _material_color(material, alpha=1.0, wavelength_um=None):
    """Glass tone: a name-keyed palette entry (SIO2 blue / CAF2 green / ...), else by index."""

    name = (getattr(material, "name", material) or "")
    entry = DEFAULT_MATERIAL_COLORS.get(str(name).upper())
    if entry is None and not isinstance(material, str):
        entry = index_color(medium_index(material, wavelength_um))
    entry = entry or ("#c9c9c9", "#8a8a8a")
    r, g, b = Color(entry[0]).rgb
    return (float(r), float(g), float(b), alpha)


class _DynamicAxes:
    """Meridional scale axes with zoom-adaptive tick labels.

    Two world-anchored rulers in the x=0 plane — z (the optical axis, at
    y=0) and y (vertical, through the current view center) — whose tick
    spacing snaps to a 1/2/5 x 10^n "nice" step chosen from the visible
    extent, and whose numeric labels regenerate on every camera zoom/pan.
    """

    COLOR_LINE = (0.55, 0.62, 0.58, 0.6)
    COLOR_TEXT = (0.72, 0.80, 0.75, 0.95)

    def __init__(self, viewer: "Viewer3D") -> None:
        self.viewer = viewer
        self.visible = True
        self._state: tuple | None = None
        self._lines = viewer._scene.visuals.Line(
            pos=np.zeros((2, 3), dtype=np.float32), connect="segments",
            color=self.COLOR_LINE, width=1.0, parent=viewer.view.scene, method="gl",
        )
        self._lines.set_gl_state(blend=True, depth_test=False)
        self._lines.order = ORDER_GLASS + 1  # overlay: on top of glass and rays
        self._text = viewer._scene.visuals.Text(
            text="", pos=(0.0, 0.0, 0.0), color=self.COLOR_TEXT, font_size=7,
            parent=viewer.view.scene,
        )
        self._text.order = ORDER_GLASS + 1
        # Rebuild is driven from the viewer's wrapped _draw_scene funnel,
        # which covers both live paints and offscreen render()/snapshot().

    def set_visible(self, on: bool) -> None:
        self.visible = on
        self._lines.visible = on
        self._text.visible = on

    @staticmethod
    def _nice_step(extent: float) -> float:
        """A 1/2/5 x 10^n step giving roughly 5-8 ticks across *extent*."""

        raw = max(extent, 1e-12) / 6.0
        magnitude = 10.0 ** np.floor(np.log10(raw))
        for mult in (1.0, 2.0, 5.0, 10.0):
            if raw <= mult * magnitude:
                return mult * magnitude
        return 10.0 * magnitude

    @staticmethod
    def _fmt(value: float, step: float) -> str:
        decimals = max(0, -int(np.floor(np.log10(step))))
        return f"{value:.{decimals}f}"

    def _maybe_rebuild(self, event=None) -> None:
        if not self.visible:
            return
        # Text re-layout every frame is too costly for smooth orbit/zoom:
        # freeze the axes during interaction; the idle-refinement pass
        # rebuilds them (crisp, current) as soon as the mouse rests.
        if self.viewer._interacting and self._state is not None:
            return
        cam = self.viewer.view.camera
        state = (float(cam.scale_factor), tuple(np.round(cam.center, 9)))
        if state == self._state:
            return
        self._state = state
        self._rebuild(cam)

    def _rebuild(self, cam) -> None:
        cx, cy, cz = (float(v) for v in cam.center)
        extent = float(cam.scale_factor)
        step = self._nice_step(extent)
        half = 0.72 * extent
        tick = 0.012 * extent

        segments: list[list[list[float]]] = []
        labels: list[str] = []
        positions: list[tuple[float, float, float]] = []

        # Both rulers follow the camera center (they cross at the middle of
        # the view), so they stay on screen wherever the user pans/zooms —
        # the tick LABELS are absolute world coordinates regardless.
        # z ruler (parallel to the optical axis, at height cy)
        segments.append([[cx, cy, cz - half], [cx, cy, cz + half]])
        z_ticks = np.arange(np.ceil((cz - half) / step) * step, cz + half, step)
        for z in z_ticks:
            segments.append([[cx, cy - tick, z], [cx, cy + tick, z]])
            labels.append(self._fmt(z, step))
            positions.append((cx, cy - 3.0 * tick, z))

        # y ruler (vertical, at axial position cz)
        segments.append([[cx, cy - half, cz], [cx, cy + half, cz]])
        y_ticks = np.arange(np.ceil((cy - half) / step) * step, cy + half, step)
        for y in y_ticks:
            if abs(y - cy) < 0.5 * step:  # would sit on the crossing point
                continue
            segments.append([[cx, y, cz - tick], [cx, y, cz + tick]])
            labels.append(self._fmt(y, step))
            positions.append((cx, y, cz + 3.0 * tick))

        pos = np.asarray(segments, dtype=np.float32).reshape(-1, 3)
        self._lines.set_data(pos=pos, connect="segments", color=self.COLOR_LINE)
        if labels:
            self._text.text = labels
            self._text.pos = np.asarray(positions, dtype=np.float32)
        else:
            self._text.text = ""


class Viewer3D:
    """Orthographic turntable viewer with cutaway lenses and a headlight."""

    def __init__(self, *, size=(1300, 820), background="#0a0a0f", title="raytracer 3D",
                 ssaa: int = 2):
        from vispy import scene

        # NOTE: no MSAA (config samples): multisample framebuffers leave stale /
        # black frames on some drivers (frozen first frame over the scene).
        # Antialiasing comes from supersampling instead: the scene renders into
        # the HDR buffer at ``ssaa``x resolution and the tonemap pass
        # downsamples bilinearly (an exact box filter at 2:1, applied in
        # linear light before the tonemap) — driver-safe and it smooths
        # every primitive, including plain GL lines that ignore LINE_SMOOTH
        # on core profiles.
        #
        # Progressive refinement: supersampling multiplies fragment work by
        # ssaa^2, which dense additive beams cannot afford per interaction
        # frame — so orbit/pan/zoom draws at 1x, and a short idle timer
        # triggers one final crisp pass at full ssaa when the mouse rests.
        self._ssaa = max(1, int(ssaa))
        self._interacting = False
        self._idle_timer = None
        self.canvas = scene.SceneCanvas(
            keys="interactive", size=size, bgcolor=background, title=title, show=False
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(fov=0.0, up="+y")  # 0 -> orthographic
        self._scene = scene
        self._bounds: list[np.ndarray] = []
        self._shading_filters: list = []
        self._wireframe_filters: list = []
        self._wireframe_on = False
        self._ray_groups: dict[str, list] = {"lines": [], "beam": [], "spectrum": []}
        self._ray_mode = "lines"
        # Wrap _draw_scene (the funnel for both live paints and offscreen
        # render()) to:
        # 1. restore glDepthMask before the clear — glClear honours it, and the
        #    glass (depth_mask=False) drawn last would otherwise freeze the
        #    first frame's depth as a black occluding silhouette;
        # 2. render the scene into a float32 HDR buffer and soft-knee map it to
        #    the display, so additive beams accumulate linearly (no 8-bit
        #    quantum, no hard clip).
        self._hdr_tex = None
        self._hdr_fbo = None
        self._hdr_prog = None
        _orig_draw_scene = self.canvas._draw_scene

        def _draw_scene_hdr(*args, **kwargs):
            from vispy import gloo

            axes = getattr(self, "_axes", None)
            if axes is not None:
                # Runs for live paints AND offscreen render(); rebuilds only
                # when the camera zoom/pan actually changed.
                axes._maybe_rebuild()
            self._reset_gl_state()
            # Mouse interaction depends on vispy's PICKING passes (visual_at
            # renders id-colors and reads a pixel). Those must NOT go through
            # the HDR buffer + tonemap quad, or the read-back is garbage and
            # the camera never engages. Bypass HDR while picking.
            scene = getattr(self.canvas, "scene", None)
            if getattr(scene, "picking", False):
                return _orig_draw_scene(*args, **kwargs)
            w, h = self.canvas.physical_size
            ss = 1 if self._interacting else self._ssaa
            self._ensure_hdr(int(w * ss), int(h * ss))
            # push_fbo (not a bare FBO activate) so the whole frame renders
            # into the larger buffer. vispy's push_fbo plumbing conflates the
            # FBO's pixel size with its logical size, which would leave the
            # scene in the lower-left 1/ss^2 of the buffer (its own
            # render(size=...) has the same flaw) — so reconfigure the canvas
            # TransformSystem with the true geometry: per-visual systems share
            # these transform objects by reference, and pixel-sized visuals
            # (text, markers, line widths) then scale into the supersampled
            # buffer and come back crisp after the downsample.
            self.canvas.push_fbo(self._hdr_fbo, (0, 0), self.canvas.size)
            self.canvas.transforms.configure(
                viewport=(0, 0, int(w * ss), int(h * ss)),
                fbo_size=(int(w * ss), int(h * ss)),
                fbo_rect=(0, 0, int(w), int(h)),
            )
            try:
                _orig_draw_scene(*args, **kwargs)
            finally:
                self.canvas.pop_fbo()  # _update_transforms restores the outer state
            gloo.set_viewport(0, 0, int(w), int(h))
            gloo.set_state(blend=False, depth_test=False, cull_face=False)
            self._hdr_prog["u_tex"] = self._hdr_tex
            self._hdr_prog.draw("triangles")

        self.canvas._draw_scene = _draw_scene_hdr
        # headlight: update the light direction whenever the camera moves
        self.canvas.events.mouse_move.connect(self._update_light)
        self.canvas.events.mouse_wheel.connect(self._update_light)
        self.canvas.events.mouse_move.connect(self._on_interaction)
        self.canvas.events.mouse_wheel.connect(self._on_interaction)
        self.canvas.events.key_press.connect(self._on_key)

    @staticmethod
    def _reset_gl_state(event=None):
        from vispy import gloo

        gloo.set_state(depth_mask=True)

    # -- progressive refinement -------------------------------------------
    def _on_interaction(self, event=None) -> None:
        if self._ssaa <= 1:
            return
        dragging = bool(getattr(event, "is_dragging", False))
        wheeling = getattr(event, "type", "") == "mouse_wheel"
        if not (dragging or wheeling):
            return
        self._interacting = True
        if self._idle_timer is None:
            from vispy.app import Timer

            self._idle_timer = Timer(interval=0.28, connect=self._end_interaction)
        self._idle_timer.stop()
        self._idle_timer.start(iterations=1)

    def _end_interaction(self, event=None) -> None:
        self._interacting = False
        self.canvas.update()  # one final crisp pass at full ssaa

    def _ensure_hdr(self, w: int, h: int) -> None:
        from vispy import gloo

        if self._hdr_prog is None:
            self._hdr_prog = gloo.Program(HDR_TONEMAP_VERT, HDR_TONEMAP_FRAG)
            self._hdr_prog["a_position"] = np.array(
                [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]], dtype=np.float32
            )  # fullscreen triangle
        if not hasattr(self, "_hdr_targets"):
            self._hdr_targets: dict = {}
        target = self._hdr_targets.get((h, w))
        if target is None:
            # Progressive refinement alternates between the 1x and ssaa-x
            # buffers on every interaction start/stop — cache both instead of
            # reallocating tens of MB of float32 texture each transition.
            if len(self._hdr_targets) >= 2:  # window resized: sizes went stale
                self._hdr_targets.clear()
            # linear: the tonemap quad also downsamples the ssaa-times-larger
            # buffer; at exactly 2:1 the bilinear tap is a perfect box filter.
            tex = gloo.Texture2D(
                shape=(h, w, 4), internalformat="rgba32f", interpolation="linear"
            )
            fbo = gloo.FrameBuffer(color=tex, depth=gloo.RenderBuffer((h, w)))
            target = self._hdr_targets[(h, w)] = (tex, fbo)
        self._hdr_tex, self._hdr_fbo = target

    # -- meshes ----------------------------------------------------------
    def _add_mesh(self, vertices, faces, color, *, style="opaque"):
        from vispy.visuals.filters import WireframeFilter

        if style == "glass":
            # Shaded translucent glass: alpha compositing, so overlapping
            # surfaces read denser (like real glass), never brighter. Smooth
            # shading + specular glint give each lens a solid body.
            tint = (color[0], color[1], color[2], GLASS_ALPHA)
            mesh = self._scene.visuals.Mesh(
                vertices=vertices, faces=faces, color=tint, shading="smooth",
                parent=self.view.scene,
            )
            mesh.set_gl_state(blend=True, depth_test=True, depth_mask=False,
                              cull_face=False,
                              blend_func=("src_alpha", "one_minus_src_alpha"))
            mesh.order = ORDER_GLASS
            sf = getattr(mesh, "shading_filter", None)
            if sf is not None:
                try:
                    sf.ambient_light = (1.0, 1.0, 1.0, 1.0)
                    sf.diffuse_light = (1.0, 1.0, 1.0, 0.9)
                    sf.ambient_coefficient = (1.0, 1.0, 1.0, 0.55)
                    sf.diffuse_coefficient = (1.0, 1.0, 1.0, 0.60)
                    sf.specular_coefficient = (1.0, 1.0, 1.0, 0.50)
                    sf.shininess = 80.0
                except Exception:
                    pass
                self._shading_filters.append(sf)  # headlight follows the camera
        else:  # opaque (mirrors)
            mesh = self._scene.visuals.Mesh(
                vertices=vertices, faces=faces, color=color, shading="smooth",
                parent=self.view.scene,
            )
            mesh.set_gl_state(blend=True, depth_test=True, cull_face=False,
                              depth_mask=True)
            mesh.order = ORDER_OPAQUE
            sf = getattr(mesh, "shading_filter", None)
            if sf is not None:
                # bright ambient floor so no mirror face ever goes black
                try:
                    sf.ambient_light = (1.0, 1.0, 1.0, 1.0)
                    sf.diffuse_light = (1.0, 1.0, 1.0, 0.85)
                    sf.ambient_coefficient = (1.0, 1.0, 1.0, 0.75)
                    sf.diffuse_coefficient = (1.0, 1.0, 1.0, 0.50)
                    sf.specular_coefficient = (1.0, 1.0, 1.0, 0.15)
                    sf.shininess = 24.0
                except Exception:
                    pass
                self._shading_filters.append(sf)
        wf = WireframeFilter(enabled=False, color=(0.9, 0.95, 1.0, 0.55), width=0.8)
        mesh.attach(wf)
        self._wireframe_filters.append(wf)
        self._bounds.append(vertices)
        return mesh

    def add_lens(self, lens: Lens, pose: Pose | None = None, *, color=None,
                 section="full", style="glass", wavelength_um=None):
        """A lens body at its world *pose* (default: its own pose)."""

        pose = lens.pose if pose is None else pose
        _, (e_front, e_back), _ = element_outline(lens, Pose())  # radial extents, Ω-joined
        rng = (0.0, np.pi) if section == "half" else (0.0, TWO_PI)
        v, f = lathe_solid(lens.front.surface, lens.back.surface, e_front, thickness=lens.thickness,
                           back_semidiameter=e_back, pose=pose, angle_range=rng)
        color = color if color is not None else _material_color(lens.material, wavelength_um=wavelength_um)
        self._add_mesh(v, f, color, style=style)

    def add_surface(self, surface, aperture, *, pose: Pose | None = None, color,
                    section="full", angular_samples=96, style="opaque", hole=None):
        """One face over its *aperture* (a radius or an aperture), placed by *pose*."""

        aperture = Aperture.coerce(aperture)
        if isinstance(aperture, Rectangle) and hole is None:
            v, f = _rectangle_sheet(surface, aperture, pose)
            self._add_mesh(v, f, color, style=style)
            return
        inner = 0.0
        if isinstance(aperture, Annulus):
            inner = aperture.inner
        if hole is not None:
            inner = max(inner, Aperture.coerce(hole).extent)
        rng = (0.0, np.pi) if section == "half" else (0.0, TWO_PI)
        v, f = lathe(surface, aperture.extent, pose=pose, inner_radius=inner,
                     angular_samples=angular_samples, angle_range=rng, aperture=aperture)
        self._add_mesh(v, f, color, style=style)

    def add_system(self, system, *, section="full", angular_samples: int = 64,
                   wavelength_um: float | None = None) -> None:
        """Every element of *system* at its world pose: lenses coloured by
        material, mirrors opaque, beam splitters glassy, blades dark, sensors
        (those with an aperture) gold. Closed solids by default;
        ``section='half'`` gives a cutaway."""

        for element, pose in system.walk():
            if isinstance(element, Lens):
                self.add_lens(element, pose, section=section, wavelength_um=wavelength_um)
            elif isinstance(element, Mirror):
                self.add_surface(element.surface, element.aperture, pose=pose, color=MIRROR_COLOR,
                                 section=section, angular_samples=angular_samples)
            elif isinstance(element, BeamSplitter):
                self.add_surface(element.interface.surface, element.aperture, pose=pose,
                                 color=_material_color("SIO2"), section=section, style="glass")
            elif isinstance(element, Stop) and not element.virtual:
                self.add_surface(Plane(), element.outer_radius, pose=pose, color=STOP_COLOR,
                                 section=section, angular_samples=angular_samples, hole=element.aperture)
            elif isinstance(element, Sensor) and element.aperture is not None:
                # an unbounded sensor (a prescription's image plane) has no
                # physical size to draw; give it an aperture to see it
                self.add_surface(element.surface, element.aperture, pose=pose, color=SENSOR_COLOR,
                                 section=section, angular_samples=angular_samples)
            elif isinstance(element, Prism):
                for facet_pose, outline in element.solid.facets():
                    corners = np.asarray(outline.vertices, dtype=float)
                    local = np.column_stack([corners, np.zeros(len(corners))])
                    v = (pose @ facet_pose).to_world(local).astype(np.float32)
                    k = np.arange(1, len(corners) - 1)
                    f = np.column_stack([np.zeros_like(k), k, k + 1]).astype(np.int32)
                    self._add_mesh(v, f, _material_color(element.material, wavelength_um=wavelength_um), style="glass")

    # -- rays ------------------------------------------------------------
    def add_paths(self, paths, *, color=None, width: float = 1.4,
                  group: str = "lines", additive: bool = False) -> None:
        """Add traced polylines. ``group`` is a toggleable layer ("lines"/"beam").

        With ``additive`` the segments blend ONE,ONE: overlapping rays sum their
        (energy-scaled) colors into a continuous beam, diffractsim-style.
        """
        paths = np.asarray(paths, dtype=float)
        if paths.ndim == 2:
            paths = paths[None, ...]
        color = color if color is not None else RAY_COLORS[0]
        positions, connections, offset = [], [], 0
        for path in paths:
            valid = path[~np.isnan(path).any(axis=1)]
            if valid.shape[0] < 2:
                continue
            positions.append(valid)
            n = valid.shape[0]
            connections.append(np.column_stack([np.arange(n - 1), np.arange(1, n)]) + offset)
            offset += n
        if not positions:
            return
        positions = np.vstack(positions).astype(np.float32)
        line = self._scene.visuals.Line(
            pos=positions, connect=np.vstack(connections).astype(np.uint32),
            color=color, width=width, parent=self.view.scene, method="gl",
            antialias=True,  # fragment AA (no MSAA needed)
        )
        if additive:
            # LIGHT adds linearly into the float32 HDR buffer (no quantization,
            # no clipping); the soft-knee display map keeps the beam body
            # exactly linear — per-pixel brightness stays proportional to the
            # number of rays crossing it — and only compresses the foci.
            line.set_gl_state(blend=True, depth_test=True, depth_mask=False,
                              blend_func=("one", "one"))
        else:
            # plain traces: alpha blending, depth-tested (occluded by mirrors,
            # seen through the glass) — no additive color summing
            line.set_gl_state(blend=True, depth_test=True)
        line.order = ORDER_RAYS
        line.visible = group == self._ray_mode
        self._ray_groups.setdefault(group, []).append(line)
        self._bounds.append(positions)

    def add_tree(self, tree, *, color=None, width: float = 1.0, group: str = "lines",
                 additive: bool = False, select: np.ndarray | None = None,
                 escape_length: float | None = None) -> None:
        """Add the segments of a non-sequential :class:`~raytracer.tracing.RayTree`.

        ``color`` is one RGBA for every segment, an ``(M, 4)`` array, or
        ``None`` for the first ray colour with alpha scaled by power. Rays
        that left the system are cut to ``escape_length`` mm when given;
        ``select`` is an optional boolean mask of segments to draw.
        """

        n = len(tree)
        if n == 0:
            return
        start = np.asarray(tree.start, dtype=float)
        end = np.asarray(tree.end, dtype=float)
        if escape_length is not None:
            escaped = np.asarray(tree.escaped, dtype=bool)
            end = np.where(escaped[:, None], start + float(escape_length) * tree.direction, end)
        keep = np.ones(n, dtype=bool) if select is None else np.asarray(select, dtype=bool)
        if color is None:
            power = np.asarray(tree.power, dtype=float)
            peak = power.max() if power.size and power.max() > 0 else 1.0
            rgba = np.tile(np.asarray(RAY_COLORS[0], dtype=float), (n, 1))
            rgba[:, 3] = np.clip(RAY_COLORS[0][3] * power / peak, 0.03, 1.0)
        else:
            rgba = np.asarray(color, dtype=float)
            rgba = np.tile(rgba, (n, 1)) if rgba.ndim == 1 else rgba
        positions = np.stack([start[keep], end[keep]], axis=1).reshape(-1, 3).astype(np.float32)
        colors = np.repeat(rgba[keep], 2, axis=0).astype(np.float32)
        if positions.shape[0] == 0:
            return
        line = self._scene.visuals.Line(pos=positions, connect="segments", color=colors, width=width,
                                        parent=self.view.scene, method="gl", antialias=True)
        if additive:
            line.set_gl_state(blend=True, depth_test=True, depth_mask=False, blend_func=("one", "one"))
        else:
            line.set_gl_state(blend=True, depth_test=True)
        line.order = ORDER_RAYS
        line.visible = group == self._ray_mode
        self._ray_groups.setdefault(group, []).append(line)
        self._bounds.append(positions)

    def add_field_bundles(self, tracer, fields, *, na_object_sine, radial=6, azimuth=32,
                          stop_index=None, mode: str = "lines",
                          wavelength_nm: float | None = None,
                          wavelengths_nm=None,
                          beam_energy: float = 12.0,
                          beam: bool = False) -> None:
        """Trace pupil bundles per field, in one of three display modes.

        mode="lines"    -> per-field colored traces (alpha, no color summing).
        mode="beam"     -> dense additive beam at the system wavelength: rays
            carry ``spectral_rgb(wavelength_nm) * beam_energy / n_rays`` and sum
            into a continuous glow (193 nm DUV shows as violet).
        mode="spectrum" -> one wavelength per field (``wavelengths_nm``, default
            633/532/473 nm — a display assignment: the patent gives indices at a
            single wavelength), each an additive spectral beam. Where beams of
            different wavelengths overlap, colors sum colorimetrically
            (CIE 1931 -> linear sRGB, as in diffractsim) -> white at the image.
        """
        from ..tracing import FieldPoint, PupilSampling, trace_pupil

        if beam:  # backward-compat flag
            mode = "beam"
        if mode == "beam" and wavelength_nm is None:
            wavelength_nm = float(tracer.wavelength_um if tracer.wavelength_um is not None else 0.55) * 1e3
        if mode == "spectrum" and wavelengths_nm is None:
            wavelengths_nm = (633.0, 532.0, 473.0)

        # Display cap for additive beams: beyond ~30k rays/field every beam
        # pixel is already covered many times over (no visual gain), and the
        # per-ray energy would drop below the 8-bit framebuffer quantum
        # (1/255) and vanish. Clamp the sampling, visually equivalent.
        MAX_BEAM_RAYS = 30_000
        if mode in ("beam", "spectrum") and radial * azimuth > MAX_BEAM_RAYS:
            import math

            f = math.sqrt(MAX_BEAM_RAYS / (radial * azimuth))
            radial = max(4, int(radial * f))
            azimuth = max(16, int(azimuth * f))
            print(f"[gl3d] {mode}: sampling clamped to {radial * azimuth:,} "
                  "rays/field (display cap; visually equivalent)",
                  flush=True)

        for k, field_y in enumerate(fields):
            pupil = trace_pupil(
                tracer, FieldPoint(y=float(field_y)), na_object_sine=na_object_sine,
                sampling=PupilSampling(kind="rings", radial=radial, azimuth=azimuth),
                keep_paths=True, stop_index=stop_index,
            )
            paths = pupil.batch.paths[pupil.valid]
            if mode in ("beam", "spectrum"):
                wl = (wavelength_nm if mode == "beam"
                      else float(wavelengths_nm[k % len(wavelengths_nm)]))
                n_rays = max(int(pupil.valid.sum()), 1)
                # Physical scaling: total beam energy is fixed, each ray
                # carries E/n. Per-pixel brightness = (rays crossing it)·E/n,
                # which is density-INVARIANT — the illumination structure
                # (caustics bright, sparse regions dim) is preserved at any
                # sampling, just smoother. The float HDR buffer makes tiny
                # per-ray energies safe (no 8-bit quantum).
                rgb = spectral_rgb(wl) * (beam_energy / n_rays)
                self.add_paths(paths, color=(*rgb.tolist(), 1.0), width=1.0,
                               group=mode, additive=True)
            else:
                self.add_paths(paths, color=RAY_COLORS[k % len(RAY_COLORS)],
                               group="lines")

    # -- interaction -----------------------------------------------------
    def add_marker(self, position, *, color=(0.3, 1.0, 0.4, 1.0), size_px: float = 10.0) -> None:
        """Screen-size point marker (constant pixel size at any zoom).

        Prefer this over hand-drawn crosshair segments for tagging a point
        (e.g. a design focus): world-sized segments turn into screen-filling
        "infinite lines" once the user zooms in tighter than their length.
        """

        position = np.asarray(position, dtype=np.float32).reshape(1, 3)
        marker = self._scene.visuals.Markers(
            pos=position, face_color=color, edge_width=0, size=size_px,
            scaling=False, parent=self.view.scene,
        )
        marker.order = ORDER_RAYS
        marker.set_gl_state(blend=True, depth_test=False)
        self._bounds.append(position)

    def add_axes(self) -> None:
        """Meridional scale axes (z along the optical axis, y vertical) with
        tick labels that re-generate as the camera zooms or pans — toggle
        with the **a** key."""

        if getattr(self, "_axes", None) is None:
            self._axes = _DynamicAxes(self)

    def _update_light(self, event=None):
        if not self._shading_filters:
            return
        cam = self.view.camera
        az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
        # camera position direction for a +y-up turntable
        d = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
        light = (-d).tolist()  # from camera into the scene (headlight)
        for sf in self._shading_filters:
            try:
                sf.light_dir = light
            except Exception:
                pass

    def _on_key(self, event):
        key = getattr(event, "key", None)
        # vispy Key objects stringify as "<Key ('W',)>"; use .name for the letter
        key = (getattr(key, "name", None) or "").lower() if key is not None else ""
        if key == "w":
            self._wireframe_on = not self._wireframe_on
            for wf in self._wireframe_filters:
                wf.enabled = self._wireframe_on
            self.canvas.update()
        elif key == "m":
            modes = [m for m in ("lines", "beam", "spectrum") if self._ray_groups.get(m)]
            if modes:
                i = modes.index(self._ray_mode) if self._ray_mode in modes else -1
                self.set_ray_mode(modes[(i + 1) % len(modes)])
        elif key == "a":
            axes = getattr(self, "_axes", None)
            if axes is not None:
                axes.set_visible(not axes.visible)
                self.canvas.update()

    def set_ray_mode(self, mode: str) -> None:
        """Show one ray layer: "lines", "beam" or "spectrum"."""
        self._ray_mode = mode
        for name, lines in self._ray_groups.items():
            for line in lines:
                line.visible = name == mode
        self.canvas.update()

    def set_beam_mode(self, on: bool) -> None:  # backward compat
        self.set_ray_mode("beam" if on else "lines")

    # -- display ---------------------------------------------------------
    def frame(self) -> None:
        if not self._bounds:
            return
        allpts = np.vstack(self._bounds)
        finite = allpts[np.isfinite(allpts).all(axis=1)]
        center = 0.5 * (finite.min(axis=0) + finite.max(axis=0))
        extent = float(np.max(finite.max(axis=0) - finite.min(axis=0)))
        cam = self.view.camera
        cam.fov = 0.0
        cam.azimuth = -18.0     # near-side 3/4 view: optical axis (z) runs ~horizontal
        cam.elevation = 15.0
        cam.center = tuple(center)
        try:
            cam.set_range(margin=0.02)   # auto-fit the long, thin system
        except Exception:
            cam.scale_factor = 0.62 * extent
        self._update_light()

    def snapshot(self) -> np.ndarray:
        self.frame()
        return self.canvas.render(alpha=True)

    def run(self) -> None:  # pragma: no cover - interactive
        from vispy import app

        self.frame()
        self.canvas.show()
        from ._window import announce_window

        announce_window(self.canvas, "[raytracer 3D]")
        app.run()


__all__ = ["lathe", "lathe_solid", "Viewer3D", "MIRROR_COLOR", "spectral_rgb"]
