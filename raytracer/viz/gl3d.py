"""3-D views of optical systems and traces (vispy.scene).

Each element is meshed in its own frame and carried to the world by its pose, so tilted, decentred and folded
systems render as built; the optical axis is z and the turntable camera keeps +y up.

- A lens is a closed solid lathed from its surfaces' sag ``s(x, y)`` — any surface — out to the extents of its
  2-D outline (:func:`~raytracer.viz.section.lens_outline`: each face's single-valued branch, the Ω join;
  the cuts between neighbours are not applied). ``section="half"`` cuts every element away below ``y = 0``.
- Mirrors, beam splitters, diaphragm blades and sensors are their faces over their apertures; prisms their facets.
- Rays come from sequential pupil bundles (:meth:`Viewer3D.add_field_bundles`), polylines
  (:meth:`Viewer3D.add_paths`) or a non-sequential :class:`~raytracer.tracing.RayTree` (:meth:`Viewer3D.add_tree`).

The light is a headlight that follows the camera, so no side stays dark while orbiting. Keys: **w** wireframe,
**m** next ray layer, **a** axes. The mesh generators :func:`lathe` and :func:`lathe_solid` are pure numpy.
"""

from __future__ import annotations

import numpy as np

from ..elements import BeamSplitter, Lens, Mirror, Prism, Sensor, Stop
from ..geometry.apertures import Annulus, Aperture, Circle, Rectangle
from ..geometry.transforms import Pose
from .bodies import DEFAULT_MATERIAL_COLORS, glass_colors, medium_index
from .color import hex_rgba, spectral_rgb
from .gl import shaders
from .section import facet_corners, fan_faces, lens_outline

TWO_PI = 2.0 * np.pi
MIRROR_COLOR = (0.62, 0.66, 0.72, 1.0)
STOP_COLOR = (0.18, 0.18, 0.20, 1.0)
SENSOR_COLOR = (0.85, 0.72, 0.30, 1.0)
GLASS_ALPHA = 0.30  # alpha blending: stacked glass reads denser, never brighter; only light adds
RAY_COLORS = [(0.95, 0.35, 0.25, 0.55), (0.30, 0.95, 0.45, 0.55), (0.35, 0.65, 1.00, 0.55), (0.80, 0.55, 0.95, 0.55)]

# Draw order: opaque mirrors, then depth-tested rays, then translucent glass over them — a ray behind glass is
# tinted, one in front stays crisp, and whatever lies behind a mirror is hidden.
ORDER_OPAQUE, ORDER_RAYS, ORDER_GLASS = 0, 1, 2

_SHADING = {  # a bright ambient floor, so no face ever goes black
    "glass": dict(ambient_coefficient=(1.0, 1.0, 1.0, 0.55), diffuse_light=(1.0, 1.0, 1.0, 0.9),
                  diffuse_coefficient=(1.0, 1.0, 1.0, 0.60), specular_coefficient=(1.0, 1.0, 1.0, 0.50), shininess=80.0),
    "opaque": dict(ambient_coefficient=(1.0, 1.0, 1.0, 0.75), diffuse_light=(1.0, 1.0, 1.0, 0.85),
                   diffuse_coefficient=(1.0, 1.0, 1.0, 0.50), specular_coefficient=(1.0, 1.0, 1.0, 0.15), shininess=24.0),
}

#: Colour and style of each kind of thin element.
_THIN = {
    Mirror: (MIRROR_COLOR, "opaque"),
    BeamSplitter: (hex_rgba(DEFAULT_MATERIAL_COLORS["SIO2"][0]), "glass"),
    Stop: (STOP_COLOR, "opaque"),
    Sensor: (SENSOR_COLOR, "opaque"),
}

#: Above this many rays per field an additive beam gains nothing visible; sampling is clamped.
MAX_BEAM_RAYS = 30_000


class Viewer3D:
    """Orthographic turntable viewer, rendered through a float HDR buffer and a soft-knee display map."""

    def __init__(self, *, size=(1300, 820), background="#0a0a0f", title="raytracer 3D", ssaa: int = 2):
        from vispy import scene

        # Antialiasing is supersampling, not MSAA (multisample buffers leave stale frames on some drivers): the
        # scene renders into the HDR buffer at ssaa× resolution and the display pass downsamples it, smoothing
        # every primitive. While the mouse moves frames render at 1×; one crisp pass follows when it rests.
        self._ssaa = max(1, int(ssaa))
        self._interacting = False
        self._idle_timer = None
        self.canvas = scene.SceneCanvas(keys="interactive", size=size, bgcolor=background, title=title, show=False)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(fov=0.0, up="+y")  # fov 0: orthographic
        self._scene = scene
        self._bounds: list[np.ndarray] = []
        self._shading_filters: list = []
        self._wireframe_filters: list = []
        self._wireframe_on = False
        self._ray_groups: dict[str, list] = {"lines": [], "beam": [], "spectrum": []}
        self._ray_mode = "lines"
        self._axes = None
        self._hdr_targets: dict = {}
        self._hdr_prog = None
        self._draw_scene = self.canvas._draw_scene  # every frame, live or offscreen, goes through _draw_hdr
        self.canvas._draw_scene = self._draw_hdr
        for event in (self.canvas.events.mouse_move, self.canvas.events.mouse_wheel):
            event.connect(self._update_light)
            event.connect(self._on_interaction)
        self.canvas.events.key_press.connect(self._on_key)

    # -- elements ----------------------------------------------------------------

    def add_system(self, system, *, section="full", angular_samples: int = 64, wavelength_um: float | None = None) -> None:
        """Every element at its world pose: lenses and prisms as glass coloured by material, mirrors opaque, beam
        splitters glassy, blades dark, sensors gold — those with an aperture (an unbounded image plane has no size)."""

        for element, pose in system.walk():
            if isinstance(element, Lens):
                self.add_lens(element, pose, section=section, wavelength_um=wavelength_um)
            elif isinstance(element, Prism):
                color = _glass_rgba(element.material, wavelength_um)
                for corners in facet_corners(element.solid, pose):
                    self._add_mesh(corners.astype(np.float32), fan_faces(len(corners)).astype(np.int32), color, style="glass")
            elif type(element) in _THIN and not getattr(element, "virtual", False):
                color, style = _THIN[type(element)]
                for face in element.faces():
                    if face.extent is not None:
                        self.add_surface(face.interface.surface, face.extent, pose=pose @ face.pose, color=color, style=style,
                                         section=section, angular_samples=angular_samples, hole=face.hole)

    def add_lens(self, lens: Lens, pose: Pose | None = None, *, color=None, section="full", style="glass", wavelength_um=None):
        """A lens body at its world *pose* (default: its own)."""

        _, (e_front, e_back), _ = lens_outline(lens, Pose())
        v, f = lathe_solid(lens.front.surface, lens.back.surface, e_front, thickness=lens.thickness, back_semidiameter=e_back,
                           pose=lens.pose if pose is None else pose, angle_range=_angle_range(section))
        self._add_mesh(v, f, _glass_rgba(lens.material, wavelength_um) if color is None else color, style=style)

    def add_surface(self, surface, aperture, *, pose: Pose | None = None, color, section="full", angular_samples=96,
                    style="opaque", hole=None):
        """One face over its *aperture* (a radius or an aperture, less its *hole*), placed by *pose*."""

        aperture = Aperture.coerce(aperture)
        if isinstance(aperture, Rectangle) and hole is None:
            v, f = _rectangle_sheet(surface, aperture, pose)
        else:
            inner = max(aperture.inner if isinstance(aperture, Annulus) else 0.0, 0.0 if hole is None else Aperture.coerce(hole).extent)
            v, f = lathe(surface, aperture.extent, pose=pose, inner_radius=inner, angular_samples=angular_samples,
                         angle_range=_angle_range(section), aperture=aperture)
        self._add_mesh(v, f, color, style=style)

    def _add_mesh(self, vertices, faces, color, *, style="opaque"):
        from vispy.visuals.filters import WireframeFilter

        glass = style == "glass"
        mesh = self._scene.visuals.Mesh(vertices=vertices, faces=faces, color=(*color[:3], GLASS_ALPHA) if glass else color,
                                        shading="smooth", parent=self.view.scene)
        if glass:
            mesh.set_gl_state(blend=True, depth_test=True, depth_mask=False, cull_face=False,
                              blend_func=("src_alpha", "one_minus_src_alpha"))
        else:
            mesh.set_gl_state(blend=True, depth_test=True, cull_face=False, depth_mask=True)
        mesh.order = ORDER_GLASS if glass else ORDER_OPAQUE
        shading = getattr(mesh, "shading_filter", None)
        if shading is not None:
            try:
                for name, value in dict(ambient_light=(1.0, 1.0, 1.0, 1.0), **_SHADING[style]).items():
                    setattr(shading, name, value)
            except Exception:
                pass
            self._shading_filters.append(shading)
        wireframe = WireframeFilter(enabled=False, color=(0.9, 0.95, 1.0, 0.55), width=0.8)
        mesh.attach(wireframe)
        self._wireframe_filters.append(wireframe)
        self._bounds.append(vertices)
        return mesh

    # -- rays ------------------------------------------------------------------------

    def add_field_bundles(self, tracer, fields, *, na_object_sine, radial=6, azimuth=32, stop_index=None, mode="lines",
                          wavelength_nm=None, wavelengths_nm=None, beam_energy: float = 12.0) -> None:
        """Trace a pupil bundle per field and add it to the ray layer *mode*.

        - ``"lines"``: per-field coloured traces, alpha-blended.
        - ``"beam"``: an additive beam at the system wavelength (193 nm DUV shows violet).
        - ``"spectrum"``: one wavelength per field (``wavelengths_nm``, default 633/532/473 nm — a display
          choice), each an additive beam; where they overlap, colours sum colorimetrically toward white.

        A beam's total energy is ``beam_energy``, each ray carrying its share, so a pixel's brightness counts
        the rays crossing it whatever the sampling: caustics bright, sparse regions dim.
        """

        from ..tracing import FieldPoint, PupilSampling, trace_pupil

        if mode == "beam" and wavelength_nm is None:
            wavelength_nm = float(tracer.wavelength_um if tracer.wavelength_um is not None else 0.55) * 1e3
        if mode == "spectrum" and wavelengths_nm is None:
            wavelengths_nm = (633.0, 532.0, 473.0)
        if mode in ("beam", "spectrum") and radial * azimuth > MAX_BEAM_RAYS:
            f = np.sqrt(MAX_BEAM_RAYS / (radial * azimuth))
            radial, azimuth = max(4, int(radial * f)), max(16, int(azimuth * f))
            print(f"[gl3d] {mode}: sampling clamped to {radial * azimuth:,} rays/field (display cap; visually equivalent)",
                  flush=True)

        for k, field_y in enumerate(fields):
            pupil = trace_pupil(tracer, FieldPoint(y=float(field_y)), na_object_sine=na_object_sine,
                                sampling=PupilSampling(kind="rings", radial=radial, azimuth=azimuth), keep_paths=True,
                                stop_index=stop_index)
            paths = pupil.batch.paths[pupil.valid]
            if mode in ("beam", "spectrum"):
                wl = wavelength_nm if mode == "beam" else float(wavelengths_nm[k % len(wavelengths_nm)])
                rgb = spectral_rgb(wl) * (beam_energy / max(int(pupil.valid.sum()), 1))
                self.add_paths(paths, color=(*rgb.tolist(), 1.0), width=1.0, group=mode, additive=True)
            else:
                self.add_paths(paths, color=RAY_COLORS[k % len(RAY_COLORS)], group="lines")

    def add_paths(self, paths, *, color=None, width: float = 1.4, group: str = "lines", additive: bool = False) -> None:
        """Traced polylines ``(N, points, 3)`` on the ray layer *group*; ``additive`` sums overlapping rays into a beam."""

        paths = np.asarray(paths, dtype=float)
        positions, connections, offset = [], [], 0
        for path in paths[None] if paths.ndim == 2 else paths:
            valid = path[~np.isnan(path).any(axis=1)]
            if valid.shape[0] >= 2:
                positions.append(valid)
                connections.append(np.column_stack([np.arange(len(valid) - 1), np.arange(1, len(valid))]) + offset)
                offset += len(valid)
        if positions:
            self._add_rays(np.vstack(positions).astype(np.float32), np.vstack(connections).astype(np.uint32),
                           RAY_COLORS[0] if color is None else color, width, group, additive)

    def add_tree(self, tree, *, color=None, width: float = 1.0, group: str = "lines", additive: bool = False,
                 select: np.ndarray | None = None, escape_length: float | None = None) -> None:
        """The segments of a non-sequential :class:`~raytracer.tracing.RayTree`.

        ``color`` is one RGBA, an ``(M, 4)`` array, or None for the first ray colour with alpha scaled by power.
        Rays that left the system end ``escape_length`` mm from their start when given; ``select`` masks segments.
        """

        n = len(tree)
        if n == 0:
            return
        start, end = np.asarray(tree.start, dtype=float), np.asarray(tree.end, dtype=float)
        if escape_length is not None:
            end = np.where(np.asarray(tree.escaped, dtype=bool)[:, None], start + float(escape_length) * tree.direction, end)
        if color is None:
            power = np.asarray(tree.power, dtype=float)
            peak = power.max() if power.size and power.max() > 0 else 1.0
            rgba = np.tile(np.asarray(RAY_COLORS[0], dtype=float), (n, 1))
            rgba[:, 3] = np.clip(RAY_COLORS[0][3] * power / peak, 0.03, 1.0)
        else:
            rgba = np.asarray(color, dtype=float)
            rgba = np.tile(rgba, (n, 1)) if rgba.ndim == 1 else rgba
        keep = np.ones(n, dtype=bool) if select is None else np.asarray(select, dtype=bool)
        positions = np.stack([start[keep], end[keep]], axis=1).reshape(-1, 3).astype(np.float32)
        if positions.shape[0]:
            self._add_rays(positions, "segments", np.repeat(rgba[keep], 2, axis=0).astype(np.float32), width, group, additive)

    def _add_rays(self, positions, connect, color, width, group, additive) -> None:
        line = self._scene.visuals.Line(pos=positions, connect=connect, color=color, width=width, parent=self.view.scene,
                                        method="gl", antialias=True)
        if additive:  # light adds linearly into the float buffer; the display map compresses only the foci
            line.set_gl_state(blend=True, depth_test=True, depth_mask=False, blend_func=("one", "one"))
        else:  # plain traces: alpha-blended, hidden by mirrors, seen through glass
            line.set_gl_state(blend=True, depth_test=True)
        line.order = ORDER_RAYS
        line.visible = group == self._ray_mode
        self._ray_groups.setdefault(group, []).append(line)
        self._bounds.append(positions)

    def add_marker(self, position, *, color=(0.3, 1.0, 0.4, 1.0), size_px: float = 10.0) -> None:
        """A point marker of constant screen size (world-sized crosshairs fill the screen when zoomed in)."""

        position = np.asarray(position, dtype=np.float32).reshape(1, 3)
        marker = self._scene.visuals.Markers(pos=position, face_color=color, edge_width=0, size=size_px, scaling=False,
                                             parent=self.view.scene)
        marker.order = ORDER_RAYS
        marker.set_gl_state(blend=True, depth_test=False)
        self._bounds.append(position)

    def add_axes(self) -> None:
        """Meridional rulers (z along the axis, y vertical) whose ticks follow zoom and pan; **a** toggles them."""

        if self._axes is None:
            self._axes = _Rulers(self)

    def set_ray_mode(self, mode: str) -> None:
        """Show one ray layer: ``"lines"``, ``"beam"`` or ``"spectrum"``."""

        self._ray_mode = mode
        for name, lines in self._ray_groups.items():
            for line in lines:
                line.visible = name == mode
        self.canvas.update()

    # -- display -----------------------------------------------------------------------

    def frame(self) -> None:
        """Fit the camera to everything added, in a near-side 3/4 view with the optical axis about horizontal."""

        if not self._bounds:
            return
        points = np.vstack(self._bounds)
        points = points[np.isfinite(points).all(axis=1)]
        cam = self.view.camera
        cam.fov, cam.azimuth, cam.elevation = 0.0, -18.0, 15.0
        cam.center = tuple(0.5 * (points.min(axis=0) + points.max(axis=0)))
        try:
            cam.set_range(margin=0.02)
        except Exception:
            cam.scale_factor = 0.62 * float(np.max(points.max(axis=0) - points.min(axis=0)))
        self._update_light()

    def snapshot(self) -> np.ndarray:
        self.frame()
        return self.canvas.render(alpha=True)

    def run(self) -> None:  # pragma: no cover - interactive
        from vispy import app

        from ._window import announce_window

        self.frame()
        self.canvas.show()
        announce_window(self.canvas, "[raytracer 3D]")
        app.run()

    def _draw_hdr(self, *args, **kwargs):
        from vispy import gloo

        if self._axes is not None:
            self._axes.rebuild()
        gloo.set_state(depth_mask=True)  # glClear honours the mask, and the glass drawn last leaves it off
        if getattr(getattr(self.canvas, "scene", None), "picking", False):
            return self._draw_scene(*args, **kwargs)  # picking reads id colours back: no HDR pass
        w, h = self.canvas.physical_size
        ss = 1 if self._interacting else self._ssaa
        texture, fbo = self._hdr_target(int(w * ss), int(h * ss))
        # push_fbo takes the buffer's pixel size for its logical size; reconfigure the shared transforms with
        # the true geometry, so pixel-sized visuals (text, markers, line widths) scale into the larger buffer.
        self.canvas.push_fbo(fbo, (0, 0), self.canvas.size)
        self.canvas.transforms.configure(viewport=(0, 0, int(w * ss), int(h * ss)), fbo_size=(int(w * ss), int(h * ss)),
                                         fbo_rect=(0, 0, int(w), int(h)))
        try:
            self._draw_scene(*args, **kwargs)
        finally:
            self.canvas.pop_fbo()
        gloo.set_viewport(0, 0, int(w), int(h))
        gloo.set_state(blend=False, depth_test=False, cull_face=False)
        self._hdr_prog["u_tex"] = texture
        self._hdr_prog.draw("triangles")

    def _hdr_target(self, w: int, h: int):
        """The float ``(texture, framebuffer)`` of a size; the 1× and ssaa× ones are kept, not reallocated."""

        from vispy import gloo

        if self._hdr_prog is None:
            self._hdr_prog = gloo.Program(shaders.TONEMAP_VERT, shaders.SOFT_KNEE_FRAG)
            self._hdr_prog["a_position"] = np.array([[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]], dtype=np.float32)
        if (h, w) not in self._hdr_targets:
            if len(self._hdr_targets) >= 2:  # the window was resized
                self._hdr_targets.clear()
            # linear sampling: at exactly 2:1 the display pass's bilinear tap is a box filter
            texture = gloo.Texture2D(shape=(h, w, 4), internalformat="rgba32f", interpolation="linear")
            self._hdr_targets[h, w] = texture, gloo.FrameBuffer(color=texture, depth=gloo.RenderBuffer((h, w)))
        return self._hdr_targets[h, w]

    def _on_interaction(self, event=None) -> None:
        if self._ssaa <= 1 or not (getattr(event, "is_dragging", False) or getattr(event, "type", "") == "mouse_wheel"):
            return
        self._interacting = True
        if self._idle_timer is None:
            from vispy.app import Timer

            self._idle_timer = Timer(interval=0.28, connect=self._end_interaction)
        self._idle_timer.stop()
        self._idle_timer.start(iterations=1)

    def _end_interaction(self, event=None) -> None:
        self._interacting = False
        self.canvas.update()

    def _update_light(self, event=None) -> None:
        cam = self.view.camera
        az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
        toward_camera = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])  # +y-up turntable
        for shading in self._shading_filters:
            try:
                shading.light_dir = (-toward_camera).tolist()
            except Exception:
                pass

    def _on_key(self, event) -> None:
        key = getattr(event, "key", None)
        key = (getattr(key, "name", None) or "").lower() if key is not None else ""
        if key == "w":
            self._wireframe_on = not self._wireframe_on
            for wireframe in self._wireframe_filters:
                wireframe.enabled = self._wireframe_on
            self.canvas.update()
        elif key == "m":
            modes = [m for m in ("lines", "beam", "spectrum") if self._ray_groups.get(m)]
            if modes:
                i = modes.index(self._ray_mode) if self._ray_mode in modes else -1
                self.set_ray_mode(modes[(i + 1) % len(modes)])
        elif key == "a" and self._axes is not None:
            self._axes.set_visible(not self._axes.visible)
            self.canvas.update()


class _Rulers:
    """Two rulers in the plane x = 0 crossing at the view's centre — z along the axis, y vertical — with ticks on a
    1/2/5 × 10ⁿ step chosen from the visible extent, labelled in world coordinates, rebuilt as the camera moves."""

    LINE = (0.55, 0.62, 0.58, 0.6)
    TEXT = (0.72, 0.80, 0.75, 0.95)

    def __init__(self, viewer: Viewer3D) -> None:
        self.viewer = viewer
        self.visible = True
        self._state: tuple | None = None
        scene = viewer.view.scene
        self._lines = viewer._scene.visuals.Line(pos=np.zeros((2, 3), dtype=np.float32), connect="segments", color=self.LINE,
                                                 width=1.0, parent=scene, method="gl")
        self._lines.set_gl_state(blend=True, depth_test=False)
        self._text = viewer._scene.visuals.Text(text="", pos=(0.0, 0.0, 0.0), color=self.TEXT, font_size=7, parent=scene)
        self._lines.order = self._text.order = ORDER_GLASS + 1  # over glass and rays

    def set_visible(self, on: bool) -> None:
        self.visible = self._lines.visible = self._text.visible = on

    def rebuild(self) -> None:
        """Rebuild when the camera moved — but not during interaction (laying text out each frame stutters):
        the crisp pass after the mouse rests brings the rulers up to date."""

        if not self.visible or (self.viewer._interacting and self._state is not None):
            return
        cam = self.viewer.view.camera
        state = (float(cam.scale_factor), tuple(np.round(cam.center, 9)))
        if state == self._state:
            return
        self._state = state
        cx, cy, cz = (float(v) for v in cam.center)
        extent = float(cam.scale_factor)
        step = _nice_step(extent)
        half, tick = 0.72 * extent, 0.012 * extent
        decimals = max(0, -int(np.floor(np.log10(step))))
        segments = [[[cx, cy, cz - half], [cx, cy, cz + half]], [[cx, cy - half, cz], [cx, cy + half, cz]]]
        labels, positions = [], []
        for z in np.arange(np.ceil((cz - half) / step) * step, cz + half, step):
            segments.append([[cx, cy - tick, z], [cx, cy + tick, z]])
            labels.append(f"{z:.{decimals}f}")
            positions.append((cx, cy - 3.0 * tick, z))
        for y in np.arange(np.ceil((cy - half) / step) * step, cy + half, step):
            if abs(y - cy) >= 0.5 * step:  # none on the crossing
                segments.append([[cx, y, cz - tick], [cx, y, cz + tick]])
                labels.append(f"{y:.{decimals}f}")
                positions.append((cx, y, cz + 3.0 * tick))
        self._lines.set_data(pos=np.asarray(segments, dtype=np.float32).reshape(-1, 3), connect="segments", color=self.LINE)
        self._text.text = labels or ""
        if labels:
            self._text.pos = np.asarray(positions, dtype=np.float32)


def _nice_step(extent: float) -> float:
    """A 1/2/5 × 10ⁿ step giving some 5 to 8 ticks across *extent*."""

    raw = max(extent, 1e-12) / 6.0
    magnitude = 10.0 ** np.floor(np.log10(raw))
    return next((m * magnitude for m in (1.0, 2.0, 5.0) if raw <= m * magnitude), 10.0 * magnitude)


def _glass_rgba(material, wavelength_um=None) -> tuple:
    """Glass tone: the palette entry of the material's name (SIO2 blue, CAF2 green, …), else by index, else grey."""

    face, _ = glass_colors(material, medium_index(material, wavelength_um), DEFAULT_MATERIAL_COLORS) or ("#c9c9c9", "#8a8a8a")
    return hex_rgba(face)


def _angle_range(section: str) -> tuple[float, float]:
    return (0.0, np.pi) if section == "half" else (0.0, TWO_PI)


# -- meshes (pure numpy) -------------------------------------------------------------------


def lathe(surface, semidiameter, *, pose: Pose | None = None, radial_samples=24, angular_samples=96, inner_radius=0.0,
          angle_range=(0.0, TWO_PI), aperture: Aperture | None = None):
    """``(vertices, faces)`` of one face over a polar grid out to *semidiameter*: an open sheet.

    The grid lies in the face's frame (``z = s(x, y)``, any surface) and *pose* carries it to the world.
    ``inner_radius`` leaves a central hole (an annular mirror, a blade); an *aperture* other than a centred
    circle drops the triangles reaching outside it.
    """

    angles, closed = _angles(angle_range, angular_samples)
    vertices = _ring_grid(surface, np.linspace(inner_radius, semidiameter, radial_samples), angles)
    faces = _grid_faces(0, radial_samples, len(angles), closed)
    if aperture is not None and not (isinstance(aperture, Circle) and aperture.center == (0.0, 0.0)):
        inside = aperture.contains(vertices[:, 0], vertices[:, 1], slack=1e-9)
        faces = faces[inside[faces].all(axis=1)]
    return _to_world(vertices, pose), faces


def lathe_solid(front, back, semidiameter, *, thickness, pose: Pose | None = None, back_semidiameter=None, radial_samples=22,
                angular_samples=96, angle_range=(0.0, TWO_PI)):
    """``(vertices, faces)`` of a closed lens body: front face, back face at ``z = thickness``, and the rim.

    The front runs to *semidiameter*, the back to *back_semidiameter* (default the same); the rim joins their
    outer rings. An ``angle_range`` short of 2π is closed by flat faces at its two end meridians: a watertight
    cutaway. Built in the lens frame and carried to the world by *pose*.
    """

    R = radial_samples
    angles, closed = _angles(angle_range, angular_samples)
    A = len(angles)
    back_semi = semidiameter if back_semidiameter is None else back_semidiameter
    vertices = np.vstack([_ring_grid(front, np.linspace(0.0, semidiameter, R), angles),
                          _ring_grid(back, np.linspace(0.0, back_semi, R), angles, z_offset=thickness)])
    faces = [_grid_faces(0, R, A, closed), _grid_faces(R * A, R, A, closed)]
    j = np.arange(A if closed else A - 1)
    jn = (j + 1) % A
    fo, bo = (R - 1) * A, R * A + (R - 1) * A  # outer rings: the rim between them
    faces.append(np.stack([fo + j, fo + jn, bo + j, fo + jn, bo + jn, bo + j], axis=1).reshape(-1, 3))
    if not closed:  # the cut faces, front to back along each end meridian
        i = np.arange(R - 1)
        for c in (0, A - 1):
            f0, f1 = i * A + c, (i + 1) * A + c
            faces.append(np.stack([f0, f1, R * A + f0, f1, R * A + f1, R * A + f0], axis=1).reshape(-1, 3))
    return _to_world(vertices, pose), np.vstack(faces).astype(np.int32)


def _rectangle_sheet(surface, rect: Rectangle, pose: Pose | None, samples: int = 12):
    """A face over a rectangular aperture, as a grid of quads."""

    cx, cy = rect.center
    xx, yy = np.meshgrid(np.linspace(cx - rect.width / 2, cx + rect.width / 2, samples),
                         np.linspace(cy - rect.height / 2, cy + rect.height / 2, samples), indexing="ij")
    vertices = np.column_stack([xx.ravel(), yy.ravel(), np.asarray(surface.sag(xx, yy)).ravel()])
    return _to_world(vertices, pose), _grid_faces(0, samples, samples, False)


def _angles(angle_range, n):
    a0, a1 = angle_range
    closed = (a1 - a0) >= TWO_PI - 1e-6
    return np.linspace(a0, a1, n, endpoint=not closed), closed


def _ring_grid(surface, radii, angles, z_offset=0.0):
    """Local points of *surface* on a polar grid: rings of *radii* (outer index) by *angles*."""

    rr, aa = np.meshgrid(radii, angles, indexing="ij")
    x, y = rr * np.cos(aa), rr * np.sin(aa)
    return np.column_stack([x.ravel(), y.ravel(), np.asarray(z_offset + surface.sag(x, y), dtype=float).ravel()])


def _grid_faces(offset, rows, columns, closed) -> np.ndarray:
    """Two triangles per cell of a ``rows × columns`` vertex grid starting at *offset*; ``closed`` wraps the columns."""

    i, j = np.meshgrid(np.arange(rows - 1), np.arange(columns if closed else columns - 1), indexing="ij")
    i, j = i.ravel(), j.ravel()
    a, c = offset + i * columns + j, offset + (i + 1) * columns + j
    b, d = offset + i * columns + (j + 1) % columns, offset + (i + 1) * columns + (j + 1) % columns
    return np.stack([a, b, c, b, d, c], axis=1).reshape(-1, 3).astype(np.int32)


def _to_world(vertices: np.ndarray, pose: Pose | None) -> np.ndarray:
    return (vertices if pose is None else pose.to_world(vertices)).astype(np.float32)


__all__ = ["Viewer3D", "lathe", "lathe_solid", "MIRROR_COLOR"]
