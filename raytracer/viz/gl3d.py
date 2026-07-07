"""3-D visualization of sequential systems (vispy.scene).

Lens elements are lathed from the shared :class:`AsphereProfile` sag. By default
each element is a closed solid of revolution; with ``section="half"`` it is a
**cutaway** (half torus + flat cut faces at the y=0 plane) so the interior and
the traced rays are visible without translucent-barrel overlap.

Lighting is a **headlight** (the light direction follows the camera, so no side
stays fixed in shadow while orbiting). Press ``w`` to toggle a wireframe overlay.
Elements are coloured by material. The camera is an orthographic turntable.

The mesh generators (:func:`lathe`, :func:`lathe_solid`) are pure numpy and
testable headless.
"""

from __future__ import annotations

import numpy as np
from vispy.color import Color

from ..geometry.sag import AsphereProfile
from .plots import DEFAULT_MATERIAL_COLORS

TWO_PI = 2.0 * np.pi
MIRROR_COLOR = (0.62, 0.66, 0.72, 1.0)
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


def _ring_grid(profile, radii, angles, vertex_z):
    sag = profile.sag(radii)
    rr, aa = np.meshgrid(radii, angles, indexing="ij")
    zz = np.repeat(sag[:, None], len(angles), axis=1) + vertex_z
    return np.column_stack([(rr * np.cos(aa)).ravel(),
                            (rr * np.sin(aa)).ravel(), zz.ravel()])


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


def lathe(profile, semidiameter, *, vertex_z=0.0, radial_samples=24,
          angular_samples=96, inner_radius=0.0, angle_range=(0.0, TWO_PI)):
    """Revolve a single profile into an (open) triangle mesh over ``angle_range``."""
    radii = np.linspace(inner_radius, semidiameter, radial_samples)
    angles, closed = _angles(angle_range, angular_samples)
    vertices = _ring_grid(profile, radii, angles, vertex_z)
    faces = _surface_faces(0, radial_samples, len(angles), closed)
    return vertices.astype(np.float32), np.asarray(faces, dtype=np.int32)


def lathe_solid(front, back, semidiameter, *, z_front, z_back,
                radial_samples=22, angular_samples=96, angle_range=(0.0, TWO_PI)):
    """Closed lens body (front + back + rim). If ``angle_range`` < 2π, adds flat
    cut faces at the two end meridians → a watertight cutaway half-solid."""
    R = radial_samples
    radii = np.linspace(0.0, semidiameter, R)
    angles, closed = _angles(angle_range, angular_samples)
    A = len(angles)

    front_v = _ring_grid(front, radii, angles, z_front)
    back_v = _ring_grid(back, radii, angles, z_back)
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
    return vertices.astype(np.float32), np.asarray(faces, dtype=np.int32)


def _material_color(material, alpha=1.0):
    # bright face tone: additive layers build a luminous crystal glow while the
    # hue keeps the media distinguishable (SIO2 blue / CAF2 green / HIINDEX amber)
    entry = DEFAULT_MATERIAL_COLORS.get(material, ("#c9c9c9", "#8a8a8a"))
    r, g, b = Color(entry[0]).rgb
    return (float(r), float(g), float(b), alpha)


class Viewer3D:
    """Orthographic turntable viewer with cutaway lenses and a headlight."""

    def __init__(self, *, size=(1300, 820), background="#0a0a0f", title="raytracer 3D"):
        from vispy import scene

        # NOTE: no MSAA (config samples): multisample framebuffers leave stale /
        # black frames on some drivers (frozen first frame over the scene).
        # Lines use their own fragment antialiasing instead.
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
        # CRITICAL: glClear honours glDepthMask. Glass draws last with
        # depth_mask=False; without re-enabling it before the next frame's
        # clear, the depth buffer never clears and the first frame stays
        # imprinted as a frozen black silhouette that occludes everything.
        # Wrap _draw_scene (the funnel for both live paints and offscreen
        # render()) so the mask is restored before every clear.
        _orig_draw_scene = self.canvas._draw_scene

        def _draw_scene_with_reset(*args, **kwargs):
            self._reset_gl_state()
            return _orig_draw_scene(*args, **kwargs)

        self.canvas._draw_scene = _draw_scene_with_reset
        # headlight: update the light direction whenever the camera moves
        self.canvas.events.mouse_move.connect(self._update_light)
        self.canvas.events.mouse_wheel.connect(self._update_light)
        self.canvas.events.key_press.connect(self._on_key)

    @staticmethod
    def _reset_gl_state(event=None):
        from vispy import gloo

        gloo.set_state(depth_mask=True)

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

    def add_lens(self, front, back, semidiameter, *, z_front, z_back, color,
                 section="full", style="glass"):
        rng = (0.0, np.pi) if section == "half" else (0.0, TWO_PI)
        v, f = lathe_solid(front, back, semidiameter, z_front=z_front, z_back=z_back,
                           angle_range=rng)
        self._add_mesh(v, f, color, style=style)

    def add_surface(self, profile, semidiameter, *, vertex_z, color, section="full",
                    angular_samples=96, style="opaque"):
        rng = (0.0, np.pi) if section == "half" else (0.0, TWO_PI)
        v, f = lathe(profile, semidiameter, vertex_z=vertex_z,
                     angular_samples=angular_samples, angle_range=rng)
        self._add_mesh(v, f, color, style=style)

    def add_system(self, system, *, section="full", angular_samples: int = 64) -> None:
        """Render lens elements (coloured by material) + mirrors as full closed
        solids by default; ``section='half'`` gives a cutaway."""
        from ..sequential.surfaces import SurfaceKind

        covered: set[int] = set()
        for i, j, material in system.solid_elements():
            ri, rj = system.rows[i], system.rows[j]
            semi = max(ri.semidiameter or 50.0, rj.semidiameter or 50.0)
            self.add_lens(ri.profile, rj.profile, semi,
                          z_front=float(system.vertices[i]), z_back=float(system.vertices[j]),
                          color=_material_color(material), section=section)
            covered.update((i, j))
        for i in system.mirror_indices:
            row = system.rows[i]
            self.add_surface(row.profile, row.semidiameter or 50.0,
                             vertex_z=float(system.vertices[i]), color=MIRROR_COLOR,
                             section=section, angular_samples=angular_samples)
            covered.add(i)
        for i, row in enumerate(system.rows):
            if i in covered or row.kind is not SurfaceKind.REFRACT:
                continue
            self.add_surface(row.profile, row.semidiameter or 50.0,
                             vertex_z=float(system.vertices[i]),
                             color=_material_color("SIO2"), section=section,
                             style="glass")

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
            # LIGHT adds: energy summation (colorimetric mixing where beams of
            # different wavelengths overlap); no depth write between beams
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
        from ..sequential.fields import FieldPoint, PupilSampling, trace_pupil

        if beam:  # backward-compat flag
            mode = "beam"
        if mode == "beam" and wavelength_nm is None:
            wavelength_nm = float(getattr(tracer.system, "wavelength_um", 0.55)) * 1e3
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
            print(f"[gl3d] {mode}: muestreo recortado a {radial * azimuth:,} "
                  "rayos/campo (tope de display; visualmente equivalente)",
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
                # Denser sampling spreads the same energy over more pixels;
                # compensate sublinearly so the beam keeps its brightness
                # regardless of density (calibrated at ~1400 rays/field), and
                # never let a ray fall below the 8-bit additive quantum.
                density_boost = max(1.0, n_rays / 1400.0) ** 0.75
                energy = max(beam_energy * density_boost / n_rays, 1.6 / 255.0)
                rgb = spectral_rgb(wl) * energy
                self.add_paths(paths, color=(*rgb.tolist(), 1.0), width=1.0,
                               group=mode, additive=True)
            else:
                self.add_paths(paths, color=RAY_COLORS[k % len(RAY_COLORS)],
                               group="lines")

    # -- interaction -----------------------------------------------------
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
