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
GLASS_ALPHA = 1.0  # opaque: avoids translucent depth-order artifacts
RAY_COLORS = [
    (0.95, 0.35, 0.25, 0.9),
    (0.30, 0.95, 0.45, 0.9),
    (0.35, 0.65, 1.00, 0.9),
    (0.80, 0.55, 0.95, 0.9),
]


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


def _material_color(material, alpha=GLASS_ALPHA):
    # use the saturated edge tone so the medium stays distinguishable under shading
    entry = DEFAULT_MATERIAL_COLORS.get(material, ("#c9c9c9", "#8a8a8a"))
    r, g, b = Color(entry[1] if len(entry) > 1 else entry[0]).rgb
    return (float(r), float(g), float(b), alpha)


class Viewer3D:
    """Orthographic turntable viewer with cutaway lenses and a headlight."""

    def __init__(self, *, size=(1300, 820), background="#0a0a0f", title="raytracer 3D"):
        from vispy import scene

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
        # headlight: update the light direction whenever the camera moves
        self.canvas.events.mouse_move.connect(self._update_light)
        self.canvas.events.mouse_wheel.connect(self._update_light)
        self.canvas.events.key_press.connect(self._on_key)

    # -- meshes ----------------------------------------------------------
    def _add_mesh(self, vertices, faces, color, *, shading="smooth"):
        from vispy.visuals.filters import WireframeFilter

        translucent = len(color) > 3 and color[3] < 1.0
        mesh = self._scene.visuals.Mesh(
            vertices=vertices, faces=faces, color=color, shading=shading,
            parent=self.view.scene,
        )
        mesh.set_gl_state(blend=True, depth_test=True, cull_face=False,
                          depth_mask=not translucent)
        sf = getattr(mesh, "shading_filter", None)
        if sf is not None:
            # Bright ambient FLOOR so no face is ever black (both the coefficient
            # AND the light must be turned up — the default ambient_light alpha is
            # only 0.25). Moderate diffuse gives gentle relief.
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
        wf = WireframeFilter(enabled=False, color=(0.9, 0.95, 1.0, 0.6), width=0.7)
        mesh.attach(wf)
        self._wireframe_filters.append(wf)
        self._bounds.append(vertices)
        return mesh

    def add_lens(self, front, back, semidiameter, *, z_front, z_back, color, section="full"):
        rng = (0.0, np.pi) if section == "half" else (0.0, TWO_PI)
        v, f = lathe_solid(front, back, semidiameter, z_front=z_front, z_back=z_back,
                           angle_range=rng)
        self._add_mesh(v, f, color)

    def add_surface(self, profile, semidiameter, *, vertex_z, color, section="full",
                    angular_samples=96):
        rng = (0.0, np.pi) if section == "half" else (0.0, TWO_PI)
        v, f = lathe(profile, semidiameter, vertex_z=vertex_z,
                     angular_samples=angular_samples, angle_range=rng)
        self._add_mesh(v, f, color)

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
                             color=_material_color("SIO2"), section=section)

    # -- rays ------------------------------------------------------------
    def add_paths(self, paths, *, color=None, width: float = 1.4) -> None:
        paths = np.asarray(paths, dtype=float)
        if paths.ndim == 2:
            paths = paths[None, ...]
        color = color or RAY_COLORS[0]
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
        )
        line.set_gl_state(blend=True, depth_test=False)  # rays always visible
        self._bounds.append(positions)

    def add_field_bundles(self, tracer, fields, *, na_object_sine, radial=6, azimuth=32,
                          stop_index=None) -> None:
        from ..sequential.fields import FieldPoint, PupilSampling, trace_pupil

        for k, field_y in enumerate(fields):
            pupil = trace_pupil(
                tracer, FieldPoint(y=float(field_y)), na_object_sine=na_object_sine,
                sampling=PupilSampling(kind="rings", radial=radial, azimuth=azimuth),
                keep_paths=True, stop_index=stop_index,
            )
            paths = pupil.batch.paths[pupil.valid]
            self.add_paths(paths, color=RAY_COLORS[k % len(RAY_COLORS)])

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
        if getattr(event, "key", None) is not None and str(event.key).lower() == "w":
            self._wireframe_on = not self._wireframe_on
            for wf in self._wireframe_filters:
                wf.enabled = self._wireframe_on
            self.canvas.update()

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
        app.run()


__all__ = ["lathe", "lathe_solid", "Viewer3D", "MIRROR_COLOR"]
