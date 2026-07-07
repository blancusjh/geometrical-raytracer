"""3-D visualization of sequential systems (vispy.scene).

Surfaces of revolution are lathed from the shared :class:`AsphereProfile`
sag; traced ray paths become line visuals. Scope: a viewer with a turntable
camera — editing stays in 2-D.

The mesh generator (:func:`lathe`) is pure numpy and testable headless.
"""

from __future__ import annotations

import numpy as np

from ..geometry.sag import AsphereProfile

GLASS_COLOR = (0.45, 0.68, 0.92, 0.16)
MIRROR_COLOR = (0.35, 0.35, 0.4, 1.0)
RAY_COLORS = [
    (0.84, 0.15, 0.16, 0.8),
    (0.17, 0.63, 0.17, 0.8),
    (0.12, 0.47, 0.71, 0.8),
    (0.58, 0.40, 0.74, 0.8),
]


def lathe(
    profile: AsphereProfile,
    semidiameter: float,
    *,
    vertex_z: float = 0.0,
    radial_samples: int = 24,
    angular_samples: int = 64,
    inner_radius: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Revolve a profile around the z axis into a triangle mesh.

    Returns ``(vertices, faces)`` with vertices shaped (N, 3) and faces
    (M, 3) int32. The surface spans ``inner_radius <= h <= semidiameter``.
    """

    radii = np.linspace(inner_radius, semidiameter, radial_samples)
    angles = np.linspace(0.0, 2.0 * np.pi, angular_samples, endpoint=False)
    sag = profile.sag(radii)

    rr, aa = np.meshgrid(radii, angles, indexing="ij")
    zz = np.repeat(sag[:, None], angular_samples, axis=1) + vertex_z
    xx = rr * np.cos(aa)
    yy = rr * np.sin(aa)
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    faces = []
    for i in range(radial_samples - 1):
        for j in range(angular_samples):
            j_next = (j + 1) % angular_samples
            a = i * angular_samples + j
            b = i * angular_samples + j_next
            c = (i + 1) * angular_samples + j
            d = (i + 1) * angular_samples + j_next
            faces.append([a, b, c])
            faces.append([b, d, c])
    return vertices.astype(np.float32), np.asarray(faces, dtype=np.int32)


class Viewer3D:
    """Turntable 3-D viewer for sequential systems and traced bundles."""

    def __init__(
        self,
        *,
        size: tuple[int, int] = (1100, 750),
        background: str = "#101014",
        title: str = "raytracer 3D",
    ) -> None:
        from vispy import scene

        self.canvas = scene.SceneCanvas(
            keys="interactive", size=size, bgcolor=background, title=title, show=False
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(fov=35.0, up="+y")
        self._scene = scene
        self._bounds: list[np.ndarray] = []

    # -- content ---------------------------------------------------------

    def add_surface(
        self,
        profile: AsphereProfile,
        semidiameter: float,
        *,
        vertex_z: float,
        color=GLASS_COLOR,
        angular_samples: int = 64,
    ) -> None:
        vertices, faces = lathe(
            profile, semidiameter, vertex_z=vertex_z, angular_samples=angular_samples
        )
        translucent = len(color) > 3 and color[3] < 1.0
        mesh = self._scene.visuals.Mesh(
            vertices=vertices,
            faces=faces,
            color=color,
            shading=None if translucent else "smooth",
            parent=self.view.scene,
        )
        # Translucent glass must not occlude the rays traced inside it.
        mesh.set_gl_state(
            blend=True, depth_test=True, cull_face=False,
            depth_mask=not translucent,
        )
        self._bounds.append(vertices)

    def add_system(self, system, *, angular_samples: int = 64) -> None:
        """Add every surface of an :class:`OpticalSystem` (stops skipped)."""

        from ..sequential.surfaces import SurfaceKind

        seen = set()
        for i, row in enumerate(system.rows):
            if row.kind is SurfaceKind.STOP:
                continue
            signature = (round(float(system.vertices[i]), 9), round(row.radius, 9))
            if signature in seen:
                continue
            seen.add(signature)
            semidiameter = row.semidiameter if row.semidiameter is not None else 50.0
            color = MIRROR_COLOR if row.kind is SurfaceKind.MIRROR else GLASS_COLOR
            self.add_surface(
                row.profile,
                semidiameter,
                vertex_z=float(system.vertices[i]),
                color=color,
                angular_samples=angular_samples,
            )

    def add_paths(self, paths: np.ndarray, *, color=None, width: float = 1.0) -> None:
        """Add traced ray polylines: (N, S, 3) array (NaN rows are skipped)."""

        paths = np.asarray(paths, dtype=float)
        if paths.ndim == 2:
            paths = paths[None, ...]
        color = color or RAY_COLORS[0]

        positions = []
        connections = []
        offset = 0
        for path in paths:
            valid = path[~np.isnan(path).any(axis=1)]
            if valid.shape[0] < 2:
                continue
            positions.append(valid)
            n = valid.shape[0]
            connections.append(
                np.column_stack([np.arange(n - 1), np.arange(1, n)]) + offset
            )
            offset += n
        if not positions:
            return
        positions = np.vstack(positions).astype(np.float32)
        connections = np.vstack(connections).astype(np.uint32)
        line = self._scene.visuals.Line(
            pos=positions,
            connect=connections,
            color=color,
            width=width,
            parent=self.view.scene,
            method="gl",
        )
        line.set_gl_state(blend=True, depth_test=True)
        self._bounds.append(positions)

    def add_field_bundles(self, tracer, fields, *, na_object_sine, samples=24,
                          stop_index=None) -> None:
        """Trace and draw a small pupil bundle per field point."""

        from ..sequential.fields import FieldPoint, PupilSampling, trace_pupil

        for k, field_y in enumerate(fields):
            pupil = trace_pupil(
                tracer,
                FieldPoint(y=float(field_y)),
                na_object_sine=na_object_sine,
                sampling=PupilSampling(kind="rings", radial=3, azimuth=samples // 3),
                keep_paths=True,
                stop_index=stop_index,
            )
            paths = pupil.batch.paths[pupil.valid]
            self.add_paths(paths, color=RAY_COLORS[k % len(RAY_COLORS)])

    # -- display -----------------------------------------------------------

    def frame(self) -> None:
        """Aim the camera at the content bounds."""

        if not self._bounds:
            return
        allpts = np.vstack(self._bounds)
        finite = allpts[np.isfinite(allpts).all(axis=1)]
        center = 0.5 * (finite.min(axis=0) + finite.max(axis=0))
        extent = float(np.max(finite.max(axis=0) - finite.min(axis=0)))
        self.view.camera.center = tuple(center)
        self.view.camera.distance = 1.4 * extent
        self.view.camera.elevation = 18.0
        self.view.camera.azimuth = -60.0

    def snapshot(self) -> np.ndarray:
        """Offscreen render to a uint8 RGBA array."""

        self.frame()
        return self.canvas.render(alpha=True)

    def run(self) -> None:  # pragma: no cover - interactive
        from vispy import app

        self.frame()
        self.canvas.show()
        app.run()


__all__ = ["lathe", "Viewer3D", "GLASS_COLOR", "MIRROR_COLOR"]
