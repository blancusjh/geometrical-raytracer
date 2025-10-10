"""VisPy-based helpers for interactive inspection of 3D ray-surface scenes."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .geometry3d import AxisymmetricCartesianSurface3D, AxisymmetricConicSurface3D
from .rays import IntersectionND, RayND


def _require_vispy():
    try:
        from vispy import app, scene
    except ImportError as exc:  # pragma: no cover - relies on optional dependency
        raise RuntimeError(
            "VisPy is required for interactive visualization. Install it with 'pip install vispy'."
        ) from exc
    return app, scene


def _revolve_generatrix(curve: np.ndarray, phi_samples: int = 128) -> tuple[np.ndarray, np.ndarray]:
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 2 or curve.shape[1] != 2:
        raise ValueError("curve must be an (N, 2) array of (x, rho) samples")
    if phi_samples < 3:
        raise ValueError("phi_samples must be >= 3")

    n_curve = curve.shape[0]
    phis = np.linspace(0.0, 2.0 * np.pi, phi_samples, endpoint=False)
    cos_phi = np.cos(phis)
    sin_phi = np.sin(phis)

    vertices = np.empty((n_curve, phi_samples, 3), dtype=float)
    for i, (x, rho) in enumerate(curve):
        vertices[i, :, 0] = x
        vertices[i, :, 1] = rho * cos_phi
        vertices[i, :, 2] = rho * sin_phi

    vertices_flat = vertices.reshape(-1, 3)
    faces: List[List[int]] = []
    for i in range(n_curve - 1):
        for j in range(phi_samples):
            jp = (j + 1) % phi_samples
            idx00 = i * phi_samples + j
            idx01 = i * phi_samples + jp
            idx10 = (i + 1) * phi_samples + j
            idx11 = (i + 1) * phi_samples + jp
            faces.append([idx00, idx10, idx11])
            faces.append([idx00, idx11, idx01])

    faces_array = np.asarray(faces, dtype=np.uint32)
    return vertices_flat, faces_array


def _surface_generatrix(
    surface: AxisymmetricConicSurface3D | AxisymmetricCartesianSurface3D,
    samples: int,
    rho_max: float | None,
) -> np.ndarray:
    if isinstance(surface, AxisymmetricConicSurface3D):
        return surface.generatrix_polyline(samples=samples, rho_max=rho_max)
    if isinstance(surface, AxisymmetricCartesianSurface3D):
        return surface.generatrix_polyline(samples=samples)
    raise TypeError(f"Unsupported surface type {type(surface)!r}")


def visualize_axisymmetric_scene(
    *,
    surfaces: Sequence[AxisymmetricConicSurface3D | AxisymmetricCartesianSurface3D],
    rays: Sequence[RayND],
    hits: Sequence[IntersectionND | None] | None = None,
    surface_options: Dict[str, Dict[str, float]] | None = None,
    ray_lengths: Sequence[float | None] | None = None,
    ray_colors: Sequence[tuple[float, float, float, float]] | None = None,
    title: str | None = "RayTracer 3D",
    default_color: tuple[float, float, float, float] = (0.2, 0.6, 0.8, 0.35),
    phi_samples: int = 128,
    camera_center: np.ndarray | None = None,
    allow_marker_toggle: bool = False,
) -> None:
    """Render surfaces and rays in an interactive VisPy canvas."""

    app, scene = _require_vispy()

    canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="#111")
    if title:
        canvas.title = title
    view = canvas.central_widget.add_view()
    turntable_cam = scene.cameras.TurntableCamera(fov=45.0, azimuth=35.0, elevation=25.0)
    fly_cam = scene.cameras.FlyCamera(fov=45.0)
    view.camera = turntable_cam  # Initial camera

    def on_key_press(event):
        if event.key == 'p':
            if view.camera is turntable_cam:
                view.camera = fly_cam
                print("Switched to Fly Camera. Use mouse and WASD/arrow keys to move.")
            else:
                view.camera = turntable_cam
                print("Switched to Turntable Camera.")
        elif event.key == 'v' and allow_marker_toggle:
            if origin_markers is not None:
                origin_markers.visible = not origin_markers.visible
            if hit_markers is not None:
                hit_markers.visible = not hit_markers.visible
            
            visible_state = origin_markers.visible if origin_markers is not None else (hit_markers.visible if hit_markers is not None else False)
            print(f"Markers visibility toggled to: {visible_state}")
            canvas.update()

    canvas.events.key_press.connect(on_key_press)
    print("Press 'c' to toggle between Turntable and Fly cameras.")

    scene.visuals.XYZAxis(parent=view.scene, width=2)

    scene.visuals.XYZAxis(parent=view.scene, width=2)

    all_points: List[np.ndarray] = []
    extent_hint = 1.0
    surface_options = surface_options or {}

    for surface in surfaces:
        opts = surface_options.get(surface.surface_id, {})
        curve_samples = int(opts.get("generatrix_samples", 256))
        rho_max = opts.get("rho_max")
        mesh_color = opts.get("color", default_color)

        curve = _surface_generatrix(surface, curve_samples, rho_max)
        if len(curve) < 2:
            continue
        vertices, faces = _revolve_generatrix(curve, phi_samples=phi_samples)
        mesh = scene.visuals.Mesh(
            vertices=vertices.astype(np.float32),
            faces=faces,
            color=mesh_color,
            shading="smooth",
            parent=view.scene,
        )
        mesh.set_gl_state("translucent", depth_test=True, cull_face=False)
        all_points.append(vertices)
        extent_hint = max(extent_hint, np.max(np.linalg.norm(vertices, axis=1)))

    default_ray_color = (0.9, 0.3, 0.2, 1.0)
    ray_lengths = list(ray_lengths) if ray_lengths is not None else [None] * len(rays)
    if ray_colors is None:
        ray_colors = [default_ray_color] * len(rays)

    origin_points: List[np.ndarray] = []
    ray_points: List[np.ndarray] = []
    segment_colors: List[tuple[float, float, float, float]] = []
    for idx, ray in enumerate(rays):
        length = ray_lengths[idx] if idx < len(ray_lengths) else None
        color_rgba = ray_colors[idx] if idx < len(ray_colors) else default_ray_color
        if length is None:
            length = 1.5 * extent_hint

        start_point = ray.origin
        end_point = ray.origin + ray.direction * length
        ray_points.extend([start_point, end_point])
        segment_colors.append(color_rgba)
        origin_points.append(ray.origin.astype(np.float32))

    origin_markers: scene.visuals.Markers | None = None
    hit_markers: scene.visuals.Markers | None = None

    if ray_points:
        # Create a single Line visual for all ray segments.
        # This is much more efficient than creating one visual per ray.
        vertex_colors = np.repeat(np.array(segment_colors, dtype=np.float32), 2, axis=0)

        line_segments = scene.visuals.Line(
            pos=np.array(ray_points, dtype=np.float32),
            color=vertex_colors,
            width=4.0,
            method="gl",
            connect="segments",
            parent=view.scene,
        )
        line_segments.set_gl_state(blend=True, depth_test=False)
        all_points.append(np.array(ray_points, dtype=np.float32))

    if origin_points:
        origin_markers = scene.visuals.Markers(parent=view.scene)
        origin_markers.set_data(
            np.vstack(origin_points),
            face_color=(0.95, 0.4, 0.4, 1.0),
            size=8.0,
        )
        origin_markers.set_gl_state(blend=True, depth_test=False)

    if hits:
        hit_points = [hit.point for hit in hits if hit is not None]
        if hit_points:
            pts = np.asarray(hit_points, dtype=np.float32)
            hit_markers = scene.visuals.Markers(parent=view.scene)
            hit_markers.set_data(pts, face_color=(0.5, 0.5, 0.5, 0.75), size=5.0)
            hit_markers.set_gl_state(blend=True, depth_test=False)
            all_points.append(pts)

    if all_points:
        stacked = np.vstack(all_points)
        extent = np.max(np.linalg.norm(stacked, axis=1))
        extent = max(extent, 1.0)
        turntable_cam.distance = 2.5 * extent
        if camera_center is not None:
            turntable_cam.center = camera_center
        else:
            turntable_cam.center = stacked.mean(axis=0)

    canvas.app.process_events()
    app.run()
