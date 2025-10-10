"""Demo tracing reflections inside a conic up to five generations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from vispy import scene

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer.geometry import EllipseConic
from raytracer.sources import PointSource2D
from raytracer.tracer import RayTracer2D, TraceConfig
from raytracer.visualization import Scene2DViewer


def main() -> None:

    semi_major = 4.0
    semi_minor = 2.5

    ellipse = EllipseConic(
        semi_major=semi_major,
        semi_minor=semi_minor,
        focus=np.array([0.0, 0.0]),
        surface_id="mirror",
    )

    primary_focus = ellipse.focus.copy()
    major_axis_dir = ellipse.frame.direction_to_world(np.array([1.0, 0.0]))
    major_axis_dir = major_axis_dir / np.linalg.norm(major_axis_dir)
    focal_offset = np.sqrt(semi_major ** 2 - semi_minor ** 2)
    secondary_focus = primary_focus - 2.0 * focal_offset * major_axis_dir


    source = PointSource2D(
        origin=np.array([0.0, 0.11]),
        axis_direction=-major_axis_dir,
        aperture=np.deg2rad(80.0),
        samples=5000,
    )

    config = TraceConfig(max_generations=3, allow_reflection=True, allow_refraction=False)
    tracer = RayTracer2D(surfaces=[ellipse], config=config)
    tree = tracer.trace([source])

    ray_alpha = 45.10/source.samples

    viewer = Scene2DViewer(
        x_lims=(-8.0, 1.0),
        y_lims=(-4.0, 4.0),
        line_method="agg",
    )
    # viewer = Scene2DViewer(x_lims=(-8.0, 1.0), y_lims=(-4.0, 4.0), line_method="gl")
    viewer.draw_surfaces([ellipse])
    viewer.draw_rays(
        tree,
        width=0.35,
        hit_color=(1.0, 1.0, 1.0, ray_alpha),
        miss_color=(1.0, 0.5, 1.0, ray_alpha),
    )

    focus_markers = np.vstack([primary_focus, secondary_focus])
    colors = np.array([[0.3, 0.9, 0.3, 1.0], [0.1, 0.7, 1.0, 1.0]])
    scene.visuals.Markers(pos=focus_markers, size=5, face_color=colors, parent=viewer.view.scene)

    viewer.run()


if __name__ == "__main__":
    main()
