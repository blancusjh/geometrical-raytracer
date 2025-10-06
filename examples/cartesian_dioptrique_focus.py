"""Demonstrate a single Cartesian dioptrique forming a stigmatic pair."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from vispy import scene

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer.dioptrics import CartesianDioptrique
from raytracer.sources import PointSource2D
from raytracer.tracer import RayTracer2D, TraceConfig
from raytracer.visualization import Scene2DViewer


def main() -> None:
    z_object = 60.0
    z_image = 30.0
    n_air = 1.0
    n_glass = 1.52
    aperture_radius = 0.7

    dioptrique = CartesianDioptrique(
        z0=z_object,
        zi=z_image,
        n_exterior=n_air,
        n_interior=n_glass,
        aperture_radius=aperture_radius,
        samples=800,
        surface_id="cartesian_surface",
    )

    source = PointSource2D(
        origin=np.array([-z_object, 0.0]),
        axis_direction=np.array([1.0, 0.0]),
        aperture=np.deg2rad(4.0),
        samples=13,
    )

    config = TraceConfig(
        max_generations=4,
        allow_reflection=False,
        allow_refraction=True,
        ambient_n=n_air,
    )

    tracer = RayTracer2D([dioptrique], config)
    tree = tracer.trace([source])

    viewer = Scene2DViewer(x_lims=(-65.0, 35.0), y_lims=(-6.0, 6.0))
    viewer.draw_surfaces([dioptrique])
    viewer.draw_rays(tree, width=1.0, extend_mode="axis", show_misses=False)

    object_point = np.array([-z_object, 0.0])
    image_point = np.array([z_image, 0.0])
    focus_markers = np.vstack([object_point, image_point])
    colors = np.array([[0.3, 0.9, 0.3, 1.0], [0.1, 0.7, 1.0, 1.0]])
    scene.visuals.Markers(pos=focus_markers, size=10, face_color=colors, parent=viewer.view.scene)

    viewer.run()


if __name__ == "__main__":
    main()
