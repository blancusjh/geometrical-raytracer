"""Trace two matched Cartesian dioptriques composing a stigmatic singlet."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from vispy import scene

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer.dioptrics import CartesianSinglet
from raytracer.sources import PointSource2D
from raytracer.tracer import RayTracer2D, TraceConfig
from raytracer.visualization import Scene2DViewer


def main() -> None:
    z_object = 60.0
    z_intermediate = 5.0
    z_image = 40.0
    thickness = 0.03
    n_air = 1.0
    n_glass = 1.52
    aperture_radius = 0.6

    singlet = CartesianSinglet(
        z0=z_object,
        zc=z_intermediate,
        zi=z_image,
        n0=n_air,
        n_lens=n_glass,
        n_out=n_air,
        thickness=thickness,
        aperture_radius=aperture_radius,
        samples=900,
    )
    front, back = singlet.surfaces()

    source = PointSource2D(
        origin=np.array([-z_object, 0.0]),
        axis_direction=np.array([1.0, 0.0]),
        aperture=np.deg2rad(3.0),
        samples=11,
    )

    config = TraceConfig(
        max_generations=4,
        allow_reflection=False,
        allow_refraction=True,
        ambient_n=n_air,
    )

    tracer = RayTracer2D([front, back], config)
    tree = tracer.trace([source])

    ray_alpha = 0.55

    viewer = Scene2DViewer(
        x_lims=(-65.0, thickness + z_image + 5.0),
        y_lims=(-6.0, 6.0),
        line_method="agg",
    )
    # viewer = Scene2DViewer(x_lims=(-65.0, thickness + z_image + 5.0), y_lims=(-6.0, 6.0), line_method="gl")
    viewer.draw_surfaces([front, back])
    viewer.draw_rays(
        tree,
        width=0.35,
        extend_mode="axis",
        show_misses=False,
        hit_color=(1.0, 1.0, 0.0, ray_alpha),
    )

    object_pt = np.array([-z_object, 0.0])
    intermediate_pt = np.array([z_intermediate, 0.0])
    final_pt = np.array([thickness + z_image, 0.0])
    markers = np.vstack([object_pt, intermediate_pt, final_pt])
    colors = np.array([
        [0.3, 0.9, 0.3, 1.0],
        [1.0, 0.5, 0.1, 1.0],
        [0.1, 0.7, 1.0, 1.0],
    ])
    scene.visuals.Markers(pos=markers, size=10, face_color=colors, parent=viewer.view.scene)

    viewer.run()


if __name__ == "__main__":
    main()
