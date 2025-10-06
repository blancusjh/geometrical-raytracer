"""Demonstrate a single Cartesian dioptrique forming a stigmatic pair."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from vispy import scene

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometries import N
from raytracer.dioptrics import CartesianDioptrique
from raytracer.sources import PointSource2D
from raytracer.tracer import RayTracer2D, TraceConfig
from raytracer.visualization import Scene2DViewer


ALPHA_RAYS = 0.55
N_RAYS = 30
N_SAMPLE_CART = 800 # Pronto en desuso, pues se usara intersección exacta.
ZO = 60.0 
ZI = 30.0 


def main() -> None:
    z_object = ZO 
    z_image = ZI 
    n_air = 1.0
    n_glass = 1.52
    aperture_radius = 0.7

    dioptrique = CartesianDioptrique(
        z0=z_object,
        zi=z_image,
        n_exterior=n_air,
        n_interior=n_glass,
        aperture_radius=aperture_radius,
        samples=N_SAMPLE_CART,
        surface_id="cartesian_surface",
    )

    source = PointSource2D(
        origin=np.array([-z_object, 0.0]),
        axis_direction=np.array([1.0, 0.0]),
        aperture=np.deg2rad(4.0),
        samples=N_RAYS,
    )

    config = TraceConfig(
        max_generations=4,
        allow_reflection=True,
        allow_refraction=True,
        ambient_n=n_air,
    )

    tracer = RayTracer2D([dioptrique], config)
    tree = tracer.trace([source])

    ray_alpha = ALPHA_RAYS 

    viewer = Scene2DViewer(
        x_lims=(-65.0, 35.0),
        y_lims=(-6.0, 6.0),
        line_method="agg",
    )
    # viewer = Scene2DViewer(x_lims=(-65.0, 35.0), y_lims=(-6.0, 6.0), line_method="gl")
    viewer.draw_surfaces([dioptrique])
    viewer.draw_rays(
        tree,
        width=0.35,
        extend_mode="axis",
        show_misses=False,
        hit_color=(1.0, 1.0, 0.0, ray_alpha),
    )

    object_point = np.array([-z_object, 0.0])
    image_point = np.array([z_image, 0.0])

    focus_markers = np.vstack([object_point, image_point])
    colors = np.array([[0.3, 0.9, 0.3, 1.0], [0.1, 0.7, 1.0, 1.0]])
    scene.visuals.Markers(pos=focus_markers, size=10, face_color=colors, parent=viewer.view.scene)

    viewer.run()


if __name__ == "__main__":
    main()
