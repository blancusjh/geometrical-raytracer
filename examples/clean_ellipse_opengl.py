"""Standalone ellipse demo using the simplified OpenGL viewer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer import EllipseConic
from raytracer import PointSource2D
from raytracer import RayTracer2D, TraceConfig
from raytracer import OpenGLViewer, RenderConfig


def build_scene(samples: int = 500):
    ellipse = EllipseConic(
        semi_major=4.0,
        semi_minor=2.5,
        focus=np.array([0.0, 0.0]),
        surface_id="mirror",
    )

    source = PointSource2D(
        origin=np.array([0.0, 0.11]),
        axis_direction=np.array([-1.0, 0.0]),
        aperture=np.deg2rad(80.0),
        samples=samples,
    )

    tracer = RayTracer2D(
        surfaces=[ellipse],
        config=TraceConfig(max_generations=3, allow_reflection=True, allow_refraction=False),
    )
    tree = tracer.trace([source])
    return ellipse, tree, source


def main() -> None:
    ellipse, tree, source = build_scene(samples=500)

    # HDR accumulation + auto-exposure: no per-sample weight tuning needed.
    render_config = RenderConfig(
        ray_width=0.008,
        sigma_factor=0.5,
        min_pixels=1.0,
    )

    viewer = OpenGLViewer(
        x_lims=(-8.0, 1.0),
        y_lims=(-4.0, 4.0),
        size=(1200, 800),
        bgcolor="black",
        render_config=render_config,
    )

    viewer.draw_surfaces([ellipse], color="white", width=2.0)

    def color_resolver(node):
        if node.intersection is None:
            return (1.0, 0.5, 1.0, 0.5)
        return (1.0, 1.0, 1.0, 1.0)

    def intensity_resolver(node):
        return 1.0 / (1.0 + 0.3 * float(node.generation))

    viewer.draw_rays(
        tree,
        color_resolver=color_resolver,
        intensity_resolver=intensity_resolver,
        show_misses=True,
        marker_color="crimson",
        marker_size=7.0,
    )

    print("\nSimple Ellipse Demo")
    print("-------------------")
    print("Drag with left mouse button to pan. Scroll to zoom.")

    viewer.run()


if __name__ == "__main__":
    main()
