"""Showcase reflections on general and special conics using the OpenGL viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer.geometry import (
    CircleConic,
    ConicalDioptrique,
    EllipseConic,
    HyperbolaConic,
    ParabolaConic,
)
from raytracer.sources import PointSource2D
from raytracer.surfaces import Surface2D
from raytracer.tracer import RayTracer2D, TraceConfig
from raytracer.visualization_opengl import OpenGLViewer, RenderConfig

SceneBuilder = Callable[[int], Tuple[Surface2D, PointSource2D, dict]]


def _ellipse_scene(samples: int) -> Tuple[Surface2D, PointSource2D, dict]:
    surface = EllipseConic(semi_major=4.0, semi_minor=2.5, surface_id="ellipse")
    source = PointSource2D(
        origin=np.array([0.0, 0.2]),
        axis_direction=np.array([-1.0, 0.0]),
        aperture=np.deg2rad(80.0),
        samples=samples,
    )
    view = dict(x_lims=(-8.0, 2.0), y_lims=(-4.0, 4.0))
    return surface, source, view


def _parabola_scene(samples: int) -> Tuple[Surface2D, PointSource2D, dict]:
    surface = ParabolaConic(p=3.0, surface_id="parabola")
    source = PointSource2D(
        origin=np.array([-4.0, 0.0]),
        axis_direction=np.array([1.0, 0.1]),
        aperture=np.deg2rad(25.0),
        samples=samples,
    )
    view = dict(x_lims=(-6.0, 6.0), y_lims=(-3.5, 3.5))
    return surface, source, view


def _circle_scene(samples: int) -> Tuple[Surface2D, PointSource2D, dict]:
    surface = CircleConic(radius=3.0, surface_id="sphere")
    source = PointSource2D(
        origin=np.array([-3.5, 0.0]),
        axis_direction=np.array([1.0, 0.0]),
        aperture=np.deg2rad(35.0),
        samples=samples,
    )
    view = dict(x_lims=(-6.0, 4.0), y_lims=(-4.0, 4.0))
    return surface, source, view


def _hyperbola_scene(samples: int) -> Tuple[Surface2D, PointSource2D, dict]:
    surface = HyperbolaConic(semi_major=3.0, semi_minor=1.5, surface_id="hyperbola")
    source = PointSource2D(
        origin=np.array([-5.0, 0.3]),
        axis_direction=np.array([1.0, -0.05]),
        aperture=np.deg2rad(20.0),
        samples=samples,
    )
    view = dict(x_lims=(-7.0, 5.0), y_lims=(-4.5, 4.5))
    return surface, source, view


def _general_scene(samples: int) -> Tuple[Surface2D, PointSource2D, dict]:
    surface = ConicalDioptrique(
        e=0.45,
        p=3.5,
        focus=np.array([0.0, 0.0]),
        angle=np.deg2rad(-12.0),
        surface_id="general_conic",
    )
    source = PointSource2D(
        origin=np.array([-4.5, 0.6]),
        axis_direction=np.array([1.0, -0.1]),
        aperture=np.deg2rad(30.0),
        samples=samples,
    )
    view = dict(x_lims=(-7.0, 5.0), y_lims=(-4.0, 4.0))
    return surface, source, view


SCENES: Dict[str, SceneBuilder] = {
    "general": _general_scene,
    "ellipse": _ellipse_scene,
    "parabola": _parabola_scene,
    "sphere": _circle_scene,
    "hyperbola": _hyperbola_scene,
}


def build_scene(kind: str, samples: int) -> Tuple[Surface2D, PointSource2D, dict]:
    builder = SCENES[kind]
    return builder(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile",
        choices=tuple(SCENES.keys()),
        default="general",
        nargs="?",
        help="Choose which conic configuration to display.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=300,
        help="Number of rays emitted by the point source.",
    )
    args = parser.parse_args()

    surface, source, view = build_scene(args.profile, args.samples)

    tracer = RayTracer2D([surface], TraceConfig(max_generations=3, allow_reflection=True))
    tree = tracer.trace([source])

    render_config = RenderConfig(
        ray_width=0.01,
        sigma_factor=0.02,
        accumulation_mode="squared",
        default_intensity=0.1,
        weight_scale=min(1.0, 80.0 / max(1, args.samples)),
        min_pixels=1.0,
    )

    viewer = OpenGLViewer(
        x_lims=view["x_lims"],
        y_lims=view["y_lims"],
        size=(1200, 800),
        bgcolor="black",
        render_config=render_config,
    )

    viewer.draw_surfaces([surface], color="white", width=2.0)

    def color_resolver(node):
        alpha = 0.9 if node.intersection is not None else 0.4
        return (0.9, 0.9, 1.0, alpha)

    def intensity_resolver(node):
        return 1.0 / (1.0 + 0.3 * (node.generation - 1))

    viewer.draw_rays(
        tree,
        tail_length=12.0,
        color_resolver=color_resolver,
        intensity_resolver=intensity_resolver,
        show_misses=True,
        marker_color="gold",
        marker_size=7.0,
    )

    print("\nConic Showcase")
    print("--------------")
    print(f"Profile: {args.profile}")
    print("Drag with left mouse button to pan. Scroll to zoom.")

    viewer.run()


if __name__ == "__main__":
    main()
