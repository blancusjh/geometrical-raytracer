# examples/refractive_ovoid.py
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer import CartesianOvoid2D
from raytracer import PointSource2D
from raytracer import RayTracer2D, TraceConfig
from raytracer import OpenGLViewer, RenderConfig

AMBIENT_N = 1.0
GLASS_N = 1.7
APERTURE_DEG = 40

def build_scene(samples: int = 120):
    """
    Cartesian Ovoid: Single refracting surface demonstrating perfect stigmatism.
    - Object at z0 in air (n=1.0)
    - Image at zi in glass (n=1.7)
    - All rays from z0 focus perfectly to zi
    """
    z0 = -30.0  # Object position (in air)
    zi = 10.0   # Image position (in glass)

    # Optional epsilon offsets for source position
    epsilon_x = 0.0
    epsilon_y = 0.0

    ovoid = CartesianOvoid2D(
        z0=z0,
        zi=zi,
        n_exterior=AMBIENT_N,
        n_interior=GLASS_N,
        origin=np.array([0.0, 0.0]),
        angle=0.0,
        surface_id="cartesian_ovoid",
    )

    # Conical (point) source at object position
    source = PointSource2D(
        origin=np.array([z0 + epsilon_x, 0.0 + epsilon_y]),
        axis_direction=np.array([1.0, 0.0]),
        aperture=np.deg2rad(APERTURE_DEG),
        samples=samples,
    )

    tracer = RayTracer2D(
        surfaces=[ovoid],
        config=TraceConfig(
            max_generations=2,  # Generation 1=incident, Generation 2=refracted (in glass)
            allow_reflection=False,
            allow_refraction=True,
            ambient_n=AMBIENT_N,
        ),
    )

    tree = tracer.trace([source])
    return ovoid, tree, source, z0, zi


def main():
    ovoid, tree, source, z0, zi = build_scene(samples=100)

    # Solid ray configuration for clear stigmatism visualization
    cfg = RenderConfig(
        ray_width=0.001,
        sigma_factor=0.5,
        min_pixels=1.0,
    )

    viewer = OpenGLViewer(
        x_lims=(-12.0, 25.0),
        y_lims=(-8.0, 8.0),
        size=(1400, 800),
        bgcolor="black",
        render_config=cfg,
    )

    # Note: Surface drawing uses parametric form which is placeholder for now
    # The actual surface (for ray tracing) is correct via Fermat principle
    viewer.draw_surfaces([ovoid], color="white", width=7.0)

    def color(node):
        # Generation 2 = refracted rays in glass (cyan)
        if node.generation == 2:
            return (0.5, 0.9, 1.0, 0.8)
        # Generation 1 = incident rays in air (white)
        return (1.0, 1.0, 1.0, 0.4)

    viewer.draw_rays(tree, color_resolver=color, marker_color="gold", marker_size=6.0)

    print(f"\n{'='*70}")
    print(f"Cartesian Ovoid - Perfect Stigmatism Demo")
    print(f"{'='*70}")
    print(f"Object at:    z0={z0} (in air, n={AMBIENT_N})")
    print(f"Image at:     zi={zi} (in glass, n={GLASS_N})")
    print(f"Source:       Point source at z0 with {source.samples} rays")
    print(f"\nAll refracted rays (cyan) focus perfectly to zi={zi}")
    print(f"{'='*70}\n")

    viewer.run()


if __name__ == "__main__":
    main()
