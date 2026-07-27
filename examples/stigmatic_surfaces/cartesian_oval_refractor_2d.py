"""Cartesian oval: a single refracting surface with perfect stigmatism.

Descartes' construction gives a surface shape (not a sphere) that images an
object at ``z0`` in air exactly onto ``zi`` inside the glass — every ray from
the object point refracts through precisely the same image point, with no
spherical aberration at any aperture. ``CartesianOvalSurface`` traces this via
Fermat's principle: ``n0 * dist(object, surface) + ni * dist(surface, image)
= const``.

The convergence claim is checked, not just asserted: the RMS perpendicular
distance from the image point to each refracted ray's line
(:func:`raytracer.analysis.point_line_distances`) is printed alongside the
diagram — near zero confirms stigmatism.

The surface's clear aperture (``semidiameter``) clips rays that would land
outside it, exactly like a real lens edge (see ``raytracer.surfaces.conic``/``cartesian_oval`` aperture clipping).

Usage:
    python -m examples.stigmatic_surfaces.cartesian_oval_refractor_2d
    python -m examples.stigmatic_surfaces.cartesian_oval_refractor_2d --save out.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer import BranchingTracer, CartesianOvalSurface, OpenGLViewer, PointSource, RenderConfig, TraceConfig
from raytracer.analysis import point_line_distances, rays_by_generation

AMBIENT_N = 1.0
GLASS_N = 1.7
APERTURE_DEG = 40.0
SEMIDIAMETER = 5.0
Z0 = -30.0  # object position, in air
ZI = 10.0  # image position, inside the glass


def build_scene(samples: int = 300):
    ovoid = CartesianOvalSurface(
        z0=Z0, zi=ZI, n_exterior=AMBIENT_N, n_interior=GLASS_N,
        semidiameter=SEMIDIAMETER, surface_id="cartesian_oval",
    )
    source = PointSource(
        origin=np.array([Z0, 0.0]),
        axis_direction=np.array([1.0, 0.0]),
        aperture=np.deg2rad(APERTURE_DEG),
        samples=samples,
    )
    tracer = BranchingTracer(
        surfaces=[ovoid],
        config=TraceConfig(
            max_generations=2,  # 1 = incident (air), 2 = refracted (glass)
            allow_reflection=False,
            allow_refraction=True,
            ambient_n=AMBIENT_N,
        ),
    )
    tree = tracer.trace([source])
    return ovoid, tree, source


def main() -> None:
    ovoid, tree, source = build_scene(samples=300)

    origins, directions = rays_by_generation(tree, generation=2)
    image_point = np.array([ZI, 0.0])
    distances_um = point_line_distances(origins, directions, image_point) * 1e3
    rms_um = float(np.sqrt(np.mean(distances_um**2))) if len(origins) else float("nan")
    print("Cartesian oval refractor: stigmatic imaging")
    print(f"  object: z0={Z0} (air, n={AMBIENT_N}); image: zi={ZI} (glass, n={GLASS_N})")
    print(f"  clear aperture (semidiameter): {SEMIDIAMETER}")
    print(f"  refracted rays within the aperture: {len(origins)}/{source.samples}")
    print(f"  RMS distance from the image point to each refracted ray: {rms_um:.4g} um")

    render_config = RenderConfig(ray_width=0.02, sigma_factor=0.5, min_pixels=1.0, use_solid_rays=True)
    viewer = OpenGLViewer(
        x_lims=(-32.0, 14.0), y_lims=(-8.0, 8.0), size=(1400, 700), bgcolor="black",
        render_config=render_config, title="Cartesian oval refractor (stigmatic) — raytracer",
    )
    viewer.draw_surfaces([ovoid], color="white", width=2.5)
    viewer.draw_markers(image_point[None, :], color="gold", size=8.0)

    def color_resolver(node):
        return (0.5, 0.9, 1.0, 0.85) if node.generation == 2 else (1.0, 1.0, 1.0, 0.45)

    viewer.draw_rays(tree, color_resolver=color_resolver, marker_color="crimson", marker_size=6.0)

    if "--save" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--save") + 1])
        import matplotlib.image as mpimg

        mpimg.imsave(out, viewer.snapshot())
        print(f"wrote {out}")
        return

    print("Drag with left mouse button to pan. Scroll to zoom.")
    viewer.run()


if __name__ == "__main__":
    main()
