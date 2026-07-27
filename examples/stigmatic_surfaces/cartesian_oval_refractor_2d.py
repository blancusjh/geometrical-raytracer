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

The source cone is matched to the surface's clear aperture, the way a real
system's stop would be, so every emitted ray lands on the surface. That
aperture has a hard geometric ceiling: the oval is a closed surface whose
half-width peaks at ``max_usable_height`` (4.858 mm for this conjugate pair)
and then turns back toward the axis, so rays wider than 7.69 deg from the
object miss it entirely no matter how large a ``semidiameter`` is declared.

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
Z0 = -30.0  # object position, in air
ZI = 10.0  # image position, inside the glass
SEMIDIAMETER = 4.5  # clear aperture, inside the oval's 4.858 geometric ceiling


def aperture_filling(oval, semidiameter: float, *, ceiling_deg: float = 45.0) -> float:
    """Full cone angle (radians) whose marginal ray lands exactly on the rim.

    Solved rather than tabulated: emitting any wider would only add rays that
    never reach the surface, and the answer moves with the conjugates.
    """

    from raytracer.optics.ray import Ray

    def height(theta: float) -> float:
        hit = oval.hit(Ray(origin=[Z0, 0.0], direction=[np.cos(theta), np.sin(theta)]))
        return abs(hit.point[1]) if hit is not None else np.inf

    lo, hi = 0.0, np.deg2rad(ceiling_deg)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if height(mid) <= semidiameter:
            lo = mid
        else:
            hi = mid
    return 2.0 * lo


def build_scene(samples: int = 300):
    oval = CartesianOvalSurface(
        z0=Z0, zi=ZI, n_exterior=AMBIENT_N, n_interior=GLASS_N,
        semidiameter=SEMIDIAMETER, surface_id="cartesian_oval",
    )
    source = PointSource(
        origin=np.array([Z0, 0.0]),
        axis_direction=np.array([1.0, 0.0]),
        aperture=aperture_filling(oval, SEMIDIAMETER),
        samples=samples,
    )
    tracer = BranchingTracer(
        surfaces=[oval],
        config=TraceConfig(
            max_generations=2,  # 1 = incident (air), 2 = refracted (glass)
            allow_reflection=False,
            allow_refraction=True,
            ambient_n=AMBIENT_N,
        ),
    )
    tree = tracer.trace([source])
    return oval, tree, source


def main() -> None:
    oval, tree, source = build_scene(samples=300)

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
    viewer.draw_surfaces([oval], color="white", width=2.5)
    viewer.draw_markers(image_point[None, :], color="gold", size=8.0)

    def color_resolver(node):
        return (0.5, 0.9, 1.0, 0.85) if node.generation == 2 else (1.0, 1.0, 1.0, 0.45)

    def leaf_extension(node):
        """Stop the refracted rays a little past the focus, so the crossing is
        visible without a long meaningless tail running off the view."""

        return np.asarray(node.ray.origin) + np.asarray(node.ray.direction) * 26.0

    viewer.draw_rays(
        tree, color_resolver=color_resolver, marker_color="crimson", marker_size=6.0,
        leaf_extension=leaf_extension,
    )

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
