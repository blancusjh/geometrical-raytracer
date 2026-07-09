"""Ellipse mirror: rays from one focus converge exactly at the other.

Classic stigmatic-mirror demonstration. An ellipse has the property that any
ray leaving one focus and reflecting off the ellipse passes exactly through
the other focus — the two foci are a perfectly stigmatic conjugate pair for
this single reflecting surface (unlike a sphere, which only focuses
paraxially).

The convergence claim is checked, not just asserted: for every reflected ray
we compute the perpendicular distance from the far focus to that ray's
infinite line (:func:`raytracer.analysis.point_line_distances`) and report
the RMS — near zero confirms stigmatism. A physical screen at the focus
would work too, but would clip the direct (unreflected) rays passing nearby
on their way to the mirror; measuring against the ray's line avoids that.

Usage:
    python -m examples.stigmatic_surfaces.ellipse_mirror
    python -m examples.stigmatic_surfaces.ellipse_mirror --save out.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer import EllipseConic, OpenGLViewer, PointSource2D, RayTracer2D, RenderConfig, TraceConfig
from raytracer.analysis import point_line_distances, rays_by_generation

SEMI_MAJOR = 4.0
SEMI_MINOR = 2.5


def build_scene(samples: int = 500):
    a, b = SEMI_MAJOR, SEMI_MINOR
    c = np.sqrt(a * a - b * b)
    near_focus = np.array([0.0, 0.0])
    far_focus = np.array([-2.0 * c, 0.0])

    ellipse = EllipseConic(
        semi_major=a, semi_minor=b, focus=near_focus, surface_id="ellipse_mirror",
    )
    source = PointSource2D(
        origin=near_focus,
        axis_direction=np.array([-1.0, 0.0]),
        aperture=np.deg2rad(80.0),
        samples=samples,
    )
    tracer = RayTracer2D(
        surfaces=[ellipse],
        config=TraceConfig(max_generations=2, allow_reflection=True, allow_refraction=False),
    )
    tree = tracer.trace([source])
    return ellipse, tree, source, far_focus


def main() -> None:
    ellipse, tree, source, far_focus = build_scene(samples=500)

    origins, directions = rays_by_generation(tree, generation=2)
    distances_um = point_line_distances(origins, directions, far_focus) * 1e3
    rms_um = float(np.sqrt(np.mean(distances_um**2)))
    print("Ellipse mirror: focus-to-focus stigmatism")
    print(f"  near focus (source): (0, 0)")
    print(f"  far focus (target):  ({far_focus[0]:.4f}, {far_focus[1]:.4f})")
    print(f"  reflected rays checked: {len(origins)}/{source.samples}")
    print(f"  RMS distance from the far focus to each reflected ray: {rms_um:.4g} um")

    render_config = RenderConfig(ray_width=0.008, sigma_factor=0.5, min_pixels=1.0, use_solid_rays=True)
    viewer = OpenGLViewer(
        x_lims=(-8.0, 1.0), y_lims=(-4.0, 4.0), size=(1200, 800), bgcolor="black",
        render_config=render_config, title="Ellipse mirror (stigmatic) — raytracer",
    )
    viewer.draw_surfaces([ellipse], color="white", width=2.0)
    viewer.draw_markers(far_focus[None, :], color="gold", size=8.0)

    def color_resolver(node):
        if node.intersection is None:
            return (1.0, 0.5, 1.0, 0.5)
        return (1.0, 1.0, 1.0, 1.0)

    viewer.draw_rays(
        tree,
        color_resolver=color_resolver,
        intensity_resolver=lambda node: 1.0 / (1.0 + 0.3 * float(node.generation)),
        show_misses=True,
        marker_color="crimson",
        marker_size=7.0,
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
