"""Newtonian telescope: parabolic primary + flat 45° secondary fold mirror.

A parabola is exactly stigmatic for the on-axis collimated-to-focus
conjugate — the primary mirror here reuses that same property. The flat
secondary, tilted 45°, intercepts the converging cone before it reaches that
focus and folds it 90° out through the side of the tube, onto a small
observation screen at the eyepiece's focal plane. The incoming beam is
annular, as in a real Newtonian: the secondary's silhouette (the central
obstruction) carries no light.

Built on the 2-D branching propagation, because the sequential propagation is
on-axis only and cannot represent a tilted fold mirror. The primary is drawn
exactly over its used clear aperture (the same ``semidiameter`` that clips
the rays), so the drawn surface and the ray reflection points coincide.

The folded focus is checked quantitatively: an exact 90° flat fold preserves
the remaining path length, so the new focus must sit at
``(secondary_x, secondary_x)`` given the primary's focus at the origin. The
RMS distance from every folded ray's line to that point
(:func:`raytracer.analysis.point_line_distances`) is printed and reads ~0.

Usage:
    python -m examples.telescopes.newtonian
    python -m examples.telescopes.newtonian --save out.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer import BranchingTracer, ParabolaSurface, ParallelSource, RenderConfig, TraceConfig
from raytracer.analysis import point_line_distances, rays_by_generation
from raytracer.optics import Mirror, Screen
from raytracer.viz import Scene, show
from raytracer.viz.scene import MarkerItem, SurfaceItem

F_PRIMARY = 20.0
SEMI_LATUS = 2.0 * F_PRIMARY  # polar parabola: r = p / (1 + cos theta), focus at origin
PRIMARY_SEMI = 4.5
SECONDARY_X = 5.0  # fold distance in front of the primary's focus
# The 45-degree tilt makes the intercepted cone asymmetric (the far-side
# marginal ray needs the most mirror — why real Newtonian secondaries are
# elliptical): the lower marginal ray here lands 2.0 from the vertex.
SECONDARY_SEMI = 2.1
ANNULUS = (1.7, 4.4)  # radial extent of the incoming beam (clears the obstruction)

BEAM_COLOR = (1.0, 1.0, 1.0, 0.9)


def primary_arc(samples: int = 400) -> np.ndarray:
    """The primary's meridional curve over exactly its clear aperture.

    For the polar parabola, height y = p*tan(theta/2), so the aperture edge
    |y| = PRIMARY_SEMI maps to theta = 2*atan(PRIMARY_SEMI / p).
    """

    theta_max = 2.0 * np.arctan(PRIMARY_SEMI / SEMI_LATUS)
    parabola = ParabolaSurface(p=SEMI_LATUS, focus=np.array([0.0, 0.0]))
    return parabola.as_points(samples=samples, theta_span=(-theta_max, theta_max))


def build_scene(samples_per_strip: int = 10):
    primary = ParabolaSurface(
        p=SEMI_LATUS, focus=np.array([0.0, 0.0]), surface_id="primary",
        semidiameter=PRIMARY_SEMI,
    )
    secondary = Mirror.from_radius(
        radius=0.0, semidiameter=SECONDARY_SEMI, vertex=(SECONDARY_X, 0.0),
        angle=np.deg2rad(45.0), name="secondary",
    ).face
    folded_focus = np.array([SECONDARY_X, SECONDARY_X])
    screen = Screen(
        p0=folded_focus + np.array([-1.4, 0.0]), p1=folded_focus + np.array([1.4, 0.0]),
        surface_id="focal_screen",
    )

    inner, outer = ANNULUS
    mid, width = 0.5 * (inner + outer), outer - inner
    strips = [
        ParallelSource(origin=np.array([-10.0, sign * mid]), direction=np.array([1.0, 0.0]),
                         width=width, samples=samples_per_strip)
        for sign in (+1.0, -1.0)
    ]

    tracer = BranchingTracer(
        [primary, secondary, screen],
        TraceConfig(max_generations=4, allow_reflection=True, fresnel_split=False),
    )
    tree = tracer.trace(strips)

    scene = Scene(x_lims=(-11.0, 23.0), y_lims=(-6.5, 8.5))
    scene.add(SurfaceItem(polyline=primary_arc(), color=(1.0, 1.0, 1.0, 1.0), width=2.5))
    scene.add(SurfaceItem(polyline=secondary.polyline(), color=(1.0, 0.82, 0.35, 1.0), width=3.0))
    scene.add_screen(screen)
    scene.add_tree(tree, default_color=BEAM_COLOR)
    scene.add(MarkerItem(points=folded_focus[None, :], color=(0.3, 1.0, 0.4, 1.0), size=7.0))
    return scene, tree, folded_focus, 2 * samples_per_strip


def main() -> None:
    scene, tree, folded_focus, n_rays = build_scene()

    origins, directions = rays_by_generation(tree, generation=3)
    distances = point_line_distances(origins, directions, folded_focus)
    rms = float(np.sqrt(np.mean(distances**2))) if len(distances) else float("nan")

    print("Newtonian telescope")
    print(f"  primary: parabola, f={F_PRIMARY}, clear aperture semidiameter={PRIMARY_SEMI}")
    print(f"  secondary: flat, 45 deg fold at x={SECONDARY_X}, semidiameter={SECONDARY_SEMI}")
    print(f"  annular beam: {n_rays} rays, radii {ANNULUS[0]}-{ANNULUS[1]} (central obstruction dark)")
    print(f"  folded focus (predicted): ({folded_focus[0]:.4f}, {folded_focus[1]:.4f})")
    print(f"  rays reaching the fold: {len(origins)}/{n_rays}")
    print(f"  RMS distance to the predicted focus: {rms:.4g}  (stigmatic primary + exact 90 deg fold)")

    render_config = RenderConfig(ray_width=0.035, min_pixels=1.0, use_solid_rays=True)
    save_path = None
    if "--save" in sys.argv:
        save_path = Path(sys.argv[sys.argv.index("--save") + 1])
    backend = show(
        scene, backend="gl", interactive=save_path is None,
        size=(1400, 640), bgcolor="black", render_config=render_config,
        title="Newtonian telescope — raytracer",
    )
    if save_path is not None:
        backend.save(save_path)
        print(f"wrote {save_path}")


if __name__ == "__main__":
    main()
