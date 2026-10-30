"""Newtonian telescope: parabolic primary + flat 45° secondary fold mirror.

A paraboloid is exactly stigmatic for the on-axis collimated-to-focus
conjugate — the primary mirror here reuses that same property. The flat
secondary, tilted 45°, intercepts the converging cone before it reaches that
focus and folds it 90° out through the side of the tube, onto a small sensor
at the eyepiece's focal plane. The incoming beam is annular, as in a real
Newtonian: the secondary's silhouette (the central obstruction) carries no
light.

The mirrors are placed elements traced non-sequentially: the light finds the
tilted fold on its own, no sequential path is stated. The figure is the
meridional section, traced with meridional (``fan="y"``) strips of the
annulus; the primary is drawn exactly over its clear aperture (the same
``aperture`` that clips the rays), so the drawn surface and the ray
reflection points coincide.

The folded focus is checked quantitatively: an exact 90° flat fold preserves
the remaining path length, so the new focus must sit at
``(y, z) = (secondary_z, secondary_z)`` given the primary's focus at the
origin. The RMS distance from every folded ray's line to that point
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

from raytracer import OpticalSystem
from raytracer.analysis import point_line_distances, rays_by_generation
from raytracer.elements import Mirror, Sensor
from raytracer.geometry.apertures import Rectangle
from raytracer.geometry.surfaces import Paraboloid, Plane
from raytracer.sources import CollimatedSource
from raytracer.tracing import NonSequentialTracer
from raytracer.viz import Drawing, MarkerItem, SurfaceItem, show
from raytracer.viz.gl.viewer import RenderConfig

F_PRIMARY = 20.0  # the primary's vertex sits at z = F_PRIMARY, its focus at the origin
PRIMARY_SEMI = 4.5
SECONDARY_Z = 5.0  # fold distance in front of the primary's focus
# The 45-degree tilt makes the intercepted cone asymmetric (the far-side
# marginal ray needs the most mirror — why real Newtonian secondaries are
# elliptical): the lower marginal ray here lands 2.0 from the vertex.
SECONDARY_SEMI = 2.1
ANNULUS = (1.7, 4.4)  # radial extent of the incoming beam (clears the obstruction)
FOLDED_FOCUS = np.array([0.0, SECONDARY_Z, SECONDARY_Z])

BEAM_COLOR = (1.0, 1.0, 1.0, 0.9)
MIRROR_COLORS = {"primary": (1.0, 1.0, 1.0, 1.0), "secondary": (1.0, 0.82, 0.35, 1.0)}


def build_telescope() -> OpticalSystem:
    return OpticalSystem([
        Mirror("primary", surface=Paraboloid(f=-F_PRIMARY), aperture=PRIMARY_SEMI, z=F_PRIMARY),
        # Light returning along -z leaves along +y: the normal bisects the two directions.
        Mirror("secondary", surface=Plane(), aperture=SECONDARY_SEMI, z=SECONDARY_Z, normal=(0.0, 1.0, 1.0)),
        Sensor("focal_plane", aperture=Rectangle(2.8, 2.8), pos=FOLDED_FOCUS, normal=(0.0, 1.0, 0.0)),
    ])


def trace(rays_per_strip: int = 10):
    """The annulus in the meridional plane: one strip above the obstruction, one below."""

    inner, outer = ANNULUS
    strips = [
        CollimatedSource((0.0, sign * 0.5 * (inner + outer), -10.0), beam_radius=0.5 * (outer - inner),
                         rays=rays_per_strip, fan="y")
        for sign in (+1.0, -1.0)
    ]
    return NonSequentialTracer(build_telescope(), max_interactions=4, split=False).trace(strips)


def build_drawing(tree) -> Drawing:
    drawing = Drawing(x_lims=(-11.0, 23.0), y_lims=(-6.5, 8.5)).add_system(build_telescope())
    for item in drawing.items:
        if isinstance(item, SurfaceItem):  # light mirrors on the black background
            item.color = MIRROR_COLORS[item.name]
    drawing.add_tree(tree, default_color=BEAM_COLOR, intensities=1.0)
    return drawing.add(MarkerItem(np.array([[FOLDED_FOCUS[2], FOLDED_FOCUS[1]]]), color=(0.3, 1.0, 0.4, 1.0), size=7.0))


def main(rays_per_strip: int = 10) -> None:
    tree = trace(rays_per_strip)
    origins, directions = rays_by_generation(tree, generation=3)  # source -> primary -> secondary -> focus
    distances = point_line_distances(origins, directions, FOLDED_FOCUS)
    rms = float(np.sqrt(np.mean(distances**2))) if len(distances) else float("nan")

    print("Newtonian telescope")
    print(f"  primary: paraboloid, f={F_PRIMARY}, clear aperture semidiameter={PRIMARY_SEMI}")
    print(f"  secondary: flat, 45 deg fold at z={SECONDARY_Z}, semidiameter={SECONDARY_SEMI}")
    print(f"  annular beam: {2 * rays_per_strip} rays, radii {ANNULUS[0]}-{ANNULUS[1]} (central obstruction dark)")
    print(f"  folded focus (predicted): (y, z) = ({FOLDED_FOCUS[1]:.4f}, {FOLDED_FOCUS[2]:.4f})")
    print(f"  rays reaching the fold: {len(origins)}/{2 * rays_per_strip}")
    print(f"  RMS distance to the predicted focus: {rms:.4g}  (stigmatic primary + exact 90 deg fold)")

    save = Path(sys.argv[sys.argv.index("--save") + 1]) if "--save" in sys.argv else None
    backend = show(
        build_drawing(tree), backend="gl", interactive=save is None,
        size=(1400, 640), bgcolor="black", title="Newtonian telescope — raytracer",
        render_config=RenderConfig(ray_width=0.035, min_pixels=1.0, use_solid_rays=True),
    )
    if save is not None:
        backend.save(save)
        print(f"wrote {save}")


if __name__ == "__main__":
    main()
