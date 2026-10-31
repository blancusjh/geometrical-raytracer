"""Ellipse mirror: rays from one focus converge exactly at the other.

Classic stigmatic-mirror demonstration. An ellipse has the property that any
ray leaving one focus and reflecting off the ellipse passes exactly through
the other focus — the two foci are a perfectly stigmatic conjugate pair for
this single reflecting surface (unlike a sphere, which only focuses
paraxially).

The mirror is a closed ellipsoid of revolution, built as its two vertex
halves (``Ellipsoid.from_semi_axes``, each out to the equator) facing each
other, and drawn in its meridional section. A fan of 80° from the right
focus, traced non-sequentially, fills the left half and spills just past the
equator onto the right one.

The convergence claim is checked, not just asserted: for every reflected ray
we compute the perpendicular distance from the far focus to that ray's
infinite line (:func:`raytracer.analysis.point_line_distances`) and report
the RMS — near zero confirms stigmatism (its ~1e-11 µm floor comes from the
few rays that land next to the equator, where the surface normal is read
from a sag slope that diverges there). A sensor at the focus would work
too, but would clip the direct (unreflected) rays passing nearby on their
way to the mirror; measuring against the ray's line avoids that.

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

from raytracer import OpticalSystem
from raytracer.analysis import point_line_distances, rays_by_generation
from raytracer.elements import Mirror
from raytracer.geometry.surfaces import Ellipsoid
from raytracer.sources import PointSource
from raytracer.tracing import NonSequentialTracer
from raytracer.viz import Drawing, MarkerItem, show
from raytracer.viz.gl.viewer import RenderConfig

SEMI_MAJOR = 4.0
SEMI_MINOR = 2.5
C = np.sqrt(SEMI_MAJOR**2 - SEMI_MINOR**2)  # centre-to-focus distance
NEAR_FOCUS = np.array([0.0, 0.0, 0.0])  # the source
FAR_FOCUS = np.array([0.0, 0.0, -2.0 * C])


def build_mirror() -> OpticalSystem:
    """The ellipsoid with its foci at z = 0 and z = -2c: two halves meeting at the equator z = -c."""

    a, b = SEMI_MAJOR, SEMI_MINOR
    return OpticalSystem([
        Mirror("left", surface=Ellipsoid.from_semi_axes(a, b), aperture=b, z=-(a + C)),
        Mirror("right", surface=Ellipsoid.from_semi_axes(-a, b), aperture=b, z=a - C),
    ])


def trace(rays: int = 500):
    source = PointSource(NEAR_FOCUS, direction=(0.0, 0.0, -1.0), half_angle_deg=40.0, rays=rays, fan="y")
    # Generation 1 runs from the focus to the mirror, generation 2 from the mirror through the far focus.
    return NonSequentialTracer(build_mirror(), max_interactions=2).trace(source)


def build_drawing(system: OpticalSystem, rays) -> Drawing:
    drawing = Drawing(x_lims=(-8.0, 1.0), y_lims=(-4.0, 4.0)).add_system(system)
    for item in drawing.items:  # the mirror's section, white on the black background
        item.color, item.width = (1.0, 1.0, 1.0, 1.0), 2.0
    drawing.add(MarkerItem(np.array([[FAR_FOCUS[2], FAR_FOCUS[1]]]), color=(1.0, 0.84, 0.0, 1.0), size=8.0))
    return drawing.add_rays(rays, intensities=lambda ray, j: 1.0 / (1.0 + 0.3 * (j + 1)), markers="crimson", marker_size=7.0)


def main() -> None:
    rays = trace()
    origins, directions = rays_by_generation(rays, generation=2)
    distances_um = point_line_distances(origins, directions, FAR_FOCUS) * 1e3
    rms_um = float(np.sqrt(np.mean(distances_um**2)))
    print("Ellipse mirror: focus-to-focus stigmatism")
    print("  near focus (source): z = 0")
    print(f"  far focus (target):  z = {FAR_FOCUS[2]:.4f}")
    print(f"  reflected rays checked: {len(origins)}/{int((rays.parent == -1).sum())}")
    print(f"  RMS distance from the far focus to each reflected ray: {rms_um:.4g} um")

    save = Path(sys.argv[sys.argv.index("--save") + 1]) if "--save" in sys.argv else None
    backend = show(
        build_drawing(build_mirror(), rays), backend="gl", interactive=save is None,
        size=(1200, 800), bgcolor="black", title="Ellipse mirror (stigmatic) — raytracer",
        render_config=RenderConfig(ray_width=0.008, sigma_factor=0.5, min_pixels=1.0, use_solid_rays=True),
    )
    if save is not None:
        backend.save(save)
        print(f"wrote {save}")


if __name__ == "__main__":
    main()
