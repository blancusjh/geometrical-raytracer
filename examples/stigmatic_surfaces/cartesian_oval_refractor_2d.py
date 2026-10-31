"""Cartesian oval: a single refracting surface with perfect stigmatism.

Descartes' construction gives a surface shape (not a sphere) that images an
object at ``z0`` in air exactly onto ``zi`` inside the glass — every ray from
the object point refracts through precisely the same image point, with no
spherical aberration at any aperture: ``n0 * dist(object, surface) +
ni * dist(surface, image) = const`` (Fermat's principle).

Here the oval (:class:`~raytracer.geometry.surfaces.CartesianSurface`) is the
front face of a glass block that reaches a little past the image point, and
a meridional fan from the object is traced non-sequentially and drawn in the
y-z plane, up to the block's flat back face.

The convergence claim is checked, not just asserted: the RMS perpendicular
distance from the image point to each refracted ray's line
(:func:`raytracer.analysis.point_line_distances`) is printed alongside the
diagram — near zero confirms stigmatism.

The source cone is matched to the surface's clear aperture, the way a real
system's stop would be, so every emitted ray lands on the surface. That
aperture has a hard geometric ceiling: the oval is a closed surface whose
half-width peaks at ``max_radius`` (4.858 mm for this conjugate pair) and
then turns back toward the axis, so rays wider than 7.69 deg from the object
miss it entirely no matter how large a semi-diameter is declared.

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

from raytracer import OpticalSystem
from raytracer.analysis import point_line_distances, rays_by_generation
from raytracer.elements import Lens
from raytracer.geometry.surfaces import CartesianSurface, Plane
from raytracer.materials import ConstantIndex
from raytracer.sources import PointSource
from raytracer.tracing import NonSequentialTracer
from raytracer.viz import Drawing, MarkerItem, show
from raytracer.viz.gl.viewer import RenderConfig

AMBIENT_N = 1.0
GLASS_N = 1.7
Z0 = -30.0  # object position, in air
ZI = 10.0  # image position, inside the glass
SEMIDIAMETER = 4.5  # clear aperture, inside the oval's 4.858 geometric ceiling
BLOCK_LENGTH = 13.0  # the glass ends 3 mm past the image point

OVAL = CartesianSurface(n0=AMBIENT_N, z0=Z0, ni=GLASS_N, zi=ZI)
IMAGE_POINT = np.array([0.0, 0.0, ZI])


def rim_half_angle_deg(surface, semi_diameter: float) -> float:
    """Half-angle of the cone from the object whose marginal ray lands exactly on the rim.

    Read off the surface itself rather than tabulated: emitting any wider would
    only add rays that never reach it, and the answer moves with the conjugates.
    """

    rim_z = float(surface.sag(np.array([semi_diameter]), np.zeros(1))[0])
    return float(np.degrees(np.arctan2(semi_diameter, rim_z - Z0)))


def build_block() -> OpticalSystem:
    return OpticalSystem([Lens("oval", surfaces=[OVAL, Plane()], thicknesses=[BLOCK_LENGTH],
                               materials=[ConstantIndex("GLASS", GLASS_N)], semi_diameter=SEMIDIAMETER)])


def trace(system: OpticalSystem, rays: int = 300):
    source = PointSource((0.0, 0.0, Z0), half_angle_deg=rim_half_angle_deg(OVAL, SEMIDIAMETER), rays=rays, fan="y")
    # Each ray runs in air to the oval (its first piece), then in the glass to the back face (its second).
    return NonSequentialTracer(system, split=False).trace(source)


def main() -> None:
    system = build_block()
    rays = trace(system)

    origins, directions = rays_by_generation(rays, generation=2)
    distances_um = point_line_distances(origins, directions, IMAGE_POINT) * 1e3
    rms_um = float(np.sqrt(np.mean(distances_um**2))) if len(origins) else float("nan")
    print("Cartesian oval refractor: stigmatic imaging")
    print(f"  object: z0={Z0} (air, n={AMBIENT_N}); image: zi={ZI} (glass, n={GLASS_N})")
    print(f"  clear aperture (semidiameter): {SEMIDIAMETER}")
    print(f"  refracted rays within the aperture: {len(origins)}/{len(rays)}")
    print(f"  RMS distance from the image point to each refracted ray: {rms_um:.4g} um")

    drawing = Drawing(x_lims=(-32.0, 14.0), y_lims=(-8.0, 8.0)).add_system(system)
    def colors(ray, j):  # the piece from each ray's first point runs in the glass
        return np.where((j == 1)[:, None], (0.5, 0.9, 1.0, 0.85), (1.0, 1.0, 1.0, 0.45))

    drawing.add_rays(rays, colors=colors, intensities=1.0, select=lambda ray, j: j <= 1,
                     markers="crimson", marker_size=6.0)
    drawing.add(MarkerItem(np.array([[ZI, 0.0]]), color=(1.0, 0.84, 0.0, 1.0), size=8.0))

    save = Path(sys.argv[sys.argv.index("--save") + 1]) if "--save" in sys.argv else None
    backend = show(
        drawing, backend="gl", interactive=save is None,
        size=(1400, 700), bgcolor="black", title="Cartesian oval refractor (stigmatic) — raytracer",
        render_config=RenderConfig(ray_width=0.02, sigma_factor=0.5, min_pixels=1.0, use_solid_rays=True),
    )
    if save is not None:
        backend.save(save)
        print(f"wrote {save}")


if __name__ == "__main__":
    main()
