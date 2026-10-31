"""One drawing, two interchangeable backends.

A biconvex singlet relays a point source onto a sensor at its paraxial
conjugate plane. The same neutral :class:`~raytracer.viz.Drawing` — the
meridional section of the 3-D system, traced non-sequentially with a
``fan="y"`` source — renders in matplotlib (publication diagram, few rays) or
OpenGL (HDR caustic accumulation, many rays). The matplotlib run also
reports the irradiance peak and RMS spot size on the sensor — at this
aperture a singlet's spherical aberration is large, so this is a nonzero
number, not a diagnostic failure.

Usage:
    python -m examples.imaging.single_lens_relay            # matplotlib
    python -m examples.imaging.single_lens_relay gl         # OpenGL viewer
    python -m examples.imaging.single_lens_relay both --save out/  # PNGs only, into the directory out/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer import OpticalSystem
from raytracer.analysis import detected_on, spot_data_from_points
from raytracer.elements import Lens, Sensor
from raytracer.geometry.apertures import Rectangle
from raytracer.geometry.surfaces import Sphere
from raytracer.materials import ConstantIndex
from raytracer.sources import PointSource
from raytracer.tracing import NonSequentialTracer
from raytracer.viz import Drawing, show

SENSOR_HALF = 12.0


def build_relay() -> OpticalSystem:
    lens = Lens("singlet", front=Sphere(60.0), back=Sphere(-60.0), thickness=9.0,
                material=ConstantIndex("N1.5168", 1.5168), semi_diameter=16.0)
    sensor = Sensor("IMG", aperture=Rectangle(2 * SENSOR_HALF, 2 * SENSOR_HALF), z=171.4)
    return OpticalSystem([lens, sensor])


def trace(system: OpticalSystem, rays: int):
    source = PointSource((0.0, 0.0, -90.0), half_angle_deg=9.0, rays=rays, fan="y")
    return NonSequentialTracer(system, max_interactions=4, split=False).trace(source)


def render(rays: int, backend: str, save_dir: Path | None, name: str):
    system = build_relay()
    traced = trace(system, rays)
    b = show(Drawing().add_system(system, rays=traced).add_rays(traced), backend=backend, interactive=save_dir is None)
    if save_dir:
        b.save(save_dir / name)
        print(f"wrote {save_dir / name}")
    return system, traced


def main() -> None:
    args = sys.argv[1:]
    backend = args[0] if args and args[0] in ("mpl", "gl", "both") else "mpl"
    save_dir = None
    if "--save" in args:
        save_dir = Path(args[args.index("--save") + 1])
        save_dir.mkdir(parents=True, exist_ok=True)

    if backend in ("mpl", "both"):
        points, power = detected_on(*render(41, "mpl", save_dir, "singlet_mpl.png"), "IMG")  # few rays: readable
        values, edges = np.histogram(points[:, 1], bins=100, range=(-SENSOR_HALF, SENSOR_HALF), weights=power)
        spot = spot_data_from_points(points[:, :2])
        print(f"sensor: {len(points)} hits, irradiance peak at y={edges[np.argmax(values)]:.2f} mm")
        print(f"RMS spot radius: {spot.rms_radius_um:.4g} um "
              "(nonzero: a spherical singlet is not stigmatic)")

    if backend in ("gl", "both"):
        render(4000, "gl", save_dir, "singlet_gl.png")  # dense: HDR caustics


if __name__ == "__main__":
    main()
