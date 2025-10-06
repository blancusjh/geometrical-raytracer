"""Generate reference renders for Cartesian optics examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer.dioptrics import CartesianDioptrique, CartesianSinglet
from raytracer.sources import PointSource2D
from raytracer.tracer import RayTracer2D, TraceConfig
from raytracer.visualization import Scene2DViewer

OUTPUT_DIR = Path("renders")


def _render_cartesian_dioptrique(line_method: str, filename: str) -> None:
    z_object = 60.0
    z_image = 30.0
    n_air = 1.0
    n_glass = 1.52

    surface = CartesianDioptrique(
        z0=z_object,
        zi=z_image,
        n_exterior=n_air,
        n_interior=n_glass,
        aperture_radius=0.7,
        samples=800,
    )

    source = PointSource2D(
        origin=np.array([-z_object, 0.0]),
        axis_direction=np.array([1.0, 0.0]),
        aperture=np.deg2rad(4.0),
        samples=21,
    )

    config = TraceConfig(
        max_generations=4,
        allow_reflection=False,
        allow_refraction=True,
        ambient_n=n_air,
    )

    tracer = RayTracer2D([surface], config)
    tree = tracer.trace([source])

    viewer = Scene2DViewer(
        x_lims=(-65.0, 35.0),
        y_lims=(-6.0, 6.0),
        show_axis=False,
        show=False,
        size=(1400, 900),
        line_method=line_method,
    )
    viewer.draw_surfaces([surface])
    viewer.draw_rays(tree, width=1.1, extend_mode="axis", show_misses=False)
    viewer.save(OUTPUT_DIR / filename, size=(1400, 900))
    viewer.close()


def _render_high_aperture_singlet(filename: str) -> None:
    z_object = 5.0
    z_intermediate = 3.0
    z_image = 5.5
    thickness = 0.2
    n_air = 1.0
    n_glass = 1.6

    singlet = CartesianSinglet(
        z0=z_object,
        zc=z_intermediate,
        zi=z_image,
        n0=n_air,
        n_lens=n_glass,
        thickness=thickness,
        aperture_radius=1.2,
        samples=1400,
    )
    front, back = singlet.surfaces()

    source = PointSource2D(
        origin=np.array([-z_object, 0.0]),
        axis_direction=np.array([1.0, 0.0]),
        aperture=np.deg2rad(40.0),
        samples=81,
    )

    config = TraceConfig(
        max_generations=3,
        allow_reflection=False,
        allow_refraction=True,
        ambient_n=n_air,
    )

    tracer = RayTracer2D([front, back], config)
    tree = tracer.trace([source])

    x_max = thickness + z_image + 1.0
    viewer = Scene2DViewer(
        x_lims=(-7.0, x_max),
        y_lims=(-4.0, 4.0),
        show_axis=False,
        show=False,
        size=(1400, 900),
        line_method="agg",
    )
    viewer.draw_surfaces([front, back])
    viewer.draw_rays(tree, width=1.0, extend_mode="axis", show_misses=False)
    viewer.save(OUTPUT_DIR / filename, size=(1400, 900))
    viewer.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _render_cartesian_dioptrique("gl", "dioptrique_gl.png")
    _render_cartesian_dioptrique("agg", "dioptrique_agg.png")
    _render_high_aperture_singlet("high_aperture_agg.png")


if __name__ == "__main__":
    main()
