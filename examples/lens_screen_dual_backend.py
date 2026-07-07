"""One scene, two interchangeable backends.

A cemented doublet focuses a point source onto an observation screen. The
same neutral Scene renders in matplotlib (publication diagram, few rays)
or OpenGL (HDR caustic accumulation, many rays).

Usage:
    python -m examples.lens_screen_dual_backend            # matplotlib
    python -m examples.lens_screen_dual_backend gl         # OpenGL viewer
    python -m examples.lens_screen_dual_backend both --save out/  # PNGs only
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer.nonseq import (
    Lens2D,
    PointSource2D,
    RayTracer2D,
    Screen2D,
    TraceConfig,
)
from raytracer.viz import Scene, show


def build_scene(samples: int):
    lens = Lens2D.from_radii(
        r1=60.0, r2=-60.0, thickness=9.0, semidiameter=16.0, n=1.5168,
        vertex=(0.0, 0.0), name="doublet",
    )
    screen = Screen2D([171.4, -12.0], [171.4, 12.0])  # conjugate of the source
    source = PointSource2D(
        origin=np.array([-90.0, 0.0]),
        axis_direction=np.array([1.0, 0.0]),
        aperture=np.deg2rad(18.0),
        samples=samples,
    )
    tracer = RayTracer2D(
        [*lens.surfaces(), screen],
        TraceConfig(max_generations=4, fresnel_split=False),
    )
    tree = tracer.trace([source])

    scene = Scene()
    scene.add_elements([lens, screen])
    scene.add_tree(tree, intensity_resolver=lambda node: node.intensity)
    return scene, screen


def main() -> None:
    args = sys.argv[1:]
    backend = args[0] if args and args[0] in ("mpl", "gl", "both") else "mpl"
    save_dir = None
    if "--save" in args:
        save_dir = Path(args[args.index("--save") + 1])
        save_dir.mkdir(parents=True, exist_ok=True)

    if backend in ("mpl", "both"):
        scene, screen = build_scene(samples=41)  # few rays: readable diagram
        b = show(scene, backend="mpl", interactive=save_dir is None)
        if save_dir:
            b.save(save_dir / "doublet_mpl.png")
            print(f"wrote {save_dir / 'doublet_mpl.png'}")
        edges, values = screen.irradiance(bins=100)
        peak = edges[np.argmax(values)]
        print(f"screen: {len(screen.hits)} hits, irradiance peak at s={peak:.2f} mm")

    if backend in ("gl", "both"):
        scene, _ = build_scene(samples=4000)  # dense: HDR caustics
        b = show(scene, backend="gl", interactive=save_dir is None)
        if save_dir:
            b.save(save_dir / "doublet_gl.png")
            print(f"wrote {save_dir / 'doublet_gl.png'}")


if __name__ == "__main__":
    main()
