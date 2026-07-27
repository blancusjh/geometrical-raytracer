"""Regenerate every image used by the README.

The OpenGL figures are produced by running the examples through their own
``--save`` entry point, so the README always shows what the code in
``examples/`` actually renders — not a separate, drifting copy of it.

Usage (needs a GL context; on a headless box use Xvfb)::

    xvfb-run -a -s "-screen 0 1920x1080x24" python docs/render_images.py

Requires the ``gl`` extra (``pip install -e ".[gl]"``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "img"
MAX_WIDTH = 1400  # keep the repository light; still sharp on a README

#: (module, extra CLI args, output name)
EXAMPLES = [
    ("examples.lithography.duv_objective_2d", [], "duv_2d"),
    ("examples.lithography.duv_objective_3d", ["--beam"], "duv_3d_beam"),
    ("examples.stigmatic_surfaces.cartesian_oval_refractor_3d", ["--beam"], "cartesian_oval_3d"),
    ("examples.stigmatic_surfaces.ellipse_mirror", [], "ellipse_mirror"),
    ("examples.stigmatic_surfaces.cartesian_oval_refractor_2d", [], "cartesian_oval_2d"),
    ("examples.telescopes.keplerian", [], "keplerian"),
    ("examples.telescopes.galilean", [], "galilean"),
]


def shrink(path: Path) -> None:
    """Downscale to MAX_WIDTH and re-encode, so the repo stays light."""

    from PIL import Image

    img = Image.open(path)
    if img.width > MAX_WIDTH:
        img = img.resize((MAX_WIDTH, round(img.height * MAX_WIDTH / img.width)), Image.LANCZOS)
    img.convert("RGB").save(path, optimize=True)
    print(f"  {path.relative_to(ROOT)}  {img.width}x{img.height}  {path.stat().st_size // 1024} KB")


def render_examples() -> None:
    print("OpenGL scenes (via each example's own --save):")
    for module, args, name in EXAMPLES:
        path = OUT / f"{name}.png"
        subprocess.run(
            [sys.executable, "-m", module, *args, "--save", str(path)],
            cwd=ROOT, check=True, capture_output=True,
        )
        shrink(path)


def render_newtonian() -> None:
    """The Newtonian, denser than the interactive demo's default sampling."""

    from examples.telescopes.newtonian import build_scene
    from raytracer import RenderConfig
    from raytracer.viz import show

    print("Newtonian (denser bundle):")
    scene, *_ = build_scene(samples_per_strip=26)
    path = OUT / "newtonian.png"
    backend = show(
        scene, backend="gl", interactive=False, size=(1400, 640), bgcolor="black",
        render_config=RenderConfig(ray_width=0.035, min_pixels=1.0, use_solid_rays=True),
    )
    backend.save(path)
    shrink(path)


def render_analysis() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from raytracer.analysis import spot_data
    from raytracer.design import OpticalSystem
    from raytracer.propagation import (
        FieldPoint,
        PupilSampling,
        SequentialTracer,
        solve_object_plane,
        trace_pupil,
    )
    from raytracer.viz import plots

    print("matplotlib analysis figures:")
    system = OpticalSystem.from_prescription(
        ROOT / "data" / "US7557996_Fig3_Table3_prescription.csv"
    )
    tracer = SequentialTracer(system)
    solve_object_plane(tracer)
    fields = [56.0, 62.0, 67.0]

    fig = plots.layout_figure(tracer, fields=fields, figsize=(15, 6))
    fig.savefig(OUT / "duv_layout.png", dpi=150, bbox_inches="tight", facecolor="white")
    shrink(OUT / "duv_layout.png")

    sampling = PupilSampling(kind="rings", radial=12, azimuth=72)
    spots = [
        spot_data(trace_pupil(tracer, FieldPoint(y=y), na_object_sine=0.3, sampling=sampling))
        for y in fields
    ]
    fig = plots.spots_figure(spots, airy_radius_um=0.61 * 193.368e-6 / 1.1977 * 1e3)
    fig.savefig(OUT / "duv_spots.png", dpi=150, bbox_inches="tight", facecolor="white")
    shrink(OUT / "duv_spots.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    render_analysis()
    render_examples()
    render_newtonian()
