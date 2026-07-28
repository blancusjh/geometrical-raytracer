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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "img"
MAX_WIDTH = 1400  # keep the repository light; still sharp on a README

#: (module, extra CLI args, output name)
EXAMPLES = [
    ("examples.lithography.duv_objective_2d", [], "duv_2d"),
    ("examples.lithography.duv_objective_3d", ["--spectrum"], "duv_3d_spectrum"),
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
        ROOT / "data" / "optical_systems/lithography/US7557996_Fig3_Table3_prescription.csv"
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


def render_distortion_grids() -> None:
    """Three systems' object-to-image maps, spanning a 4700x range.

    Each panel is drawn at its own exaggeration and titled with the true
    magnitude: the dial that makes the DUV objective's 44 nm residual
    visible would push the Double Gauss's 206 um clean off the page.
    """

    import matplotlib

    matplotlib.use("Agg")
    from raytracer.analysis import distortion_grid
    from raytracer.design import OpticalSystem
    from raytracer.propagation import SequentialTracer, solve_object_plane
    from raytracer.viz import plots

    print("distortion grids:")
    #: (prescription, panel title, object half-field in degrees, exaggeration)
    SYSTEMS = [
        ("optical_systems/photographic/cooke_triplet_prescription.csv", "Cooke triplet, EFL 50 mm", 10.0, 25.0),
        ("optical_systems/photographic/double_gauss_prescription.csv", "Double Gauss, EFL 99 mm", 20.6, 15.0),
    ]

    grids, titles, exaggerations = [], [], []
    for csv, title, half_deg, exaggeration in SYSTEMS:
        tracer = SequentialTracer(OpticalSystem.from_prescription(ROOT / "data" / csv))
        conjugate = solve_object_plane(tracer)
        half = abs(conjugate.object_z) * np.tan(np.deg2rad(half_deg))
        grids.append(
            distortion_grid(
                tracer, magnification=conjugate.magnification, half_field=half, n=11
            )
        )
        titles.append(f"{title}\n±{half_deg:g}° object half-field")
        exaggerations.append(exaggeration)

    # The DUV objective is referred to the patent's own 4x reduction, so the
    # ideal map here is the design intent rather than the recovered paraxial
    # value. A 50 mm half-field square puts its corners at 50*sqrt(2) = 70.7
    # mm, past the ~68 mm usable field radius: those are the vignetted points.
    tracer = SequentialTracer(
        OpticalSystem.from_prescription(
            ROOT / "data" / "optical_systems/lithography/US7557996_Fig3_Table3_prescription.csv"
        )
    )
    solve_object_plane(tracer)
    grids.append(distortion_grid(tracer, magnification=0.25, half_field=50.0, n=11))
    titles.append("US7557996 DUV objective\n±50 mm object half-field")
    exaggerations.append(2000.0)

    fig = plots.distortion_grids_figure(
        grids, titles=titles, exaggerations=exaggerations,
        suptitle="Chief-ray distortion grids — dashed: the ideal linear map; "
                 "solid: where the traced chief rays actually land",
    )
    fig.savefig(OUT / "distortion_grids.png", dpi=150, bbox_inches="tight", facecolor="white")
    shrink(OUT / "distortion_grids.png")
    for title, grid in zip(titles, grids):
        print(f"    {title.splitlines()[0]}: max {grid.max_distortion_um:.4g} µm "
              f"({grid.max_relative_distortion_percent:.3g} %), "
              f"coverage {grid.valid_fraction:.1%}")


def render_aplanatism() -> None:
    """Sphere vs stigmatic vs aplanatic singlet: the sine condition made visible.

    Delegates to the example's own figure so the README shows exactly what
    ``examples.stigmatic_surfaces.aplanatic_sol --save`` produces.
    """

    import warnings

    import matplotlib

    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from examples.stigmatic_surfaces.aplanatic_sol import field_blur, figure

    print("aplanatism comparison:")
    path = OUT / "aplanatism.png"
    figure(field_blur(), str(path))
    shrink(path)


def render_stigmatic_spots() -> None:
    """Sphere vs Cartesian oval: same conjugates, same vertex curvature.

    The oval is not "a better sphere" -- it is exact, and the only way to show
    that honestly is to plot both and let the axis scales speak.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from raytracer.analysis import spot_data
    from raytracer.design import OpticalSystem, SurfaceRow
    from raytracer.optics.materials import AIR, ConstantIndex
    from raytracer.propagation import FieldPoint, PupilSampling, SequentialTracer, trace_pupil
    from raytracer.surfaces.cartesian_oval import CartesianOvalProfile

    print("stigmatic spot comparison:")
    n0, z0, ni, zi, semi, na = 1.0, -30.0, 1.7, 10.0, 3.0, 0.1
    glass = ConstantIndex("DENSE_GLASS", ni)
    radius = CartesianOvalProfile(n0=n0, z0=z0, ni=ni, zi=zi).radius

    def spot(row):
        tracer = SequentialTracer(OpticalSystem([row], object_space=AIR, object_z=z0))
        pupil = trace_pupil(
            tracer, FieldPoint(y=0.0), na_object_sine=na,
            sampling=PupilSampling(kind="rings", radial=14, azimuth=64), chief_slope=0.0,
        )
        return spot_data(pupil)

    sphere = spot(SurfaceRow.refracting(radius=radius, thickness=zi, material=glass,
                                        semidiameter=semi))
    oval = spot(SurfaceRow.cartesian_oval(n0=n0, z0=z0, ni=ni, zi=zi, thickness=zi,
                                          material=glass, semidiameter=semi))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    scale = 1.15 * np.abs(sphere.relative_um).max()

    axes[0].scatter(sphere.relative_um[:, 0], sphere.relative_um[:, 1], s=5, alpha=0.5,
                    color="#c1440e")
    axes[0].set(xlim=(-scale, scale), ylim=(-scale, scale),
                title=f"Sphere, R = {radius:.3f} mm\nRMS = {sphere.rms_radius_um:,.1f} µm")

    axes[1].scatter(oval.relative_um[:, 0], oval.relative_um[:, 1], s=5, alpha=0.9,
                    color="#2878b5")
    axes[1].set(xlim=(-scale, scale), ylim=(-scale, scale),
                title="Cartesian oval, same scale\n(every ray inside one pixel)")

    # Panel 3 is not a small spot -- it is the double-precision floor. The
    # scatter is round-off, so label it as such instead of dressing it up in a
    # physical unit it does not deserve.
    relative_to_conjugate = oval.rms_radius_um / (abs(z0) * 1e3)
    axes[2].scatter(oval.relative_um[:, 0], oval.relative_um[:, 1], s=5, alpha=0.9,
                    color="#2878b5")
    axes[2].ticklabel_format(style="sci", scilimits=(0, 0))
    axes[2].set(
        title=f"Cartesian oval, own scale\nRMS = {oval.rms_radius_um:.1e} µm "
              f"= {relative_to_conjugate:.0e} of the object distance"
    )

    for ax in axes:
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xlabel("Δx (µm)")
    axes[0].set_ylabel("Δy (µm)")
    fig.suptitle(
        "Same conjugates, same vertex curvature, NA 0.1 — spherical aberration vs none "
        "(the oval's residual is double-precision round-off, not a spot)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "stigmatic_spots.png", dpi=150, bbox_inches="tight", facecolor="white")
    shrink(OUT / "stigmatic_spots.png")
    print(f"    sphere {sphere.rms_radius_um:.4g} µm  vs  oval {oval.rms_radius_um:.4g} µm"
          f"  ({sphere.rms_radius_um / oval.rms_radius_um:.2g}x)")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    render_analysis()
    render_distortion_grids()
    render_aplanatism()
    render_stigmatic_spots()
    render_examples()
    render_newtonian()
