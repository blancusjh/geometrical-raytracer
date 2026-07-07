"""Image formation through an optical system, two complementary ways.

1. **Geometric (2-D engine):** a bar-pattern object emits weighted ray fans
   (`ImageSource2D`); a doublet relays it onto a screen whose irradiance
   histogram reveals the magnified, inverted image.
2. **Diffractive (sequential analysis):** any picture becomes a binary mask
   (`BinaryMask.from_image`) and is imaged through a (perfect or fitted)
   pupil with the partially coherent Abbe method.

Usage:
    python -m examples.image_formation           # show both figures
    python -m examples.image_formation --save out/
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer.analysis import BinaryMask, abbe_image, pupil_function
from raytracer.nonseq import (
    ImageSource2D,
    Lens2D,
    RayTracer2D,
    Screen2D,
    TraceConfig,
)


def geometric_relay():
    """Bar-pattern object -> doublet -> screen irradiance."""

    # Three-bar test object, 12 mm tall, at the doublet's finite conjugate.
    profile = np.zeros(240)
    for start in (30, 105, 180):
        profile[start : start + 30] = 1.0
    profile[105:135] *= 0.55  # middle bar dimmer: gray levels survive imaging

    lens = Lens2D.from_radii(
        r1=60.0, r2=-60.0, thickness=9.0, semidiameter=16.0, n=1.5168,
        vertex=(0.0, 0.0), name="relay",
    )
    source = ImageSource2D(
        profile=profile,
        p0=np.array([-90.0, -6.0]),
        p1=np.array([-90.0, 6.0]),
        axis_direction=np.array([1.0, 0.0]),
        aperture=np.deg2rad(10.0),
        rays_per_point=31,
    )
    screen = Screen2D([171.4, -18.0], [171.4, 18.0])

    tracer = RayTracer2D([*lens.surfaces(), screen], TraceConfig(max_generations=4))
    tracer.trace([source])
    edges, values = screen.irradiance(bins=180)
    centers = 0.5 * (edges[:-1] + edges[1:]) - screen.length / 2.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    object_y = np.linspace(-6.0, 6.0, profile.size)
    axes[0].plot(object_y, profile, color="#20509e")
    axes[0].set(title="Object intensity profile", xlabel="y (mm)", ylabel="I")
    axes[1].plot(centers, values / values.max(), color="#b3421b")
    axes[1].set(
        title=f"Screen irradiance ({len(screen.hits)} rays) — inverted, magnified",
        xlabel="y (mm)",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def diffractive_imaging(image_path: Path):
    """Arbitrary picture -> binary mask -> Abbe aerial image (DUV pupil)."""

    NA, WAVELENGTH_MM, PIXEL_NM = 1.2, 193.368e-6, 10.0
    mask = BinaryMask.from_image(
        image_path, pixel_nm=PIXEL_NM, size=512, threshold=0.5, invert=True
    )
    grid = pupil_function(
        None, na=NA, wavelength_mm=WAVELENGTH_MM, size=512, pixel_mm=PIXEL_NM * 1e-6
    )
    aerial = abbe_image(grid, mask, sigma=0.7, source_points=9)

    size = mask.data.shape[0]
    x_nm = (np.arange(size) - size // 2) * PIXEL_NM
    extent = [x_nm[0], x_nm[-1], x_nm[0], x_nm[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(mask.data, extent=extent, origin="lower", cmap="gray_r")
    axes[0].set_title("Mask from image file")
    shown = axes[1].imshow(aerial, extent=extent, origin="lower", cmap="inferno")
    axes[1].set_title("Abbe aerial image (NA 1.2, σ=0.7)")
    fig.colorbar(shown, ax=axes[1], fraction=0.046)
    axes[2].imshow(aerial > 0.3, extent=extent, origin="lower", cmap="gray_r")
    axes[2].set_title("Thresholded")
    for ax in axes:
        ax.set_xlabel("x (nm)")
    fig.tight_layout()
    return fig


def _make_demo_image(path: Path) -> None:
    """Render a letter into a small PNG used as the diffractive object."""

    fig = plt.figure(figsize=(2, 2), dpi=128)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.text(0.5, 0.5, "R", fontsize=150, ha="center", va="center", family="serif")
    ax.axis("off")
    fig.savefig(path, dpi=128)
    plt.close(fig)


def main() -> None:
    save_dir = None
    if "--save" in sys.argv:
        save_dir = Path(sys.argv[sys.argv.index("--save") + 1])
        save_dir.mkdir(parents=True, exist_ok=True)

    fig_geo = geometric_relay()

    demo = Path(save_dir or ".") / "demo_letter.png"
    _make_demo_image(demo)
    fig_diff = diffractive_imaging(demo)

    if save_dir:
        fig_geo.savefig(save_dir / "geometric_relay.png", dpi=160)
        fig_diff.savefig(save_dir / "diffractive_imaging.png", dpi=160)
        print(f"wrote figures to {save_dir}")
    else:  # pragma: no cover - interactive
        plt.show()


if __name__ == "__main__":
    main()
