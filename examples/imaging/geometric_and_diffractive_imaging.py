"""Image formation through an optical system, two complementary ways.

1. **Geometric (non-sequential engine):** a bar-pattern object emits weighted
   meridional ray fans (`ImageSource`); a biconvex singlet relays it onto a
   sensor whose irradiance histogram reveals the magnified, inverted image.
   The RMS spot radius of the rays landing near the centre of that image is
   also reported, quantifying the singlet's (nonzero) aberration.
2. **Diffractive (sequential analysis):** any picture becomes a binary mask
   (`BinaryMask.from_image`) and is imaged through a (perfect or fitted)
   pupil with the partially coherent Abbe method.

Usage:
    python -m examples.imaging.geometric_and_diffractive_imaging           # show both figures
    python -m examples.imaging.geometric_and_diffractive_imaging --save out/
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer import OpticalSystem
from raytracer.analysis import BinaryMask, abbe_image, detected_on, pupil_function, spot_data_from_points
from raytracer.elements import Lens, Sensor
from raytracer.geometry.apertures import Rectangle
from raytracer.geometry.surfaces import Sphere
from raytracer.materials import ConstantIndex
from raytracer.sources import ImageSource
from raytracer.tracing import NonSequentialTracer


def geometric_relay():
    """Bar-pattern object -> singlet -> sensor irradiance."""

    # Three-bar test object, 12 mm tall, at the doublet's finite conjugate.
    profile = np.zeros(240)
    for start in (30, 105, 180):
        profile[start : start + 30] = 1.0
    profile[105:135] *= 0.55  # middle bar dimmer: gray levels survive imaging

    lens = Lens("relay", front=Sphere(60.0), back=Sphere(-60.0), thickness=9.0,
                material=ConstantIndex("N1.5168", 1.5168), semi_diameter=16.0)
    sensor = Sensor("IMG", aperture=Rectangle(36.0, 36.0), z=171.4)
    source = ImageSource(profile, p0=(0.0, -6.0, -90.0), p1=(0.0, 6.0, -90.0), half_angle_deg=5.0, rays_per_point=31)

    system = OpticalSystem([lens, sensor])
    points, power = detected_on(system, NonSequentialTracer(system, max_interactions=4, split=False).trace(source), "IMG")
    values, edges = np.histogram(points[:, 1], bins=180, range=(-18.0, 18.0), weights=power)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # RMS spot size for rays landing near the object's mid-height bar (the
    # dimmer one), as a proxy for the relay's on-axis aberration.
    near_center = points[np.abs(points[:, 1]) < 2.0, :2]
    rms_um = spot_data_from_points(near_center).rms_radius_um if len(near_center) > 1 else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    object_y = np.linspace(-6.0, 6.0, profile.size)
    axes[0].plot(object_y, profile, color="#20509e")
    axes[0].set(title="Object intensity profile", xlabel="y (mm)", ylabel="I")
    axes[1].plot(centers, values / values.max(), color="#b3421b")
    axes[1].set(
        title=f"Sensor irradiance ({len(points)} rays) — inverted, magnified\n"
              f"RMS spot near center: {rms_um:.3g} um",
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
    import tempfile

    save_dir = None
    if "--save" in sys.argv:
        save_dir = Path(sys.argv[sys.argv.index("--save") + 1])
        save_dir.mkdir(parents=True, exist_ok=True)

    fig_geo = geometric_relay()

    # The demo letter is an intermediate input, not a deliverable: keep it in
    # a temp dir (never the CWD) unless the user asked to save outputs.
    with tempfile.TemporaryDirectory() as tmp:
        demo = Path(save_dir or tmp) / "demo_letter.png"
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
