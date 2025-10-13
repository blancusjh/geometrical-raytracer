"""Compute a high-resolution intensity map for the ellipse setup without OpenGL.

This script mirrors the default parameters from ``examples/ellipse_interactive.py``
and accumulates the Gaussian-weighted ray contributions on the CPU.  The goal is
to provide a trustworthy baseline image (and zoomed crops) that help debug the
OpenGL viewer without relying on GPU rendering.

The output is written to ``renders/opengl_debug/cpu_baseline`` by default and
includes:

- A full-frame intensity image (squared accumulation -> sqrt before display).
- Focused crops near the source, mid-body, and second focus.
- The raw ``.npz`` field for further analysis.

The script only depends on ``numpy`` and ``Pillow``.  It avoids importing
``raytracer.visualization_opengl`` so no ``vispy`` context is needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer.geometry import EllipseConic
from raytracer.sources import PointSource2D
from raytracer.tracer import RayTracer2D, TraceConfig


# -------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------


def _gather_rays(tree, tail_length: float, ray_weight: float) -> list[Tuple[np.ndarray, np.ndarray, float, float]]:
    """Return a list of (start, end, intensity, alpha_weight) tuples in 2D."""

    rays: list[Tuple[np.ndarray, np.ndarray, float, float]] = []
    for node in tree.nodes():
        start = np.asarray(node.ray.origin[:2], dtype=np.float32)
        direction = np.asarray(node.ray.direction[:2], dtype=np.float32)

        if node.intersection is None:
            end = start + direction * tail_length
            alpha = 0.5 * ray_weight
        else:
            end = np.asarray(node.intersection.point[:2], dtype=np.float32)
            alpha = ray_weight

        intensity = 1.0 / (1.0 + 0.3 * float(node.generation))
        rays.append((start, end, float(intensity), float(alpha)))

    return rays


def _accumulate_gaussian_field(
    rays: Sequence[Tuple[np.ndarray, np.ndarray, float, float]],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Accumulate squared intensities over the grid using a Gaussian tube."""

    xx, yy = np.meshgrid(grid_x, grid_y)
    field = np.zeros_like(xx, dtype=np.float32)

    sigma_sq = max(1e-12, float(sigma) ** 2)

    for start, end, intensity, alpha in rays:
        ax, ay = start
        bx, by = end

        pax = xx - ax
        pay = yy - ay
        bax = bx - ax
        bay = by - ay

        denom = bax * bax + bay * bay
        denom = np.where(denom < 1e-12, 1e-12, denom)

        h = np.clip((pax * bax + pay * bay) / denom, 0.0, 1.0)
        closest_x = ax + h * bax
        closest_y = ay + h * bay

        dist_sq = (xx - closest_x) ** 2 + (yy - closest_y) ** 2
        gaussian = np.exp(-dist_sq / sigma_sq)

        field += alpha * intensity * gaussian

    return field


def _export_image(arr: np.ndarray, path: Path) -> None:
    arr = np.clip(arr, 0.0, 1.0)
    img = (arr * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def _save_zoom(
    norm_field: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    center: Tuple[float, float],
    span: Tuple[float, float],
    out_path: Path,
) -> None:
    cx, cy = center
    sx, sy = span

    x_mask = (xs >= cx - sx) & (xs <= cx + sx)
    y_mask = (ys >= cy - sy) & (ys <= cy + sy)

    if not x_mask.any() or not y_mask.any():
        raise ValueError("Zoom window outside sampled domain")

    sub = norm_field[np.ix_(y_mask, x_mask)]
    if sub.max() > 0:
        sub = sub / sub.max()

    img = (sub * 255.0).astype(np.uint8)
    Image.fromarray(img).resize((800, 800), Image.BILINEAR).save(out_path)


# -------------------------------------------------------------------------------------
# Main routine
# -------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=500, help="Number of rays emitted by the source")
    parser.add_argument("--resolution", type=str, default="1200x800", help="Output grid resolution W×H")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "renders" / "opengl_debug" / "cpu_baseline",
        help="Directory for saved images",
    )
    parser.add_argument("--tail-length", type=float, default=12.0, help="Length for rays that miss the ellipse")
    args = parser.parse_args(argv)

    try:
        width_str, height_str = args.resolution.lower().split("x")
        width = int(width_str)
        height = int(height_str)
    except Exception as exc:  # pragma: no cover - CLI guard
        raise SystemExit(f"Invalid resolution '{args.resolution}'. Use the form WIDTHxHEIGHT.") from exc

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ Trace rays
    ellipse = EllipseConic(semi_major=4.0, semi_minor=2.5, focus=np.array([0.0, 0.0]))
    source = PointSource2D(
        origin=np.array([0.0, 0.11]),
        axis_direction=np.array([-1.0, 0.0]),
        aperture=np.deg2rad(80.0),
        samples=int(args.samples),
    )

    tracer = RayTracer2D(
        surfaces=[ellipse],
        config=TraceConfig(max_generations=3, allow_reflection=True, allow_refraction=False),
    )
    tree = tracer.trace([source])

    ray_weight = min(1.0, 50.0 / float(source.samples))
    rays = _gather_rays(tree, tail_length=float(args.tail_length), ray_weight=ray_weight)

    # ------------------------------------------------------------------ Grid setup
    x_min, x_max = -8.0, 1.0
    y_min, y_max = -4.0, 4.0

    xs = np.linspace(x_min, x_max, width, dtype=np.float32)
    ys = np.linspace(y_min, y_max, height, dtype=np.float32)

    requested_width = 0.001
    sigma_factor = 0.5

    units_per_pixel = (x_max - x_min) / float(width)
    geometry_width = max(requested_width, units_per_pixel)
    sigma = geometry_width * sigma_factor

    field = _accumulate_gaussian_field(rays, xs, ys, sigma=sigma)

    # Squared accumulation -> convert to radiant intensity for display
    field_display = np.sqrt(field)
    if field_display.max() > 0:
        field_display /= field_display.max()

    np.savez_compressed(outdir / "cpu_field.npz", field=field, xs=xs, ys=ys)
    _export_image(field_display, outdir / "field_full.png")

    _save_zoom(field_display, xs, ys, center=(0.0, 0.11), span=(0.6, 0.6), out_path=outdir / "zoom_source.png")
    _save_zoom(field_display, xs, ys, center=(-2.0, 0.0), span=(1.0, 0.7), out_path=outdir / "zoom_mid_body.png")
    _save_zoom(field_display, xs, ys, center=(-6.5, 0.0), span=(0.8, 0.8), out_path=outdir / "zoom_second_focus.png")

    print(f"Saved CPU intensity baseline to {outdir}")


if __name__ == "__main__":
    main()

