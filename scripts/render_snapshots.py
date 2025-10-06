"""Render reference images for the optics examples and collect metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import imageio.v2 as imageio
from vispy import scene

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer.dioptrics import CartesianDioptrique, CartesianSinglet
from raytracer.geometry import EllipseConic
from raytracer.sources import PointSource2D
from raytracer.tracer import RayTracer2D, TraceConfig
from raytracer.visualization import Scene2DViewer

OUTPUT_DIR = ROOT / "renders"
REPORTS_DIR = ROOT / "reports"


@dataclass
class MarkerSpec:
    position: np.ndarray
    color: np.ndarray
    size: float


@dataclass
class ExampleData:
    surfaces: list
    tree: RayTracer2D


@dataclass
class ExampleBuild:
    surfaces: list
    tree: RayTracer2D
    markers: list[MarkerSpec]


@dataclass
class CaptureSpec:
    name: str
    builder: Callable[[], ExampleBuild]
    full_xlim: tuple[float, float]
    full_ylim: tuple[float, float]
    zoom_xlim: tuple[float, float]
    zoom_ylim: tuple[float, float]
    ray_kwargs: dict
    markers: bool = True
    full_size: tuple[int, int] = (1400, 900)
    zoom_size: tuple[int, int] = (1200, 900)
    zoom_regions: list[tuple[str, tuple[float, float, float, float]]] = None


@dataclass
class RenderResult:
    path: Path
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    size: tuple[int, int]


# ---------------------------------------------------------------------------
def _build_ellipse() -> ExampleBuild:
    semi_major = 4.0
    semi_minor = 2.5
    ellipse = EllipseConic(
        semi_major=semi_major,
        semi_minor=semi_minor,
        focus=np.array([0.0, 0.0]),
        surface_id="mirror",
    )

    source = PointSource2D(
        origin=np.array([0.0, 0.0]),
        axis_direction=np.array([1.0, 0.0]),
        aperture=np.deg2rad(80.0),
        samples=25,
    )

    config = TraceConfig(max_generations=5, allow_reflection=True, allow_refraction=False)
    tracer = RayTracer2D([ellipse], config)
    tree = tracer.trace([source])

    primary_focus = ellipse.focus.copy()
    major_axis_dir = ellipse._R @ np.array([1.0, 0.0])
    focal_offset = np.sqrt(semi_major**2 - semi_minor**2)
    secondary_focus = primary_focus - 2.0 * focal_offset * major_axis_dir

    markers = [
        MarkerSpec(primary_focus, np.array([0.3, 0.9, 0.3, 1.0]), 10.0),
        MarkerSpec(secondary_focus, np.array([0.1, 0.7, 1.0, 1.0]), 10.0),
    ]
    return ExampleBuild([ellipse], tree, markers)


def _build_dioptrique() -> ExampleBuild:
    z_object = 60.0
    z_image = 30.0
    n_air = 1.0
    n_glass = 1.52

    dioptrique = CartesianDioptrique(
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
    tracer = RayTracer2D([dioptrique], config)
    tree = tracer.trace([source])

    markers = [
        MarkerSpec(np.array([-z_object, 0.0]), np.array([0.3, 0.9, 0.3, 1.0]), 10.0),
        MarkerSpec(np.array([z_image, 0.0]), np.array([0.1, 0.7, 1.0, 1.0]), 10.0),
    ]
    return ExampleBuild([dioptrique], tree, markers)


def _build_singlet() -> ExampleBuild:
    z_object = 60.0
    z_intermediate = 5.0
    z_image = 40.0
    thickness = 0.03
    n_air = 1.0
    n_glass = 1.52

    singlet = CartesianSinglet(
        z0=z_object,
        zc=z_intermediate,
        zi=z_image,
        n0=n_air,
        n_lens=n_glass,
        n_out=n_air,
        thickness=thickness,
        aperture_radius=0.6,
        samples=900,
    )
    front, back = singlet.surfaces()

    source = PointSource2D(
        origin=np.array([-z_object, 0.0]),
        axis_direction=np.array([1.0, 0.0]),
        aperture=np.deg2rad(3.0),
        samples=21,
    )

    config = TraceConfig(
        max_generations=4,
        allow_reflection=False,
        allow_refraction=True,
        ambient_n=n_air,
    )

    tracer = RayTracer2D([front, back], config)
    tree = tracer.trace([source])

    markers = [
        MarkerSpec(np.array([-z_object, 0.0]), np.array([0.3, 0.9, 0.3, 1.0]), 10.0),
        MarkerSpec(np.array([z_intermediate, 0.0]), np.array([1.0, 0.5, 0.1, 1.0]), 10.0),
        MarkerSpec(np.array([thickness + z_image, 0.0]), np.array([0.1, 0.7, 1.0, 1.0]), 10.0),
    ]
    return ExampleBuild([front, back], tree, markers)


def _build_high_aperture() -> ExampleBuild:
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

    markers = [
        MarkerSpec(np.array([-z_object, 0.0]), np.array([0.3, 0.9, 0.3, 1.0]), 10.0),
        MarkerSpec(np.array([z_intermediate, 0.0]), np.array([1.0, 0.5, 0.1, 1.0]), 10.0),
        MarkerSpec(np.array([thickness + z_image, 0.0]), np.array([0.1, 0.7, 1.0, 1.0]), 10.0),
    ]
    return ExampleBuild([front, back], tree, markers)


CAPTURES: list[CaptureSpec] = [
    CaptureSpec(
        name="ellipse",
        builder=_build_ellipse,
        full_xlim=(-8.0, 1.0),
        full_ylim=(-4.0, 4.0),
        zoom_xlim=(-1.2, 1.2),
        zoom_ylim=(-1.2, 1.2),
        ray_kwargs=dict(width=0.35, extend_mode="none", show_misses=True),
        zoom_regions=[
            ("focus", (-0.15, 0.15, -0.15, 0.15)),
            ("axis_left", (-0.8, -0.3, -0.1, 0.1)),
            ("axis_right", (0.3, 0.8, -0.1, 0.1)),
        ],
    ),
    CaptureSpec(
        name="cartesian_dioptrique",
        builder=_build_dioptrique,
        full_xlim=(-65.0, 35.0),
        full_ylim=(-6.0, 6.0),
        zoom_xlim=(-3.0, 3.0),
        zoom_ylim=(-3.0, 3.0),
        ray_kwargs=dict(width=0.35, extend_mode="axis", show_misses=False),
        zoom_regions=[
            ("surface", (-0.2, 0.2, -1.5, 1.5)),
            ("post_surface", (0.5, 1.0, -0.5, 0.5)),
            ("pre_surface", (-1.0, -0.5, -0.5, 0.5)),
        ],
    ),
    CaptureSpec(
        name="cartesian_singlet",
        builder=_build_singlet,
        full_xlim=(-65.0, 46.0),
        full_ylim=(-6.0, 6.0),
        zoom_xlim=(-2.0, 6.0),
        zoom_ylim=(-3.0, 3.0),
        ray_kwargs=dict(width=0.35, extend_mode="axis", show_misses=False),
        zoom_regions=[
            ("front_surface", (-0.3, 0.1, -1.2, 1.2)),
            ("back_surface", (0.2, 0.6, -1.2, 1.2)),
            ("final_focus", (3.5, 4.5, -0.5, 0.5)),
        ],
    ),
    CaptureSpec(
        name="high_aperture_singlet",
        builder=_build_high_aperture,
        full_xlim=(-7.0, 7.0),
        full_ylim=(-4.0, 4.0),
        zoom_xlim=(-1.0, 5.0),
        zoom_ylim=(-3.0, 3.0),
        ray_kwargs=dict(width=0.35, extend_mode="axis", show_misses=False),
        zoom_regions=[
            ("front_surface", (-0.3, 0.1, -2.0, 2.0)),
            ("back_surface", (0.2, 0.6, -2.0, 2.0)),
            ("intermediate_axis", (1.5, 2.5, -0.5, 0.5)),
        ],
    ),
]


# ---------------------------------------------------------------------------
def _add_markers(viewer: Scene2DViewer, markers: Iterable[MarkerSpec]) -> None:
    positions = []
    colors = []
    sizes = []
    for spec in markers:
        positions.append(spec.position)
        colors.append(spec.color)
        sizes.append(spec.size)
    marker_visual = scene.visuals.Markers(
        pos=np.asarray(positions, dtype=np.float32),
        face_color=np.asarray(colors, dtype=np.float32),
        size=np.asarray(sizes, dtype=np.float32),
        parent=viewer.view.scene,
    )
    viewer._marker_visuals.append(marker_visual)


def _save(viewer: Scene2DViewer, path: Path, size: tuple[int, int]) -> np.ndarray:
    image_path = viewer.save(path, size=size)
    return imageio.imread(image_path)


def _world_to_pixels(x, y, xlim, ylim, size):
    x_min, x_max = xlim
    y_min, y_max = ylim
    width, height = size
    col = (x - x_min) / (x_max - x_min) * (width - 1)
    row = (y_max - y) / (y_max - y_min) * (height - 1)
    return int(np.clip(col, 0, width - 1)), int(np.clip(row, 0, height - 1))


def _region_pixels(region, xlim, ylim, size):
    x0, x1, y0, y1 = region
    x_min, x_max = min(x0, x1), max(x0, x1)
    y_min, y_max = min(y0, y1), max(y0, y1)
    width, height = size
    col0, row0 = _world_to_pixels(x_min, y_max, xlim, ylim, size)
    col1, row1 = _world_to_pixels(x_max, y_min, xlim, ylim, size)
    c0, c1 = sorted((col0, col1))
    r0, r1 = sorted((row0, row1))
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def _compute_metrics(image: np.ndarray, xlim, ylim, size, regions):
    data = {}
    rgb = image[..., :3].astype(np.float32) / 255.0
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    data["global"] = {
        "mean": float(luminance.mean()),
        "std": float(luminance.std()),
        "min": float(luminance.min()),
        "max": float(luminance.max()),
    }
    if regions:
        for label, bounds in regions:
            rows, cols = _region_pixels(bounds, xlim, ylim, size)
            region_data = luminance[rows, cols]
            data[label] = {
                "mean": float(region_data.mean()),
                "std": float(region_data.std()),
                "min": float(region_data.min()),
                "max": float(region_data.max()),
            }
    return data


# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    for spec in CAPTURES:
        example = spec.builder()
        metrics[spec.name] = {}

        for method in ("gl", "agg"):
            viewer = Scene2DViewer(
                x_lims=spec.full_xlim,
                y_lims=spec.full_ylim,
                show_axis=False,
                show=False,
                size=spec.full_size,
                line_method=method,
            )
            viewer.draw_surfaces(example.surfaces)
            viewer.draw_rays(example.tree, **spec.ray_kwargs)
            _add_markers(viewer, example.markers)

            # Full-frame render
            full_filename = OUTPUT_DIR / f"{spec.name}_full_{method}.png"
            image = _save(viewer, full_filename, spec.full_size)
            metrics[spec.name][f"full_{method}"] = _compute_metrics(
                image, spec.full_xlim, spec.full_ylim, spec.full_size, None
            )

            # Zoom render
            viewer.view.camera.set_range(x=spec.zoom_xlim, y=spec.zoom_ylim)
            zoom_filename = OUTPUT_DIR / f"{spec.name}_zoom_{method}.png"
            image = _save(viewer, zoom_filename, spec.zoom_size)
            metrics[spec.name][f"zoom_{method}"] = _compute_metrics(
                image, spec.zoom_xlim, spec.zoom_ylim, spec.zoom_size, spec.zoom_regions
            )
            viewer.close()

    report_path = REPORTS_DIR / "scene_metrics.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"Rendered snapshots written to {OUTPUT_DIR}")
    print(f"Metrics saved to {report_path}")


if __name__ == "__main__":
    main()
