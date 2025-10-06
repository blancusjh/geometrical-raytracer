"""Generate a markdown report summarising rendered scenes and metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "scene_metrics.json"

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer.dioptrics import CartesianDioptrique, CartesianSinglet
from raytracer.geometry import EllipseConic
from raytracer.sources import PointSource2D
from raytracer.tracer import RayTracer2D, TraceConfig


@dataclass
class FocusResult:
    count: int
    mean_miss: float
    max_miss: float


def _line_point_distance(ray, target: np.ndarray) -> float:
    origin = ray.origin
    direction = ray.direction
    diff = target - origin
    return float(abs(direction[0] * diff[1] - direction[1] * diff[0]))


def _ellipse_focus() -> FocusResult:
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
    major_axis_dir = ellipse._R @ np.array([1.0, 0.0])
    focal_offset = np.sqrt(semi_major**2 - semi_minor**2)
    secondary_focus = ellipse.focus.copy() - 2.0 * focal_offset * major_axis_dir

    final_nodes = [node for node in tree.nodes() if node.generation == 2]
    misses = np.array([_line_point_distance(node.ray, secondary_focus) for node in final_nodes])
    return FocusResult(len(final_nodes), float(misses.mean()), float(misses.max()))


def _dioptrique_focus() -> FocusResult:
    z_object = 60.0
    z_image = 30.0
    n_air = 1.0
    n_glass = 1.52
    surface = CartesianDioptrique(z0=z_object, zi=z_image, n_exterior=n_air, n_interior=n_glass, aperture_radius=0.7, samples=800)
    source = PointSource2D(origin=np.array([-z_object, 0.0]), axis_direction=np.array([1.0, 0.0]), aperture=np.deg2rad(4.0), samples=21)
    config = TraceConfig(max_generations=4, allow_reflection=False, allow_refraction=True, ambient_n=n_air)
    tracer = RayTracer2D([surface], config)
    tree = tracer.trace([source])
    final_nodes = [node for node in tree.nodes() if node.generation == 2]
    target = np.array([z_image, 0.0])
    misses = np.array([_line_point_distance(node.ray, target) for node in final_nodes])
    return FocusResult(len(final_nodes), float(misses.mean()), float(misses.max()))


def _singlet_focus(aperture_radius: float, samples: int, aperture_deg: float) -> FocusResult:
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
        aperture_radius=aperture_radius,
        samples=900,
    )
    front, back = singlet.surfaces()
    source = PointSource2D(
        origin=np.array([-z_object, 0.0]),
        axis_direction=np.array([1.0, 0.0]),
        aperture=np.deg2rad(aperture_deg),
        samples=samples,
    )
    config = TraceConfig(max_generations=4, allow_reflection=False, allow_refraction=True, ambient_n=n_air)
    tracer = RayTracer2D([front, back], config)
    tree = tracer.trace([source])
    final_nodes = [node for node in tree.nodes() if node.generation == 3]
    target = np.array([thickness + z_image, 0.0])
    misses = np.array([_line_point_distance(node.ray, target) for node in final_nodes])
    return FocusResult(len(final_nodes), float(misses.mean()), float(misses.max()))


def _high_aperture_focus() -> FocusResult:
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
    config = TraceConfig(max_generations=3, allow_reflection=False, allow_refraction=True, ambient_n=n_air)
    tracer = RayTracer2D([front, back], config)
    tree = tracer.trace([source])
    final_nodes = [node for node in tree.nodes() if node.generation == 3]
    target = np.array([thickness + z_image, 0.0])
    misses = np.array([_line_point_distance(node.ray, target) for node in final_nodes])
    return FocusResult(len(final_nodes), float(misses.mean()), float(misses.max()))


FOCUS_BUILDERS = {
    "ellipse": _ellipse_focus,
    "cartesian_dioptrique": _dioptrique_focus,
    "cartesian_singlet": lambda: _singlet_focus(0.6, 21, 3.0),
    "high_aperture_singlet": _high_aperture_focus,
}


def _format_focus(result: FocusResult) -> str:
    return f"rays={result.count}, mean miss={result.mean_miss:.4f}, max miss={result.max_miss:.4f}"


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text())
    focus_results = {name: builder() for name, builder in FOCUS_BUILDERS.items()}

    lines = ["# Scene Observations", ""]

    def add_section(title: str):
        lines.append(f"## {title}")
        lines.append("")

    for example_name in ("ellipse", "cartesian_dioptrique", "cartesian_singlet", "high_aperture_singlet"):
        add_section(example_name.replace("_", " ").title())
        example_metrics = metrics[example_name]
        focus_result = focus_results.get(example_name)

        def format_region(method: str, region: str) -> str:
            data = example_metrics[f"zoom_{method}"][region]
            return f"{data['mean']:.4f} (±{data['std']:.4f})"

        if f"zoom_gl" in example_metrics and f"zoom_agg" in example_metrics:
            lines.append("**Zoom luminance (mean ± std)**")
            regions = [r for r in example_metrics["zoom_gl"].keys() if r != "global"]
            if not regions:
                regions = ["global"]
            lines.append("| Region | GL | AGG | Δ (AGG-GL) |")
            lines.append("| --- | --- | --- | --- |")
            for region in regions:
                gl_val = example_metrics["zoom_gl"][region]
                agg_val = example_metrics["zoom_agg"][region]
                delta = agg_val["mean"] - gl_val["mean"]
                lines.append(
                    f"| {region} | {gl_val['mean']:.4f} ± {gl_val['std']:.4f} | {agg_val['mean']:.4f} ± {agg_val['std']:.4f} | {delta:.4f} |"
                )
            lines.append("")

        if focus_result is not None:
            lines.append(f"**Focus miss**: {_format_focus(focus_result)}")
            lines.append("")

        lines.append(
            "**Notes:**"\
            "\n- See renders in `renders/` for side-by-side GL/AGG comparisons."\
            "\n- AGG yields smoother edges but reduces luminance where many segments overlap."\
        )
        lines.append("")

    report_path = REPORTS_DIR / "scene_observations.md"
    report_path.write_text("\n".join(lines))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
