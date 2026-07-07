"""Minimal refractive sphere demo with rays extended to the optical axis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer import CircleConic
from raytracer import Intersection2D, Ray2D
from raytracer import ParallelSource2D
from raytracer import RayTracer2D, TraceConfig
from raytracer import OpenGLViewer, RenderConfig

AMBIENT_N = 1.0
SPHERE_N = 1.52
_AXIS_TAIL_LENGTH = 8.0  # updated during scene construction


def build_scene(samples: int = 80):
    sphere = CircleConic(
        radius=2.0,
        center=np.array([0.0, 0.0]),
        surface_id="glass_sphere",
        n_exterior=AMBIENT_N,
        n_interior=SPHERE_N,
    )

    source = ParallelSource2D(
        origin=np.array([-5.0, 0.0]),
        direction=np.array([1.0, 0.0]),
        width=2.5,
        samples=samples,
    )

    tracer = RayTracer2D(
        surfaces=[sphere],
        config=TraceConfig(
            max_generations=2,
            allow_reflection=False,
            allow_refraction=True,
            ambient_n=AMBIENT_N,
        ),
    )

    tree = tracer.trace([source])
    _update_axis_tail_length([sphere])
    _freeze_internal_segments(tree)
    return sphere, tree, source


def _freeze_internal_segments(tree) -> None:
    for node in list(tree.nodes()):
        if node.parent_label is None:
            continue
        parent = tree[node.parent_label]
        parent_hit = parent.intersection
        child_hit = node.intersection
        if parent_hit is None or child_hit is None:
            continue
        new_origin = parent_hit.point
        node.ray = Ray2D(origin=new_origin, direction=node.ray.direction)

        exit_point = np.array(child_hit.point, dtype=float, copy=True)
        exit_normal = np.array(child_hit.normal, dtype=float, copy=True)
        exit_distance = float(np.linalg.norm(exit_point - new_origin))

        axis_lambda = _lambda_to_axis(new_origin, node.ray.direction)
        if axis_lambda is not None:
            if np.isclose(axis_lambda, 0.0, atol=1e-9) and exit_distance > 0.0:
                axis_point = exit_point
                axis_lambda = exit_distance
            else:
                axis_point = new_origin + node.ray.direction * axis_lambda
            axis_normal = np.array([0.0, 1.0])
            node.intersection = Intersection2D(
                point=axis_point,
                normal=axis_normal,
                distance=axis_lambda,
                surface_id="optical_axis",
                meta={
                    "axis_lambda": axis_lambda,
                    "axis_point": np.array(axis_point, copy=True),
                    "exit": {
                        "point": exit_point,
                        "normal": exit_normal,
                        "distance": exit_distance,
                        "surface_id": child_hit.surface_id,
                    },
                },
            )
        else:
            node.intersection = None


def _lambda_to_axis(origin: np.ndarray, direction: np.ndarray, *, axis_y: float = 0.0, eps: float = 1e-9) -> float | None:
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    dy = float(direction[1])
    y_offset = origin[1] - float(axis_y)

    if abs(dy) <= eps:
        if abs(y_offset) <= eps:
            return 0.0
        return None

    lam = (float(axis_y) - origin[1]) / dy
    if lam < -eps:
        return None
    if lam < 0.0:
        return 0.0
    return float(lam)


def _extend_to_axis(node):
    if node.parent_label is None:
        return None
    start = node.ray.origin
    direction = node.ray.direction
    lam = _lambda_to_axis(start, direction)
    if lam is None:
        return start + direction * _AXIS_TAIL_LENGTH
    return start + direction * lam


def _update_axis_tail_length(surfaces) -> None:
    """Derive a fallback tail length from supplied surfaces."""

    global _AXIS_TAIL_LENGTH
    extent = 0.0
    for surface in surfaces:
        try:
            pts = surface.polyline(samples=256)
        except Exception:  # pragma: no cover - defensive guard
            pts = np.empty((0, 2), dtype=float)
        if pts.size:
            extent = max(extent, float(np.max(np.linalg.norm(pts, axis=1))))
    if extent <= 0.0:
        extent = 5.0
    _AXIS_TAIL_LENGTH = extent + 2.0


def _verify_lambda_helper() -> None:
    origin = np.array([-0.5, 0.3])
    direction = np.array([0.2, -0.4])
    lam = _lambda_to_axis(origin, direction)
    if lam is None or not np.isclose(origin[1] + lam * direction[1], 0.0, atol=1e-9):
        raise AssertionError("Axis lambda solver failed for generic case")

    parallel = _lambda_to_axis(np.array([1.0, 0.2]), np.array([1.0, 0.0]))
    if parallel is not None:
        raise AssertionError("Parallel ray should not report an axis hit")

    on_axis = _lambda_to_axis(np.array([2.0, 0.0]), np.array([1.0, 0.0]))
    if on_axis != 0.0:
        raise AssertionError("Ray already on axis should yield zero distance")


if __debug__:
    _verify_lambda_helper()


def main() -> None:
    sphere, tree, source = build_scene(samples=120)

    config = RenderConfig(
        ray_width=0.03,
        sigma_factor=0.02,
        accumulation_mode="squared",
        default_intensity=0.1,
        weight_scale=min(1.0, 80.0 / max(1, source.samples)),
        min_pixels=1.0,
    )

    viewer = OpenGLViewer(
        x_lims=(-6.0, 6.0),
        y_lims=(-4.0, 4.0),
        size=(1200, 800),
        bgcolor="black",
        render_config=config,
    )

    viewer.draw_surfaces([sphere], color="white", width=2.0)

    def color_resolver(node):
        if node.parent_label is None:
            return (1.0, 1.0, 1.0, 0.45)
        return (0.8, 0.9, 1.0, 0.9)

    viewer.draw_rays(
        tree,
        tail_length=_AXIS_TAIL_LENGTH,
        color_resolver=color_resolver,
        intensity_resolver=lambda n: 1.0,
        show_misses=False,
        marker_color="gold",
        marker_size=6.5,
        leaf_extension=_extend_to_axis,
    )

    print("\nRefractive Sphere Demo (Minimal)")
    print("--------------------------------")
    print("Rays refract on entry and extend straight until they reach the optical axis (x = 0).")
    viewer.run()


if __name__ == "__main__":
    main()
