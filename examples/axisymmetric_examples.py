"""Illustrative 3D axisymmetric surface examples with ray tracing and VisPy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# Allow running the script directly from the repository root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from raytracer.dioptrics import SigmaCurve, _rho_for_radius
from raytracer.geometry import CircleConic, EllipseConic, ParabolaConic
from raytracer.geometry3d import AxisymmetricCartesianSurface3D, AxisymmetricConicSurface3D
from raytracer.rays import IntersectionND, RayNode, RayTree
from raytracer.sources import PointSource3D
from raytracer.tracer import RayTracer3D, TraceConfig

ExampleResult = Tuple[
    List[AxisymmetricConicSurface3D | AxisymmetricCartesianSurface3D],
    RayTree,
    Dict[str, Dict[str, float]],
    Dict[str, Any],
]


def example_conic_ellipse(curve_samples: int = 256) -> ExampleResult:
    a = 1.5
    b = 1.0
    base = EllipseConic(semi_major=a, semi_minor=b)
    surface = AxisymmetricConicSurface3D(
        profile=base.profile,
        profile_curve=base,
        surface_id="ellipse_3d",
        n_exterior=1.0,
        n_interior=1.0,
    )
    surfaces = [surface]

    focus = np.array([0.0, 0.0, 0.0])
    eps = 0.06
    source_point = focus + eps*np.array([0, 0, 1])

    k = 30

    axis = np.array([-1.0, 0.0, 0.0])
    source = PointSource3D(
        origin=source_point,
        axis_direction=axis,
        aperture=np.deg2rad(65.0),
        theta_samples= k, 
        phi_samples= k*2,
    )

    tracer = RayTracer3D(surfaces, 
                         TraceConfig(max_generations=2, 
                                     allow_reflection=True))

    tree = tracer.trace([source])

    c = np.sqrt(a**2 - b**2)
    target_focus = np.array([2.0 * c, 0.0, 0.0])

    options = {
        surface.surface_id: {
            "generatrix_samples": curve_samples,
            "color": (0.6, 0.6, 0.6, 0.32),
        }
    }
    metadata = {"focus": focus, "target_focus": target_focus}
    return surfaces, tree, options, metadata



def example_conic_paraboloid(curve_samples: int = 256) -> ExampleResult:
    base = ParabolaConic(p=1.5, focus=np.array([0.0, 0.0]))
    surface = AxisymmetricConicSurface3D(
        profile=base.profile,
        profile_curve=base,
        surface_id="paraboloid_3d",
        n_exterior=1.0,
        n_interior=1.0,
    )
    surfaces = [surface]

    focus = np.array([0.0, 0.0, 0.0])
    axis = np.array([1.0, 0.0, 0.0])
    source = PointSource3D(
        origin=focus,
        axis_direction=axis,
        aperture=np.deg2rad(45.0),
        theta_samples=9,
        phi_samples=18,
    )

    tracer = RayTracer3D(surfaces, TraceConfig(max_generations=2, allow_reflection=True))
    tree = tracer.trace([source])

    options = {
        surface.surface_id: {
            "rho_max": 1.5,
            "generatrix_samples": curve_samples,
            "color": (0.25, 0.65, 0.95, 0.38),
        }
    }
    metadata = {"focus": focus}
    return surfaces, tree, options, metadata


def example_conic_sphere(curve_samples: int = 256) -> ExampleResult:
    base = CircleConic(radius=1.0)
    surface = AxisymmetricConicSurface3D(
        profile=base.profile,
        profile_curve=base,
        surface_id="sphere_3d",
        n_exterior=1.0,
        n_interior=1.5,
    )
    surfaces = [surface]

    focus = np.array([0.0, 0.0, 0.0])
    axis = np.array([1.0, 0.0, 0.0])
    source = PointSource3D(
        origin=focus,
        axis_direction=axis,
        aperture=np.deg2rad(70.0),
        theta_samples=10,
        phi_samples=20,
    )

    tracer = RayTracer3D(
        surfaces,
        TraceConfig(max_generations=2, allow_reflection=True, allow_refraction=True),
    )
    tree = tracer.trace([source])

    options = {
        surface.surface_id: {
            "generatrix_samples": curve_samples,
            "color": (0.2, 0.8, 0.6, 0.35),
        }
    }
    metadata = {"focus": focus}
    return surfaces, tree, options, metadata


def _build_cartesian_surface(
    z0: float,
    zi: float,
    n0: float,
    ni: float,
    aperture_radius: float,
    surface_id: str,
) -> AxisymmetricCartesianSurface3D:
    curve = SigmaCurve(z0=z0, zi=zi, n0=n0, ni=ni)
    rho_max = _rho_for_radius(z0, zi, n0, ni, aperture_radius)
    return AxisymmetricCartesianSurface3D(curve=curve, rho_max=rho_max, surface_id=surface_id)


def example_cartesian_positive_lens(curve_samples: int = 256) -> ExampleResult:
    surface = _build_cartesian_surface(
        z0=2.0,
        zi=4.0,
        n0=1.0,
        ni=1.5,
        aperture_radius=0.6,
        surface_id="cartesian_positive",
    )
    surfaces = [surface]

    source = PointSource3D(
        origin=np.array([-1.0, 0.0, 0.0]),
        axis_direction=np.array([1.0, 0.0, 0.0]),
        aperture=np.deg2rad(18.0),
        theta_samples=5,
        phi_samples=12,
    )

    tracer = RayTracer3D(
        surfaces,
        TraceConfig(max_generations=3, allow_reflection=True, allow_refraction=True),
    )
    tree = tracer.trace([source])

    options = {
        surface.surface_id: {
            "generatrix_samples": curve_samples,
            "color": (0.85, 0.5, 0.2, 0.3),
        }
    }
    metadata = {"focus": np.array([-1.0, 0.0, 0.0])}
    return surfaces, tree, options, metadata


def example_cartesian_negative_lens(curve_samples: int = 256) -> ExampleResult:
    surface = _build_cartesian_surface(
        z0=-1.5,
        zi=2.5,
        n0=1.5,
        ni=1.0,
        aperture_radius=0.5,
        surface_id="cartesian_negative",
    )
    surfaces = [surface]

    source = PointSource3D(
        origin=np.array([1.0, 0.0, 0.0]),
        axis_direction=np.array([-1.0, 0.0, 0.0]),
        aperture=np.deg2rad(18.0),
        theta_samples=5,
        phi_samples=12,
    )

    tracer = RayTracer3D(
        surfaces,
        TraceConfig(max_generations=3, allow_reflection=True, allow_refraction=True),
    )
    tree = tracer.trace([source])

    options = {
        surface.surface_id: {
            "generatrix_samples": curve_samples,
            "color": (0.95, 0.75, 0.25, 0.3),
        }
    }
    metadata = {"focus": np.array([1.0, 0.0, 0.0])}
    return surfaces, tree, options, metadata


EXAMPLES: Dict[str, Callable[[int], ExampleResult]] = {
    "ellipse": example_conic_ellipse,
    "paraboloid": example_conic_paraboloid,
    "sphere": example_conic_sphere,
    "cartesian-positive": example_cartesian_positive_lens,
    "cartesian-negative": example_cartesian_negative_lens,
}


def _sorted_nodes(tree: RayTree) -> List[RayNode]:
    return sorted(tree.nodes(), key=lambda node: node.label)


def _collect_segments(tree: RayTree) -> Tuple[List[RayNode], List[IntersectionND | None], List[float | None]]:
    nodes = _sorted_nodes(tree)
    hits: List[IntersectionND | None] = []
    lengths: List[float | None] = []
    for node in nodes:
        hit = node.intersection
        hits.append(hit)
        lengths.append(hit.distance if hit is not None else None)
    return nodes, hits, lengths


def _print_trace_summary(name: str, tree: RayTree, metadata: Dict[str, Any]) -> None:
    title = name.replace("-", " ").title()
    print(f"\n=== {title} ===")
    focus = metadata.get("focus")
    target_focus = metadata.get("target_focus")
    if focus is not None:
        print(f"Focus position: {focus}")
    if target_focus is not None:
        print(f"Target focus: {target_focus}")

    nodes = _sorted_nodes(tree)
    hit_nodes = [node for node in nodes if node.intersection is not None]
    miss_nodes = [node for node in nodes if node.intersection is None]

    max_print = 12

    for node in hit_nodes[:max_print]:
        hit = node.intersection
        label = node.label
        print(f"[{label}] gen={node.generation} point {hit.point}")
        print(f"    normal {hit.normal}")
        if hit.parameters is not None:
            print(f"    parameters [x, rho, phi]: {hit.parameters}")

    if len(hit_nodes) > max_print:
        print(f"    … {len(hit_nodes) - max_print} additional hits omitted")

    if target_focus is not None and miss_nodes:
        for node in miss_nodes[:max_print]:
            direction = node.ray.direction
            vec = target_focus - node.ray.origin
            t = np.dot(vec, direction)
            if t <= 0:
                continue
            closest = node.ray.origin + direction * t
            miss = np.linalg.norm(closest - target_focus)
            print(f"[{node.label}] gen={node.generation} passes target focus | miss = {miss:.3e}")
        if len(miss_nodes) > max_print:
            print(f"    … {len(miss_nodes) - max_print} additional rays without hits")


def _run_examples(
    names: Iterable[str],
    *,
    show_vispy: bool,
    curve_samples: int,
    phi_samples: int,
) -> None:
    for name in names:
        surfaces, tree, options, metadata = EXAMPLES[name](curve_samples)
        _print_trace_summary(name, tree, metadata)

        if show_vispy:
            from raytracer.visualization_vispy import visualize_axisymmetric_scene

            nodes, hits, lengths = _collect_segments(tree)

            num_source_rays = sum(1 for node in nodes if node.generation == 1)
            ray_alpha = 40.10 / num_source_rays if num_source_rays > 0 else 0.1
            hit_color = (1.0, 1.0, 1.0, ray_alpha)
            miss_color = (1.0, 0.5, 1.0, ray_alpha)
            ray_colors = [hit_color if hit is not None else miss_color for hit in hits]

            camera_center = None
            if name == "ellipse":
                camera_center = np.array([0.0, 0.0, 0.0])

            rays = [node.ray for node in nodes]
            visualize_axisymmetric_scene(
                surfaces=surfaces,
                rays=rays,
                hits=hits,
                surface_options=options,
                ray_lengths=lengths,
                ray_colors=ray_colors,
                title=f"{name.replace('-', ' ').title()}",
                phi_samples=phi_samples,
                camera_center=camera_center,
                allow_marker_toggle=True,
            )


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="3D axisymmetric surfaces demo")
    parser.add_argument(
        "--example",
        choices=list(EXAMPLES.keys()) + ["all"],
        default="ellipse",
        help="Choose a specific example to run",
    )
    parser.add_argument(
        "--vispy",
        action="store_true",
        help="Open an interactive VisPy window for the selected example(s)",
    )
    parser.add_argument(
        "--curve-samples",
        type=int,
        default=256,
        help="Number of samples along the generating curve",
    )
    parser.add_argument(
        "--phi-samples",
        type=int,
        default=128,
        help="Number of azimuthal samples for surface revolution",
    )
    args = parser.parse_args(argv)

    selected: Sequence[str]
    if args.example == "all":
        selected = EXAMPLES.keys()
    else:
        selected = [args.example]

    _run_examples(
        selected,
        show_vispy=args.vispy,
        curve_samples=args.curve_samples,
        phi_samples=args.phi_samples,
    )


if __name__ == "__main__":
    main()
