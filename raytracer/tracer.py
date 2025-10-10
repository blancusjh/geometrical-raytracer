"""Breadth-first multi-generation ray tracer for 2D and 3D geometries."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence, Type

import numpy as np

from .physics import reflect, refract
from .rays import Ray2D, Ray3D, RayLabeler, RayND, RayNode, RayTree
from .sources import Source2D, Source3D, SourceND

try:  # pragma: no cover - avoid circular import at runtime
    from .surfaces import SurfaceND
except ImportError:  # pragma: no cover - only for type checkers
    SurfaceND = object  # type: ignore


@dataclass
class TraceConfig:
    max_generations: int = 5
    allow_reflection: bool = True
    allow_refraction: bool = False
    ambient_n: float = 1.0
    exit_on_no_hit: bool = True
    epsilon: float = 1e-6


class RayTracerND:
    """Iteratively propagates rays through a list of generic surfaces."""

    def __init__(
        self,
        surfaces: Sequence[SurfaceND],
        *,
        ray_type: Type[RayND],
        config: TraceConfig | None = None,
    ) -> None:
        if not surfaces:
            raise ValueError("At least one surface must be provided")
        self.surfaces = list(surfaces)
        self.ray_type = ray_type
        self.config = config or TraceConfig()
        self._labeler = RayLabeler()

        surface_dims = {surface.dimension for surface in self.surfaces}
        if len(surface_dims) != 1:
            raise ValueError("All surfaces must share the same embedding dimension")
        self._dimension = surface_dims.pop()

        try:
            test_origin = np.zeros(self._dimension)
            test_direction = np.zeros(self._dimension)
            test_direction[0] = 1.0
            candidate = self.ray_type(test_origin, test_direction)
        except Exception as exc:  # pragma: no cover - defensive programming
            raise ValueError("ray_type must accept vectors matching the surface dimension") from exc
        if getattr(candidate, "dimension", None) != self._dimension:
            raise ValueError("ray_type dimension does not match surface dimension")

    @property
    def dimension(self) -> int:
        return self._dimension

    def trace(self, sources: Iterable[SourceND]) -> RayTree:
        tree = RayTree()
        queue: deque[str] = deque()

        for source in sources:
            if source.dimension != self.dimension:
                raise ValueError(
                    f"Source dimension {source.dimension} incompatible with tracer dimension {self.dimension}"
                )
            for seed in source.emit():
                label = self._labeler.new_primary()
                ray = self.ray_type(seed.origin, seed.direction)
                node = RayNode(
                    label=label,
                    ray=ray,
                    parent_label=None,
                    generation=1,
                    medium_n=self.config.ambient_n,
                )
                tree.add(node)
                queue.append(label)

        while queue:
            label = queue.popleft()
            node = tree[label]

            if node.generation > self.config.max_generations:
                continue

            hit = self._first_hit(node.ray)
            node.intersection = hit

            if hit is None:
                if self.config.exit_on_no_hit:
                    continue
                continue

            surface = getattr(hit, "surface", None)
            surface_normal = hit.normal
            incident_medium = node.medium_n
            transmit_medium = incident_medium

            if surface is not None and hasattr(surface, "n_exterior") and hasattr(surface, "n_interior"):
                exterior = float(surface.n_exterior)
                interior = float(surface.n_interior)
                if np.isclose(incident_medium, exterior, atol=1e-7):
                    transmit_medium = interior
                    if np.dot(node.ray.direction, surface_normal) > 0.0:
                        surface_normal = -surface_normal
                        hit.normal = surface_normal
                elif np.isclose(incident_medium, interior, atol=1e-7):
                    transmit_medium = exterior
                    if np.dot(node.ray.direction, surface_normal) < 0.0:
                        surface_normal = -surface_normal
                        hit.normal = surface_normal
                else:
                    if np.dot(node.ray.direction, surface_normal) > 0.0:
                        surface_normal = -surface_normal
                        hit.normal = surface_normal
                    transmit_medium = interior if abs(incident_medium - exterior) < abs(incident_medium - interior) else exterior
            else:
                if np.dot(node.ray.direction, surface_normal) > 0.0:
                    surface_normal = -surface_normal
                    hit.normal = surface_normal

            if node.generation >= self.config.max_generations:
                continue

            if self.config.allow_reflection:
                refl_dir = reflect(node.ray.direction, surface_normal)
                refl_origin = hit.point + refl_dir * self.config.epsilon
                self._enqueue_child(
                    tree,
                    queue,
                    node,
                    refl_origin,
                    refl_dir,
                    branch_index=0,
                    medium_n=incident_medium,
                )

            if self.config.allow_refraction:
                refr_dir = refract(node.ray.direction, surface_normal, incident_medium, transmit_medium)
                if refr_dir is not None:
                    refr_origin = hit.point + refr_dir * self.config.epsilon
                    self._enqueue_child(
                        tree,
                        queue,
                        node,
                        refr_origin,
                        refr_dir,
                        branch_index=1,
                        medium_n=transmit_medium,
                    )

        return tree

    def _first_hit(self, ray: RayND):
        best_hit = None
        for surface in self.surfaces:
            hit = surface.first_intersection(ray)
            if hit is None:
                continue
            if best_hit is None or hit.distance < best_hit.distance:
                best_hit = hit
        return best_hit

    def _enqueue_child(
        self,
        tree: RayTree,
        queue: deque[str],
        parent: RayNode,
        origin,
        direction,
        branch_index: int,
        *,
        medium_n: float,
    ) -> None:
        label = self._labeler.child(parent.label, branch_index)
        child = RayNode(
            label=label,
            ray=self.ray_type(origin, direction),
            parent_label=parent.label,
            generation=parent.generation + 1,
            medium_n=medium_n,
        )
        tree.add(child)
        queue.append(label)


class RayTracer2D(RayTracerND):
    """Compatibility wrapper for 2D ray tracing."""

    def __init__(self, surfaces: Sequence[SurfaceND], config: TraceConfig | None = None) -> None:
        super().__init__(surfaces, ray_type=Ray2D, config=config)

    def trace(self, sources: Iterable[Source2D]) -> RayTree:
        return super().trace(sources)


class RayTracer3D(RayTracerND):
    """3D ray tracer for axisymmetric and general surfaces."""

    def __init__(self, surfaces: Sequence[SurfaceND], config: TraceConfig | None = None) -> None:
        super().__init__(surfaces, ray_type=Ray3D, config=config)

    def trace(self, sources: Iterable[Source3D]) -> RayTree:
        return super().trace(sources)
