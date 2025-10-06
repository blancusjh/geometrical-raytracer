"""Breadth-first multi-generation ray tracer for 2D geometries."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .geometry import Surface2D
from .physics import reflect, refract
from .rays import Ray, RayLabeler, RayNode, RayTree
from .sources import Source2D


@dataclass
class TraceConfig:
    max_generations: int = 5
    allow_reflection: bool = True
    allow_refraction: bool = False
    n1: float = 1.0
    n2: float = 1.0
    ambient_n: float = 1.0
    exit_on_no_hit: bool = True
    epsilon: float = 1e-6


class RayTracer2D:
    """Iteratively propagates rays through a list of 2D surfaces."""

    def __init__(self, surfaces: Sequence[Surface2D], config: TraceConfig | None = None) -> None:
        self.surfaces = list(surfaces)
        self.config = config or TraceConfig()
        self._labeler = RayLabeler()

    def trace(self, sources: Iterable[Source2D]) -> RayTree:
        tree = RayTree()
        queue: deque[str] = deque()

        for source in sources:
            for seed in source.emit():
                label = self._labeler.new_primary()
                node = RayNode(
                    label=label,
                    ray=Ray(seed.origin, seed.direction),
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
                self._enqueue_child(tree, queue, node, refl_origin, refl_dir, branch_index=0, medium_n=incident_medium)

            if self.config.allow_refraction:
                refr_dir = refract(node.ray.direction, surface_normal, incident_medium, transmit_medium)
                if refr_dir is not None:
                    refr_origin = hit.point + refr_dir * self.config.epsilon
                    self._enqueue_child(tree, queue, node, refr_origin, refr_dir, branch_index=1, medium_n=transmit_medium)

        return tree

    def _first_hit(self, ray: Ray):
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
            ray=Ray(origin, direction),
            parent_label=parent.label,
            generation=parent.generation + 1,
            medium_n=medium_n,
        )
        tree.add(child)
        queue.append(label)
