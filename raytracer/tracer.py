"""Breadth-first 2-D ray tracer supporting reflections and refractions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .physics import reflect, refract
from .rays import Ray2D, RayLabeler, RayNode, RayTree
from .sources import Source2D
from .surfaces import Surface2D


@dataclass
class TraceConfig:
    """Configuration for the simplified ray tracer."""

    max_generations: int = 5
    allow_reflection: bool = True
    allow_refraction: bool = False
    ambient_n: float = 1.0
    epsilon: float = 1e-6


class RayTracer2D:
    """Iteratively propagates rays across a set of 2-D surfaces."""

    def __init__(self, surfaces: Sequence[Surface2D], config: TraceConfig | None = None) -> None:
        if not surfaces:
            raise ValueError("At least one surface must be provided")
        self.surfaces = list(surfaces)
        self.config = config or TraceConfig()
        self._labeler = RayLabeler()

    def trace(self, sources: Iterable[Source2D]) -> RayTree:
        tree = RayTree()
        queue: deque[str] = deque()

        for source in sources:
            for seed in source.emit():
                label = self._labeler.new_primary()
                ray = Ray2D(seed.origin, seed.direction)
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

            hit = self._first_hit(node.ray)
            node.intersection = hit

            if hit is None or node.generation >= self.config.max_generations:
                continue

            surface = hit.surface if hit.surface is not None else None
            normal = hit.normal.copy()
            incident_n = node.medium_n

            exterior = getattr(surface, "n_exterior", incident_n) if surface else incident_n
            interior = getattr(surface, "n_interior", incident_n) if surface else incident_n

            if surface is not None and hit.surface is None:
                hit.surface = surface

            if np.dot(node.ray.direction, normal) > 0.0:
                normal = -normal
                hit.normal = normal

            if surface is not None:
                if np.isclose(incident_n, exterior, atol=1e-7):
                    transmit_medium = interior
                elif np.isclose(incident_n, interior, atol=1e-7):
                    transmit_medium = exterior
                else:
                    transmit_medium = interior
            else:
                transmit_medium = incident_n

            if self.config.allow_reflection:
                refl_dir = reflect(node.ray.direction, normal)
                refl_origin = hit.point + refl_dir * self.config.epsilon
                self._spawn_child(tree, queue, node, refl_origin, refl_dir, incident_n)

            if self.config.allow_refraction:
                refr_dir = refract(node.ray.direction, normal, incident_n, transmit_medium)
                if refr_dir is not None:
                    refr_origin = hit.point + refr_dir * self.config.epsilon
                    self._spawn_child(tree, queue, node, refr_origin, refr_dir, transmit_medium)

        return tree

    def _first_hit(self, ray: Ray2D):
        best_hit = None
        min_distance = max(self.config.epsilon * 5.0, 1e-9)
        for surface in self.surfaces:
            hit = surface.intersect(ray)
            if hit is None:
                continue
            if hit.distance <= min_distance:
                continue
            if hit.surface is None:
                hit.surface = surface
            if best_hit is None or hit.distance < best_hit.distance:
                best_hit = hit
        return best_hit

    def _spawn_child(
        self,
        tree: RayTree,
        queue: deque[str],
        parent: RayNode,
        origin,
        direction,
        medium_n: float,
    ) -> None:
        label = self._labeler.child(parent.label)
        child = RayNode(
            label=label,
            ray=Ray2D(origin, direction),
            parent_label=parent.label,
            generation=parent.generation + 1,
            medium_n=medium_n,
        )
        tree.add(child)
        queue.append(label)


__all__ = ["TraceConfig", "RayTracer2D"]
