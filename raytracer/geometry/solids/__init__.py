"""Solids: closed regions of space, as geometry only.

A solid answers two questions in its local frame: does it contain a point,
and where does a ray first cross its boundary (with the outward normal
there). What fills it is not the solid's business — an element pairs a
solid with a material.

Polyhedra (:class:`Box`, :class:`Mesh`) also expose their planar
:meth:`~Polyhedron.facets`, each a pose and a polygon. An element made of
one turns every facet into an optical interface with its own coating.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from ..apertures import Polygon
from ..transforms import Pose, Rotation
from ..vectors import normalize


class Solid(ABC):
    """A closed region of space in its local frame."""

    @abstractmethod
    def contains(self, points: ArrayLike) -> np.ndarray:
        """Whether each point ``(N, 3)`` is inside (boundary included)."""

    @abstractmethod
    def intersect(self, origins: np.ndarray, directions: np.ndarray, *, t_min: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
        """First boundary crossing ahead of each ray: ``(t, outward normal)``; ``t = inf`` on a miss."""

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self._key() == other._key()  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._key()))

    def _key(self) -> tuple:
        return tuple(vars(self).values())


def _first_positive(roots: np.ndarray, t_min: float) -> np.ndarray:
    roots = np.where(np.isfinite(roots) & (roots > t_min), roots, np.inf)
    return roots.min(axis=1)


class Sphere(Solid):
    """Ball of *radius* centred on the local origin."""

    def __init__(self, radius: float) -> None:
        if not float(radius) > 0.0:
            raise ValueError("a sphere has a positive radius")
        self.radius = float(radius)

    def contains(self, points):
        return np.linalg.norm(np.asarray(points, dtype=float), axis=-1) <= self.radius

    def intersect(self, origins, directions, *, t_min=1e-9):
        b = np.einsum("ij,ij->i", origins, directions)
        c = np.einsum("ij,ij->i", origins, origins) - self.radius**2
        disc = b * b - c
        root = np.sqrt(np.where(disc >= 0.0, disc, np.nan))
        t = _first_positive(np.stack([-b - root, -b + root], axis=1), t_min)
        p = origins + np.where(np.isfinite(t), t, 0.0)[:, None] * directions
        return t, p / self.radius

    def __repr__(self) -> str:
        return f"Sphere(radius={self.radius})"


class Cylinder(Solid):
    """Right circular cylinder of *radius* along local z, from ``z = 0`` to ``z = length``."""

    def __init__(self, radius: float, length: float) -> None:
        if not (float(radius) > 0.0 and float(length) > 0.0):
            raise ValueError("a cylinder has positive radius and length")
        self.radius, self.length = float(radius), float(length)

    def contains(self, points):
        p = np.asarray(points, dtype=float)
        return (np.hypot(p[..., 0], p[..., 1]) <= self.radius) & (p[..., 2] >= 0.0) & (p[..., 2] <= self.length)

    def wall_hits(self, origins: np.ndarray, directions: np.ndarray, *, t_min: float = 1e-9) -> np.ndarray:
        """Ray parameters ``(N, 2)`` where each ray crosses the curved wall; ``inf`` where it does not."""

        o, d = origins, directions
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        b = o[:, 0] * d[:, 0] + o[:, 1] * d[:, 1]
        c = o[:, 0] ** 2 + o[:, 1] ** 2 - self.radius**2
        with np.errstate(divide="ignore", invalid="ignore"):
            root = np.sqrt(np.where(b * b - a * c >= 0.0, b * b - a * c, np.nan))
            roots = np.stack([(-b - root) / a, (-b + root) / a], axis=1)
        z = o[:, 2:3] + roots * d[:, 2:3]
        on_wall = np.isfinite(roots) & (roots > t_min) & (z >= 0.0) & (z <= self.length)
        return np.where(on_wall, roots, np.inf)

    def intersect(self, origins, directions, *, t_min=1e-9):
        o, d = origins, directions
        n = o.shape[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            caps = np.stack([-o[:, 2] / d[:, 2], (self.length - o[:, 2]) / d[:, 2]], axis=1)
            pc = o[:, None, :2] + caps[:, :, None] * d[:, None, :2]
            caps = np.where(np.hypot(pc[..., 0], pc[..., 1]) <= self.radius, caps, np.inf)
        roots = np.concatenate([self.wall_hits(o, d, t_min=t_min), caps], axis=1)
        roots = np.where(np.isfinite(roots) & (roots > t_min), roots, np.inf)
        which = roots.argmin(axis=1)
        t = roots[np.arange(n), which]
        p = o + np.where(np.isfinite(t), t, 0.0)[:, None] * d
        normal = np.zeros((n, 3))
        on_wall = which < 2
        normal[on_wall, :2] = p[on_wall, :2] / self.radius
        normal[which == 2, 2] = -1.0
        normal[which == 3, 2] = 1.0
        return t, normal

    def __repr__(self) -> str:
        return f"Cylinder(radius={self.radius}, length={self.length})"


class Polyhedron(Solid):
    """A solid bounded by planar polygonal facets."""

    @abstractmethod
    def facets(self) -> list[tuple[Pose, Polygon]]:
        """Each facet as the pose of its plane (local +z outward) and its outline in that plane."""

    def contains(self, points):
        p = np.atleast_2d(np.asarray(points, dtype=float))
        inside = np.ones(p.shape[0], dtype=bool)
        for pose, _ in self.facets():
            inside &= pose.to_local(p)[:, 2] <= 1e-12
        return inside

    def intersect(self, origins, directions, *, t_min=1e-9):
        n = origins.shape[0]
        best = np.full(n, np.inf)
        normal = np.zeros((n, 3))
        for pose, polygon in self.facets():
            o = pose.to_local(origins)
            d = pose.direction_to_local(directions)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = -o[:, 2] / d[:, 2]
            q = o + np.where(np.isfinite(t), t, 0.0)[:, None] * d
            hit = np.isfinite(t) & (t > t_min) & polygon.contains(q[:, 0], q[:, 1], slack=1e-12) & (t < best)
            best = np.where(hit, t, best)
            normal[hit] = pose.axis
        return best, normal


def _facet(points: np.ndarray) -> tuple[Pose, Polygon]:
    """Pose and outline of the planar polygon through *points* (counter-clockwise seen from outside)."""

    p = np.asarray(points, dtype=float)
    centroid = p.mean(axis=0)
    n = normalize(np.cross(p[1] - p[0], p[2] - p[0]))
    rotation = Rotation.align_z(n)
    pose = Pose(centroid, rotation)
    local = pose.to_local(p)[:, :2]
    return pose, Polygon(local)


class Box(Polyhedron):
    """Rectangular block: ``|x| ≤ width/2``, ``|y| ≤ height/2``, ``0 ≤ z ≤ depth``."""

    def __init__(self, width: float, height: float, depth: float) -> None:
        if not (float(width) > 0.0 and float(height) > 0.0 and float(depth) > 0.0):
            raise ValueError("a box has positive width, height and depth")
        self.width, self.height, self.depth = float(width), float(height), float(depth)

    def vertices(self) -> np.ndarray:
        w, h, d = self.width / 2, self.height / 2, self.depth
        return np.array([[x, y, z] for z in (0.0, d) for y in (-h, h) for x in (-w, w)])

    def facets(self):
        v = self.vertices()
        quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
        return [_facet(v[list(q)]) for q in quads]

    def __repr__(self) -> str:
        return f"Box(width={self.width}, height={self.height}, depth={self.depth})"


class Mesh(Polyhedron):
    """Closed polyhedron from vertices and faces (vertex indices, counter-clockwise from outside)."""

    def __init__(self, vertices: ArrayLike, faces: list[tuple[int, ...]]) -> None:
        v = np.asarray(vertices, dtype=float)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError("mesh vertices are (N, 3)")
        if any(len(f) < 3 for f in faces):
            raise ValueError("every face has at least three vertices")
        self.vertices = v
        self.faces = tuple(tuple(int(i) for i in f) for f in faces)

    def facets(self):
        return [_facet(self.vertices[list(f)]) for f in self.faces]

    def _key(self) -> tuple:
        return (self.vertices.tobytes(), self.faces)

    @classmethod
    def prism(cls, outline: ArrayLike, depth: float) -> "Mesh":
        """Extrude a counter-clockwise polygon of the x-y plane along +z by *depth*."""

        o = np.asarray(outline, dtype=float)
        m = o.shape[0]
        verts = np.vstack([np.column_stack([o, np.zeros(m)]), np.column_stack([o, np.full(m, float(depth))])])
        faces = [tuple(range(m - 1, -1, -1)), tuple(range(m, 2 * m))]
        faces += [(i, (i + 1) % m, m + (i + 1) % m, m + i) for i in range(m)]
        return cls(verts, faces)

    def __repr__(self) -> str:
        return f"Mesh({self.vertices.shape[0]} vertices, {len(self.faces)} faces)"


__all__ = ["Solid", "Sphere", "Cylinder", "Polyhedron", "Box", "Mesh"]
