"""Segment- and profile-based 2-D surfaces: lens faces, rims, baffles.

Both classes are thin ``Surface`` wrappers: they hold a coordinate frame
and clear aperture, and delegate the actual hit-testing to
:mod:`raytracer.math.intersections`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..math.intersections import intersect_profile, intersect_segment
from ..math.transforms import RigidTransform
from ..math.vectors import normalize, perpendicular_2d
from ..shapes.profile import AsphereProfile
from .surface import Intersection, Surface
from .ray import Ray


@dataclass
class LineSegment(Surface):
    """Straight segment between two points (baffles, rims, screens)."""

    p0: np.ndarray = field(default_factory=lambda: np.zeros(2))
    p1: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    surface_id: str = "segment"
    n_exterior: float = 1.0
    n_interior: float = 1.0
    absorbing: bool = False

    def __post_init__(self) -> None:
        Surface.__init__(
            self,
            surface_id=self.surface_id,
            n_exterior=self.n_exterior,
            n_interior=self.n_interior,
            absorbing=self.absorbing,
        )
        self.p0 = np.asarray(self.p0, dtype=float)
        self.p1 = np.asarray(self.p1, dtype=float)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    def polyline(self, samples: int = 2) -> np.ndarray:
        return np.array([self.p0, self.p1])

    def hit(self, ray: Ray) -> Optional[Intersection]:
        result = intersect_segment(ray.origin, ray.direction, self.p0, self.p1)
        if result is None:
            return None
        point, t, s = result
        normal = normalize(perpendicular_2d(self.p1 - self.p0))
        return Intersection(
            point=point,
            normal=normal,
            distance=t,
            surface_id=self.surface_id,
            parameters=np.array([s]),
            surface=self,
        )


@dataclass
class ProfileSurface(Surface):
    """Meridional cut of a rotationally-symmetric surface profile.

    The local frame puts the optical axis along +x: a point at transverse
    height ``h`` sits at local ``(sag(h), h)``. This is the 2-D counterpart
    of a sequential ``SurfaceRow`` and reuses the same :class:`AsphereProfile`
    sag mathematics (spheres, conics, aspheres).
    """

    profile: AsphereProfile = field(default_factory=AsphereProfile)
    vertex: np.ndarray = field(default_factory=lambda: np.zeros(2))
    angle: float = 0.0
    semidiameter: float = 10.0
    surface_id: str = "face"
    n_exterior: float = 1.0
    n_interior: float = 1.5
    interaction: str = "refract"  # "refract" | "reflect" | "config"

    def __post_init__(self) -> None:
        Surface.__init__(
            self,
            surface_id=self.surface_id,
            n_exterior=self.n_exterior,
            n_interior=self.n_interior,
        )
        self.vertex = np.asarray(self.vertex, dtype=float)
        self.frame = RigidTransform.from_angle_2d(origin=self.vertex, angle=self.angle)

    def polyline(self, samples: int = 256) -> np.ndarray:
        h = np.linspace(-self.semidiameter, self.semidiameter, samples)
        sag = self.profile.sag(np.abs(h))
        pts_local = np.column_stack([sag, h])
        return np.array([self.frame.to_world(p) for p in pts_local])

    def hit(self, ray: Ray) -> Optional[Intersection]:
        o = self.frame.to_local(ray.origin)
        d = self.frame.direction_to_local(ray.direction)
        result = intersect_profile(o, d, self.profile)
        if result is None:
            return None
        t, h, slope = result
        if h > self.semidiameter + 1e-9:
            return None
        p = o + t * d
        normal_local = normalize(np.array([1.0, -slope * np.sign(p[1])]))
        # Convention: the normal points toward -x (exterior side).
        normal_local = -normal_local
        return Intersection(
            point=self.frame.to_world(p),
            normal=normalize(self.frame.direction_to_world(normal_local)),
            distance=t,
            surface_id=self.surface_id,
            parameters=np.asarray(p, dtype=float),
            surface=self,
        )


__all__ = ["LineSegment", "ProfileSurface"]
