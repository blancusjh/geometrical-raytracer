"""Segment and profile-based 2-D surfaces: lens faces, rims, screens."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..geometry.sag import AsphereProfile
from ..math.frames import LocalFrame
from ..math.vectors import normalize, perpendicular_2d
from .rays import Intersection2D, Ray2D
from .surfaces import Surface2D


@dataclass
class LineSegment2D(Surface2D):
    """Straight segment between two points (baffles, rims, screens)."""

    p0: np.ndarray = field(default_factory=lambda: np.zeros(2))
    p1: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    surface_id: str = "segment"
    n_exterior: float = 1.0
    n_interior: float = 1.0
    absorbing: bool = False

    def __post_init__(self) -> None:
        Surface2D.__init__(
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

    def intersect(self, ray: Ray2D) -> Optional[Intersection2D]:
        d = ray.direction
        e = self.p1 - self.p0
        denom = d[0] * (-e[1]) - d[1] * (-e[0])
        if abs(denom) < 1e-14:
            return None  # parallel
        rhs = self.p0 - ray.origin
        t = (rhs[0] * (-e[1]) - rhs[1] * (-e[0])) / denom
        s = (d[0] * rhs[1] - d[1] * rhs[0]) / denom
        if t <= 1e-12 or s < 0.0 or s > 1.0:
            return None
        point = ray.origin + t * d
        normal = normalize(perpendicular_2d(e))
        return Intersection2D(
            point=point,
            normal=normal,
            distance=float(t),
            surface_id=self.surface_id,
            parameters=np.array([s]),
            surface=self,
        )


@dataclass
class ProfileFace2D(Surface2D):
    """Meridional cut of a rotationally-symmetric surface profile.

    The local frame puts the optical axis along +x: a point at transverse
    height ``h`` sits at local ``(sag(h), h)``. This is the 2-D counterpart
    of a sequential :class:`SurfaceRow` and reuses the same
    :class:`AsphereProfile` sag mathematics (spheres, conics, aspheres).
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
        Surface2D.__init__(
            self,
            surface_id=self.surface_id,
            n_exterior=self.n_exterior,
            n_interior=self.n_interior,
        )
        self.vertex = np.asarray(self.vertex, dtype=float)
        self.frame = LocalFrame.from_angle_2d(origin=self.vertex, angle=self.angle)

    def polyline(self, samples: int = 256) -> np.ndarray:
        h = np.linspace(-self.semidiameter, self.semidiameter, samples)
        sag = self.profile.sag(np.abs(h))
        pts_local = np.column_stack([sag, h])
        return np.array([self.frame.to_world(p) for p in pts_local])

    def intersect(self, ray: Ray2D) -> Optional[Intersection2D]:
        o = self.frame.to_local(ray.origin)
        d = self.frame.direction_to_local(ray.direction)
        if abs(d[0]) < 1e-14 and self.profile.is_plane:
            return None

        # Newton from the vertex-plane guess, matching the sequential tracer.
        t = (0.0 - o[0]) / d[0] if abs(d[0]) > 1e-14 else 0.0
        converged = False
        for _ in range(30):
            p = o + t * d
            h = abs(p[1])
            sag, slope = self.profile.sag_and_slope(h)
            f = p[0] - float(sag)
            signed_slope = float(slope) * np.sign(p[1])
            derivative = d[0] - signed_slope * d[1]
            if abs(derivative) < 1e-14:
                return None
            step = f / derivative
            t -= step
            if abs(step) < 1e-12:
                converged = True
                break
        if not converged and abs(step) > 1e-9:
            return None
        if t <= 1e-12:
            return None
        p = o + t * d
        if abs(p[1]) > self.semidiameter + 1e-9:
            return None
        _, slope = self.profile.sag_and_slope(abs(p[1]))
        normal_local = normalize(np.array([1.0, -float(slope) * np.sign(p[1])]))
        # Convention: the normal points toward -x (exterior side).
        normal_local = -normal_local
        return Intersection2D(
            point=self.frame.to_world(p),
            normal=normalize(self.frame.direction_to_world(normal_local)),
            distance=float(t),
            surface_id=self.surface_id,
            parameters=np.asarray(p, dtype=float),
            surface=self,
        )


__all__ = ["LineSegment2D", "ProfileFace2D"]
