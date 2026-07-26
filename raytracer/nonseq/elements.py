"""Physical 2-D elements built from profile faces, and the sequential bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..geometry.sag import AsphereProfile
from ..math.vectors import rotation_2d
from .segments2d import LineSegment2D, ProfileFace2D
from .surfaces import Surface2D


@dataclass
class Lens2D:
    """A solid lens: two refracting faces closed by absorbing rim edges.

    Local convention: the optical axis runs along +x, ``vertex`` is the
    front-face vertex, ``thickness`` the axial distance to the back vertex.
    """

    front: ProfileFace2D
    back: ProfileFace2D
    rims: list[LineSegment2D] = field(default_factory=list)
    name: str = "lens"

    @classmethod
    def from_radii(
        cls,
        *,
        r1: float,
        r2: float,
        thickness: float,
        semidiameter: float,
        n: float,
        n_ambient: float = 1.0,
        vertex=(0.0, 0.0),
        angle: float = 0.0,
        name: str = "lens",
        conic1: float = 0.0,
        conic2: float = 0.0,
        coefficients1: tuple[float, ...] = (),
        coefficients2: tuple[float, ...] = (),
    ) -> "Lens2D":
        vertex = np.asarray(vertex, dtype=float)
        rot = rotation_2d(angle)
        axis = rot @ np.array([1.0, 0.0])
        back_vertex = vertex + thickness * axis

        front = ProfileFace2D(
            profile=AsphereProfile.from_radius(r1, conic1, coefficients1),
            vertex=vertex,
            angle=angle,
            semidiameter=semidiameter,
            surface_id=f"{name}.front",
            n_exterior=n_ambient,
            n_interior=n,
            interaction="refract",
        )
        back = ProfileFace2D(
            profile=AsphereProfile.from_radius(r2, conic2, coefficients2),
            vertex=back_vertex,
            angle=angle,
            semidiameter=semidiameter,
            surface_id=f"{name}.back",
            n_exterior=n,       # ray arrives from inside the glass
            n_interior=n_ambient,
            interaction="refract",
        )

        # Rim edges join the face boundaries at +/- semidiameter.
        rims = []
        for sign in (1.0, -1.0):
            h = sign * semidiameter
            sag1 = float(front.profile.sag(abs(h)))
            sag2 = float(back.profile.sag(abs(h)))
            p0 = vertex + rot @ np.array([sag1, h])
            p1 = back_vertex + rot @ np.array([sag2, h])
            rims.append(
                LineSegment2D(
                    p0=p0, p1=p1, surface_id=f"{name}.rim{'+' if sign > 0 else '-'}",
                    absorbing=True,
                )
            )
        return cls(front=front, back=back, rims=rims, name=name)

    @classmethod
    def spherical(
        cls,
        *,
        R1: float,
        R2: float,
        thickness: float,
        semidiameter: float,
        n: float,
        n_ambient: float = 1.0,
        vertex=(0.0, 0.0),
        angle: float = 0.0,
        name: str = "lens",
    ) -> "Lens2D":
        """Lens bounded by two *spherical* surfaces of radii ``R1``/``R2``.

        Named by the surface shape (a bare radius is only the first-order
        curvature and does not determine a shape by itself); ``R = 0`` or
        ``inf`` denotes a flat face. Sign convention: positive ``R`` curves
        toward +axis (center of curvature after the vertex), so a biconvex
        lens is ``R1 > 0, R2 < 0``.
        """

        return cls.from_radii(
            r1=R1, r2=R2, thickness=thickness, semidiameter=semidiameter,
            n=n, n_ambient=n_ambient, vertex=vertex, angle=angle, name=name,
        )

    @classmethod
    def biconvex(cls, *, focal_hint_radius: float, **kwargs) -> "Lens2D":
        return cls.from_radii(r1=focal_hint_radius, r2=-focal_hint_radius, **kwargs)

    def surfaces(self) -> list[Surface2D]:
        return [self.front, self.back, *self.rims]

    def body_polygon(self, samples: int = 120) -> np.ndarray:
        """Closed outline (front face down, back face up) for filled rendering."""

        front_pts = self.front.polyline(samples)
        back_pts = self.back.polyline(samples)
        return np.vstack([front_pts, back_pts[::-1]])


@dataclass
class Mirror2D:
    """A reflective profile face."""

    face: ProfileFace2D
    name: str = "mirror"

    @classmethod
    def from_radius(
        cls,
        *,
        radius: float,
        semidiameter: float,
        vertex=(0.0, 0.0),
        angle: float = 0.0,
        conic: float = 0.0,
        coefficients: tuple[float, ...] = (),
        name: str = "mirror",
    ) -> "Mirror2D":
        face = ProfileFace2D(
            profile=AsphereProfile.from_radius(radius, conic, coefficients),
            vertex=np.asarray(vertex, dtype=float),
            angle=angle,
            semidiameter=semidiameter,
            surface_id=name,
            interaction="reflect",
        )
        return cls(face=face, name=name)

    def surfaces(self) -> list[Surface2D]:
        return [self.face]


def system_to_2d(system) -> list[Surface2D]:
    """Meridional (y-z) slice of a sequential system as 2-D surfaces.

    The sequential z axis maps to 2-D x. Every refractive row becomes a
    ``ProfileFace2D`` carrying its before/after indices; mirrors become
    reflective faces; stops are skipped (pure markers). Double-passed
    surfaces on folded return paths (duplicated rows with matching vertex
    and radius) collapse to a single physical face.

    Caveat: this bridge is exact for dioptric (unfolded) systems. In a
    folded catadioptric, the non-sequential tracer will let the forward
    beam interact with any mirror whose full aperture crosses its path —
    which sequential ordering deliberately ignores. Clip mirror
    semidiameters to the physically used sub-aperture in that case, or
    analyse folded systems with the sequential engine.
    """

    from ..sequential.surfaces import SurfaceKind

    surfaces: list[Surface2D] = []
    seen: set[tuple[float, float]] = set()
    for i, row in enumerate(system.rows):
        if row.kind is SurfaceKind.STOP:
            continue
        signature = (round(float(system.vertices[i]), 9), round(row.radius, 9))
        if signature in seen:
            continue  # double-passed surface: keep the first physical copy
        seen.add(signature)
        semidiameter = row.semidiameter if row.semidiameter is not None else 100.0
        face = ProfileFace2D(
            profile=row.profile,
            vertex=np.array([system.vertices[i], 0.0]),
            angle=0.0,
            semidiameter=semidiameter,
            surface_id=f"s{i + 1}",
            n_exterior=float(system.n_before[i]),
            n_interior=float(system.n_after[i]),
            interaction="reflect" if row.kind is SurfaceKind.MIRROR else "refract",
        )
        surfaces.append(face)
    return surfaces


__all__ = ["Lens2D", "Mirror2D", "system_to_2d"]
