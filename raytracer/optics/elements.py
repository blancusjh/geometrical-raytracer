"""Built optical elements: solids and mirrors composed of surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..math.vectors import rotation_2d
from ..shapes.profile import AsphereProfile
from .profiles import LineSegment, ProfileSurface
from .surface import Surface


@dataclass
class Lens:
    """A solid lens: two refracting faces closed by absorbing rim edges.

    Local convention: the optical axis runs along +x, ``vertex`` is the
    front-face vertex, ``thickness`` the axial distance to the back vertex.
    """

    front: ProfileSurface
    back: ProfileSurface
    rims: list[LineSegment] = field(default_factory=list)
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
    ) -> "Lens":
        vertex = np.asarray(vertex, dtype=float)
        rot = rotation_2d(angle)
        axis = rot @ np.array([1.0, 0.0])
        back_vertex = vertex + thickness * axis

        front = ProfileSurface(
            profile=AsphereProfile.from_radius(r1, conic1, coefficients1),
            vertex=vertex,
            angle=angle,
            semidiameter=semidiameter,
            surface_id=f"{name}.front",
            n_exterior=n_ambient,
            n_interior=n,
            interaction="refract",
        )
        back = ProfileSurface(
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
                LineSegment(
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
    ) -> "Lens":
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
    def biconvex(cls, *, focal_hint_radius: float, **kwargs) -> "Lens":
        return cls.from_radii(r1=focal_hint_radius, r2=-focal_hint_radius, **kwargs)

    def surfaces(self) -> list[Surface]:
        return [self.front, self.back, *self.rims]

    def body_polygon(self, samples: int = 120) -> np.ndarray:
        """Closed outline (front face down, back face up) for filled rendering."""

        front_pts = self.front.polyline(samples)
        back_pts = self.back.polyline(samples)
        return np.vstack([front_pts, back_pts[::-1]])


@dataclass
class Mirror:
    """A reflective profile face."""

    face: ProfileSurface
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
    ) -> "Mirror":
        face = ProfileSurface(
            profile=AsphereProfile.from_radius(radius, conic, coefficients),
            vertex=np.asarray(vertex, dtype=float),
            angle=angle,
            semidiameter=semidiameter,
            surface_id=name,
            interaction="reflect",
        )
        return cls(face=face, name=name)

    def surfaces(self) -> list[Surface]:
        return [self.face]


__all__ = ["Lens", "Mirror"]
