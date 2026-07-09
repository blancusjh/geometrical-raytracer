"""Surface rows for sequential prescription-defined systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..core.materials import AIR, Material
from ..geometry.cartesian_oval import CartesianOvalProfile
from ..geometry.sag import AsphereProfile


class SurfaceKind(Enum):
    REFRACT = "refractive"
    MIRROR = "mirror"
    STOP = "aperture_stop"


@dataclass
class SurfaceRow:
    """One row of a sequential prescription.

    ``thickness`` is the signed axial distance to the next vertex (negative
    on the folded return path after an odd number of mirrors).
    ``material_after`` names the medium following the surface and is ignored
    for mirrors (the ray stays in its current medium).
    ``semidiameter`` is the clear aperture; ``None`` disables clipping.
    """

    profile: AsphereProfile | CartesianOvalProfile
    thickness: float = 0.0
    material_after: Material = AIR
    kind: SurfaceKind = SurfaceKind.REFRACT
    semidiameter: float | None = None
    comment: str = ""

    @classmethod
    def refracting(
        cls,
        *,
        radius: float,
        thickness: float,
        material: Material,
        semidiameter: float | None = None,
        conic: float = 0.0,
        coefficients: tuple[float, ...] | list[float] = (),
        comment: str = "",
    ) -> "SurfaceRow":
        return cls(
            profile=AsphereProfile.from_radius(radius, conic, coefficients),
            thickness=thickness,
            material_after=material,
            kind=SurfaceKind.REFRACT,
            semidiameter=semidiameter,
            comment=comment,
        )

    @classmethod
    def cartesian_oval(
        cls,
        *,
        n0: float,
        z0: float,
        ni: float,
        zi: float,
        thickness: float,
        material: Material,
        semidiameter: float | None = None,
        comment: str = "",
    ) -> "SurfaceRow":
        """Stigmatic refracting surface between object (n0, z0) and image (ni, zi).

        See :class:`raytracer.geometry.cartesian_oval.CartesianOvalProfile`;
        keep ``semidiameter`` within ``profile.max_usable_height`` for a
        physically meaningful (single-valued) surface.
        """

        return cls(
            profile=CartesianOvalProfile(n0=n0, z0=z0, ni=ni, zi=zi),
            thickness=thickness,
            material_after=material,
            kind=SurfaceKind.REFRACT,
            semidiameter=semidiameter,
            comment=comment,
        )

    @classmethod
    def mirror(
        cls,
        *,
        radius: float,
        thickness: float,
        semidiameter: float | None = None,
        conic: float = 0.0,
        coefficients: tuple[float, ...] | list[float] = (),
        comment: str = "",
    ) -> "SurfaceRow":
        return cls(
            profile=AsphereProfile.from_radius(radius, conic, coefficients),
            thickness=thickness,
            kind=SurfaceKind.MIRROR,
            semidiameter=semidiameter,
            comment=comment,
        )

    @classmethod
    def stop(
        cls,
        *,
        thickness: float = 0.0,
        semidiameter: float | None = None,
        comment: str = "",
    ) -> "SurfaceRow":
        return cls(
            profile=AsphereProfile.plane(),
            thickness=thickness,
            kind=SurfaceKind.STOP,
            semidiameter=semidiameter,
            comment=comment,
        )

    @property
    def reflective(self) -> bool:
        return self.kind is SurfaceKind.MIRROR

    @property
    def radius(self) -> float:
        return self.profile.radius


__all__ = ["SurfaceKind", "SurfaceRow"]
