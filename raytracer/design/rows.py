"""Surface rows for sequential prescription-defined systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..math.transforms import RigidTransform
from ..optics.materials import AIR, Material
from ..surfaces.apertures import Aperture, CircularAperture
from ..surfaces.cartesian_oval import CartesianOvalProfile
from ..surfaces.profile import AsphereProfile


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
    for mirrors and stops (the ray stays in its current medium).
    ``aperture`` takes precedence over the circular ``semidiameter``;
    leaving both unset disables clipping. ``is_stop`` marks an interface as
    the reference stop without changing its optical interaction.
    """

    profile: AsphereProfile | CartesianOvalProfile
    thickness: float = 0.0
    material_after: Material = AIR
    kind: SurfaceKind = SurfaceKind.REFRACT
    semidiameter: float | None = None
    comment: str = ""
    placement: RigidTransform = field(default_factory=RigidTransform.identity)
    aperture: Aperture | None = None
    is_stop: bool = False

    def __post_init__(self):
        if self.placement.origin.shape != (3,):
            raise ValueError("a sequential surface needs a three-dimensional placement")
        if not np.isfinite(self.thickness):
            raise ValueError("surface thickness must be finite")
        if self.semidiameter is not None and (
            not np.isfinite(self.semidiameter) or self.semidiameter <= 0
        ):
            raise ValueError("semidiameter must be finite and positive")

    @property
    def clear_aperture(self) -> Aperture | None:
        if self.aperture is not None:
            return self.aperture
        return None if self.semidiameter is None else CircularAperture(self.semidiameter)

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
        placement: RigidTransform | None = None,
        aperture: Aperture | None = None,
        is_stop: bool = False,
    ) -> "SurfaceRow":
        return cls(
            profile=AsphereProfile.from_radius(radius, conic, coefficients),
            thickness=thickness,
            material_after=material,
            kind=SurfaceKind.REFRACT,
            semidiameter=semidiameter,
            comment=comment,
            placement=placement or RigidTransform.identity(),
            aperture=aperture,
            is_stop=is_stop,
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
        placement: RigidTransform | None = None,
        aperture: Aperture | None = None,
        is_stop: bool = False,
    ) -> "SurfaceRow":
        """Stigmatic refracting surface between object (n0, z0) and image (ni, zi).

        See :class:`raytracer.surfaces.cartesian_oval.CartesianOvalProfile`;
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
            placement=placement or RigidTransform.identity(),
            aperture=aperture,
            is_stop=is_stop,
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
        placement: RigidTransform | None = None,
        aperture: Aperture | None = None,
        is_stop: bool = False,
    ) -> "SurfaceRow":
        return cls(
            profile=AsphereProfile.from_radius(radius, conic, coefficients),
            thickness=thickness,
            kind=SurfaceKind.MIRROR,
            semidiameter=semidiameter,
            comment=comment,
            placement=placement or RigidTransform.identity(),
            aperture=aperture,
            is_stop=is_stop,
        )

    @classmethod
    def stop(
        cls,
        *,
        thickness: float = 0.0,
        semidiameter: float | None = None,
        comment: str = "",
        placement: RigidTransform | None = None,
        aperture: Aperture | None = None,
        is_stop: bool = False,
    ) -> "SurfaceRow":
        return cls(
            profile=AsphereProfile.plane(),
            thickness=thickness,
            kind=SurfaceKind.STOP,
            semidiameter=semidiameter,
            comment=comment,
            placement=placement or RigidTransform.identity(),
            aperture=aperture,
            is_stop=is_stop,
        )

    @property
    def reflective(self) -> bool:
        return self.kind is SurfaceKind.MIRROR

    @property
    def radius(self) -> float:
        return self.profile.radius


__all__ = ["SurfaceKind", "SurfaceRow"]
