"""Shared primitives: vectors, frames, materials, and optical laws."""

from .vectors import as_vector, normalize, direction_from_angle, perpendicular_2d, rotation_2d
from .frames import LocalFrame
from .materials import (
    AIR,
    VACUUM,
    ConstantIndex,
    Material,
    MaterialLibrary,
    default_materials,
)
from .physics import (
    FresnelCoefficients,
    SnellResult,
    fresnel_coefficients,
    reflect,
    refract,
    snell,
)

__all__ = [
    "as_vector",
    "normalize",
    "direction_from_angle",
    "perpendicular_2d",
    "rotation_2d",
    "LocalFrame",
    "Material",
    "ConstantIndex",
    "MaterialLibrary",
    "default_materials",
    "AIR",
    "VACUUM",
    "SnellResult",
    "FresnelCoefficients",
    "snell",
    "reflect",
    "refract",
    "fresnel_coefficients",
]
