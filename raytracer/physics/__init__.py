"""The physics of light: refractive materials and the laws at an interface.

This is the crown of the package — a raytracer exists to simulate these
laws. It is built on :mod:`raytracer.math` (vectors, frames) and nothing
else; :mod:`raytracer.geometry` and the sequential/non-sequential engines
are built on *it*, not the reverse.
"""

from .laws import (
    FresnelCoefficients,
    SnellResult,
    fresnel_coefficients,
    reflect,
    refract,
    snell,
)
from .materials import (
    AIR,
    VACUUM,
    AbbeMaterial,
    ConstantIndex,
    Material,
    MaterialLibrary,
    SellmeierMaterial,
    default_materials,
    sellmeier_glass,
)

__all__ = [
    "SnellResult",
    "FresnelCoefficients",
    "snell",
    "reflect",
    "refract",
    "fresnel_coefficients",
    "Material",
    "ConstantIndex",
    "AbbeMaterial",
    "SellmeierMaterial",
    "sellmeier_glass",
    "MaterialLibrary",
    "default_materials",
    "AIR",
    "VACUUM",
]
