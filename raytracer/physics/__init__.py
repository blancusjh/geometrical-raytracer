"""The physics of light: refractive materials, direction laws, and power laws.

This is the crown of the package — a raytracer exists to simulate these
laws. It is built on :mod:`raytracer.math` alone; :mod:`raytracer.shapes`,
:mod:`raytracer.optics`, and the propagation engines are built on *it*, not
the reverse.
"""

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
from .radiometry import FresnelCoefficients, fresnel_coefficients
from .refraction import reflect, reflect_batch, refract, refract_batch

__all__ = [
    "reflect",
    "refract",
    "reflect_batch",
    "refract_batch",
    "FresnelCoefficients",
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
