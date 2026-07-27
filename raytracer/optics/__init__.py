"""Light, the media it travels through, and the laws it obeys.

``ray`` (the light primitive -- it neither detects its own intersections
nor decides what happens when it lands), ``laws`` (reflection, refraction),
``radiometry`` (Fresnel power), ``materials``, ``sources`` (emitters),
``instruments`` (a detector is an instrument, which is a surface), and
``elements`` (built ``Lens``/``Mirror``).

The shapes these act on live in :mod:`raytracer.surfaces`.
"""

from .elements import Lens, Mirror
from .instruments import Instrument, Screen, ScreenHit
from .laws import reflect, reflect_batch, refract, refract_batch
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
from .ray import Ray, ray_from_angle
from .sources import ImageSource, ParallelSource, PointSource, RaySeed, Source

__all__ = [
    "Ray",
    "ray_from_angle",
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
    "Source",
    "PointSource",
    "ParallelSource",
    "ImageSource",
    "RaySeed",
    "Instrument",
    "Screen",
    "ScreenHit",
    "Lens",
    "Mirror",
]
