"""Deprecated: split into :mod:`raytracer.math` (utility) and
:mod:`raytracer.physics` (the crown — materials and optical laws).

Importing from ``raytracer.core`` (or its submodules) still works but emits
a :class:`DeprecationWarning`; update call sites to the new locations.
"""

from __future__ import annotations

import warnings

from ..math.frames import LocalFrame
from ..math.vectors import as_vector, direction_from_angle, normalize, perpendicular_2d, rotation_2d
from ..physics.laws import (
    FresnelCoefficients,
    SnellResult,
    fresnel_coefficients,
    reflect,
    refract,
    snell,
)
from ..physics.materials import (
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

warnings.warn(
    "raytracer.core is deprecated; import from raytracer.math (vectors, frames) "
    "or raytracer.physics (materials, optical laws) instead.",
    DeprecationWarning,
    stacklevel=2,
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
    "AbbeMaterial",
    "SellmeierMaterial",
    "sellmeier_glass",
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
