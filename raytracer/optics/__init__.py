"""Optics: light, the media it travels through, and the laws it obeys.

- :mod:`~raytracer.optics.ray` — ``Ray``, the light primitive. It extends
  through space and nothing else: it deliberately cannot detect its own
  intersections (that is the surface's description plus
  :mod:`raytracer.math.intersections`) and cannot decide what happens when
  it lands (that is :mod:`raytracer.propagation`).
- :mod:`~raytracer.optics.laws` — reflection and refraction: which way the
  light goes at an interface.
- :mod:`~raytracer.optics.radiometry` — Fresnel coefficients: how much power
  goes each way.
- :mod:`~raytracer.optics.materials` — the refractive media themselves.
- :mod:`~raytracer.optics.sources` — emitters, which produce the rays.
- :mod:`~raytracer.optics.instruments` — surfaces that observe light rather
  than redirect it (a detector is an instrument, which is a surface).
- :mod:`~raytracer.optics.elements` — built ``Lens``/``Mirror`` assemblies.

The shapes these act on live in :mod:`raytracer.surfaces`, which this
package builds on and which knows nothing about any of the above.
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
