"""Physical-object abstractions: what an optical system is made of.

``Ray`` (the light primitive), ``Surface``/``Intersection`` (a medium
boundary and what hitting one produces), ``Source`` (an emitter),
``Instrument`` (a surface that observes rather than redirects), and the
built ``Lens``/``Mirror`` elements all live here, once, regardless of which
propagation algorithm in :mod:`raytracer.propagation` ends up driving them.
None of these classes know how to propagate a ray or decide what happens
after a hit -- that is deliberately not their responsibility.
"""

from .cartesian_oval import CartesianOvalSurface
from .conics import CircleSurface, ConicSurface, EllipseSurface, HyperbolaSurface, ParabolaSurface
from .elements import Lens, Mirror
from .instruments import Instrument, Screen, ScreenHit
from .profiles import LineSegment, ProfileSurface
from .ray import Ray, ray_from_angle
from .sources import ImageSource, ParallelSource, PointSource, RaySeed, Source
from .surface import Intersection, Surface

__all__ = [
    "Ray",
    "ray_from_angle",
    "Surface",
    "Intersection",
    "Source",
    "PointSource",
    "ParallelSource",
    "ImageSource",
    "RaySeed",
    "Instrument",
    "Screen",
    "ScreenHit",
    "LineSegment",
    "ProfileSurface",
    "ConicSurface",
    "EllipseSurface",
    "CircleSurface",
    "ParabolaSurface",
    "HyperbolaSurface",
    "CartesianOvalSurface",
    "Lens",
    "Mirror",
]
