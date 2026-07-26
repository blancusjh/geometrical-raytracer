"""Non-sequential 2-D engine: rays, surfaces, sources, and the tree tracer.

This is the engineering layer for the 2-D engine: it turns the pure shapes
in :mod:`raytracer.geometry` into traceable ``Surface2D`` objects
(``conics2d``, ``segments2d``, ``ovoid2d``) and drives them with
:mod:`raytracer.physics`'s reflection/refraction/Fresnel laws.
"""

from .conics2d import CircleConic, ConicInterface2D, EllipseConic, HyperbolaConic, ParabolaConic
from .detectors import Screen2D, ScreenHit
from .elements import Lens2D, Mirror2D, system_to_2d
from .ovoid2d import CartesianOvoid2D
from .rays import Intersection2D, Ray2D, RayLabeler, RayNode, RayTree, ray_from_angle
from .segments2d import LineSegment2D, ProfileFace2D
from .sources import ImageSource2D, ParallelSource2D, PointSource2D, RaySeed, Source2D
from .surfaces import Surface2D
from .tracer import RayTracer2D, TraceConfig

__all__ = [
    "Screen2D",
    "ScreenHit",
    "Lens2D",
    "Mirror2D",
    "system_to_2d",
    "Ray2D",
    "Intersection2D",
    "RayNode",
    "RayTree",
    "RayLabeler",
    "ray_from_angle",
    "RaySeed",
    "Source2D",
    "PointSource2D",
    "ParallelSource2D",
    "Surface2D",
    "LineSegment2D",
    "ProfileFace2D",
    "ConicInterface2D",
    "EllipseConic",
    "CircleConic",
    "ParabolaConic",
    "HyperbolaConic",
    "CartesianOvoid2D",
    "RayTracer2D",
    "TraceConfig",
]
