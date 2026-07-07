"""Non-sequential 2-D engine: rays, surfaces, sources, and the tree tracer."""

from .detectors import Screen2D, ScreenHit
from .elements import Lens2D, Mirror2D, system_to_2d
from .rays import Intersection2D, Ray2D, RayLabeler, RayNode, RayTree, ray_from_angle
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
    "RayTracer2D",
    "TraceConfig",
]
