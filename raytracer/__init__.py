"""2D geometric ray tracing components."""

from .rays import Ray, RayNode, RayTree, RayLabeler, Intersection2D
from .sources import Source2D, PointSource2D, ParallelSource2D, RaySeed
from .geometry import Surface2D, ConicalDioptrique, EllipseConic, CircleConic, ParabolaConic
from .physics import reflect, refract
from .tracer import RayTracer2D, TraceConfig

__all__ = [
    "Ray",
    "RayNode",
    "RayTree",
    "RayLabeler",
    "Intersection2D",
    "Source2D",
    "PointSource2D",
    "ParallelSource2D",
    "RaySeed",
    "Surface2D",
    "ConicalDioptrique",
    "EllipseConic",
    "CircleConic",
    "ParabolaConic",
    "reflect",
    "refract",
    "RayTracer2D",
    "TraceConfig",
]
