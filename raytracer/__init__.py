"""Slim 2-D ray tracing toolkit with OpenGL visualisation."""

from .geometry import (
    ConicalDioptrique,
    EllipseConic,
    CircleConic,
    ParabolaConic,
    HyperbolaConic,
)
from .physics import SnellResult, snell, reflect, refract
from .rays import (
    Intersection2D,
    Ray2D,
    RayNode,
    RayTree,
    RayLabeler,
    direction_from_angle,
    ray_from_angle,
)
from .sources import PointSource2D, ParallelSource2D, RaySeed, Source2D
from .surfaces import LocalFrame, Surface2D
from .tracer import RayTracer2D, TraceConfig
from .visualization_opengl import OpenGLViewer, RenderConfig

__all__ = [
    "Ray2D",
    "RayNode",
    "RayTree",
    "RayLabeler",
    "Intersection2D",
    "direction_from_angle",
    "ray_from_angle",
    "Source2D",
    "PointSource2D",
    "ParallelSource2D",
    "RaySeed",
    "Surface2D",
    "LocalFrame",
    "ConicalDioptrique",
    "EllipseConic",
    "CircleConic",
    "ParabolaConic",
    "HyperbolaConic",
    "SnellResult",
    "snell",
    "reflect",
    "refract",
    "TraceConfig",
    "RayTracer2D",
    "OpenGLViewer",
    "RenderConfig",
]
