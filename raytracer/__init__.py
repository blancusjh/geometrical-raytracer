"""Ray tracer primitives spanning 2D and 3D geometries."""

from .geometry import CircleConic, ConicalDioptrique, EllipseConic, ParabolaConic, Surface2D
from .geometry3d import AxisymmetricCartesianSurface3D, AxisymmetricConicSurface3D
from .physics import (
    fresnel_coefficients,
    fresnel_unpolarised,
    reflect,
    refract,
    snell,
)
from .rays import (
    Intersection2D,
    IntersectionND,
    Ray,
    Ray2D,
    Ray3D,
    RayLabeler,
    RayND,
    RayNode,
    RayTree,
    direction_from_angles,
    ray_from_angles,
)
from .sources import (
    PointSource2D,
    PointSource3D,
    ParallelSource2D,
    RaySeed,
    Source2D,
    Source3D,
)
from .surfaces import LocalFrame, SurfaceIntersection, SurfaceND
from .tracer import RayTracer2D, RayTracer3D, RayTracerND, TraceConfig
from .visualization_opengl import OpenGLViewer, RenderConfig
from .interactive import InteractiveController, SliderConfig

__all__ = [
    "Ray",
    "Ray2D",
    "Ray3D",
    "RayND",
    "RayNode",
    "RayTree",
    "RayLabeler",
    "Intersection2D",
    "IntersectionND",
    "direction_from_angles",
    "ray_from_angles",
    "Source2D",
    "PointSource2D",
    "PointSource3D",
    "ParallelSource2D",
    "RaySeed",
    "Source3D",
    "SurfaceND",
    "SurfaceIntersection",
    "LocalFrame",
    "Surface2D",
    "ConicalDioptrique",
    "EllipseConic",
    "CircleConic",
    "ParabolaConic",
    "AxisymmetricConicSurface3D",
    "AxisymmetricCartesianSurface3D",
    "reflect",
    "refract",
    "snell",
    "fresnel_coefficients",
    "fresnel_unpolarised",
    "RayTracer2D",
    "RayTracer3D",
    "RayTracerND",
    "TraceConfig",
    "OpenGLViewer",
    "RenderConfig",
    "InteractiveController",
    "SliderConfig",
]
