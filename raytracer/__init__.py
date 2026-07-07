"""Optical ray-tracing toolkit.

Two engines share the same primitives:

- ``raytracer.sequential`` — sequential 3-D tracer for prescription-defined
  systems (aspheres, mirrors, stops) with professional analysis.
- ``raytracer.nonseq`` — non-sequential 2-D tracer producing full
  reflection/refraction trees (caustics, exploration).

Visualization lives in ``raytracer.viz`` with interchangeable matplotlib
and OpenGL backends.
"""

from .core.frames import LocalFrame
from .core.materials import AIR, VACUUM, ConstantIndex, MaterialLibrary, default_materials
from .core.physics import (
    FresnelCoefficients,
    SnellResult,
    fresnel_coefficients,
    reflect,
    refract,
    snell,
)
from .core.vectors import direction_from_angle, normalize
from .geometry.conics2d import (
    CircleConic,
    ConicInterface2D,
    EllipseConic,
    HyperbolaConic,
    ParabolaConic,
)
from .geometry.ovoid2d import CartesianOvoid2D
from .nonseq.rays import Intersection2D, Ray2D, RayLabeler, RayNode, RayTree, ray_from_angle
from .nonseq.sources import ParallelSource2D, PointSource2D, RaySeed, Source2D
from .nonseq.surfaces import Surface2D
from .nonseq.tracer import RayTracer2D, TraceConfig

__version__ = "0.2.0"

__all__ = [
    # core
    "LocalFrame",
    "normalize",
    "direction_from_angle",
    "SnellResult",
    "FresnelCoefficients",
    "snell",
    "reflect",
    "refract",
    "fresnel_coefficients",
    "ConstantIndex",
    "MaterialLibrary",
    "default_materials",
    "AIR",
    "VACUUM",
    # geometry
    "ConicInterface2D",
    "EllipseConic",
    "CircleConic",
    "ParabolaConic",
    "HyperbolaConic",
    "CartesianOvoid2D",
    # non-sequential engine
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


def __getattr__(name: str):
    # Lazy imports: the OpenGL viewer pulls in vispy, which should not be a
    # hard requirement for headless/analysis use.
    if name in ("OpenGLViewer", "RenderConfig"):
        from .visualization_opengl import OpenGLViewer, RenderConfig

        return {"OpenGLViewer": OpenGLViewer, "RenderConfig": RenderConfig}[name]
    if name == "ConicalDioptrique":
        from .compat import ConicalDioptrique

        return ConicalDioptrique
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
