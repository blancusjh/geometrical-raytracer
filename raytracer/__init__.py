"""Optical ray-tracing toolkit, layered by what each piece knows about:

- ``raytracer.math`` — dimension-agnostic vector/frame utilities. Knows
  nothing about optics.
- ``raytracer.physics`` — the crown: refractive materials and the laws
  (Snell, Fresnel) a ray obeys at an interface. Everything below exists to
  give these laws something to act on.
- ``raytracer.geometry`` — pure shape math (sag profiles, conic/implicit-form
  algebra). Knows nothing about rays or tracers.
- ``raytracer.sequential`` / ``raytracer.nonseq`` — the engineering layer:
  two tracer engines built from the above. ``sequential`` is the 3-D tracer
  for prescription-defined systems (aspheres, mirrors, stops), including
  reading/writing prescriptions from files; ``nonseq`` is the 2-D tracer
  producing full reflection/refraction trees (caustics, exploration).
- ``raytracer.analysis`` — aberration and image-quality analysis (Seidel,
  Zernike, spot diagrams, PSF) built on top of a traced system.
- ``raytracer.viz`` — presentation, with interchangeable matplotlib and
  OpenGL backends.
"""

from .math.frames import LocalFrame
from .math.vectors import direction_from_angle, normalize
from .nonseq.conics2d import (
    CircleConic,
    ConicInterface2D,
    EllipseConic,
    HyperbolaConic,
    ParabolaConic,
)
from .nonseq.ovoid2d import CartesianOvoid2D
from .nonseq.rays import Intersection2D, Ray2D, RayLabeler, RayNode, RayTree, ray_from_angle
from .nonseq.sources import ParallelSource2D, PointSource2D, RaySeed, Source2D
from .nonseq.surfaces import Surface2D
from .nonseq.tracer import RayTracer2D, TraceConfig
from .physics.laws import (
    FresnelCoefficients,
    SnellResult,
    fresnel_coefficients,
    reflect,
    refract,
    snell,
)
from .physics.materials import AIR, VACUUM, ConstantIndex, MaterialLibrary, default_materials

__version__ = "0.2.0"

__all__ = [
    # math
    "LocalFrame",
    "normalize",
    "direction_from_angle",
    # physics
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
    # non-sequential surfaces (geometry adapted to the 2-D engine)
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
        from .viz.gl.viewer import OpenGLViewer, RenderConfig

        return {"OpenGLViewer": OpenGLViewer, "RenderConfig": RenderConfig}[name]
    if name == "ConicalDioptrique":
        from .compat import ConicalDioptrique

        return ConicalDioptrique
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
