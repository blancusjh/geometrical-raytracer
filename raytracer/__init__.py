"""Optical ray-tracing toolkit, layered strictly by the nature of each piece:

- ``raytracer.math`` — dimension-agnostic vectors, rigid transforms, and the
  pure numeric ray/shape intersection solvers. A utility layer: it knows
  nothing about optics.
- ``raytracer.physics`` — the crown: refractive materials, the geometric-
  optics direction laws (reflect/refract), and the separate energetic
  (Fresnel) laws. Everything else exists to give these laws something to
  act on.
- ``raytracer.shapes`` — pure shape math (sag profiles, conic and Cartesian-
  oval/Fermat-oval definitions). No notion of a ray, a medium, or a
  traceable surface.
- ``raytracer.optics`` — the physical-object abstractions an optical system
  is made of: ``Ray`` (the light primitive, which cannot detect its own
  intersections), ``Surface``/``Instrument``, ``Source``, and the built
  ``Lens``/``Mirror`` elements.
- ``raytracer.propagation`` — the algorithms that actually emit rays,
  prolong them to a collision, and decide what happens next:
  ``sequential`` (every ray through every surface in order, vectorized) and
  ``branching`` (a breadth-first tree of reflected/refracted rays).
- ``raytracer.design`` — the data model of an engineered system
  (``SurfaceRow``/``OpticalSystem``), independent of any propagation engine.
- ``raytracer.io`` — reading/writing a design (prescription CSVs today).
- ``raytracer.analysis`` — aberration and image-quality analysis (Seidel,
  Zernike, spot diagrams, PSF) built on top of a traced system.
- ``raytracer.viz`` — presentation, with interchangeable matplotlib and
  OpenGL backends.
"""

from .design import OpticalSystem, SurfaceKind, SurfaceRow, to_branching_surfaces
from .io import read_csv, write_csv
from .math.transforms import RigidTransform
from .math.vectors import direction_from_angle, normalize
from .optics import (
    CartesianOvalSurface,
    CircleSurface,
    ConicSurface,
    EllipseSurface,
    HyperbolaSurface,
    ImageSource,
    Instrument,
    Intersection,
    Lens,
    LineSegment,
    Mirror,
    ParabolaSurface,
    ParallelSource,
    PointSource,
    ProfileSurface,
    Ray,
    RaySeed,
    Screen,
    ScreenHit,
    Source,
    Surface,
    ray_from_angle,
)
from .physics import (
    AIR,
    VACUUM,
    ConstantIndex,
    FresnelCoefficients,
    MaterialLibrary,
    default_materials,
    fresnel_coefficients,
    reflect,
    refract,
)
from .propagation import BranchingTracer, RayLabeler, RayNode, RayTree, TraceConfig

__version__ = "0.3.0"

__all__ = [
    # math
    "RigidTransform",
    "normalize",
    "direction_from_angle",
    # physics
    "FresnelCoefficients",
    "reflect",
    "refract",
    "fresnel_coefficients",
    "ConstantIndex",
    "MaterialLibrary",
    "default_materials",
    "AIR",
    "VACUUM",
    # optics: the physical-object abstractions
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
    # propagation: the branching (tree) algorithm
    "BranchingTracer",
    "TraceConfig",
    "RayNode",
    "RayTree",
    "RayLabeler",
    # design + io
    "SurfaceKind",
    "SurfaceRow",
    "OpticalSystem",
    "to_branching_surfaces",
    "read_csv",
    "write_csv",
]


def __getattr__(name: str):
    # Lazy imports: the OpenGL viewer pulls in vispy, which should not be a
    # hard requirement for headless/analysis use.
    if name in ("OpenGLViewer", "RenderConfig"):
        from .viz.gl.viewer import OpenGLViewer, RenderConfig

        return {"OpenGLViewer": OpenGLViewer, "RenderConfig": RenderConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
