"""Optical ray-tracing toolkit, layered strictly by the nature of each piece:

- ``raytracer.math`` — dimension-agnostic vectors, rigid transforms, and the
  solvers that find where a ray meets a surface. A utility layer: it knows
  nothing about optics.
- ``raytracer.surfaces`` — the shapes a ray can meet. Each states its own
  geometry, either implicitly (``f_Sigma(x) = 0``) or parametrically
  (``x = P(t)``), plus its placement, extent, and the media it separates.
  It never solves for an intersection itself.
- ``raytracer.optics`` — light and the laws it obeys: ``Ray`` (the light
  primitive, which cannot detect its own intersections), the laws of
  reflection and refraction (``optics.laws``), the energetic Fresnel laws
  (``optics.radiometry``), refractive materials, emitters, instruments, and
  built ``Lens``/``Mirror`` elements.
- ``raytracer.propagation`` — the algorithms that emit rays, prolong them to
  a collision, and decide what happens next: ``sequential`` (every ray
  through every surface in order, vectorized) and ``branching`` (a
  breadth-first tree of reflected/refracted rays).
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
    AIR,
    VACUUM,
    ConstantIndex,
    FresnelCoefficients,
    ImageSource,
    Instrument,
    Lens,
    MaterialLibrary,
    Mirror,
    ParallelSource,
    PointSource,
    Ray,
    RaySeed,
    Screen,
    ScreenHit,
    Source,
    default_materials,
    fresnel_coefficients,
    ray_from_angle,
    reflect,
    refract,
)
from .propagation import BranchingTracer, RayLabeler, RayNode, RayTree, TraceConfig
from .surfaces import (
    CartesianOvalSurface,
    CircleSurface,
    ConicSurface,
    EllipseSurface,
    HyperbolaSurface,
    Intersection,
    LineSegment,
    ParabolaSurface,
    ProfileSurface,
    Surface,
)

__version__ = "0.3.0"

__all__ = [
    # math
    "RigidTransform",
    "normalize",
    "direction_from_angle",
    # optics: light, media, and the laws
    "Ray",
    "ray_from_angle",
    "reflect",
    "refract",
    "FresnelCoefficients",
    "fresnel_coefficients",
    "ConstantIndex",
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
    # surfaces
    "Surface",
    "Intersection",
    "LineSegment",
    "ProfileSurface",
    "ConicSurface",
    "EllipseSurface",
    "CircleSurface",
    "ParabolaSurface",
    "HyperbolaSurface",
    "CartesianOvalSurface",
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
