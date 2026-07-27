"""Optical ray-tracing toolkit, layered by the nature of each piece:

- ``math`` — vectors, rigid transforms, and the ray/surface intersection
  solvers. Knows nothing about optics.
- ``surfaces`` — the shapes a ray can meet. Each declares its own geometry,
  implicitly (``f_Sigma(x) = 0``) or parametrically (``x = P(t)``); none
  solves its own intersection.
- ``optics`` — light and the laws it obeys: ``Ray``, ``laws`` (reflection,
  refraction), ``radiometry`` (Fresnel), materials, sources, instruments,
  elements.
- ``propagation`` — the algorithms that emit rays, prolong them to a
  collision, and decide what happens next: ``sequential`` and ``branching``.
- ``design`` — the data model of a system, independent of any engine.
- ``io`` — reading/writing a design. ``analysis`` — aberrations and image
  quality. ``viz`` — matplotlib and OpenGL presentation.

Every public name from ``surfaces``, ``optics``, ``propagation``, ``design``
and ``io`` is re-exported here for convenience, derived from those packages'
own ``__all__`` so the lists cannot drift apart.
"""

from . import design, io, optics, propagation, surfaces
from .design import *  # noqa: F401,F403
from .io import *  # noqa: F401,F403
from .math.transforms import RigidTransform
from .math.vectors import direction_from_angle, normalize
from .optics import *  # noqa: F401,F403
from .propagation import *  # noqa: F401,F403
from .surfaces import *  # noqa: F401,F403

__version__ = "0.3.0"

__all__ = [
    "RigidTransform",
    "normalize",
    "direction_from_angle",
    *surfaces.__all__,
    *optics.__all__,
    *propagation.__all__,
    *design.__all__,
    *io.__all__,
]


def __getattr__(name: str):
    # Lazy: the OpenGL viewer pulls in vispy, which should not be a hard
    # requirement for headless/analysis use.
    if name in ("OpenGLViewer", "RenderConfig"):
        from .viz.gl.viewer import OpenGLViewer, RenderConfig

        return {"OpenGLViewer": OpenGLViewer, "RenderConfig": RenderConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
