"""Visualization: analysis figures, the neutral 2-D drawing, and dual backends.

Every 2-D figure is the meridional section ``x = 0`` of the 3-D model,
drawn in ``(z, y)`` with the optical axis horizontal
(:mod:`~raytracer.viz.sag_drawing`). :mod:`~raytracer.viz.bodies` draws a
system's lens bodies on matplotlib axes; :mod:`~raytracer.viz.plots` holds
the analysis figures; a :class:`Drawing` is the backend-neutral form shown
by :func:`show` in matplotlib or the OpenGL viewer (:mod:`~raytracer.viz.gl`);
:mod:`~raytracer.viz.gl3d` renders systems and traces in 3-D and
:mod:`~raytracer.viz.interactive` edits element poses live.

Nothing outside this package imports it.
"""

from . import plots
from .protocol import GLBackend, show
from .scene import (
    Drawing,
    LensBodyItem,
    MarkerItem,
    RaySegmentsItem,
    SensorItem,
    SurfaceItem,
)

__all__ = [
    "plots",
    "show",
    "GLBackend",
    "Drawing",
    "RaySegmentsItem",
    "SurfaceItem",
    "LensBodyItem",
    "SensorItem",
    "MarkerItem",
]
