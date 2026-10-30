"""Visualization: the drawing plane, analysis figures, and the matplotlib and OpenGL backends.

Every 2-D figure is the meridional section ``x = 0`` of the 3-D model, drawn in ``(z, y)`` with the optical
axis horizontal; :mod:`~raytracer.viz.section` computes every section, outline and beam focus once.
:mod:`~raytracer.viz.bodies` draws a system's bodies and traced fans on matplotlib axes, and
:mod:`~raytracer.viz.plots` holds the analysis figures. A :class:`Drawing` is the backend-neutral form, shown by
:func:`show` in matplotlib or in the OpenGL viewer (:mod:`~raytracer.viz.gl`). :mod:`~raytracer.viz.gl3d`
renders systems and traces in 3-D, and :mod:`~raytracer.viz.interactive` edits them live.

Nothing outside this package imports it.
"""

from . import plots
from .protocol import GLBackend, show
from .drawing import Drawing, LensBodyItem, MarkerItem, RaySegmentsItem, SensorItem, SurfaceItem

__all__ = ["plots", "show", "GLBackend", "Drawing", "RaySegmentsItem", "SurfaceItem", "LensBodyItem", "SensorItem", "MarkerItem"]
