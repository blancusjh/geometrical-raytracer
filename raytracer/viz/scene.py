"""Backend-neutral 2-D drawing model.

A :class:`Drawing` is pure data: items describing ray segments, surface
curves, filled bodies, sensors and markers, all in **drawing coordinates**
``(z, y)`` — the meridional section ``x = 0`` of the 3-D model, optical axis
horizontal (see :mod:`raytracer.viz.sag_drawing`). Both the matplotlib
backend (:mod:`raytracer.viz.mpl`) and the OpenGL backend consume the same
drawing, making the two visualizations interchangeable.

What goes in:

- :meth:`Drawing.add_system` — every element of an
  :class:`~raytracer.system.OpticalSystem`: lens and prism bodies filled by
  index (neighbouring lenses cut apart where they interpenetrate),
  mirrors, beam splitters, diaphragm blades and sensors;
- :meth:`Drawing.add_tree` — the segments of a non-sequential
  :class:`~raytracer.tracing.RayTree`, weighted by power;
- :meth:`Drawing.add_elements` / :meth:`Drawing.add_sensor` — loose
  elements at their own (or given) poses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Union

import numpy as np

from ..elements import BeamSplitter, Element, Lens, Mirror, Prism, Sensor, Stop
from ..geometry.apertures import Circle
from ..geometry.surfaces import Plane
from ..geometry.transforms import Pose
from .bodies import BodyOutline, body_outlines, body_colors, element_outline, medium_index, system_extent
from .sag_drawing import DEFAULT_SEMIDIAMETER, face_section, fan_faces, polyhedron_section, strip_faces, to_drawing


def _rgba(hex_color: str, alpha: float = 1.0) -> tuple:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0, float(alpha))


@dataclass
class RaySegmentsItem:
    segments: np.ndarray  # (N, 2, 2) drawing-plane endpoints (z, y)
    colors: np.ndarray  # (N, 4) rgba; alpha = per-ray weight
    intensities: np.ndarray  # (N,)
    escaping: np.ndarray  # (N,) bool — extend to the view edge
    directions: np.ndarray  # (N, 2) unit directions (for escaping rays)


@dataclass
class SurfaceItem:
    polyline: np.ndarray  # (M, 2)
    color: tuple = (1.0, 1.0, 1.0, 1.0)
    width: float = 2.0
    name: str = ""


@dataclass
class LensBodyItem:
    """A filled body: closed outline, and optionally its triangles for GL fills.

    Without ``faces`` the outline is taken to be laid out as by
    :func:`~raytracer.viz.sag_drawing.lens_section` (one face, then the
    other back) and strip-triangulated.
    """

    polygon: np.ndarray  # (M, 2) closed outline
    facecolor: tuple = (0.74, 0.88, 1.0, 0.55)
    edgecolor: tuple = (0.16, 0.47, 0.71, 1.0)
    faces: np.ndarray | None = None  # (K, 3) triangles over polygon vertices
    name: str = ""

    def triangles(self) -> tuple[np.ndarray, np.ndarray]:
        """``(vertices, faces)`` filling the body."""

        polygon = np.asarray(self.polygon, dtype=float)
        if self.faces is not None:
            return polygon, np.asarray(self.faces)
        if polygon.shape[0] > 1 and np.allclose(polygon[0], polygon[-1]):
            polygon = polygon[:-1]
        return polygon, strip_faces(polygon.shape[0])


@dataclass
class SensorItem:
    """A detector's section, with the drawing-plane points it recorded."""

    polyline: np.ndarray  # (M, 2)
    color: tuple = (1.0, 0.85, 0.3, 1.0)
    width: float = 3.0
    hits: np.ndarray | None = None  # (K, 2) recorded points (z, y)
    name: str = ""

    @property
    def p0(self) -> np.ndarray:
        return np.asarray(self.polyline[0])

    @property
    def p1(self) -> np.ndarray:
        return np.asarray(self.polyline[-1])


@dataclass
class MarkerItem:
    points: np.ndarray  # (M, 2)
    color: tuple = (0.86, 0.08, 0.24, 1.0)
    size: float = 6.0


Resolver = Union[np.ndarray, Callable, None]

MIRROR_RGBA = (0.13, 0.13, 0.13, 1.0)
STOP_RGBA = (0.35, 0.35, 0.38, 1.0)
SPLITTER_RGBA = (0.55, 0.55, 0.6, 1.0)


class Drawing:
    """Container of drawable items in drawing coordinates ``(z, y)``, plus optional view limits."""

    def __init__(
        self,
        x_lims: tuple[float, float] | None = None,
        y_lims: tuple[float, float] | None = None,
    ) -> None:
        self.items: list = []
        self.x_lims = x_lims
        self.y_lims = y_lims

    def add(self, item) -> "Drawing":
        self.items.append(item)
        return self

    # -- rays -------------------------------------------------------------

    def add_tree(
        self,
        tree,
        *,
        colors: Resolver = None,
        intensities: Resolver = None,
        show_misses: bool = True,
        markers: bool = False,
        default_color: tuple = (1.0, 1.0, 1.0, 1.0),
        select: np.ndarray | None = None,
    ) -> "Drawing":
        """Add the segments of a non-sequential :class:`~raytracer.tracing.RayTree`.

        Each segment is projected onto the drawing plane. ``colors`` is an
        ``(M, 4)`` array or a callable ``tree -> (M, 4)``; ``intensities``
        likewise ``(M,)`` (default: the segment's power). Segments that left
        the system (``tree.escaped``) are drawn to the edge of the view, or
        dropped with ``show_misses=False``. ``markers`` marks where segments
        end on a boundary. ``select`` is an optional boolean mask of the
        segments to draw (e.g. ``tree.generation <= 3``).
        """

        n = len(tree)
        if n == 0:
            return self
        segments = to_drawing(tree.segments())
        direction = to_drawing(tree.direction)
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = direction / np.where(norms == 0.0, 1.0, norms)
        escaping = np.asarray(tree.escaped, dtype=bool)

        rgba = colors(tree) if callable(colors) else colors
        rgba = np.tile(np.asarray(default_color, dtype=float), (n, 1)) if rgba is None else np.asarray(rgba, dtype=float)
        weight = intensities(tree) if callable(intensities) else intensities
        weight = np.asarray(tree.power, dtype=float) if weight is None else np.asarray(weight, dtype=float)

        keep = np.ones(n, dtype=bool) if select is None else np.asarray(select, dtype=bool).copy()
        if not show_misses:
            keep &= ~escaping
        if not keep.any():
            return self
        self.add(
            RaySegmentsItem(
                segments=segments[keep],
                colors=rgba[keep],
                intensities=weight[keep],
                escaping=escaping[keep],
                directions=direction[keep],
            )
        )
        if markers:
            ends = segments[keep & ~escaping, 1, :]
            if ends.size:
                self.add(MarkerItem(points=ends))
        return self

    # -- elements -----------------------------------------------------------

    def add_system(
        self,
        system,
        *,
        wavelength_um: float | None = None,
        material_colors: dict | None = None,
        tree=None,
        sensors: bool = True,
    ) -> "Drawing":
        """Add every element of *system* at its world pose.

        Lenses and prisms come from :func:`~raytracer.viz.bodies.body_outlines`
        (neighbouring lenses cut apart, filled by index); pass the traced
        ``tree`` to overlay what each sensor recorded.
        """

        for body in body_outlines(system, wavelength_um=wavelength_um):
            self._add_body(body, material_colors)
        scale = system_extent(system)
        for element, pose in system.walk():
            if isinstance(element, (Lens, Prism)):
                continue
            if isinstance(element, Sensor):
                if sensors:
                    self.add_sensor(element, pose, hits=_sensor_hits(tree, element.name), fallback=scale)
                continue
            self._add_thin(element, pose, scale)
        return self

    def add_elements(self, elements: Iterable, *, wavelength_um: float | None = None) -> "Drawing":
        """Add loose elements, each at its own pose, or ``(element, pose)`` pairs.

        No neighbour cuts are made (use :meth:`add_system` for a system).
        """

        pairs = [(e, e.pose) if isinstance(e, Element) else (e[0], e[1]) for e in elements]
        sizes = [getattr(e, "semi_diameter", None) or getattr(getattr(e, "aperture", None), "extent", None) for e, _ in pairs]
        scale = max([s for s in sizes if s] or [DEFAULT_SEMIDIAMETER])
        for element, pose in pairs:
            if isinstance(element, Lens):
                polygon, extents, _ = element_outline(element, pose)
                n = medium_index(element.material, wavelength_um)
                self._add_body(BodyOutline(element.name, element.material, n, polygon, extents, True), None)
            elif isinstance(element, Prism):
                polygon, exact = polyhedron_section(element.solid, pose)
                n = medium_index(element.material, wavelength_um)
                self._add_body(BodyOutline(element.name, element.material, n, np.vstack([polygon, polygon[:1]]), None, exact), None)
            elif isinstance(element, Sensor):
                self.add_sensor(element, pose, fallback=scale)
            else:
                self._add_thin(element, pose, scale)
        return self

    def add_sensor(self, sensor: Sensor, pose: Pose | None = None, *, hits=None, fallback: float = DEFAULT_SEMIDIAMETER) -> "Drawing":
        """Add a sensor's section; *hits* are its recorded points (world ``(K, 3)`` or a ``DetectorHits``)."""

        pose = sensor.pose if pose is None else pose
        if hits is not None and hasattr(hits, "points"):
            hits = hits.points
        hits2d = None if hits is None or len(hits) == 0 else to_drawing(np.asarray(hits, dtype=float))
        for curve in face_section(sensor.surface, pose, sensor.aperture, fallback=fallback):
            self.add(SensorItem(polyline=curve, hits=hits2d, name=sensor.name))
            hits2d = None  # one overlay per sensor
        return self

    def _add_body(self, body: BodyOutline, material_colors) -> None:
        colors = body_colors(body, material_colors)
        if colors is None:
            return
        face, edge = colors
        faces = None
        if body.extents is None:  # a prism: convex section, fan-triangulated
            faces = fan_faces(body.polygon.shape[0] - 1)
        self.add(
            LensBodyItem(polygon=body.polygon, facecolor=_rgba(face, 0.55), edgecolor=_rgba(edge), faces=faces, name=body.name)
        )

    def _add_thin(self, element: Element, pose: Pose, scale: float) -> None:
        if isinstance(element, Mirror):
            for curve in face_section(element.surface, pose, element.aperture):
                self.add(SurfaceItem(polyline=curve, color=MIRROR_RGBA, width=3.0, name=element.name))
        elif isinstance(element, BeamSplitter):
            for curve in face_section(element.interface.surface, pose, element.aperture):
                self.add(SurfaceItem(polyline=curve, color=SPLITTER_RGBA, width=2.0, name=element.name))
        elif isinstance(element, Stop) and not element.virtual:
            for curve in face_section(Plane(), pose, Circle(element.outer_radius), hole=element.aperture):
                self.add(SurfaceItem(polyline=curve, color=STOP_RGBA, width=3.0, name=element.name))

    # -- view -----------------------------------------------------------------

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """(x_lims, y_lims): explicit limits or data bounds with margin."""

        if self.x_lims is not None and self.y_lims is not None:
            return self.x_lims, self.y_lims
        points = []
        for item in self.items:
            if isinstance(item, RaySegmentsItem):
                points.append(item.segments[~item.escaping].reshape(-1, 2))
                points.append(item.segments[:, 0, :])
            elif isinstance(item, (SurfaceItem, SensorItem)):
                points.append(item.polyline)
            elif isinstance(item, LensBodyItem):
                points.append(item.polygon)
            elif isinstance(item, MarkerItem):
                points.append(item.points)
        points = [np.asarray(p, dtype=float).reshape(-1, 2) for p in points if np.size(p)]
        if not points:
            return (-10.0, 10.0), (-10.0, 10.0)
        allpts = np.vstack(points)
        finite = allpts[np.isfinite(allpts).all(axis=1)]
        x0, y0 = finite.min(axis=0)
        x1, y1 = finite.max(axis=0)
        mx = 0.05 * max(x1 - x0, 1e-6)
        my = 0.05 * max(y1 - y0, 1e-6)
        x_lims = self.x_lims or (x0 - mx, x1 + mx)
        y_lims = self.y_lims or (y0 - my, y1 + my)
        return x_lims, y_lims


def _sensor_hits(tree, name: str):
    if tree is None:
        return None
    hits = tree.detectors.get(name)
    return None if hits is None else hits.points


__all__ = [
    "Drawing",
    "RaySegmentsItem",
    "SurfaceItem",
    "LensBodyItem",
    "SensorItem",
    "MarkerItem",
]
