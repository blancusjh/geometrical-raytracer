"""The backend-neutral 2-D drawing: items in drawing coordinates ``(z, y)``.

A :class:`Drawing` is pure data — ray segments, surface curves, filled bodies, sensors, markers — in the
meridional plane of :mod:`~raytracer.viz.sag_drawing`. The matplotlib backend (:mod:`~raytracer.viz.mpl`) and
the OpenGL viewer (:mod:`~raytracer.viz.gl`) draw the same drawing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..elements import BeamSplitter, Element, Mirror, Sensor, Stop
from .bodies import body_outline, body_outlines, glass_colors
from .color import hex_rgba
from .sag_drawing import DEFAULT_SEMIDIAMETER, element_sections, extent_of, fan_faces, strip_faces, to_drawing

_THIN_STYLES = {
    Mirror: ((0.13, 0.13, 0.13, 1.0), 3.0),
    BeamSplitter: ((0.55, 0.55, 0.6, 1.0), 2.0),
    Stop: ((0.35, 0.35, 0.38, 1.0), 3.0),
}


class Drawing:
    """Drawable items in drawing coordinates ``(z, y)``, with optional view limits."""

    def __init__(self, x_lims: tuple[float, float] | None = None, y_lims: tuple[float, float] | None = None) -> None:
        self.items: list = []
        self.x_lims = x_lims
        self.y_lims = y_lims

    def add(self, item) -> "Drawing":
        self.items.append(item)
        return self

    def add_system(self, system, *, wavelength_um=None, material_colors=None, tree=None, sensors=True) -> "Drawing":
        """Every element of *system* at its world pose.

        Bodies come from :func:`~raytracer.viz.bodies.body_outlines` (neighbours cut apart, filled by index);
        a traced *tree* adds what each sensor recorded.
        """

        hits = {} if tree is None else {name: d.points for name, d in tree.detectors.items()}
        bodies = body_outlines(system, wavelength_um=wavelength_um)
        return self._add_placed(list(system.walk()), bodies, material_colors, hits, sensors)

    def add_elements(self, elements, *, wavelength_um=None) -> "Drawing":
        """Loose elements, each at its own pose, or ``(element, pose)`` pairs; no neighbour cuts."""

        placed = [(e, e.pose) if isinstance(e, Element) else (e[0], e[1]) for e in elements]
        bodies = [body_outline(e, p, wavelength_um=wavelength_um) for e, p in placed]
        return self._add_placed(placed, [b for b in bodies if b is not None], None, {}, True)

    def _add_placed(self, placed, bodies, material_colors, hits, sensors) -> "Drawing":
        for body in bodies:
            colors = glass_colors(body.material, body.n, material_colors)
            if colors is not None:  # a prism's section is convex: fan-triangulated
                faces = fan_faces(body.polygon.shape[0] - 1) if body.extents is None else None
                self.add(LensBodyItem(body.polygon, hex_rgba(colors[0], 0.55), hex_rgba(colors[1]), faces, body.name))
        scale = extent_of(e for e, _ in placed)
        for element, pose in placed:
            if isinstance(element, Sensor):
                if sensors:
                    self.add_sensor(element, pose, hits=hits.get(element.name), fallback=scale)
            elif type(element) in _THIN_STYLES:
                color, width = _THIN_STYLES[type(element)]
                for curve in element_sections(element, pose, fallback=scale):
                    self.add(SurfaceItem(curve, color, width, element.name))
        return self

    def add_sensor(self, sensor, pose=None, *, hits=None, fallback: float = DEFAULT_SEMIDIAMETER) -> "Drawing":
        """A sensor's section; *hits* are its recorded points (world ``(K, 3)``, or a ``DetectorHits``)."""

        hits = getattr(hits, "points", hits)
        hits = None if hits is None or len(hits) == 0 else to_drawing(hits)
        for curve in element_sections(sensor, sensor.pose if pose is None else pose, fallback=fallback):
            self.add(SensorItem(curve, hits=hits, name=sensor.name))
            hits = None  # one overlay per sensor
        return self

    def add_tree(
        self,
        tree,
        *,
        colors=None,
        intensities=None,
        show_misses: bool = True,
        markers=False,
        marker_size: float = 6.0,
        default_color: tuple = (1.0, 1.0, 1.0, 1.0),
        select: np.ndarray | None = None,
        escape_length: float | None = None,
    ) -> "Drawing":
        """The segments of a non-sequential :class:`~raytracer.tracing.RayTree`, projected on the drawing plane.

        ``colors`` is ``(M, 4)`` or a callable ``tree -> (M, 4)``; ``intensities`` likewise ``(M,)`` (default:
        each segment's power). Rays that left the system run to the edge of the view, or end ``escape_length``
        mm from their start, or are dropped with ``show_misses=False``. ``markers`` (True, or a colour) marks
        where segments end on a boundary; ``select`` is a boolean mask of the segments to draw.
        """

        n = len(tree)
        if n == 0:
            return self
        segments = to_drawing(tree.segments())
        direction = to_drawing(tree.direction)
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = direction / np.where(norms == 0.0, 1.0, norms)
        escaped = np.asarray(tree.escaped, dtype=bool)

        rgba = colors(tree) if callable(colors) else colors
        rgba = np.tile(np.asarray(default_color, dtype=float), (n, 1)) if rgba is None else np.asarray(rgba, dtype=float)
        weight = intensities(tree) if callable(intensities) else intensities
        weight = np.broadcast_to(np.asarray(tree.power if weight is None else weight, dtype=float), (n,))

        keep = np.ones(n, dtype=bool) if select is None else np.asarray(select, dtype=bool).copy()
        if not show_misses:
            keep &= ~escaped
        if not keep.any():
            return self
        escaping = escaped
        if escape_length is not None:
            segments[escaped, 1] = segments[escaped, 0] + float(escape_length) * direction[escaped]
            escaping = np.zeros(n, dtype=bool)
        self.add(RaySegmentsItem(segments[keep], rgba[keep], weight[keep], escaping[keep], direction[keep]))
        ends = segments[keep & ~escaped, 1, :]
        if markers is not False and markers is not None and ends.size:
            self.add(MarkerItem(ends, size=marker_size) if markers is True else MarkerItem(ends, markers, marker_size))
        return self

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """``(x_lims, y_lims)``: the explicit limits, else the data's bounds with a 5 % margin."""

        if self.x_lims is not None and self.y_lims is not None:
            return self.x_lims, self.y_lims
        points = []
        for item in self.items:
            if isinstance(item, RaySegmentsItem):
                points += [item.segments[~item.escaping].reshape(-1, 2), item.segments[:, 0, :]]
            else:
                points.append(getattr(item, "polyline", getattr(item, "polygon", getattr(item, "points", None))))
        points = [np.asarray(p, dtype=float).reshape(-1, 2) for p in points if p is not None and np.size(p)]
        if not points:
            return (-10.0, 10.0), (-10.0, 10.0)
        allpts = np.vstack(points)
        finite = allpts[np.isfinite(allpts).all(axis=1)]
        (x0, y0), (x1, y1) = finite.min(axis=0), finite.max(axis=0)
        mx, my = 0.05 * max(x1 - x0, 1e-6), 0.05 * max(y1 - y0, 1e-6)
        return self.x_lims or (x0 - mx, x1 + mx), self.y_lims or (y0 - my, y1 + my)


@dataclass
class RaySegmentsItem:
    segments: np.ndarray  # (N, 2, 2) end points (z, y)
    colors: np.ndarray  # (N, 4) rgba; alpha = per-ray weight
    intensities: np.ndarray  # (N,)
    escaping: np.ndarray  # (N,) bool: extend to the view's edge
    directions: np.ndarray  # (N, 2) unit directions, for escaping rays


@dataclass
class SurfaceItem:
    polyline: np.ndarray  # (M, 2)
    color: tuple = (1.0, 1.0, 1.0, 1.0)
    width: float = 2.0
    name: str = ""


@dataclass
class LensBodyItem:
    """A filled body: its closed outline and, for GL fills, its triangles.

    Without ``faces`` the outline is laid out as by :func:`~raytracer.viz.sag_drawing.lens_outline` (one face
    up, the other back down) and strip-triangulated.
    """

    polygon: np.ndarray  # (M, 2) closed outline
    facecolor: tuple = (0.74, 0.88, 1.0, 0.55)
    edgecolor: tuple = (0.16, 0.47, 0.71, 1.0)
    faces: np.ndarray | None = None  # (K, 3) triangles over polygon vertices
    name: str = ""

    def triangles(self) -> tuple[np.ndarray, np.ndarray]:
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
    hits: np.ndarray | None = None  # (K, 2)
    name: str = ""


@dataclass
class MarkerItem:
    points: np.ndarray  # (M, 2)
    color: tuple = (0.86, 0.08, 0.24, 1.0)
    size: float = 6.0


__all__ = ["Drawing", "RaySegmentsItem", "SurfaceItem", "LensBodyItem", "SensorItem", "MarkerItem"]
