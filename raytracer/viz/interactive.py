"""Interactive editing: drag sources and elements, tune parameters, see the light follow.

The model — :class:`Parameter`, :class:`Handle`, :class:`Editable` and :class:`InteractiveSession` — needs no
toolkit and is tested headless. :meth:`InteractiveSession.run` wires it to the OpenGL viewer: **drag** a handle
(a source's origin, a lens or mirror vertex, a sensor's end) in the drawing plane ``(z, y)``; **Tab** cycles
the parameters and **←/→** adjust the active one (Shift: coarse). Every edit retraces with the
:class:`~raytracer.tracing.NonSequentialTracer` (milliseconds for a meridional fan) and redraws in place.

Elements and systems are immutable values: an edit rebuilds the element and swaps it into a new system
(``system.with_component``), so element editables address their element *by name* through the session.
Sources are plain emitters, edited in place. In notebooks, :func:`ipywidgets_panel` makes sliders of the
same parameters for the matplotlib backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from ..elements import Lens, Mirror, Sensor
from ..geometry.apertures import Rectangle
from ..geometry.surfaces import Conic, Plane, Sphere
from ..geometry.transforms import Pose, Rotation
from ..materials import ConstantIndex
from ..sources import CollimatedSource, PointSource
from ..system import OpticalSystem
from ..tracing import NonSequentialTracer
from .bodies import COLOR_WAVELENGTH_UM, medium_index
from .section import to_drawing
from .drawing import Drawing


@dataclass
class Parameter:
    """One editable scalar with bounds and a step."""

    name: str
    get: Callable[[], float]
    set: Callable[[float], None]
    lo: float
    hi: float
    step: float

    def nudge(self, direction: float, *, coarse: bool = False) -> float:
        value = float(np.clip(self.get() + direction * self.step * (10.0 if coarse else 1.0), self.lo, self.hi))
        self.set(value)
        return value


@dataclass
class Handle:
    """A draggable point of the drawing plane, ``(z, y)``."""

    label: str
    get: Callable[[], np.ndarray]
    set: Callable[[np.ndarray], None]


@dataclass
class Editable:
    """What one object exposes to interaction: ``obj`` is a source, or the name of an element of the session."""

    obj: object
    label: str
    handles: list[Handle] = field(default_factory=list)
    parameters: list[Parameter] = field(default_factory=list)


class InteractiveSession:
    """An editable optical system, retraced non-sequentially after every edit.

    The session owns the sources and the current :class:`~raytracer.system.OpticalSystem`. :meth:`retrace`
    traces them (``split=False`` follows the one branch a sequential trace would take — the clean picture;
    ``split=True`` every Fresnel branch) and rebuilds the :class:`~raytracer.viz.drawing.Drawing`. Sources are
    best given a meridional fan (``fan="y"``); ``wavelength_um`` is where dispersive media are read for rays
    whose source names no wavelength (default: the d line).
    """

    def __init__(
        self,
        *,
        sources: Sequence,
        system: OpticalSystem,
        split: bool = False,
        max_interactions: int = 6,
        wavelength_um: float | None = None,
        editables: Optional[Iterable[Editable]] = None,
        on_update: Optional[Callable[[Drawing], None]] = None,
    ) -> None:
        self.sources = list(sources)
        self.system = system
        self.split = bool(split)
        self.max_interactions = int(max_interactions)
        self.wavelength_um = wavelength_um
        self.on_update = on_update
        if editables is None:
            editables = [editable_for(source) for source in self.sources]
            editables += [editable_for(e.name, self) for e in system.elements if isinstance(e, (Lens, Mirror, Sensor))]
        self.editables = list(editables)
        self.drawing: Drawing | None = None
        self.rays = None
        self._selection = 0  # index over every (editable, parameter) pair

    def replace_element(self, name: str, element, *, world: Pose | None = None) -> None:
        """Swap element *name* for *element*, placed at the world pose *world* when given."""

        if world is not None:
            parent = self.system.pose_of(name) @ self.system[name].pose.inverse()
            element = element.placed(parent.inverse() @ world)
        self.system = self.system.with_component(name, element)

    def place(self, name: str, world: Pose) -> None:
        """Move element *name* to the world pose *world*; its assembly, if any, stays put."""

        self.replace_element(name, self.system[name], world=world)

    def tracer(self) -> NonSequentialTracer:
        wavelength = self.wavelength_um if self.wavelength_um is not None else COLOR_WAVELENGTH_UM
        return NonSequentialTracer(self.system, wavelength_um=wavelength, split=self.split, max_interactions=self.max_interactions)

    def hits(self, sensor: str):
        """The rays *sensor* detected in the last trace (None before the first)."""

        return None if self.rays is None else self.rays[self.rays.detected & (self.rays.boundary == sensor)]

    def retrace(self) -> Drawing:
        self.rays = self.tracer().trace(self.sources)
        self.drawing = Drawing().add_system(self.system, wavelength_um=self.wavelength_um, rays=self.rays).add_rays(self.rays)
        if self.on_update is not None:
            self.on_update(self.drawing)
        return self.drawing

    # -- selection ------------------------------------------------------------------

    def _parameters(self) -> list[tuple[Editable, Parameter]]:
        return [(e, p) for e in self.editables for p in e.parameters]

    @property
    def selected(self) -> tuple[Editable, Parameter] | None:
        pairs = self._parameters()
        return pairs[self._selection % len(pairs)] if pairs else None

    def cycle_selection(self, step: int = 1) -> tuple[Editable, Parameter] | None:
        pairs = self._parameters()
        if pairs:
            self._selection = (self._selection + step) % len(pairs)
        return self.selected

    def adjust_selected(self, direction: float, *, coarse: bool = False) -> str:
        if self.selected is None:
            return "no parameters"
        editable, parameter = self.selected
        value = parameter.nudge(direction, coarse=coarse)
        self.retrace()
        return f"{editable.label}.{parameter.name} = {value:.4g}"

    def status(self) -> str:
        if self.selected is None:
            return "drag handles to edit"
        editable, parameter = self.selected
        return f"[Tab] {editable.label}.{parameter.name} = {parameter.get():.4g}  [←/→ adjust, Shift=coarse]"

    # -- handles ------------------------------------------------------------------------

    def all_handles(self) -> list[tuple[Editable, Handle]]:
        return [(e, h) for e in self.editables for h in e.handles]

    def pick_handle(self, world_pos: np.ndarray, *, radius_world: float) -> tuple[Editable, Handle] | None:
        """The handle nearest *world_pos* within *radius_world*."""

        best, best_dist = None, radius_world
        for editable, handle in self.all_handles():
            dist = float(np.linalg.norm(handle.get() - world_pos))
            if dist <= best_dist:
                best, best_dist = (editable, handle), dist
        return best

    def drag_handle(self, handle: Handle, world_pos: np.ndarray) -> None:
        handle.set(np.asarray(world_pos, dtype=float))
        self.retrace()

    def run(self, **viewer_kwargs) -> None:  # pragma: no cover - interactive
        """Open the OpenGL viewer wired to this session."""

        from vispy import app

        from .protocol import GLBackend

        self.retrace()
        backend = GLBackend(**viewer_kwargs)
        viewer = backend.render(self.drawing)
        dragging: list[Handle] = []

        def title(message):
            viewer.title = f"raytracer interactive — {message}"

        def at(event):
            return np.asarray(viewer.camera.screen_to_world(viewer._mouse_to_gl(event.pos)))

        def refresh(message=None):
            backend.update(self.drawing)
            viewer.clear_markers()
            points = np.array([h.get() for _, h in self.all_handles()])
            if points.size:
                viewer.draw_markers(points, color="cyan", size=9.0)
            title(message or self.status())

        @viewer.events.mouse_press.connect
        def _on_press(event):
            picked = self.pick_handle(at(event), radius_world=12.0 / viewer.camera.pixels_per_world) if event.button == 1 else None
            if picked is not None:
                dragging[:] = [picked[1]]
                event.handled = True
                viewer._drag_start = None  # no camera pan

        @viewer.events.mouse_move.connect
        def _on_move(event):
            if dragging and event.is_dragging:
                self.drag_handle(dragging[0], at(event))
                refresh()
                event.handled = True
                viewer._drag_start = None

        @viewer.events.mouse_release.connect
        def _on_release(event):
            dragging.clear()

        @viewer.events.key_press.connect
        def _on_key(event):
            name = event.key.name if event.key is not None else None
            if name == "Tab":
                self.cycle_selection()
                title(self.status())
            elif name in ("Left", "Right"):
                coarse = "Shift" in [m.name for m in event.modifiers]
                refresh(self.adjust_selected(1.0 if name == "Right" else -1.0, coarse=coarse))

        refresh()
        viewer.show()
        app.run()


def editable_for(obj, session: InteractiveSession | None = None) -> Editable:
    """The default :class:`Editable` of a source, or of an element (or element name) of *session*'s system.

    Sources (:class:`~raytracer.sources.PointSource`, :class:`~raytracer.sources.CollimatedSource`) are edited
    in place; a :class:`~raytracer.elements.Lens`, :class:`~raytracer.elements.Mirror` or
    :class:`~raytracer.elements.Sensor` needs the session that holds its system.
    """

    if isinstance(obj, PointSource):
        def set_axis(v, o=obj):
            o.direction = _tilt(v).apply(np.array([0.0, 0.0, 1.0]))

        parameters = [
            Parameter("half_angle_deg", lambda: float(obj.half_angle_deg), lambda v: setattr(obj, "half_angle_deg", float(v)), 0.5, 89.0, 0.5),
            Parameter("axis_angle_deg", lambda: _angle_deg(obj.direction), set_axis, -180.0, 180.0, 1.0),
            Parameter("rays", lambda: float(obj.rays), lambda v: setattr(obj, "rays", max(3, int(round(v)))), 3, 5000, 10),
        ]
        return Editable(obj, "source", [_point_handle(obj, "position")], parameters)
    if isinstance(obj, CollimatedSource):
        radius = Parameter("beam_radius", lambda: float(obj.beam_radius), lambda v: setattr(obj, "beam_radius", float(v)), 0.05, 250.0, 0.25)
        return Editable(obj, "beam", [_point_handle(obj, "origin")], [radius])
    if isinstance(obj, (str, Lens, Mirror, Sensor)):
        if session is None:
            raise TypeError(f"editing {type(obj).__name__} {getattr(obj, 'name', obj)!r} needs the session that holds its system")
        return _element_editable(session.system[obj] if isinstance(obj, str) else obj, session)
    raise TypeError(f"No editable adapter for {type(obj).__name__}")


def _point_handle(source, attribute: str) -> Handle:
    """The source's origin as a handle: dragging moves it in ``(z, y)`` and keeps its x."""

    def move(zy):
        x = float(getattr(source, attribute)[0])
        setattr(source, attribute, np.array([x, float(zy[1]), float(zy[0])]))

    return Handle("origin", lambda: to_drawing(getattr(source, attribute)), move)


def _element_editable(element, session: InteractiveSession) -> Editable:
    name = element.name

    def current():
        return session.system[name]

    def world():
        return session.system.pose_of(name)

    def change(**arguments):
        session.replace_element(name, current().replace(**arguments))

    def move_vertex(zy):
        pose = world()
        session.place(name, Pose(np.array([float(pose.position[0]), float(zy[1]), float(zy[0])]), pose.rotation))

    vertex = Handle("vertex", lambda: to_drawing(world().position), move_vertex)
    tilt = Parameter("angle_deg", lambda: _angle_deg(world().axis), lambda v: session.place(name, Pose(world().position, _tilt(v))),
                     -180.0, 180.0, 1.0)

    if isinstance(element, Lens):
        parameters = [
            Parameter(
                label,
                lambda side=side: float(getattr(current(), side).surface.radius),
                lambda v, side=side: change(**{side: getattr(current(), side).replace(surface=_with_radius(getattr(current(), side).surface, v))}),
                -1000.0, 1000.0, 1.0,
            )
            for side, label in (("front", "r1"), ("back", "r2"))
            if isinstance(getattr(element, side).surface, Conic)
        ]
        parameters += [
            Parameter("thickness", lambda: float(current().thickness), lambda v: change(thickness=max(0.1, float(v))), 0.1, 100.0, 0.5),
            Parameter("n", lambda: medium_index(current().material, session.wavelength_um),
                      lambda v: change(material=ConstantIndex(getattr(current().material, "name", "glass"), float(v))), 1.0, 3.0, 0.01),
            tilt,
        ]
        return Editable(name, name, [vertex], parameters)

    if isinstance(element, Mirror):
        parameters = [tilt]
        if isinstance(element.surface, Conic):
            radius = Parameter("radius", lambda: float(current().surface.radius),
                               lambda v: change(surface=_with_radius(current().surface, v)),
                               -2000.0, 2000.0, 5.0)
            parameters.insert(0, radius)
        return Editable(name, name, [vertex], parameters)

    if not isinstance(element, Sensor):
        raise TypeError(f"No editable adapter for {type(element).__name__}")

    def move_end(k, zy):
        ends = list(_sensor_ends(current(), world()))
        ends[k] = np.asarray(zy, dtype=float)
        pose, length = _segment_pose(*ends)
        sensor = current()
        width = sensor.aperture.width if isinstance(sensor.aperture, Rectangle) else length
        session.replace_element(name, sensor.replace(aperture=Rectangle(width, length)), world=pose)

    # a sensor's two ends: dragging one turns and resizes it
    handles = [Handle(f"p{k}", lambda k=k: _sensor_ends(current(), world())[k], lambda zy, k=k: move_end(k, zy)) for k in (0, 1)]
    return Editable(name, name, handles, [])


def _angle_deg(vector) -> float:
    """Angle of a direction in the drawing plane, from +z toward +y (degrees)."""

    return float(np.degrees(np.arctan2(vector[1], vector[2])))


def _tilt(angle_deg: float) -> Rotation:
    """The rotation about x that turns +z by *angle_deg* toward +y."""

    return Rotation.axis_angle((1.0, 0.0, 0.0), -float(angle_deg))


def _with_radius(surface, radius: float):
    """The same conic family with a new vertex radius (``inf`` or 0 is a plane)."""

    if not np.isfinite(radius) or radius == 0.0:
        return Plane()
    if isinstance(surface, (Sphere, Plane)):
        return Sphere(radius)
    return Conic(radius, surface.k).classify()


def _sensor_ends(sensor: Sensor, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    """The drawn ends ``(z, y)`` of a sensor: its local y axis across the aperture's height."""

    aperture = sensor.aperture
    half = aperture.height / 2 if isinstance(aperture, Rectangle) else aperture.extent if aperture is not None else 10.0
    return to_drawing(pose.to_world(np.array([0.0, -half, 0.0]))), to_drawing(pose.to_world(np.array([0.0, half, 0.0])))


def _segment_pose(p0, p1) -> tuple[Pose, float]:
    """The pose (centred, local +y along p0→p1, local x the world x) and length of a drawn segment."""

    a, b = np.array([0.0, float(p0[1]), float(p0[0])]), np.array([0.0, float(p1[1]), float(p1[0])])
    length = float(np.linalg.norm(b - a))
    if length == 0.0:
        raise ValueError("a sensor segment has two distinct end points")
    e_x, e_y = np.array([1.0, 0.0, 0.0]), (b - a) / length
    return Pose(0.5 * (a + b), Rotation(np.column_stack([e_x, e_y, np.cross(e_x, e_y)]))), length


def ipywidgets_panel(session: InteractiveSession, *, backend=None):
    """An ipywidgets slider panel bound to *session*; an already-rendered ``MplBackend`` is refreshed on every
    change (with ``%matplotlib widget``)."""

    import ipywidgets as widgets

    controls = []
    for editable, parameter in session._parameters():
        slider = widgets.FloatSlider(value=parameter.get(), min=parameter.lo, max=parameter.hi, step=parameter.step,
                                     description=f"{editable.label}.{parameter.name}", continuous_update=False,
                                     style={"description_width": "initial"})

        def on_change(change, parameter=parameter):
            parameter.set(change["new"])
            drawing = session.retrace()
            if backend is not None:
                backend.update(drawing)
                backend.fig.canvas.draw_idle()

        slider.observe(on_change, names="value")
        controls.append(slider)
    return widgets.VBox(controls)


__all__ = ["Parameter", "Handle", "Editable", "editable_for", "InteractiveSession", "ipywidgets_panel"]
