"""Interactive editing: drag sources and elements, tune parameters live.

The model layer (:class:`Parameter`, :class:`Handle`, adapters, and
:class:`InteractiveSession`) is toolkit-agnostic and fully testable headless;
the GL front end wires it to mouse/keyboard events on the OpenGL viewer:

- **drag** a handle (a source's origin, a lens or mirror vertex, a sensor's
  end points) to move it in the drawing plane ``(z, y)``;
- **Tab** cycles the editable parameters; **←/→** adjust the active one
  (Shift for coarse steps); the window title shows the live value;
- every edit retraces the system with the
  :class:`~raytracer.tracing.NonSequentialTracer` (a meridional fan of a few
  hundred rays takes milliseconds) and refreshes the display in place.

Elements and systems are immutable values, so an edit never mutates them:
it rebuilds the element (``element.replace(...)`` / ``.placed(pose)``) and
swaps it into a new :class:`~raytracer.system.OpticalSystem`
(``system.with_component``), which the session then traces. Element
editables therefore address their element *by name* through the session;
sources are plain emitters and are edited in place.

In notebooks, :func:`ipywidgets_panel` builds sliders from the same
Parameter objects for use with the matplotlib backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Sequence

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
from .scene import Drawing

if TYPE_CHECKING:  # pragma: no cover
    from ..tracing import RayTree


@dataclass
class Parameter:
    """One editable scalar with bounds and a step size."""

    name: str
    get: Callable[[], float]
    set: Callable[[float], None]
    lo: float
    hi: float
    step: float

    def nudge(self, direction: float, *, coarse: bool = False) -> float:
        value = self.get() + direction * self.step * (10.0 if coarse else 1.0)
        value = float(np.clip(value, self.lo, self.hi))
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
    """An object exposed to interaction: its handles and parameters.

    ``obj`` is the source itself, or the *name* of an element of the
    session's system (elements are immutable; the session holds the
    current one).
    """

    obj: object
    label: str
    handles: list[Handle] = field(default_factory=list)
    parameters: list[Parameter] = field(default_factory=list)


# -- the drawing plane <-> poses ----------------------------------------------


def _zy(point3) -> np.ndarray:
    return np.array([float(point3[2]), float(point3[1])])


def _with_zy(point3, zy) -> np.ndarray:
    return np.array([float(point3[0]), float(zy[1]), float(zy[0])])


def tilt_deg(rotation: Rotation) -> float:
    """Angle of a frame's +z axis in the drawing plane, from +z toward +y (degrees)."""

    axis = rotation.axis
    return float(np.degrees(np.arctan2(axis[1], axis[2])))


def tilt_rotation(angle_deg: float) -> Rotation:
    """The rotation about x that turns +z by *angle_deg* toward +y."""

    return Rotation.axis_angle((1.0, 0.0, 0.0), -float(angle_deg))


def _with_radius(surface, radius: float):
    """The same conic family with a new vertex radius (``inf`` is a plane)."""

    if not np.isfinite(radius) or radius == 0.0:
        return Plane()
    if isinstance(surface, Sphere) or isinstance(surface, Plane):
        return Sphere(radius)
    return Conic(radius, surface.k).classify()


def _sensor_pose_from_segment(p0, p1) -> tuple[Pose, float]:
    """Pose (centre, local +y along p0→p1, local x = world x) and length of a drawn segment."""

    a = np.array([0.0, float(p0[1]), float(p0[0])])
    b = np.array([0.0, float(p1[1]), float(p1[0])])
    span = b - a
    length = float(np.linalg.norm(span))
    if length == 0.0:
        raise ValueError("a sensor segment has two distinct end points")
    e_y = span / length
    e_x = np.array([1.0, 0.0, 0.0])
    e_z = np.cross(e_x, e_y)
    return Pose(0.5 * (a + b), Rotation(np.column_stack([e_x, e_y, e_z]))), length


def _sensor_ends(sensor: Sensor, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    half = sensor.aperture.height / 2 if isinstance(sensor.aperture, Rectangle) else (
        sensor.aperture.extent if sensor.aperture is not None else 10.0
    )
    p0 = pose.to_world(np.array([0.0, -half, 0.0]))
    p1 = pose.to_world(np.array([0.0, half, 0.0]))
    return _zy(p0), _zy(p1)


# -- adapters -----------------------------------------------------------------


def _source_editable(obj) -> Editable:
    if isinstance(obj, PointSource):

        def move(p, o=obj):
            o.position = _with_zy(o.position, p)

        def set_angle(v, o=obj):
            o.direction = tilt_rotation(v).apply(np.array([0.0, 0.0, 1.0]))

        return Editable(
            obj=obj,
            label="source",
            handles=[Handle("origin", lambda o=obj: _zy(o.position), move)],
            parameters=[
                Parameter(
                    "half_angle_deg",
                    lambda o=obj: float(o.half_angle_deg),
                    lambda v, o=obj: setattr(o, "half_angle_deg", float(v)),
                    0.5, 89.0, 0.5,
                ),
                Parameter(
                    "axis_angle_deg",
                    lambda o=obj: float(np.degrees(np.arctan2(o.direction[1], o.direction[2]))),
                    set_angle,
                    -180.0, 180.0, 1.0,
                ),
                Parameter(
                    "rays",
                    lambda o=obj: float(o.rays),
                    lambda v, o=obj: setattr(o, "rays", max(3, int(round(v)))),
                    3, 5000, 10,
                ),
            ],
        )

    if isinstance(obj, CollimatedSource):

        def move(p, o=obj):
            o.origin = _with_zy(o.origin, p)

        return Editable(
            obj=obj,
            label="beam",
            handles=[Handle("origin", lambda o=obj: _zy(o.origin), move)],
            parameters=[
                Parameter(
                    "beam_radius",
                    lambda o=obj: float(o.beam_radius),
                    lambda v, o=obj: setattr(o, "beam_radius", float(v)),
                    0.05, 250.0, 0.25,
                ),
            ],
        )
    raise TypeError(f"No editable adapter for {type(obj).__name__}")


def _element_editable(element, session: "InteractiveSession") -> Editable:
    name = element.name

    def current():
        return session.system[name]

    def world_pose() -> Pose:
        return session.system.pose_of(name)

    def place(world: Pose) -> None:
        session.place(name, world)

    def change(**kwargs) -> None:
        session.replace_element(name, current().replace(**kwargs))

    def move_vertex(p) -> None:
        pose = world_pose()
        place(Pose(_with_zy(pose.position, p), pose.rotation))

    def set_tilt(v) -> None:
        pose = world_pose()
        place(Pose(pose.position, tilt_rotation(v)))

    tilt = Parameter("angle_deg", lambda: tilt_deg(world_pose().rotation), set_tilt, -180.0, 180.0, 1.0)
    vertex = Handle("vertex", lambda: _zy(world_pose().position), move_vertex)

    if isinstance(element, Lens):
        parameters = []
        for side in ("front", "back"):
            if isinstance(getattr(element, side).surface, Conic):
                parameters.append(
                    Parameter(
                        "r1" if side == "front" else "r2",
                        lambda side=side: float(getattr(current(), side).surface.radius),
                        lambda v, side=side: change(
                            **{side: getattr(current(), side).replace(surface=_with_radius(getattr(current(), side).surface, v))}
                        ),
                        -1000.0, 1000.0, 1.0,
                    )
                )
        parameters += [
            Parameter(
                "thickness",
                lambda: float(current().thickness),
                lambda v: change(thickness=max(0.1, float(v))),
                0.1, 100.0, 0.5,
            ),
            Parameter(
                "n",
                lambda: medium_index(current().material, session.wavelength_um),
                lambda v: change(material=ConstantIndex(getattr(current().material, "name", "glass"), float(v))),
                1.0, 3.0, 0.01,
            ),
            tilt,
        ]
        return Editable(obj=name, label=name, handles=[vertex], parameters=parameters)

    if isinstance(element, Mirror):
        parameters = []
        if isinstance(element.surface, Conic):
            parameters.append(
                Parameter(
                    "radius",
                    lambda: float(current().surface.radius),
                    lambda v: change(surface=current().interface.replace(surface=_with_radius(current().surface, v))),
                    -2000.0, 2000.0, 5.0,
                )
            )
        parameters.append(tilt)
        return Editable(obj=name, label=name, handles=[vertex], parameters=parameters)

    if isinstance(element, Sensor):

        def end(k):
            return _sensor_ends(current(), world_pose())[k]

        def move_end(k, p) -> None:
            ends = list(_sensor_ends(current(), world_pose()))
            ends[k] = np.asarray(p, dtype=float)
            pose, length = _sensor_pose_from_segment(*ends)
            sensor = current()
            width = sensor.aperture.width if isinstance(sensor.aperture, Rectangle) else length
            session.replace_element(name, sensor.replace(aperture=Rectangle(width, length)), world=pose)

        return Editable(
            obj=name,
            label=name,
            handles=[
                Handle("p0", lambda: end(0), lambda p: move_end(0, p)),
                Handle("p1", lambda: end(1), lambda p: move_end(1, p)),
            ],
            parameters=[],
        )
    raise TypeError(f"No editable adapter for {type(element).__name__}")


def editable_for(obj, session: "InteractiveSession | None" = None) -> Editable:
    """Build the default Editable for a source, or for an element of *session*'s system.

    Sources (:class:`~raytracer.sources.PointSource`,
    :class:`~raytracer.sources.CollimatedSource`) are edited in place.
    Elements (:class:`~raytracer.elements.Lens`, :class:`~raytracer.elements.Mirror`,
    :class:`~raytracer.elements.Sensor`, or an element's name) need the
    *session* whose system they belong to: every edit rebuilds the element
    and the system.
    """

    if isinstance(obj, (PointSource, CollimatedSource)):
        return _source_editable(obj)
    if session is not None and isinstance(obj, (str, Lens, Mirror, Sensor)):
        element = session.system[obj] if isinstance(obj, str) else obj
        return _element_editable(element, session)
    if isinstance(obj, (Lens, Mirror, Sensor)):
        raise TypeError(f"editing {type(obj).__name__} {obj.name!r} needs the session that holds its system")
    raise TypeError(f"No editable adapter for {type(obj).__name__}")


# -- session ------------------------------------------------------------------


class InteractiveSession:
    """An editable optical system with live non-sequential retrace.

    The session owns the sources and the current
    :class:`~raytracer.system.OpticalSystem`; every edit replaces the system
    with a rebuilt one. ``retrace()`` traces the sources through it with a
    :class:`~raytracer.tracing.NonSequentialTracer` (``split=False``
    follows the one branch a sequential trace would take — refraction
    through glass, reflection off mirrors — the clean picture for drawing;
    ``split=True`` follows every Fresnel branch) and rebuilds the neutral
    :class:`~raytracer.viz.scene.Drawing`. Attach a display with ``run()``
    (OpenGL) or drive it headless / from widgets by calling ``retrace()``
    after edits. Sources are best given a meridional fan (``fan="y"``);
    ``wavelength_um`` is where dispersive media are evaluated for rays
    whose source names no wavelength (the d line by default).
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
            editables += [
                editable_for(element.name, self)
                for element in system.elements
                if isinstance(element, (Lens, Mirror, Sensor))
            ]
        self.editables = list(editables)
        self.drawing: Drawing | None = None
        self.tree: "RayTree | None" = None
        self._selection = 0  # flat index over (editable, parameter) pairs

    # -- editing the system ------------------------------------------------

    def replace_element(self, name: str, element, *, world: Pose | None = None) -> None:
        """Swap element *name* for *element* (at world pose *world* when given)."""

        if world is not None:
            parent = self.system.pose_of(name) @ self.system[name].pose.inverse()
            element = element.placed(parent.inverse() @ world)
        self.system = self.system.with_component(name, element)

    def place(self, name: str, world: Pose) -> None:
        """Move element *name* to the world pose *world* (its assembly, if any, stays put)."""

        self.replace_element(name, self.system[name], world=world)

    # -- tracing ---------------------------------------------------------

    def tracer(self) -> NonSequentialTracer:
        return NonSequentialTracer(
            self.system,
            wavelength_um=self.wavelength_um if self.wavelength_um is not None else COLOR_WAVELENGTH_UM,
            split=self.split,
            max_interactions=self.max_interactions,
        )

    def hits(self, sensor: str):
        """What *sensor* recorded in the last trace (``None`` when nothing reached it)."""

        return None if self.tree is None else self.tree.detectors.get(sensor)

    def retrace(self) -> Drawing:
        self.tree = self.tracer().trace(self.sources)
        drawing = Drawing()
        drawing.add_system(self.system, wavelength_um=self.wavelength_um, tree=self.tree)
        drawing.add_tree(self.tree)
        self.drawing = drawing
        if self.on_update is not None:
            self.on_update(drawing)
        return drawing

    # -- selection / parameters -------------------------------------------

    def _flat_parameters(self) -> list[tuple[Editable, Parameter]]:
        return [
            (editable, parameter)
            for editable in self.editables
            for parameter in editable.parameters
        ]

    @property
    def selected(self) -> tuple[Editable, Parameter] | None:
        flat = self._flat_parameters()
        if not flat:
            return None
        return flat[self._selection % len(flat)]

    def cycle_selection(self, step: int = 1) -> tuple[Editable, Parameter] | None:
        flat = self._flat_parameters()
        if not flat:
            return None
        self._selection = (self._selection + step) % len(flat)
        return self.selected

    def adjust_selected(self, direction: float, *, coarse: bool = False) -> str:
        pair = self.selected
        if pair is None:
            return "no parameters"
        editable, parameter = pair
        value = parameter.nudge(direction, coarse=coarse)
        self.retrace()
        return f"{editable.label}.{parameter.name} = {value:.4g}"

    def status(self) -> str:
        pair = self.selected
        if pair is None:
            return "drag handles to edit"
        editable, parameter = pair
        return (
            f"[Tab] {editable.label}.{parameter.name} = {parameter.get():.4g}  "
            f"[←/→ adjust, Shift=coarse]"
        )

    # -- handle picking ----------------------------------------------------

    def all_handles(self) -> list[tuple[Editable, Handle]]:
        return [
            (editable, handle)
            for editable in self.editables
            for handle in editable.handles
        ]

    def pick_handle(
        self, world_pos: np.ndarray, *, radius_world: float
    ) -> tuple[Editable, Handle] | None:
        best, best_dist = None, radius_world
        for editable, handle in self.all_handles():
            dist = float(np.linalg.norm(handle.get() - world_pos))
            if dist <= best_dist:
                best, best_dist = (editable, handle), dist
        return best

    def drag_handle(self, handle: Handle, world_pos: np.ndarray) -> None:
        handle.set(np.asarray(world_pos, dtype=float))
        self.retrace()

    # -- GL front end -------------------------------------------------------

    def run(self, **viewer_kwargs) -> None:  # pragma: no cover - interactive
        """Open the OpenGL viewer wired to this session."""

        from vispy import app

        from .protocol import GLBackend

        self.retrace()
        backend = GLBackend(**viewer_kwargs)
        viewer = backend.render(self.drawing)
        base_title = "raytracer interactive"
        viewer.title = f"{base_title} — {self.status()}"
        drag_state: dict = {"handle": None}

        # Draw handle markers on top.
        def refresh_handles():
            viewer.clear_markers()
            points = np.array([h.get() for _, h in self.all_handles()])
            if points.size:
                viewer.draw_markers(points, color="cyan", size=9.0)

        def full_refresh():
            backend.update(self.drawing)
            refresh_handles()
            viewer.title = f"{base_title} — {self.status()}"

        @viewer.events.mouse_press.connect
        def _on_press(event):
            if event.button != 1:
                return
            gl_pos = viewer._mouse_to_gl(event.pos)
            world = viewer.camera.screen_to_world(gl_pos)
            pick_radius = 12.0 / viewer.camera.pixels_per_world
            picked = self.pick_handle(np.asarray(world), radius_world=pick_radius)
            if picked is not None:
                drag_state["handle"] = picked[1]
                event.handled = True
                viewer._drag_start = None  # suppress camera pan

        @viewer.events.mouse_move.connect
        def _on_move(event):
            handle = drag_state["handle"]
            if handle is None or not event.is_dragging:
                return
            gl_pos = viewer._mouse_to_gl(event.pos)
            world = viewer.camera.screen_to_world(gl_pos)
            self.drag_handle(handle, world)
            full_refresh()
            event.handled = True
            viewer._drag_start = None

        @viewer.events.mouse_release.connect
        def _on_release(event):
            drag_state["handle"] = None

        @viewer.events.key_press.connect
        def _on_key(event):
            if event.key is None:
                return
            name = event.key.name
            if name == "Tab":
                self.cycle_selection()
                viewer.title = f"{base_title} — {self.status()}"
            elif name in ("Left", "Right"):
                coarse = "Shift" in [m.name for m in event.modifiers]
                message = self.adjust_selected(
                    1.0 if name == "Right" else -1.0, coarse=coarse
                )
                full_refresh()
                viewer.title = f"{base_title} — {message}"

        refresh_handles()
        viewer.show()
        app.run()


def ipywidgets_panel(session: InteractiveSession, *, backend=None):
    """Build an ipywidgets slider panel bound to the session (notebooks).

    Requires ipywidgets; pass an already-rendered ``MplBackend`` to refresh
    its figure on every change (use the ``%matplotlib widget`` magic).
    """

    import ipywidgets as widgets

    controls = []
    for editable in session.editables:
        for parameter in editable.parameters:
            slider = widgets.FloatSlider(
                value=parameter.get(),
                min=parameter.lo,
                max=parameter.hi,
                step=parameter.step,
                description=f"{editable.label}.{parameter.name}",
                continuous_update=False,
                style={"description_width": "initial"},
            )

            def _on_change(change, parameter=parameter):
                parameter.set(change["new"])
                drawing = session.retrace()
                if backend is not None:
                    backend.update(drawing)
                    backend.fig.canvas.draw_idle()

            slider.observe(_on_change, names="value")
            controls.append(slider)
    return widgets.VBox(controls)


__all__ = [
    "Parameter",
    "Handle",
    "Editable",
    "editable_for",
    "InteractiveSession",
    "ipywidgets_panel",
]
