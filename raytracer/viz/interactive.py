"""Interactive editing: drag sources/elements and tune parameters live.

The model layer (:class:`Parameter`, :class:`Handle`, adapters, and
:class:`InteractiveSession`) is toolkit-agnostic and fully testable headless;
the GL front end wires it to mouse/keyboard events on the OpenGL viewer:

- **drag** a handle (source origin, lens vertex, screen endpoint) to move it;
- **Tab** cycles the editable parameters; **←/→** adjust the active one
  (Shift for coarse steps); the window title shows the live value;
- every edit retraces the scene (a 2-D trace of hundreds of rays takes
  milliseconds) and refreshes the display in place.

In notebooks, :func:`ipywidgets_panel` builds sliders from the same
Parameter objects for use with the matplotlib backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from ..optics.instruments import Screen
from ..optics.elements import Lens, Mirror
from ..propagation.branching import BranchingTracer, TraceConfig
from .scene import Scene


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
    """A draggable world-space point."""

    label: str
    get: Callable[[], np.ndarray]
    set: Callable[[np.ndarray], None]


@dataclass
class Editable:
    """An object exposed to interaction: its handles and parameters."""

    obj: object
    label: str
    handles: list[Handle] = field(default_factory=list)
    parameters: list[Parameter] = field(default_factory=list)


# -- adapters -----------------------------------------------------------------


def _rebuild_lens(lens: Lens, **overrides) -> None:
    """Rebuild a lens in place from its current geometry plus overrides."""

    front = lens.front
    back = lens.back
    axis = np.array([np.cos(front.angle), np.sin(front.angle)])
    current = {
        "r1": front.profile.radius,
        "r2": back.profile.radius,
        "thickness": float((back.vertex - front.vertex) @ axis),
        "semidiameter": front.semidiameter,
        "n": front.n_interior,
        "n_ambient": front.n_exterior,
        "vertex": front.vertex.copy(),
        "angle": front.angle,
        "name": lens.name,
    }
    current.update(overrides)
    rebuilt = Lens.from_radii(**current)
    lens.front = rebuilt.front
    lens.back = rebuilt.back
    lens.rims = rebuilt.rims


def editable_for(obj) -> Editable:
    """Build the default Editable wrapper for a known scene object."""

    from ..optics.sources import ParallelSource, PointSource

    if isinstance(obj, PointSource):
        def move(p, o=obj):
            o.origin = np.asarray(p, dtype=float)

        def set_angle(v, o=obj):
            o.axis_direction = np.array([np.cos(v), np.sin(v)])

        return Editable(
            obj=obj,
            label="source",
            handles=[Handle("origin", lambda o=obj: np.asarray(o.origin), move)],
            parameters=[
                Parameter(
                    "aperture_deg",
                    lambda o=obj: float(np.rad2deg(o.aperture)),
                    lambda v, o=obj: setattr(o, "aperture", float(np.deg2rad(v))),
                    1.0, 178.0, 1.0,
                ),
                Parameter(
                    "axis_angle_deg",
                    lambda o=obj: float(
                        np.rad2deg(np.arctan2(o.axis_direction[1], o.axis_direction[0]))
                    ),
                    lambda v, o=obj: set_angle(np.deg2rad(v)),
                    -180.0, 180.0, 1.0,
                ),
                Parameter(
                    "samples",
                    lambda o=obj: float(o.samples),
                    lambda v, o=obj: setattr(o, "samples", max(3, int(round(v)))),
                    3, 5000, 10,
                ),
            ],
        )

    if isinstance(obj, ParallelSource):
        def move(p, o=obj):
            o.origin = np.asarray(p, dtype=float)

        return Editable(
            obj=obj,
            label="beam",
            handles=[Handle("origin", lambda o=obj: np.asarray(o.origin), move)],
            parameters=[
                Parameter(
                    "width",
                    lambda o=obj: float(o.width),
                    lambda v, o=obj: setattr(o, "width", float(v)),
                    0.1, 500.0, 0.5,
                ),
            ],
        )

    if isinstance(obj, Lens):
        def move(p, lens=obj):
            _rebuild_lens(lens, vertex=np.asarray(p, dtype=float))

        return Editable(
            obj=obj,
            label=obj.name,
            handles=[
                Handle("vertex", lambda lens=obj: lens.front.vertex.copy(), move)
            ],
            parameters=[
                Parameter(
                    "r1",
                    lambda lens=obj: lens.front.profile.radius,
                    lambda v, lens=obj: _rebuild_lens(lens, r1=float(v)),
                    -1000.0, 1000.0, 1.0,
                ),
                Parameter(
                    "r2",
                    lambda lens=obj: lens.back.profile.radius,
                    lambda v, lens=obj: _rebuild_lens(lens, r2=float(v)),
                    -1000.0, 1000.0, 1.0,
                ),
                Parameter(
                    "thickness",
                    lambda lens=obj: float(
                        (lens.back.vertex - lens.front.vertex)
                        @ np.array([np.cos(lens.front.angle), np.sin(lens.front.angle)])
                    ),
                    lambda v, lens=obj: _rebuild_lens(lens, thickness=max(0.1, float(v))),
                    0.1, 100.0, 0.5,
                ),
                Parameter(
                    "n",
                    lambda lens=obj: lens.front.n_interior,
                    lambda v, lens=obj: _rebuild_lens(lens, n=float(v)),
                    1.0, 3.0, 0.01,
                ),
                Parameter(
                    "angle_deg",
                    lambda lens=obj: float(np.rad2deg(lens.front.angle)),
                    lambda v, lens=obj: _rebuild_lens(lens, angle=float(np.deg2rad(v))),
                    -90.0, 90.0, 1.0,
                ),
            ],
        )

    if isinstance(obj, Mirror):
        def move(p, m=obj):
            m.face.vertex = np.asarray(p, dtype=float)
            m.face.__post_init__()

        def set_radius(v, m=obj):
            from ..shapes.profile import AsphereProfile

            m.face.profile = AsphereProfile.from_radius(
                float(v), m.face.profile.conic, m.face.profile.coefficients
            )

        return Editable(
            obj=obj,
            label=obj.name,
            handles=[Handle("vertex", lambda m=obj: m.face.vertex.copy(), move)],
            parameters=[
                Parameter(
                    "radius",
                    lambda m=obj: m.face.profile.radius,
                    set_radius,
                    -2000.0, 2000.0, 5.0,
                ),
            ],
        )

    if isinstance(obj, Screen):
        def move0(p, s=obj):
            s.p0 = np.asarray(p, dtype=float)

        def move1(p, s=obj):
            s.p1 = np.asarray(p, dtype=float)

        return Editable(
            obj=obj,
            label="screen",
            handles=[
                Handle("p0", lambda s=obj: s.p0.copy(), move0),
                Handle("p1", lambda s=obj: s.p1.copy(), move1),
            ],
            parameters=[],
        )

    raise TypeError(f"No editable adapter for {type(obj).__name__}")


# -- session ------------------------------------------------------------------


class InteractiveSession:
    """Editable optical scene with live retrace.

    The session owns sources, elements (lenses/mirrors/raw surfaces), and
    screens; ``retrace()`` rebuilds the ray tree and the neutral Scene.
    Attach a display with ``run()`` (OpenGL) or drive it headless/from
    widgets by calling ``retrace()`` after edits.
    """

    def __init__(
        self,
        *,
        sources: Sequence,
        elements: Sequence = (),
        screens: Sequence[Screen] = (),
        trace_config: TraceConfig | None = None,
        editables: Optional[Iterable[Editable]] = None,
        on_update: Optional[Callable[[Scene], None]] = None,
    ) -> None:
        self.sources = list(sources)
        self.elements = list(elements)
        self.screens = list(screens)
        self.trace_config = trace_config or TraceConfig(max_generations=6)
        self.on_update = on_update
        if editables is None:
            editables = [
                editable_for(obj)
                for obj in (*self.sources, *self.elements, *self.screens)
            ]
        self.editables = list(editables)
        self.scene: Scene | None = None
        self.tree = None
        self._selection = 0  # flat index over (editable, parameter) pairs

    # -- tracing ---------------------------------------------------------

    def _surfaces(self):
        surfaces = []
        for element in self.elements:
            if hasattr(element, "surfaces"):
                surfaces.extend(element.surfaces())
            else:
                surfaces.append(element)
        surfaces.extend(self.screens)
        return surfaces

    def retrace(self) -> Scene:
        for screen in self.screens:
            screen.clear()
        tracer = BranchingTracer(self._surfaces(), self.trace_config)
        self.tree = tracer.trace(self.sources)

        scene = Scene()
        scene.add_elements([*self.elements, *self.screens])
        scene.add_tree(self.tree, intensity_resolver=lambda n: n.intensity)
        self.scene = scene
        if self.on_update is not None:
            self.on_update(scene)
        return scene

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
        viewer = backend.render(self.scene)
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
            backend.update(self.scene)
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
                scene = session.retrace()
                if backend is not None:
                    backend.update(scene)
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
