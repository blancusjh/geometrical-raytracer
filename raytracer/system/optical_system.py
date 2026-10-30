"""The optical system: placed components in an ambient medium."""

from __future__ import annotations

from typing import Any, Iterator, Sequence

from ..elements import Element, Stop
from ..geometry.transforms import Pose
from ..materials import AIR, Material
from .assembly import Assembly, Component
from .faces import PhysicalFace, Step, infer_path, parse_path, resolve_faces


class OpticalSystem:
    """Elements and assemblies placed in space, immersed in an ambient medium.

    This is the single model every engine reads: the non-sequential tracer
    visits its boundaries in whatever order light finds them; the
    sequential tracer follows its :attr:`path`.

    - ``ambient`` fills every region no element occupies.
    - ``path`` is the designed sequential route, as boundary names with an
      optional event (``"BS:reflect"``). Omitted, it is inferred for a
      coaxial system, and is an error to ask for otherwise.
    - ``stop`` names the aperture stop; omitted, it is the system's only
      :class:`~raytracer.elements.Stop`, if it has exactly one.

    Systems are values: :meth:`rotated`, :meth:`translated` and
    :meth:`replace` return new systems.
    """

    def __init__(
        self,
        components: Sequence[Component],
        *,
        name: str = "",
        ambient: Material = AIR,
        path: Sequence[str | Step] | None = None,
        stop: str | None = None,
    ) -> None:
        root = Assembly("system", components)
        names = [element.name for element, _, _ in root.walk()]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"element names must be unique; repeated: {duplicates}")
        object.__setattr__(self, "components", root.components)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "ambient", ambient)
        object.__setattr__(self, "_path_spec", None if path is None else tuple(path))
        object.__setattr__(self, "_stop", stop)
        object.__setattr__(self, "_cache", {})
        if stop is not None and stop not in names:
            raise KeyError(f"stop={stop!r} names no component")

    def __setattr__(self, name, value):
        raise AttributeError("OpticalSystem is immutable; use .replace(...)")

    # -- builders -------------------------------------------------------

    @classmethod
    def compose(cls, *parts, name: str = "", ambient: Material = AIR, stop: str | None = None) -> "OpticalSystem":
        """Elements and :class:`Gap` s in the order light meets them, along +z (see :mod:`.table`)."""

        from .table import compose

        return compose(cls, parts, name=name, ambient=ambient, stop=stop)

    @classmethod
    def from_table(cls, rows, **kwargs) -> "OpticalSystem":
        """A prescription table, read into placed elements (see :func:`.table.from_table`)."""

        from .table import from_table

        return from_table(cls, rows, **kwargs)

    def table(self) -> list:
        """The prescription rows of a coaxial path (see :func:`.table.to_table`)."""

        from .table import to_table

        return to_table(self)

    # -- what the system is made of ---------------------------------------

    def walk(self) -> Iterator[tuple[Element, Pose]]:
        """Every element with its world pose."""

        for element, pose, _ in Assembly("system", self.components).walk():
            yield element, pose

    @property
    def elements(self) -> tuple[Element, ...]:
        return tuple(e for e, _ in self.walk())

    def __getitem__(self, name: str) -> Element:
        for element, _ in self.walk():
            if element.name == name:
                return element
        raise KeyError(f"no element named {name!r}")

    def __contains__(self, name: str) -> bool:
        return any(e.name == name for e, _ in self.walk())

    def pose_of(self, name: str) -> Pose:
        """World pose of the element called *name*."""

        for element, pose in self.walk():
            if element.name == name:
                return pose
        raise KeyError(f"no element named {name!r}")

    # -- derived facts ----------------------------------------------------

    @property
    def faces(self) -> tuple[PhysicalFace, ...]:
        """Every physical boundary, with the media on its two sides."""

        if "faces" not in self._cache:
            self._cache["faces"] = resolve_faces(list(self.walk()), self.ambient)
        return self._cache["faces"]

    def face(self, name: str) -> PhysicalFace:
        for f in self.faces:
            if name in f.names:
                return f
        raise KeyError(f"no boundary named {name!r}")

    @property
    def path(self) -> tuple[Step, ...]:
        """The sequential route: stated, or inferred for a coaxial system."""

        if "path" not in self._cache:
            spec = self._path_spec
            self._cache["path"] = infer_path(self.faces) if spec is None else parse_path(spec, self.faces)
        return self._cache["path"]

    @property
    def has_stated_path(self) -> bool:
        return self._path_spec is not None

    @property
    def stop(self) -> str | None:
        """Name of the aperture stop."""

        if self._stop is not None:
            return self._stop
        stops = [e.name for e in self.elements if isinstance(e, Stop)]
        return stops[0] if len(stops) == 1 else None

    def step_index(self, name: str) -> int:
        """Position in :attr:`path` of the first visit to the boundary called *name*."""

        for i, step in enumerate(self.path):
            if name in step.face.names:
                return i
        raise KeyError(f"the path does not visit {name!r}")

    @property
    def stop_index(self) -> int | None:
        stop = self.stop
        return None if stop is None else self.step_index(stop)

    # -- values ---------------------------------------------------------------

    def replace(self, **changes: Any) -> "OpticalSystem":
        args = {
            "components": self.components,
            "name": self.name,
            "ambient": self.ambient,
            "path": self._path_spec,
            "stop": self._stop,
        }
        unknown = set(changes) - set(args)
        if unknown:
            raise TypeError(f"OpticalSystem has no argument(s) {sorted(unknown)}")
        args.update(changes)
        return OpticalSystem(args.pop("components"), **args)

    def with_component(self, name: str, component: Component) -> "OpticalSystem":
        """This system with the top-level or nested component *name* swapped for *component*."""

        def swap(items):
            out = []
            for c in items:
                if c.name == name:
                    out.append(component)
                elif isinstance(c, Assembly):
                    out.append(c.replace(components=swap(c.components)))
                else:
                    out.append(c)
            return tuple(out)

        if name not in self and not any(isinstance(c, Assembly) and c.name == name for c in self.components):
            raise KeyError(f"no component named {name!r}")
        return self.replace(components=swap(self.components))

    def translated(self, offset) -> "OpticalSystem":
        return self.replace(components=tuple(c.translated(offset) for c in self.components))

    def rotated(self, angle: float, axis, about=(0.0, 0.0, 0.0)) -> "OpticalSystem":
        """The whole system turned rigidly about the world line through *about* along *axis*."""

        return self.replace(components=tuple(c.rotated(angle, axis, about) for c in self.components))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OpticalSystem) and (
            self.components,
            self.ambient,
            self._path_spec,
            self._stop,
        ) == (other.components, other.ambient, other._path_spec, other._stop)

    def __hash__(self) -> int:
        return hash((self.components, self.ambient, self._path_spec, self._stop))

    def __repr__(self) -> str:
        label = f"{self.name!r}, " if self.name else ""
        return f"OpticalSystem({label}{len(self.elements)} elements, ambient={self.ambient.name})"

    def summary(self) -> str:
        """One line per boundary of the path: name, event, media, position."""

        lines = [repr(self)]
        for i, step in enumerate(self.path):
            f = step.face
            p = f.pose.position
            lines.append(
                f"{i:3d}  {'='.join(f.names):<22s} {step.event:<9s} "
                f"{f.medium_minus.name:>9s} | {f.medium_plus.name:<9s} at ({p[0]:.4g}, {p[1]:.4g}, {p[2]:.6g})"
            )
        return "\n".join(lines)


__all__ = ["OpticalSystem"]
