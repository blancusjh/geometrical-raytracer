"""Optical elements: the physical things an optical system is made of.

An element is a named, placed object: geometry (its surfaces, or a solid)
plus what it is made of (its material) plus its optical interfaces. It
knows its own **pose** — position and orientation — but nothing about the
other elements around it: which medium lies outside it, whether it is
cemented to a neighbour, and in which order light visits it are facts of
the system, resolved there.

Every element accepts the same placement keywords (see
:meth:`raytracer.geometry.Pose.from_spec`):

- ``pos=(x, y, z)`` or ``z=`` for the position of its local origin;
- ``rot=`` (Euler degrees ``(x, y, z)``, ``(angle, axis)``, a matrix or a
  :class:`~raytracer.geometry.Rotation`), or ``axis=`` / ``normal=`` (where
  its local +z points), for its orientation;
- or a complete ``pose=``.

The local origin of an element is the vertex of its first face, and its
local +z axis runs from that face into the element: for a lens, from the
front vertex to the back vertex.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..coatings import Absorbing, Coating, IdealBeamSplitter, PerfectMirror
from ..geometry.apertures import Aperture, Circle
from ..geometry.solids import Polyhedron
from ..geometry.surfaces import Plane, Surface
from ..geometry.transforms import Pose
from ..interfaces import Interface
from ..materials import Material

#: What a face is for, as far as tracing is concerned.
OPTICAL, EDGE, BLADE, DETECTOR = "optical", "edge", "blade", "detector"


@dataclass(frozen=True)
class EdgeWall:
    """The cylindrical rim of a lens, between its two faces at ``radius``.

    ``z_front`` and ``z_back`` are the axial positions (element frame) where
    the rim meets the front and back faces.
    """

    radius: float
    z_front: float
    z_back: float


@dataclass(frozen=True)
class ElementFace:
    """One boundary of an element, in the element's own frame.

    ``bulk_side`` is +1 when the element's material lies on the face's local
    +z side, −1 on its −z side, 0 when the element has no bulk. ``extent``
    bounds the physical patch of surface; ``hole`` removes a region from it
    (the opening of a diaphragm).
    """

    name: str
    interface: Interface | None
    pose: Pose
    bulk_side: int
    extent: Aperture | None
    role: str = OPTICAL
    hole: Aperture | None = None
    wall: EdgeWall | None = None


class Element(ABC):
    """A named, placed physical component."""

    name: str
    pose: Pose
    material: Material | None = None

    def _place(self, name: str, placement: dict[str, Any]) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("every element has a non-empty name")
        if "." in name or ":" in name:
            raise ValueError(f"element names cannot contain '.' or ':' (got {name!r})")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "pose", Pose.from_spec(**placement))

    @abstractmethod
    def faces(self) -> list[ElementFace]:
        """The element's boundaries, in its own frame."""

    @abstractmethod
    def _arguments(self) -> dict[str, Any]:
        """The constructor arguments that rebuild this element (pose excluded)."""

    # -- values ---------------------------------------------------------

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__} is immutable; use .replace(...)")

    def replace(self, **changes) -> "Element":
        """A copy with some arguments changed; ``pose=`` (or pos/rot/…) re-places it."""

        args = self._arguments()
        placement = {k: changes.pop(k) for k in ("pose", "pos", "z", "rot", "axis", "normal") if k in changes}
        unknown = set(changes) - set(args) - {"name"}
        if unknown:
            raise TypeError(f"{type(self).__name__} has no argument(s) {sorted(unknown)}")
        args.update(changes)
        name = args.pop("name", self.name)
        return type(self)(name, **args, **(placement or {"pose": self.pose}))

    def placed(self, pose: Pose) -> "Element":
        return self.replace(pose=pose)

    def translated(self, offset) -> "Element":
        return self.placed(self.pose.translated(offset))

    def rotated(self, angle: float, axis, about=(0.0, 0.0, 0.0)) -> "Element":
        """This element turned rigidly about the world line through *about* along *axis*."""

        return self.placed(self.pose.rotated(angle, axis, about))

    def _key(self) -> tuple:
        return (type(self).__name__, self.name, self.pose, tuple(sorted(self._arguments().items(), key=lambda kv: kv[0])))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Element) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(repr(self._key()))

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self._arguments().items())
        pose = "" if self.pose.is_identity() else f", pose={self.pose!r}"
        return f"{type(self).__name__}({self.name!r}, {args}{pose})"


def _semi_diameter(value: float) -> float:
    value = float(value)
    if not value > 0.0:
        raise ValueError("a semi-diameter is positive")
    return value


class Lens(Element):
    """A body of one material between two optical surfaces.

    ``front`` sits at the element's origin and ``back`` at ``z = thickness``
    on its axis; each is a :class:`~raytracer.geometry.surfaces.Surface` or
    an :class:`~raytracer.interfaces.Interface` (to give it a coating or a
    clear aperture of its own). ``semi_diameter`` is the physical radius of
    the body, closed by a cylindrical rim that absorbs (a ground edge).

    A cemented doublet is two lenses in contact; an immersion layer is a
    lens of liquid in contact with its neighbours. The system recognises
    contact where a face of one element coincides with a face of another.
    """

    def __init__(
        self,
        name: str,
        *,
        front: Interface | Surface,
        back: Interface | Surface,
        thickness: float,
        material: Material,
        semi_diameter: float,
        **placement,
    ) -> None:
        object.__setattr__(self, "front", Interface.coerce(front))
        object.__setattr__(self, "back", Interface.coerce(back))
        if not float(thickness) > 0.0:
            raise ValueError(f"lens {name!r}: the centre thickness is positive, got {thickness}")
        object.__setattr__(self, "thickness", float(thickness))
        object.__setattr__(self, "material", material)
        object.__setattr__(self, "semi_diameter", _semi_diameter(semi_diameter))
        self._place(name, placement)

    def _arguments(self):
        return {
            "front": self.front,
            "back": self.back,
            "thickness": self.thickness,
            "material": self.material,
            "semi_diameter": self.semi_diameter,
        }

    def edge_heights(self) -> tuple[float, float]:
        """Axial positions where the rim meets the front and back faces."""

        h = np.array([self.semi_diameter])
        zf = float(self.front.surface.sag(h, np.zeros(1))[0])
        zb = float(self.thickness + self.back.surface.sag(h, np.zeros(1))[0])
        return zf, zb

    @property
    def edge_thickness(self) -> float:
        zf, zb = self.edge_heights()
        return zb - zf

    def faces(self):
        sd = Circle(self.semi_diameter)
        zf, zb = self.edge_heights()
        return [
            ElementFace("front", self.front, Pose(), +1, sd),
            ElementFace("back", self.back, Pose((0.0, 0.0, self.thickness)), -1, sd),
            ElementFace("edge", None, Pose(), 0, None, role=EDGE, wall=EdgeWall(self.semi_diameter, zf, zb)),
        ]


class Mirror(Element):
    """A single reflecting surface (coated :class:`~raytracer.coatings.PerfectMirror` unless told otherwise).

    ``aperture=None`` is an unbounded mirror: the whole surface reflects.
    """

    def __init__(
        self,
        name: str,
        *,
        surface: Surface | Interface,
        aperture: Aperture | float | None,
        coating: Coating | None = None,
        **placement,
    ) -> None:
        interface = Interface.coerce(surface)
        if coating is not None:
            interface = interface.replace(coating=coating)
        elif type(interface.coating).__name__ == "Uncoated":
            interface = interface.replace(coating=PerfectMirror())
        object.__setattr__(self, "interface", interface)
        object.__setattr__(self, "aperture", Aperture.coerce(aperture))
        self._place(name, placement)

    @property
    def surface(self) -> Surface:
        return self.interface.surface

    def _arguments(self):
        return {"surface": self.interface, "aperture": self.aperture}

    def faces(self):
        return [ElementFace("", self.interface, Pose(), 0, self.aperture)]


class BeamSplitter(Element):
    """A thin partially reflecting surface (an :class:`~raytracer.coatings.IdealBeamSplitter` by default)."""

    def __init__(
        self,
        name: str,
        *,
        aperture: Aperture | float,
        surface: Surface | Interface | None = None,
        coating: Coating | None = None,
        **placement,
    ) -> None:
        interface = Interface.coerce(surface if surface is not None else Plane())
        if coating is not None or type(interface.coating).__name__ == "Uncoated":
            interface = interface.replace(coating=coating if coating is not None else IdealBeamSplitter())
        object.__setattr__(self, "interface", interface)
        object.__setattr__(self, "aperture", Aperture.coerce(aperture))
        self._place(name, placement)

    def _arguments(self):
        return {"surface": self.interface, "aperture": self.aperture}

    def faces(self):
        return [ElementFace("", self.interface, Pose(), 0, self.aperture)]


class Prism(Element):
    """A polyhedral body of one material; every facet is an optical interface.

    ``coatings`` maps facet indices (in the order of
    :meth:`~raytracer.geometry.solids.Polyhedron.facets`) to coatings;
    facets not listed are uncoated.
    """

    def __init__(
        self,
        name: str,
        *,
        solid: Polyhedron,
        material: Material,
        coatings: dict[int, Coating] | None = None,
        **placement,
    ) -> None:
        if not isinstance(solid, Polyhedron):
            raise TypeError("a prism is a polyhedral solid (Box or Mesh)")
        object.__setattr__(self, "solid", solid)
        object.__setattr__(self, "material", material)
        object.__setattr__(self, "coatings", dict(coatings or {}))
        self._place(name, placement)

    def _arguments(self):
        return {"solid": self.solid, "material": self.material, "coatings": self.coatings}

    def _key(self):
        return (type(self).__name__, self.name, self.pose, self.solid, self.material, tuple(sorted(self.coatings.items())))

    def faces(self):
        faces = []
        for i, (pose, outline) in enumerate(self.solid.facets()):
            interface = Interface(Plane(), coating=self.coatings.get(i))
            faces.append(ElementFace(f"facet{i}", interface, pose, -1, outline))
        return faces


class Stop(Element):
    """A diaphragm: an opaque plane with an opening.

    ``aperture`` is the opening (a radius, or any aperture shape);
    ``outer_radius`` the outer edge of the blade (default: twice the
    opening). A **virtual** stop is a reference plane only — it defines
    the aperture stop of a sequential design (as patent prescriptions do,
    sometimes inside a lens) but is not hardware, so non-sequential light
    passes it untouched.
    """

    def __init__(
        self,
        name: str,
        *,
        aperture: Aperture | float,
        outer_radius: float | None = None,
        virtual: bool = False,
        **placement,
    ) -> None:
        opening = Aperture.coerce(aperture)
        object.__setattr__(self, "aperture", opening)
        object.__setattr__(self, "outer_radius", float(outer_radius) if outer_radius is not None else 2.0 * opening.extent)
        object.__setattr__(self, "virtual", bool(virtual))
        self._place(name, placement)

    def _arguments(self):
        return {"aperture": self.aperture, "outer_radius": self.outer_radius, "virtual": self.virtual}

    def faces(self):
        blade = Interface(Plane(), coating=Absorbing(), clear_aperture=self.aperture)
        return [ElementFace("", blade, Pose(), 0, Circle(self.outer_radius), role=BLADE, hole=self.aperture)]


class Sensor(Element):
    """A detector surface: it absorbs and records the light reaching it."""

    def __init__(
        self,
        name: str,
        *,
        surface: Surface | None = None,
        aperture: Aperture | float | None = None,
        **placement,
    ) -> None:
        object.__setattr__(self, "surface", surface if surface is not None else Plane())
        object.__setattr__(self, "aperture", Aperture.coerce(aperture))
        self._place(name, placement)

    def _arguments(self):
        return {"surface": self.surface, "aperture": self.aperture}

    def faces(self):
        return [ElementFace("", Interface(self.surface, coating=Absorbing()), Pose(), 0, self.aperture, role=DETECTOR)]


__all__ = [
    "Element",
    "ElementFace",
    "EdgeWall",
    "Lens",
    "Mirror",
    "BeamSplitter",
    "Prism",
    "Stop",
    "Sensor",
    "OPTICAL",
    "EDGE",
    "BLADE",
    "DETECTOR",
]
