"""Optical elements: the physical things an optical system is made of.

An element is a named, placed object: geometry (its surfaces, or a solid)
plus what it is made of (its materials) plus its optical interfaces. It
knows its own **pose** — position and orientation — but nothing about the
other elements around it: which medium lies outside it, whether it touches
a neighbour, and in which order light visits it are facts of the system,
resolved there.

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

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..coatings import Absorbing, Coating, IdealBeamSplitter, PerfectMirror, Uncoated
from ..geometry.apertures import Aperture, Circle
from ..geometry.solids import Cylinder, Polyhedron
from ..geometry.surfaces import Plane, Surface
from ..geometry.transforms import Pose
from ..interfaces import Interface
from ..materials import Material

#: What a face is for, as far as tracing is concerned.
OPTICAL, EDGE, BLADE, DETECTOR = "optical", "edge", "blade", "detector"


@dataclass(frozen=True)
class ElementFace:
    """One boundary of an element, in the element's own frame.

    ``media`` are what the element itself fills on the face's local −z and
    +z sides (None: outside it, for the system to resolve). ``extent``
    bounds the physical patch of surface; ``hole`` removes a region from it
    (the opening of a diaphragm). A lens rim is the curved ``wall`` of a
    cylinder, placed by ``pose``, instead of an interface; its inside is
    ``media[0]``.
    """

    name: str
    interface: Interface | None
    pose: Pose
    media: tuple[Material | None, Material | None]
    extent: Aperture | None
    role: str = OPTICAL
    hole: Aperture | None = None
    wall: Cylinder | None = None


class Element(ABC):
    """A named, placed physical component."""

    name: str
    pose: Pose

    def _place(self, name: str, placement: dict[str, Any]) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("every element has a non-empty name")
        if "." in name or ":" in name:
            raise ValueError(f"element names cannot contain '.' or ':' (got {name!r})")
        self._set(name=name, pose=Pose.from_spec(**placement))

    @abstractmethod
    def faces(self) -> list[ElementFace]:
        """The element's boundaries, in its own frame."""

    def _set(self, **attributes: Any) -> None:
        """Store the constructor's arguments, each under its own name."""

        for key, value in attributes.items():
            object.__setattr__(self, key, value)

    def _arguments(self) -> dict[str, Any]:
        """The constructor arguments that rebuild this element (name and placement excluded)."""

        names = [
            p.name
            for p in inspect.signature(type(self).__init__).parameters.values()
            if p.kind is p.KEYWORD_ONLY
        ]
        return {n: getattr(self, n) for n in names}

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
    """Glass between optical surfaces: one body, or several cemented into one.

    The ``surfaces`` follow one another along the element's axis, the first
    at its origin; ``thicknesses[k]`` is the centre thickness of the body
    between surfaces ``k`` and ``k + 1``, made of ``materials[k]``. A singlet
    has two surfaces, one thickness and one material; a cemented doublet
    three, two and two. Each surface is a
    :class:`~raytracer.geometry.surfaces.Surface` or an
    :class:`~raytracer.interfaces.Interface` (to give it a coating or a clear
    aperture of its own). ``semi_diameter`` is the physical radius of the
    whole, closed by a cylindrical rim that absorbs (a ground edge).

    Its faces are numbered from 1 in the order of the surfaces (``L1.1``,
    ``L1.2``, …); the rim of body ``k`` is ``edge{k}``.
    """

    def __init__(
        self,
        name: str,
        *,
        surfaces: list[Interface | Surface],
        thicknesses: list[float],
        materials: list[Material],
        semi_diameter: float,
        **placement,
    ) -> None:
        surfaces = tuple(Interface.coerce(s) for s in surfaces)
        thicknesses, materials = tuple(float(t) for t in thicknesses), tuple(materials)
        if not len(surfaces) - 1 == len(thicknesses) == len(materials) >= 1:
            raise ValueError(
                f"lens {name!r}: n + 1 surfaces bound n bodies, each with a thickness and a material "
                f"(got {len(surfaces)} surfaces, {len(thicknesses)} thicknesses, {len(materials)} materials)"
            )
        if not all(t > 0.0 for t in thicknesses):
            raise ValueError(f"lens {name!r}: centre thicknesses are positive, got {thicknesses}")
        self._set(surfaces=surfaces, thicknesses=thicknesses, materials=materials, semi_diameter=_semi_diameter(semi_diameter))
        self._place(name, placement)

    @property
    def vertices(self) -> np.ndarray:
        """Axial position of each surface's vertex."""

        return np.concatenate([[0.0], np.cumsum(self.thicknesses)])

    @property
    def thickness(self) -> float:
        """Centre thickness of the whole lens."""

        return float(self.vertices[-1])

    def edge_heights(self) -> np.ndarray:
        """Axial position where each surface meets the rim."""

        h = np.array([self.semi_diameter])
        return np.array([z + float(s.surface.sag(h, np.zeros(1))[0]) for z, s in zip(self.vertices, self.surfaces)])

    @property
    def edge_thickness(self) -> float:
        edges = self.edge_heights()
        return float(edges[-1] - edges[0])

    def faces(self):
        sd = Circle(self.semi_diameter)
        around = (None, *self.materials, None)  # what lies on either side of each surface, within the lens
        faces = [
            ElementFace(str(k + 1), surface, Pose((0.0, 0.0, z)), (around[k], around[k + 1]), sd)
            for k, (surface, z) in enumerate(zip(self.surfaces, self.vertices))
        ]
        edges = self.edge_heights()
        for k, material in enumerate(self.materials):  # a body's rim, absent where its faces meet before the edge
            if edges[k + 1] > edges[k]:
                wall = Cylinder(self.semi_diameter, edges[k + 1] - edges[k])
                faces.append(ElementFace(f"edge{k + 1}", None, Pose((0.0, 0.0, edges[k])), (material, None), None, role=EDGE, wall=wall))
        return faces


def _one_face(surface: Surface | Interface, coating: Coating | None, clear_aperture, default: Coating):
    """The (surface, coating, clear aperture) of a single-face element, an interface unpacked."""

    if isinstance(surface, Interface):
        if coating is not None or clear_aperture is not None:
            raise TypeError("give coating and clear_aperture either in the Interface or as arguments, not both")
        surface, coating, clear_aperture = surface.surface, surface.coating, surface.clear_aperture
        coating = None if isinstance(coating, Uncoated) else coating
    return surface, coating if coating is not None else default, Aperture.coerce(clear_aperture)


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
        clear_aperture: Aperture | float | None = None,
        **placement,
    ) -> None:
        surface, coating, clear_aperture = _one_face(surface, coating, clear_aperture, PerfectMirror())
        self._set(surface=surface, aperture=Aperture.coerce(aperture), coating=coating, clear_aperture=clear_aperture)
        self._place(name, placement)

    def faces(self):
        interface = Interface(self.surface, coating=self.coating, clear_aperture=self.clear_aperture)
        return [ElementFace("", interface, Pose(), (None, None), self.aperture)]


class BeamSplitter(Element):
    """A thin partially reflecting surface (an :class:`~raytracer.coatings.IdealBeamSplitter` by default)."""

    def __init__(
        self,
        name: str,
        *,
        aperture: Aperture | float,
        surface: Surface | Interface | None = None,
        coating: Coating | None = None,
        clear_aperture: Aperture | float | None = None,
        **placement,
    ) -> None:
        surface, coating, clear_aperture = _one_face(surface if surface is not None else Plane(), coating, clear_aperture, IdealBeamSplitter())
        self._set(surface=surface, aperture=Aperture.coerce(aperture), coating=coating, clear_aperture=clear_aperture)
        self._place(name, placement)

    def faces(self):
        interface = Interface(self.surface, coating=self.coating, clear_aperture=self.clear_aperture)
        return [ElementFace("", interface, Pose(), (None, None), self.aperture)]


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
        self._set(solid=solid, material=material, coatings=dict(coatings or {}))
        self._place(name, placement)



    def faces(self):
        faces = []
        for i, (pose, outline) in enumerate(self.solid.facets()):
            interface = Interface(Plane(), coating=self.coatings.get(i))
            faces.append(ElementFace(f"facet{i}", interface, pose, (self.material, None), outline))
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
        outer_radius = float(outer_radius) if outer_radius is not None else 2.0 * opening.extent
        self._set(aperture=opening, outer_radius=outer_radius, virtual=bool(virtual))
        self._place(name, placement)


    def faces(self):
        blade = Interface(Plane(), coating=Absorbing(), clear_aperture=self.aperture)
        return [ElementFace("", blade, Pose(), (None, None), Circle(self.outer_radius), role=BLADE, hole=self.aperture)]


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
        self._set(surface=surface if surface is not None else Plane(), aperture=Aperture.coerce(aperture))
        self._place(name, placement)


    def faces(self):
        return [ElementFace("", Interface(self.surface, coating=Absorbing()), Pose(), (None, None), self.aperture, role=DETECTOR)]


__all__ = [
    "Element",
    "ElementFace",
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
