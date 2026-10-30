"""From placed elements to physical boundaries, and the route light takes through them.

Two facts follow from the elements of a system and are computed here, never
stated:

**Which medium lies on each side of each boundary.** Every element knows
on which side of each of its faces its own material lies. Where a face of
one element coincides with a face of another — same world pose, same
surface — the two are one physical boundary between two materials:
cemented lenses, an immersion liquid against a lens, a detector at the
back of a liquid layer. Everywhere else, the other side is the ambient
medium.

**The sequential path**: the ordered boundaries a sequential trace visits,
each with the event that happens there. It is stated by the designer
(``path=[...]``) or, for a coaxial system whose every boundary has one
obvious event, inferred by ordering the boundaries along the common axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..coatings import Coating, Uncoated
from ..elements import BLADE, DETECTOR, EDGE, Element, ElementFace, EdgeWall, Stop
from ..geometry.apertures import Aperture
from ..geometry.transforms import Pose
from ..interfaces import Interface
from ..materials import Material

#: Events of a sequential step.
TRANSMIT, REFLECT, STOP, DETECT = "transmit", "reflect", "stop", "detect"


@dataclass(frozen=True, eq=False)
class PhysicalFace:
    """One physical boundary, in world coordinates.

    ``medium_minus`` / ``medium_plus`` fill the local −z / +z sides of the
    surface. ``names`` lists every element face that coincides here (two
    for a cemented face), the first being the canonical one.
    """

    names: tuple[str, ...]
    interface: Interface | None
    pose: Pose
    extent: Aperture | None
    role: str
    medium_minus: Material
    medium_plus: Material
    elements: tuple[str, ...]
    hole: Aperture | None = None
    wall: EdgeWall | None = None
    virtual: bool = False

    @property
    def name(self) -> str:
        return self.names[0]

    @property
    def surface(self):
        return None if self.interface is None else self.interface.surface

    @property
    def coating(self) -> Coating | None:
        return None if self.interface is None else self.interface.coating

    @property
    def clear_aperture(self) -> Aperture | None:
        """Where light counts as passing: the clear aperture if given, else the physical extent."""

        if self.interface is not None and self.interface.clear_aperture is not None:
            return self.interface.clear_aperture
        return self.extent

    def __repr__(self) -> str:
        return f"PhysicalFace({'='.join(self.names)}, {self.medium_minus.name}|{self.medium_plus.name})"


@dataclass(frozen=True)
class Step:
    """One visit of the sequential path: a boundary and what happens there.

    ``clear_aperture``, when set, is the clear aperture of this visit alone
    — narrower than the boundary's own, as a prescription lists it for each
    pass through a double-pass lens.
    """

    face: PhysicalFace
    event: str
    clear_aperture: Aperture | None = None

    @property
    def name(self) -> str:
        return self.face.name

    def __repr__(self) -> str:
        return f"Step({self.face.name}:{self.event})"


def face_name(element: Element, face: ElementFace) -> str:
    return f"{element.name}.{face.name}" if face.name else element.name


def _merge_coatings(a: Coating, b: Coating, where: str) -> Coating:
    if isinstance(a, Uncoated):
        return b
    if isinstance(b, Uncoated) or a == b:
        return a
    raise ValueError(f"{where}: two coincident faces carry different coatings ({a!r} vs {b!r})")


def _larger(a: Aperture | None, b: Aperture | None) -> Aperture | None:
    if a is None or b is None:
        return None
    return a if a.extent >= b.extent else b


def resolve_faces(placed: list[tuple[Element, Pose]], ambient: Material) -> tuple[PhysicalFace, ...]:
    """Merge coincident element faces and assign the media on both sides."""

    groups: list[dict] = []
    for element, pose in placed:
        for face in element.faces():
            world = pose @ face.pose
            name = face_name(element, face)
            if face.role == EDGE:
                groups.append({"names": [name], "faces": [(element, face)], "pose": world})
                continue
            for g in groups:
                other_element, other = g["faces"][0]
                if (
                    other.role != EDGE
                    and g["pose"].isclose(world, atol=1e-9)
                    and other.interface.surface == face.interface.surface
                ):
                    g["names"].append(name)
                    g["faces"].append((element, face))
                    break
            else:
                groups.append({"names": [name], "faces": [(element, face)], "pose": world})

    faces = []
    for g in groups:
        (element0, face0), *rest = g["faces"]
        if face0.role == EDGE:
            m = element0.material or ambient
            faces.append(
                PhysicalFace((g["names"][0],), None, g["pose"], None, EDGE, m, ambient, (element0.name,), wall=face0.wall)
            )
            continue
        sides = {+1: None, -1: None}
        interface, extent, role, hole = face0.interface, face0.extent, face0.role, face0.hole
        virtual = isinstance(element0, Stop) and element0.virtual
        for element, face in g["faces"]:
            if face.bulk_side:
                if sides[face.bulk_side] is not None:
                    raise ValueError(
                        f"{'='.join(g['names'])}: two elements claim the same side of one boundary "
                        "(their bodies overlap)"
                    )
                sides[face.bulk_side] = element.material
            if (element, face) is not g["faces"][0]:
                interface = interface.replace(coating=_merge_coatings(interface.coating, face.interface.coating, g["names"][0]))
                extent = _larger(extent, face.extent)
                if face.role in (DETECTOR, BLADE):
                    role = face.role
        faces.append(
            PhysicalFace(
                tuple(g["names"]),
                interface,
                g["pose"],
                extent,
                role,
                sides[-1] or ambient,
                sides[+1] or ambient,
                tuple(e.name for e, _ in g["faces"]),
                hole=hole,
                virtual=virtual,
            )
        )
    return tuple(faces)


def default_event(face: PhysicalFace) -> str | None:
    """The event a sequential path takes at *face* when the path does not say."""

    if face.role == DETECTOR:
        return DETECT
    if face.role == BLADE:
        return STOP
    event = face.coating.sequential_event
    return None if event == "absorb" else event


def parse_path(spec, faces: tuple[PhysicalFace, ...]) -> tuple[Step, ...]:
    """Read ``["L1.front", "BS:reflect", ...]`` against the system's boundaries.

    An item may also be a pair ``("L3.front", aperture)`` giving the clear
    aperture of that visit (see :class:`Step`).
    """

    by_name = {n: f for f in faces for n in f.names}
    steps = []
    for item in spec:
        if isinstance(item, Step):
            steps.append(item)
            continue
        visit_aperture = None
        if isinstance(item, tuple):
            item, visit_aperture = item
        name, _, event = str(item).partition(":")
        if name not in by_name:
            raise KeyError(f"path step {item!r}: no boundary named {name!r}; known: {sorted(by_name)}")
        face = by_name[name]
        if face.role == EDGE:
            raise ValueError(f"path step {item!r}: a lens edge is not a sequential boundary")
        event = event or default_event(face)
        if event is None:
            raise ValueError(
                f"path step {item!r}: {face.coating!r} both reflects and transmits; "
                f"say which branch the path follows, e.g. '{name}:reflect'"
            )
        if event not in (TRANSMIT, REFLECT, STOP, DETECT):
            raise ValueError(f"path step {item!r}: unknown event {event!r}")
        steps.append(Step(face, event, visit_aperture))
    return tuple(steps)


def infer_path(faces: tuple[PhysicalFace, ...]) -> tuple[Step, ...]:
    """The path of a coaxial system with one obvious event per boundary."""

    optical = [f for f in faces if f.role != EDGE]
    if not optical:
        return ()
    axis = optical[0].pose.axis
    origin = optical[0].pose.position
    for f in optical:
        if not np.allclose(f.pose.axis, axis, atol=1e-12):
            raise ValueError(
                "the boundaries do not share one axis, so the sequential path cannot be inferred; "
                "state it with path=[...]"
            )
        offset = f.pose.position - origin
        if np.linalg.norm(offset - (offset @ axis) * axis) > 1e-9:
            raise ValueError(
                "the boundaries are not centred on one axis, so the sequential path cannot be inferred; "
                "state it with path=[...]"
            )
    steps = []
    for f in sorted(optical, key=lambda f: float((f.pose.position - origin) @ axis)):
        event = default_event(f)
        if event in (None, REFLECT):
            raise ValueError(
                f"{f.name} reflects, so the order of a sequential trace is not the order along the axis; "
                "state the path with path=[...]"
            )
        steps.append(Step(f, event))
    return tuple(steps)


__all__ = [
    "PhysicalFace",
    "Step",
    "resolve_faces",
    "parse_path",
    "infer_path",
    "default_event",
    "TRANSMIT",
    "REFLECT",
    "STOP",
    "DETECT",
]
