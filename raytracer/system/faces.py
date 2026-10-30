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
from ..elements import BLADE, DETECTOR, EDGE, Element, ElementFace, Stop
from ..geometry.apertures import Aperture
from ..geometry.solids import Cylinder
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
    wall: Cylinder | None = None
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
    """One visit of the sequential path: a boundary and what happens there."""

    face: PhysicalFace
    event: str

    @property
    def name(self) -> str:
        return self.face.name

    def __repr__(self) -> str:
        return f"Step({self.face.name}:{self.event})"


def face_name(element: Element, face: ElementFace) -> str:
    return f"{element.name}.{face.name}" if face.name else element.name


def _coincide(a: tuple[Element, ElementFace, Pose], b: tuple[Element, ElementFace, Pose]) -> bool:
    """Two element faces are one physical boundary: same place, same surface."""

    (_, fa, pa), (_, fb, pb) = a, b
    return EDGE not in (fa.role, fb.role) and pa.isclose(pb, atol=1e-9) and fa.interface.surface == fb.interface.surface


def _canonical_order(member: tuple[Element, ElementFace, Pose]) -> tuple:
    """A merged boundary is named after the sensor or stop in it, else alphabetically."""

    element, face, _ = member
    return (face.role not in (DETECTOR, BLADE), face_name(element, face))


def _merge_coatings(a: Coating, b: Coating, where: str) -> Coating:
    if isinstance(a, Uncoated):
        return b
    if isinstance(b, Uncoated) or a == b:
        return a
    raise ValueError(f"{where}: two coincident faces carry different coatings ({a!r} vs {b!r})")


def _physical_face(group: list[tuple[Element, ElementFace, Pose]], ambient: Material) -> PhysicalFace:
    group = sorted(group, key=_canonical_order)
    names = tuple(face_name(e, f) for e, f, _ in group)
    elements = tuple(e.name for e, _, _ in group)
    element, face, pose = group[0]
    if face.role == EDGE:
        return PhysicalFace(names, None, pose, None, EDGE, element.material or ambient, ambient, elements, wall=face.wall)

    sides: dict[int, Material] = {}
    interface, extent = face.interface, face.extent
    for e, f, _ in group:
        if f.bulk_side:
            if f.bulk_side in sides:
                raise ValueError(f"{'='.join(names)}: two elements claim the same side of one boundary (their bodies overlap)")
            sides[f.bulk_side] = e.material
        if f is not face:
            interface = interface.replace(coating=_merge_coatings(interface.coating, f.interface.coating, names[0]))
            if extent is not None and (f.extent is None or f.extent.extent > extent.extent):
                extent = f.extent
    return PhysicalFace(
        names,
        interface,
        pose,
        extent,
        face.role,
        sides.get(-1, ambient),
        sides.get(+1, ambient),
        elements,
        hole=face.hole,
        virtual=isinstance(element, Stop) and element.virtual,
    )


def resolve_faces(placed: list[tuple[Element, Pose]], ambient: Material) -> tuple[PhysicalFace, ...]:
    """Merge coincident element faces into physical boundaries and assign the media on both sides."""

    groups: list[list[tuple[Element, ElementFace, Pose]]] = []
    for element, pose in placed:
        for face in element.faces():
            member = (element, face, pose @ face.pose)
            for group in groups:
                if _coincide(group[0], member):
                    group.append(member)
                    break
            else:
                groups.append([member])
    return tuple(_physical_face(group, ambient) for group in groups)


def default_event(face: PhysicalFace) -> str | None:
    """The event a sequential path takes at *face* when the path does not say."""

    if face.role == DETECTOR:
        return DETECT
    if face.role == BLADE:
        return STOP
    event = face.coating.sequential_event
    return None if event == "absorb" else event


def parse_path(spec, faces: tuple[PhysicalFace, ...]) -> tuple[Step, ...]:
    """Read ``["L1.front", "BS:reflect", ...]`` against the system's boundaries."""

    by_name = {n: f for f in faces for n in f.names}
    steps = []
    for item in spec:
        if isinstance(item, Step):
            steps.append(item)
            continue
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
        steps.append(Step(face, event))
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
