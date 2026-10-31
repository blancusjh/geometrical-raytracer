"""Concise ways to write a system: composition along an axis, and prescription tables.

Both build the same thing every other constructor builds — placed elements
in an ambient medium — so nothing downstream knows how a system was
written.

**Composition** (:func:`compose`) lists elements and :class:`Gap` s in the
order light meets them; each element is placed at the running position
on the +z axis.

**Prescription tables** (:func:`from_table`) are the designer's
surface-by-surface listing, as in patents and lens files: each row is a
surface, the signed gap to the next vertex, and the medium after it. The
reader recovers the physical elements the table describes:

- every run of non-ambient medium between two surfaces is a body of glass
  (or of liquid);
- bodies that share a surface are cemented into one
  :class:`~raytracer.elements.Lens` (an immersion layer against a lens
  reads the same way);
- a surface met twice (a double-pass lens in a catadioptric) is one
  physical surface: the second visit reuses the element;
- mirror rows become :class:`~raytracer.elements.Mirror` s, or a reflective
  coating on a lens face when they sit on glass (a Mangin mirror);
- stop rows become :class:`~raytracer.elements.Stop` s — *virtual* when the
  plane lies inside a lens, as patent tables sometimes put it;
- rows that change nothing (air to air, no stop) are dropped;
- the image plane becomes a :class:`~raytracer.elements.Sensor`.

The row order becomes the system's stated sequential path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..coatings import PerfectMirror
from ..elements import Element, Lens, Mirror, Sensor, Stop
from ..geometry.apertures import Circle
from ..geometry.surfaces import Plane, Surface
from ..geometry.transforms import Pose
from ..interfaces import Interface
from ..materials import AIR, Material

REFRACT, MIRROR, STOP = "refract", "mirror", "stop"


@dataclass(frozen=True)
class Gap:
    """A stretch of ambient medium in a composition, *distance* mm along the light."""

    distance: float


@dataclass(frozen=True)
class TableRow:
    """One row of a prescription table.

    ``gap`` is the signed axial distance to the next vertex (negative after
    an odd number of mirrors). ``medium`` follows the surface and is ignored
    on mirror rows. ``semi_diameter`` is the clear semi-diameter.
    """

    surface: Surface | Interface
    gap: float
    medium: Material | None = None
    semi_diameter: float | None = None
    kind: str = REFRACT
    name: str | None = None


# -- composition ---------------------------------------------------------------


def compose(cls, parts, *, name="", ambient=AIR, stop=None):
    z = 0.0
    direction = 1.0
    placed: list[Element] = []
    path: list[str] = []
    last_lens: Lens | None = None
    for part in parts:
        if isinstance(part, Gap):
            z += float(part.distance)
            last_lens = None
            continue
        if not isinstance(part, Element):
            raise TypeError(f"compose takes elements and Gaps, got {part!r}")
        if not part.pose.is_identity():
            raise ValueError(f"compose places {part.name!r} itself; give it no pose")
        if direction < 0.0 and isinstance(part, Lens):
            raise ValueError(
                f"{part.name!r}: compose places lenses only before the first mirror; "
                "after it, give elements explicit poses or use a prescription table"
            )
        element = part.placed(Pose((0.0, 0.0, z)))
        placed.append(element)
        if isinstance(element, Lens):
            faces = [f"{element.name}.{k}" for k in range(1, len(element.surfaces) + 1)]
            if last_lens is not None:
                if last_lens.surfaces[-1].surface != element.surfaces[0].surface:
                    raise ValueError(
                        f"{last_lens.name!r} and {element.name!r} touch (no Gap between them), so they must "
                        "share the contact surface; glass cemented together is one Lens"
                    )
                faces = faces[1:]  # the contact face is on the path already
            path += faces
            z += element.thickness
            last_lens = element
        else:
            last_lens = None
            if isinstance(element, Mirror):
                path.append(f"{element.name}:reflect")
                direction = -direction
            else:
                path.append(element.name)
    return cls(placed, name=name, ambient=ambient, path=path, stop=stop)


# -- prescription tables ------------------------------------------------------


@dataclass(eq=False)
class _Surface:
    """A physical surface the table meets, once however many rows visit it."""

    z: float
    interface: Interface
    kind: str
    semi_diameter: float
    name: str | None = None
    on_glass: bool = False  # a mirror row with glass in front of it (a Mangin mirror)

    def visited_by(self, row: TableRow, z: float) -> bool:
        return self.kind == row.kind and abs(self.z - z) < 1e-9 and self.interface.surface == _surface(row.surface)

    def as_face(self) -> Interface:
        """The interface an element carries here: the row's clear aperture, a mirror coating on glass."""

        interface = self.interface
        if self.on_glass:
            interface = interface.replace(coating=PerfectMirror())
        if self.semi_diameter > 0.0 and interface.clear_aperture is None:
            interface = interface.replace(clear_aperture=Circle(self.semi_diameter))
        return interface


def _surface(spec: Surface | Interface) -> Surface:
    return spec.surface if isinstance(spec, Interface) else spec


def _read_surfaces(rows: list[TableRow], ambient: Material) -> tuple[list, float]:
    """Walk the table: each row's vertex and media, folded into physical surfaces.

    Returns the visits ``(row, surface, medium after)`` in row order — rows
    that change nothing (ambient on both sides, not a stop) leave no visit —
    and the axial position of the image plane.
    """

    visits, surfaces = [], []
    z, medium = 0.0, ambient
    for row in rows:
        before = medium
        if row.kind != MIRROR and row.medium is not None:
            medium = row.medium
        if not (row.kind == REFRACT and before == medium == ambient):
            surface = next((s for s in surfaces if s.visited_by(row, z)), None)
            if surface is None:
                surface = _Surface(z, Interface.coerce(row.surface), row.kind, 0.0, row.name, row.kind == MIRROR and before != ambient)
                surfaces.append(surface)
            surface.semi_diameter = max(surface.semi_diameter, row.semi_diameter or 0.0)
            visits.append((row, surface, medium))
        z += float(row.gap)
    return visits, z


def _bodies(visits: list, image: _Surface, ambient: Material) -> list[tuple[_Surface, _Surface, Material]]:
    """Each run of non-ambient medium between two surfaces, once, in order of first visit."""

    chain = [(s, m) for _, s, m in visits if s.kind != STOP] + [(image, None)]
    bodies: dict[frozenset, tuple] = {}
    for (a, medium), (b, _) in zip(chain, chain[1:]):
        if medium == ambient or a is b:
            continue
        front, back = (a, b) if a.z < b.z else (b, a)
        known = bodies.setdefault(frozenset((id(a), id(b))), (front, back, medium))
        if known[2] != medium:
            raise ValueError(f"the body between z = {front.z} and z = {back.z} is read with two materials")
    return list(bodies.values())


def _cemented(bodies: list) -> list[list]:
    """The bodies grouped into lenses: a body whose back surface is another's front is cemented to it."""

    behind = {id(front): (front, back, material) for front, back, material in bodies}
    backs = {id(back) for _, back, _ in bodies}
    lenses = []
    for body in bodies:
        if id(body[0]) in backs:
            continue  # cemented behind another body: its lens starts there
        lens = [body]
        while id(lens[-1][1]) in behind:
            lens.append(behind[id(lens[-1][1])])
        lenses.append(lens)
    return lenses


def _lens(name: str, bodies: list) -> Lens:
    """A lens of cemented bodies, as wide as its widest face; a table that gives no width falls back on what its surfaces allow."""

    surfaces = [bodies[0][0]] + [back for _, back, _ in bodies]
    semi = max(s.semi_diameter for s in surfaces)
    if semi <= 0.0:
        semi = min(s.interface.surface.max_radius for s in surfaces)
        if not np.isfinite(semi):
            raise ValueError(f"{name}: the table gives no semi-diameter and its surfaces do not bound one")
    return Lens(
        name,
        surfaces=[s.as_face() for s in surfaces],
        thicknesses=[back.z - front.z for front, back, _ in bodies],
        materials=[material for _, _, material in bodies],
        semi_diameter=semi,
        z=surfaces[0].z,
    )


def _cuts(stop: Stop, lens: Lens) -> bool:
    """Whether the stop's plane, out to the blade's edge, passes through the lens body.

    A table may place its stop plane where no diaphragm can be: inside a lens,
    or at a curved vertex with glass beyond it. Such a stop is a reference
    plane of the design, and is read as virtual.
    """

    z = stop.pose.position[2] - lens.pose.position[2]
    r = np.linspace(0.0, min(stop.outer_radius, lens.semi_diameter), 129)
    zero = np.zeros_like(r)
    front = lens.surfaces[0].surface.sag(r, zero)
    back = lens.thickness + lens.surfaces[-1].surface.sag(r, zero)
    return bool(np.any((front + 1e-9 < z) & (z < back - 1e-9)))


def from_table(cls, rows, *, name: str = "", ambient: Material = AIR, image_surface: Surface | None = None,
               image_name: str = "IMG", stop: str | None = None):
    """Read prescription rows into placed elements; the row order becomes the stated path.

    The table starts and ends in the ambient medium (an immersion layer is a
    row of its own: a liquid lens in contact with its neighbours).
    """

    rows = [r if isinstance(r, TableRow) else TableRow(*r) for r in rows]
    if not rows:
        raise ValueError("a prescription has at least one row")
    visits, image_z = _read_surfaces(rows, ambient)
    image = _Surface(image_z, Interface.coerce(image_surface or Plane()), "image", 0.0, image_name)

    elements: list[Element] = []
    names: dict[int, str] = {}
    for n, bodies in enumerate(_cemented(_bodies(visits, image, ambient)), 1):
        elements.append(_lens(f"L{n}", bodies))
        for k, surface in enumerate([bodies[0][0]] + [back for _, back, _ in bodies], 1):
            names.setdefault(id(surface), f"L{n}.{k}")

    mirrors = 0
    for surface in {id(s): s for _, s, _ in visits}.values():
        if surface.kind == MIRROR and not surface.on_glass:
            mirrors += 1
            names[id(surface)] = surface.name or f"M{mirrors}"
            aperture = Circle(surface.semi_diameter) if surface.semi_diameter > 0.0 else None
            elements.append(Mirror(names[id(surface)], surface=surface.interface, aperture=aperture, z=surface.z))
        elif surface.kind == STOP:
            names[id(surface)] = surface.name or "STO"
            if surface.semi_diameter <= 0.0:
                raise ValueError(f"the stop at z = {surface.z} has no semi-diameter: a stop's opening is its definition")
            stop_element = Stop(names[id(surface)], aperture=Circle(surface.semi_diameter), z=surface.z)
            if any(_cuts(stop_element, lens) for lens in elements if isinstance(lens, Lens)):
                stop_element = stop_element.replace(virtual=True)
            elements.append(stop_element)
        elif id(surface) not in names:
            raise ValueError(f"the surface at z = {surface.z} bounds no body and is neither a mirror nor a stop")
    elements.append(Sensor(image_name, surface=image_surface, z=image_z))
    names[id(image)] = image_name

    path: list[str] = []
    for row, surface, _ in visits:
        step = names[id(surface)] + (":reflect" if row.kind == MIRROR else "")
        if not path or path[-1] != step:
            path.append(step)
    path.append(image_name)
    return cls(elements, name=name, ambient=ambient, path=path, stop=stop)


def to_table(system) -> list[TableRow]:
    """The prescription rows of a coaxial path — the inverse of :func:`from_table`.

    One row per step before the detector, the gap measured to the next
    step's vertex along z (negative after an odd number of reflections).
    """

    from .faces import DETECT, REFLECT

    steps = system.path
    rows = []
    sign = 1.0
    for step, following in zip(steps, steps[1:]):
        face = step.face
        pose = face.pose
        if not (np.allclose(pose.axis, (0.0, 0.0, 1.0), atol=1e-12) and np.allclose(pose.position[:2], 0.0)):
            raise ValueError(f"{step.name} is off the z axis; only a coaxial path is a prescription table")
        arriving = face.medium_minus if sign > 0 else face.medium_plus
        if step.event == REFLECT:
            kind, medium = MIRROR, None
            sign = -sign
        elif step.event == "stop":
            kind, medium = STOP, arriving
        else:
            kind, medium = REFRACT, face.medium_plus if sign > 0 else face.medium_minus
        region = face.interface.clear_aperture if step.event == "stop" else face.clear_aperture
        semi = None if region is None else region.extent
        gap = float(following.face.pose.position[2] - pose.position[2])
        rows.append(TableRow(face.interface, gap, medium, semi, kind, step.name))
        if following.event == DETECT:
            break
    return rows


__all__ = ["Gap", "TableRow", "compose", "from_table", "to_table"]
