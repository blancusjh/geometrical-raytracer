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

- every run of non-ambient medium between two surfaces becomes a
  :class:`~raytracer.elements.Lens` (a liquid layer included);
- a surface met twice (a double-pass lens in a catadioptric) is one
  physical surface: the second visit reuses the element;
- consecutive bodies sharing a surface are in contact (cemented);
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
            if last_lens is not None:
                if last_lens.back.surface != element.front.surface:
                    raise ValueError(
                        f"{last_lens.name!r} and {element.name!r} touch (no Gap between them), "
                        "so they must share the contact surface"
                    )
            else:
                path.append(f"{element.name}.front")
            path.append(f"{element.name}.back")
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


def _surface(spec: Surface | Interface) -> Surface:
    return spec.surface if isinstance(spec, Interface) else spec


def from_table(
    cls,
    rows,
    *,
    name: str = "",
    ambient: Material = AIR,
    object_medium: Material | None = None,
    image_surface: Surface | None = None,
    image_name: str = "IMG",
    stop: str | None = None,
):
    """Read prescription rows into a placed-element :class:`OpticalSystem`."""

    rows = [r if isinstance(r, TableRow) else TableRow(*r) for r in rows]
    if not rows:
        raise ValueError("a prescription has at least one row")
    object_medium = object_medium or ambient

    # 1. Vertex positions and the medium on each side of each row.
    zs, before, after = [], [], []
    z, medium = 0.0, object_medium
    for r in rows:
        zs.append(z)
        before.append(medium)
        if r.kind != MIRROR:
            medium = r.medium if r.medium is not None else medium
        after.append(medium)
        z += float(r.gap)
    image_z = z

    # 2. Physical surfaces: a row at the same vertex with the same surface is a second visit.
    face_of: list[int | None] = []
    faces: list[dict] = []
    for i, r in enumerate(rows):
        dummy = r.kind == REFRACT and before[i] == after[i] and before[i] == ambient
        if dummy:
            face_of.append(None)
            continue
        for k, f in enumerate(faces):
            if f["kind"] == r.kind and abs(f["z"] - zs[i]) < 1e-9 and f["surface"] == _surface(r.surface):
                f["sd"] = max(f["sd"], r.semi_diameter or 0.0)
                face_of.append(k)
                break
        else:
            faces.append(
                {
                    "z": zs[i],
                    "surface": _surface(r.surface),
                    "interface": Interface.coerce(r.surface),
                    "sd": r.semi_diameter or 0.0,
                    "kind": r.kind,
                    "name": r.name,
                    "on_glass": r.kind == MIRROR and before[i] != ambient,
                }
            )
            face_of.append(len(faces) - 1)
    IMAGE = len(faces)

    # 3. Bodies: each run of non-ambient medium between two physical surfaces.
    visits = [(face_of[i], i) for i in range(len(rows)) if face_of[i] is not None and rows[i].kind != STOP]
    visits.append((IMAGE, len(rows)))
    bodies: dict[frozenset, dict] = {}
    for (fa, ia), (fb, _) in zip(visits, visits[1:]):
        m = after[ia]
        if m == ambient or fa == fb:
            continue
        za = faces[fa]["z"]
        zb = image_z if fb == IMAGE else faces[fb]["z"]
        key = frozenset((fa, fb))
        if key not in bodies:
            front, back = (fa, fb) if za < zb else (fb, fa)
            bodies[key] = {"front": front, "back": back, "material": m}
        elif bodies[key]["material"] != m:
            raise ValueError(f"rows {ia}: the same body is read with two materials")

    # 4. Elements.
    def z_of(f: int) -> float:
        return image_z if f == IMAGE else faces[f]["z"]

    def interface_of(f: int) -> Interface:
        if f == IMAGE:
            return Interface.coerce(image_surface or Plane())
        face = faces[f]
        interface = face["interface"]
        if face["on_glass"]:
            interface = interface.replace(coating=PerfectMirror())
        if face["sd"] > 0.0 and interface.clear_aperture is None:
            # The row's clear semi-diameter belongs to this surface, not to the
            # larger body that carries it (a cemented face, a lens's smaller side).
            interface = interface.replace(clear_aperture=Circle(face["sd"]))
        return interface

    elements: list[Element] = []
    names: dict[int, str] = {}
    first_visit: dict[int, int] = {}
    for i, f in enumerate(face_of):
        if f is not None:
            first_visit.setdefault(f, i)
    ordered = sorted(
        bodies.values(),
        key=lambda b: min(first_visit.get(b["front"], len(rows)), first_visit.get(b["back"], len(rows))),
    )
    for n, b in enumerate(ordered, 1):
        lens_name = f"L{n}"
        sd = max(faces[f]["sd"] for f in (b["front"], b["back"]) if f != IMAGE)
        if sd <= 0.0:
            raise ValueError(f"{lens_name}: the table gives no semi-diameter for its faces")
        elements.append(
            Lens(
                lens_name,
                front=interface_of(b["front"]),
                back=interface_of(b["back"]),
                thickness=z_of(b["back"]) - z_of(b["front"]),
                material=b["material"],
                semi_diameter=sd,
                z=z_of(b["front"]),
            )
        )
        names.setdefault(b["front"], f"{lens_name}.front")
        names.setdefault(b["back"], f"{lens_name}.back")

    spans = [(z_of(b["front"]), z_of(b["back"])) for b in bodies.values()]
    n_mirrors = 0
    for k, f in enumerate(faces):
        if f["kind"] == MIRROR and not f["on_glass"]:
            n_mirrors += 1
            mirror_name = f["name"] or f"M{n_mirrors}"
            elements.append(Mirror(mirror_name, surface=f["interface"], aperture=Circle(f["sd"]), z=f["z"]))
            names[k] = mirror_name
        elif f["kind"] == STOP:
            stop_name = f["name"] or "STO"
            inside = any(a + 1e-9 < f["z"] < b - 1e-9 for a, b in spans)
            elements.append(Stop(stop_name, aperture=Circle(f["sd"]), virtual=inside, z=f["z"]))
            names[k] = stop_name
        elif k not in names:
            raise ValueError(f"surface at z = {f['z']} bounds no body and is neither a mirror nor a stop")

    elements.append(Sensor(image_name, surface=image_surface, z=image_z))
    names[IMAGE] = image_name

    # 5. The stated path, in row order.
    path: list[str] = []
    for i, r in enumerate(rows):
        f = face_of[i]
        if f is None:
            continue
        step = names[f] + (":reflect" if r.kind == MIRROR else "")
        if path and (path[-1][0] if isinstance(path[-1], tuple) else path[-1]) == step:
            continue
        if r.semi_diameter and r.semi_diameter < faces[f]["sd"]:
            # A surface met twice keeps the narrower clear aperture listed for this pass.
            path.append((step, Circle(r.semi_diameter)))
        else:
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
        if step.clear_aperture is not None:
            region = step.clear_aperture
        else:
            region = face.interface.clear_aperture if step.event == "stop" else face.clear_aperture
        semi = None if region is None else region.extent
        gap = float(following.face.pose.position[2] - pose.position[2])
        rows.append(TableRow(face.interface, gap, medium, semi, kind, step.name))
        if following.event == DETECT:
            break
    return rows


__all__ = ["Gap", "TableRow", "compose", "from_table", "to_table"]
