"""Lens bodies in the drawing plane: outlines, cuts between neighbours, colour by index.

A system states its bodies: every :class:`~raytracer.elements.Lens` is a
body of one material between its front and back faces, closed by a
cylindrical rim at its semi-diameter, placed by its world pose
(``system.pose_of(name)``); cemented contact and immersion are resolved by
the system (``system.faces``). This module draws those bodies as the
meridional section of the 3-D model (see :mod:`raytracer.viz.sag_drawing`
for the section and when it is exact):

- :func:`element_outline` builds the closed drawing-plane polygon of one
  lens. Each face runs to its own extent — the semi-diameter's chord,
  clamped to the face's single-valued branch (a strong Cartesian ovoid
  closes back on itself) — so a ray refracting near the rim always
  refracts *on* the drawn surface. Faces that cross below the rim (a lens
  with a negative edge thickness) are joined at the crossing height, found
  by bisection — the Ω join of the author's
  ``cartesian-surfaces-stl-generator``; otherwise the outline closes with
  the section of the rim.
- :func:`body_overlaps` finds **neighbouring lenses that interpenetrate**:
  coaxial neighbours not in contact whose facing surfaces cross off-axis.
  The space between the crossed faces is claimed by both solids — two
  materials at once, an inconsistency of the design rather than of the
  drawing — and :func:`body_outlines` cuts both parts there, assigning the
  ambiguous region to *neither* (it stays hollow), while each part keeps
  its unambiguous glass. The cut height bounds the design's **valid
  aperture**: light refracting on a face beyond the drawn part meets no
  defined medium there.
- :func:`index_color` maps refractive index onto the celestial-blue ramp:
  nothing for n = 1, then sky blue deepening with density.
- :func:`draw_system` renders a system (a :class:`SequentialTracer`, a
  stigmatic train, or a bare :class:`OpticalSystem`) on a matplotlib axes:
  bodies, mirrors (over the sub-aperture the rays use, when rays are
  traced), stops, beam splitters, prisms, sensors, and meridional ray
  fans with each field's focus marked. When traced rays refract beyond the
  drawn glass it warns instead of decorating: there is no medium there to
  refract against, and no drawing can make that physical.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np

from ..elements import BeamSplitter, Lens, Mirror, Prism, Sensor, Stop
from ..geometry.apertures import Circle
from ..geometry.surfaces import Plane
from ..geometry.transforms import Pose
from ..materials import Material
from ..system import OpticalSystem
from ..system.faces import REFLECT, TRANSMIT
from ..tracing import FieldPoint, SequentialTracer, chief_ray_slopes, trace_from_object
from .sag_drawing import (
    DEFAULT_SEMIDIAMETER,
    beam_foci_from_paths,
    face_section,
    lens_section,
    mirror_arcs_from_paths,
    polyhedron_section,
    section_offset,
    usable_half_chord,
)

FIELD_COLORS = ["#d62728", "#2878b5", "#1a6e3c", "#9467bd", "#8c564b"]

#: Wavelength at which a body's index is read for its colour when none is given (d line).
COLOR_WAVELENGTH_UM = 0.58756

#: Celestial-blue ramp endpoints (the package's own palette family:
#: ``#bde0fe``/``#2878b5`` is the DUV figure's SIO2 pair).
_SKY = (0xDF, 0xF1, 0xFF)
_DEEP = (0x28, 0x78, 0xB5)
_EDGE_DEEP = (0x1A, 0x52, 0x76)

MIRROR_COLOR = "#202020"
STOP_COLOR = "#444444"
SPLITTER_COLOR = "#555555"
SENSOR_COLOR = "#999999"


class BodyOutline(NamedTuple):
    """One drawn body: element name, material, index, closed ``(z, y)`` polygon.

    ``extents`` are the drawn half-chords of the front and back faces
    (``None`` for a prism); ``exact`` is False when the element's pose is
    not meridional and the polygon is a projection.
    """

    name: str
    material: Material
    n: float
    polygon: np.ndarray
    extents: tuple[float, float] | None
    exact: bool


def medium_index(material: Material | None, wavelength_um: float | None = None) -> float:
    """Refractive index of *material* for colouring (d line unless told otherwise)."""

    if material is None:
        return 1.0
    for wl in (wavelength_um, COLOR_WAVELENGTH_UM):
        if wl is None:
            continue
        try:
            return float(material.index(wl))
        except (ValueError, TypeError):
            continue
    return 1.5


# -- one lens ----------------------------------------------------------------


def _local_curves(lens: Lens, u0: float):
    """Local z of the front and back faces along the chord ``x = u0``, as functions of signed v."""

    def front(v):
        v = np.asarray(v, dtype=float)
        return lens.front.surface.sag(np.full_like(v, u0), v)

    def back(v):
        v = np.asarray(v, dtype=float)
        return lens.thickness + lens.back.surface.sag(np.full_like(v, u0), v)

    return front, back


def _crossing_height(early, late, upper: float, *, samples: int = 240) -> float | None:
    """Smallest ``h in (0, upper]`` where curve *late* falls behind *early* (at ±h).

    ``None`` when the curves keep a positive gap over the whole range. The
    root is refined by bisection — the Ω-join construction. Raises when the
    curves already cross at ``h = 0``.
    """

    def gap(h):
        h = np.asarray(h, dtype=float)
        return np.minimum(late(h) - early(h), late(-h) - early(-h))

    probe = np.linspace(0.0, upper, samples)
    gaps = gap(probe)
    if gaps[0] <= 0.0:
        raise ValueError("the faces already cross on the axis; not a physical body")
    crossing = np.nonzero(gaps <= 0.0)[0]
    if crossing.size == 0:
        return None
    lo, hi = probe[crossing[0] - 1], probe[crossing[0]]
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(gap(np.asarray([mid]))[0]) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def usable_extents(lens: Lens, pose: Pose | None = None) -> tuple[float, float]:
    """Half-chords of the front and back faces before any join or cut.

    The semi-diameter's chord in the drawing plane, each clamped to its
    face's single-valued branch.
    """

    pose = lens.pose if pose is None else pose
    u0, _ = section_offset(pose)
    half = float(np.sqrt(max(lens.semi_diameter**2 - u0 * u0, 0.0)))
    return (
        usable_half_chord(lens.front.surface, u0, half),
        usable_half_chord(lens.back.surface, u0, half),
    )


def element_outline(
    lens: Lens,
    pose: Pose | None = None,
    *,
    front_extent: float | None = None,
    back_extent: float | None = None,
    samples: int = 240,
) -> tuple[np.ndarray, tuple[float, float], bool]:
    """Closed drawing-plane polygon of one lens body at its world *pose*.

    Each face runs to its own usable extent (:func:`usable_extents`),
    further capped by ``front_extent`` / ``back_extent`` — the hooks
    :func:`body_outlines` uses for inter-lens cuts. The Ω join is sought
    over the *raw* extents, before the caps apply: a cap on one face must
    not hide the height where the two faces of this body meet, or the
    other face would be drawn past their intersection. Returns
    ``(polygon, (front_edge, back_edge), joined)``: *joined* is True when the
    outline closes at the sharp Ω edge where the faces meet; otherwise the
    rim's section connects the two face edges. *pose* defaults to the
    lens's own pose (pass ``system.pose_of(name)`` for a lens in an
    assembly).
    """

    pose = lens.pose if pose is None else pose
    u0, _ = section_offset(pose)
    e_front, e_back = usable_extents(lens, pose)
    front, back = _local_curves(lens, u0)
    crossing = _crossing_height(front, back, min(e_front, e_back), samples=samples)
    if crossing is not None:
        e_front = e_back = crossing
    if front_extent is not None:
        e_front = min(e_front, front_extent)
    if back_extent is not None:
        e_back = min(e_back, back_extent)
    joined = crossing is not None and e_front == e_back == crossing
    polygon = lens_section(lens, pose, front_extent=e_front, back_extent=e_back, samples=samples)
    return polygon, (e_front, e_back), joined


# -- neighbours ---------------------------------------------------------------


def _coaxial(pose: Pose) -> bool:
    """On the world z axis, pointing +z: where neighbouring lenses can be compared by height."""

    return bool(np.allclose(pose.position[:2], 0.0, atol=1e-12) and pose.rotation.is_identity(1e-12))


def _coaxial_lenses(system: OpticalSystem) -> list[tuple[Lens, Pose]]:
    lenses = [(e, p) for e, p in system.walk() if isinstance(e, Lens) and _coaxial(p)]
    return sorted(lenses, key=lambda lp: float(lp[1].position[2]))


def _world_curve(lens: Lens, pose: Pose, side: str):
    """World z of one face of a coaxial lens as a function of signed height."""

    surface = lens.front.surface if side == "front" else lens.back.surface
    z0 = float(pose.position[2]) + (0.0 if side == "front" else lens.thickness)

    def curve(h):
        h = np.asarray(h, dtype=float)
        return z0 + surface.sag(np.zeros_like(h), h)

    return curve


def _in_contact(system: OpticalSystem, a: str, b: str) -> bool:
    return any(f"{a}.back" in f.names and f"{b}.front" in f.names for f in system.faces)


def body_overlaps(system: OpticalSystem) -> list[tuple[str, str, float]]:
    """Interpenetrations between neighbouring lenses on the common axis.

    Returns ``[(upstream, downstream, h_cut), ...]`` (element names): below
    ``h_cut`` the two bodies are separate; above it, the upstream lens's
    back face crosses the downstream lens's front face. The overlap region
    is claimed by two materials at once — :func:`body_outlines` cuts both
    parts at ``h_cut``, and a design meant to be built should be reworked to
    remove the overlap. Lenses in contact (a cemented face) never overlap;
    ``h_cut = 0`` means the two collide on the axis itself. Only coaxial
    lenses (on the z axis, unrotated) are compared.
    """

    lenses = _coaxial_lenses(system)
    overlaps = []
    for (a, pa), (b, pb) in zip(lenses, lenses[1:]):
        if _in_contact(system, a.name, b.name):
            continue
        upper = min(usable_extents(a, pa)[1], usable_extents(b, pb)[0])
        try:
            crossing = _crossing_height(_world_curve(a, pa, "back"), _world_curve(b, pb, "front"), upper)
        except ValueError:
            crossing = 0.0
        if crossing is not None:
            overlaps.append((a.name, b.name, crossing))
    return overlaps


def _chord_clear(hs, z_a, h_a, z_b, h_b, curve, *, side: float) -> bool:
    """True when the chord (z_a,h_a)→(z_b,h_b) stays on *side* of *curve*.

    ``side=+1`` requires the chord at or behind the curve (larger z),
    ``side=-1`` in front of it — sampled over the heights *hs*.
    """

    chord = z_a + (z_b - z_a) * (hs - h_a) / (h_b - h_a)
    return bool((side * (chord - curve(hs)) >= -1e-9).all())


def body_outlines(system: OpticalSystem, *, wavelength_um: float | None = None, samples: int = 240) -> list[BodyOutline]:
    """Every body of *system* (lenses and prisms), neighbouring lenses cut apart.

    Where :func:`body_overlaps` finds lens A's back face crossing lens B's
    front face, the space between the crossed faces above the cut is
    ambiguous — both solids claim it — so it is assigned to *neither* part:
    both crossed faces are trimmed to the cut. Each body keeps the rest of
    its **unambiguous** glass: A its full front face while its rim
    verifiably stays in front of the ambiguous region, B its back face over
    the band behind A's back face, up to where that band closes —
    refractions on B's back face above the cut are physical, glass sits in
    front of them. The parts never interpenetrate and no part covers
    ambiguous space; checks that fail degrade to capping at the cut, never
    to over-claiming. ``wavelength_um`` sets where each body's index ``n``
    is read (the d line by default).
    """

    placed = {e.name: (e, p) for e, p in system.walk()}
    caps: dict[tuple[str, str], float] = {}

    def cap(key: tuple[str, str], height: float) -> None:
        caps[key] = min(caps.get(key, np.inf), height)

    def curve(key):
        lens, pose = placed[key[0]]
        return _world_curve(lens, pose, key[1])

    def usable(key):
        lens, pose = placed[key[0]]
        return usable_extents(lens, pose)[0 if key[1] == "front" else 1]

    for a, b, h_cut in body_overlaps(system):
        if h_cut <= 0.0:
            warnings.warn(f"lenses {a!r} and {b!r} collide on the axis; drawn uncut", stacklevel=2)
            continue
        jb, p, q, i = (a, "back"), (b, "front"), (b, "back"), (a, "front")
        cap(jb, h_cut)
        cap(p, h_cut)

        # downstream body: the unambiguous band between A's back face and its own back face
        e_q = usable(q)
        upper = min(usable(jb), e_q)
        try:
            h_star = _crossing_height(curve(jb), curve(q), upper)
        except ValueError:
            h_star = 0.0
        candidate = min(h_star if h_star is not None else e_q, e_q)
        keep = h_cut
        if candidate > h_cut:
            z_x = float(curve(p)(np.asarray([h_cut]))[0])
            z_top = float(curve(q)(np.asarray([candidate]))[0])
            hs = np.linspace(h_cut, min(candidate, upper), 120)
            if _chord_clear(hs, z_x, h_cut, z_top, candidate, curve(jb), side=+1.0):
                keep = candidate
        cap(q, keep)

        # upstream body: keep its front face while its rim stays in front of the region
        e_i = usable(i)
        if e_i > h_cut:
            z_cut = float(curve(p)(np.asarray([h_cut]))[0])
            z_top = float(curve(i)(np.asarray([e_i]))[0])
            upper = min(e_i, usable(jb), usable(p))
            if upper > h_cut:
                hs = np.linspace(h_cut, upper, 120)
                if not _chord_clear(hs, z_cut, h_cut, z_top, e_i, curve(p), side=-1.0):
                    cap(i, h_cut)

    outlines = []
    for element, pose in system.walk():
        exact = section_offset(pose)[1]
        if isinstance(element, Lens):
            polygon, extents, _ = element_outline(
                element,
                pose,
                front_extent=caps.get((element.name, "front")),
                back_extent=caps.get((element.name, "back")),
                samples=samples,
            )
            outlines.append(
                BodyOutline(element.name, element.material, medium_index(element.material, wavelength_um), polygon, extents, exact)
            )
        elif isinstance(element, Prism):
            polygon, exact = polyhedron_section(element.solid, pose)
            if polygon.shape[0] >= 3:
                polygon = np.vstack([polygon, polygon[:1]])
                outlines.append(
                    BodyOutline(element.name, element.material, medium_index(element.material, wavelength_um), polygon, None, exact)
                )
    return outlines


def index_color(n: float, *, n_max: float = 2.0) -> tuple[str, str] | None:
    """``(face, edge)`` colors for a medium of index *n*; None for n = 1.

    The celestial-blue ramp: sky blue for light media deepening toward the
    package's ``#2878b5`` for dense ones — luminous at every density.
    Media at or beyond ``n_max`` saturate at the deepest shade.
    """

    if n <= 1.0 + 1e-9:
        return None
    t = min(max((n - 1.0) / (n_max - 1.0), 0.0), 1.0)

    def blend(a, b, level):
        return "#" + "".join(f"{round(ca + (cb - ca) * level):02x}" for ca, cb in zip(a, b))

    return blend(_SKY, _DEEP, 0.25 + 0.75 * t), blend(_SKY, _EDGE_DEEP, 0.45 + 0.55 * t)


def body_colors(outline: BodyOutline, material_colors: dict | None = None) -> tuple[str, str] | None:
    """``(face, edge)`` for a body: a name-keyed override, else :func:`index_color`."""

    name = getattr(outline.material, "name", "") or ""
    return (material_colors or {}).get(name.upper()) or index_color(outline.n)


# -- drawing on matplotlib axes -------------------------------------------------


def system_extent(system: OpticalSystem) -> float:
    """Largest aperture extent of any element: the scale for unbounded faces."""

    sizes = []
    for element in system.elements:
        if isinstance(element, Lens):
            sizes.append(element.semi_diameter)
        elif isinstance(element, (Mirror, BeamSplitter)):
            sizes.append(element.aperture.extent)
        elif isinstance(element, Stop):
            sizes.append(element.outer_radius)
        elif isinstance(element, Sensor) and element.aperture is not None:
            sizes.append(element.aperture.extent)
    return max(sizes) if sizes else DEFAULT_SEMIDIAMETER


def _is_axial_plane(surface, pose: Pose) -> bool:
    """A plane perpendicular to the z axis: drawn as a vertical line across the axes."""

    return isinstance(surface, Plane) and abs(abs(float(pose.axis[2])) - 1.0) < 1e-12 and section_offset(pose)[1]


def draw_elements(
    ax,
    system: OpticalSystem,
    *,
    wavelength_um: float | None = None,
    material_colors: dict | None = None,
    alpha: float = 0.8,
    paths: np.ndarray | None = None,
    mirror_labels: bool = False,
    sensors: bool = True,
) -> dict[str, float]:
    """Draw every element of *system* on *ax* in drawing coordinates ``(z, y)``.

    Lenses and prisms fill by index (:func:`body_outlines`, neighbours cut
    apart); mirrors draw over the sub-aperture the traced *paths* use when
    given (:func:`~raytracer.viz.sag_drawing.mirror_arcs_from_paths`), over
    their whole aperture otherwise; diaphragm blades, beam splitters
    (dashed) and sensors (dotted) as their sections. Returns the drawn
    half-chord of every lens face by boundary name (``"L2.front"``), the
    extent of glass a refraction there can be physical within.
    """

    drawn: dict[str, float] = {}
    scale = system_extent(system)
    for body in body_outlines(system, wavelength_um=wavelength_um):
        colors = body_colors(body, material_colors)
        if colors is None:
            continue
        face, edge = colors
        ax.fill(body.polygon[:, 0], body.polygon[:, 1], facecolor=face, edgecolor=edge, lw=1.1, alpha=alpha, zorder=1)
        if body.extents is not None:
            for side, extent in zip(("front", "back"), body.extents):
                key = f"{body.name}.{side}"
                drawn[key] = max(drawn.get(key, 0.0), extent)

    reflecting = [i for i, s in enumerate(system.path) if s.event == REFLECT] if _has_path(system) else []
    arcs_drawn: set[str] = set()
    if reflecting and paths is not None and len(paths):
        for count, z, y in mirror_arcs_from_paths(system, paths):
            step = system.path[reflecting[count - 1]]
            ax.plot(z, y, color=MIRROR_COLOR, lw=2.6, zorder=3)
            arcs_drawn.add(step.face.name)
            if mirror_labels:
                ax.text(float(np.mean(z)), float(np.max(y)) + 0.04 * scale, step.face.name, ha="center", fontsize=9)

    for element, pose in system.walk():
        if isinstance(element, Mirror) and element.name not in arcs_drawn:
            for curve in face_section(element.surface, pose, element.aperture):
                ax.plot(curve[:, 0], curve[:, 1], color=MIRROR_COLOR, lw=2.6, zorder=3)
                if mirror_labels:
                    ax.text(float(np.mean(curve[:, 0])), float(np.max(curve[:, 1])) + 0.04 * scale, element.name, ha="center", fontsize=9)
        elif isinstance(element, BeamSplitter):
            for curve in face_section(element.interface.surface, pose, element.aperture):
                ax.plot(curve[:, 0], curve[:, 1], color=SPLITTER_COLOR, lw=1.8, ls="--", zorder=3)
        elif isinstance(element, Stop) and not element.virtual:
            for curve in face_section(Plane(), pose, Circle(element.outer_radius), hole=element.aperture):
                ax.plot(curve[:, 0], curve[:, 1], color=STOP_COLOR, lw=2.2, solid_capstyle="butt", zorder=3)
        elif isinstance(element, Sensor) and sensors:
            if element.aperture is None and _is_axial_plane(element.surface, pose):
                ax.axvline(float(pose.position[2]), color=SENSOR_COLOR, ls=":", lw=0.8)
                continue
            for curve in face_section(element.surface, pose, element.aperture, fallback=scale):
                ax.plot(curve[:, 0], curve[:, 1], color=SENSOR_COLOR, ls=":", lw=1.2, zorder=2)
    return drawn


def _has_path(system: OpticalSystem) -> bool:
    try:
        return bool(system.path)
    except ValueError:
        return False


def warn_outside_glass(system: OpticalSystem, paths: np.ndarray, drawn: dict[str, float], *, stacklevel: int = 3) -> None:
    """Warn when traced refractions land on a lens face beyond its drawn glass.

    The sequential formalism refracts at every surface wherever a ray meets
    it — but a refraction is only physical where glass lies against the
    face. Beyond a drawn part (past an inter-lens cut, or past the usable
    branch of a face) there is no medium to refract against, so a hit there
    marks rays outside the design's valid aperture.
    """

    paths = np.asarray(paths, dtype=float)
    problems = []
    for i, step in enumerate(system.path):
        if step.event != TRANSMIT:
            continue
        extents = [drawn[n] for n in step.face.names if n in drawn]
        if not extents:
            continue
        edge = max(extents)
        hits = paths[:, i + 1, :]
        hits = hits[np.isfinite(hits).all(axis=1)]
        if hits.shape[0] == 0:
            continue
        u0, _ = section_offset(step.face.pose)
        local = step.face.pose.to_local(hits)
        reach = float(np.sqrt(np.max(local[:, 0] ** 2 + local[:, 1] ** 2)))
        limit = float(np.hypot(edge, u0))
        if reach > limit * 1.001:
            problems.append(f"{step.face.name}: hits to h = {reach:.3f} mm, glass ends at {limit:.3f}")
    if problems:
        warnings.warn(
            "rays refract beyond the drawn glass — outside the design's valid aperture, "
            f"the system defines no medium there ({'; '.join(problems)})",
            stacklevel=stacklevel,
        )


def as_tracer(subject) -> SequentialTracer:
    """A :class:`SequentialTracer` from a tracer, a stigmatic train, or a bare system.

    A bare system gets a tracer with no wavelength and no object plane —
    enough to draw it; tracing fields needs ``object_z`` (pass a tracer).
    """

    if isinstance(subject, SequentialTracer):
        return subject
    if isinstance(subject, OpticalSystem):
        return SequentialTracer(subject)
    if hasattr(subject, "tracer"):
        return subject.tracer()
    raise TypeError(f"cannot draw {subject!r}: pass a SequentialTracer, an OpticalSystem or a StigmaticTrain")


def draw_system(
    ax,
    subject,
    *,
    fields=(),
    na: float = 0.1,
    rays: int = 7,
    aim_z: float = 0.0,
    stop_index: int | None = None,
    launch: float | None = None,
    field_colors=FIELD_COLORS,
    material_colors: dict | None = None,
    xlim=None,
    ylim=None,
    image_plane: bool = True,
) -> np.ndarray | None:
    """Standard meridional rendering: bodies, mirrors, stops, sensors, ray fans.

    *subject* is a :class:`~raytracer.tracing.SequentialTracer` (system,
    wavelength and object plane), a stigmatic train (its own tracer), or a
    bare :class:`~raytracer.system.OpticalSystem` (drawn without fields).

    ``fields`` are object heights traced as fans of numerical aperture
    ``na``, aimed at the axial point ``z = aim_z`` — or, when ``stop_index``
    (a path index) is given, centred on each field's solved chief ray
    through that step. In virtual-object mode (``launch`` set to a
    launch-plane z in front of the lens) each field value is instead the
    *sine of an aim angle* and converging fans aimed at the virtual object
    point are launched from that plane.

    Bodies come from :func:`body_outlines` (neighbouring parts cut apart)
    and are coloured by :func:`index_color`; ``material_colors``
    (name-keyed ``(face, edge)`` pairs) overrides individual media. Where
    the light focuses is part of the trace and is drawn by default: each
    field's convergence points (:func:`~raytracer.viz.sag_drawing.beam_foci_from_paths`)
    are marked in the field's colour, the aperture stop draws as a dotted
    vertical (for a coaxial stop), and when the final focus lies beyond
    ``xlim`` a corner note says where the image is instead of silently
    cropping it.

    Returns the traced ray paths, stacked ``(rays, points, 3)``, so
    annotation layers can reuse them, or ``None`` when no ray survived.
    """

    tracer = as_tracer(subject)
    system = tracer.system
    object_z = tracer.object_z
    if fields and launch is None and object_z is None:
        raise ValueError("tracing fields needs the object plane: pass a SequentialTracer with object_z")

    # -- rays first, so mirror arcs can follow the real hit envelope -------
    traced: list[tuple[np.ndarray, str]] = []
    beams: list[tuple[list[np.ndarray], str]] = []
    smax = na / np.sqrt(1.0 - na * na)
    for h_field, color in zip(fields, field_colors):
        bundle: list[np.ndarray] = []
        if launch is None:
            if stop_index is not None:
                _, chief = chief_ray_slopes(tracer, FieldPoint(y=float(h_field)), stop_index=stop_index)
            else:
                chief = (0.0 - h_field) / (aim_z - object_z)
            for s in np.linspace(-smax, smax, rays):
                r = trace_from_object(tracer, (0.0, h_field), (0.0, chief + s), keep_path=True)
                if r.ok:
                    bundle.append(r.path)
        else:
            reference = object_z if object_z is not None else 0.0
            t0 = np.tan(np.arcsin(h_field))
            for dt in np.linspace(-0.04, 0.04, rays):
                t = t0 + dt
                origin = np.array([0.0, (reference - launch) * t, launch])
                direction = np.array([0.0, -t, 1.0]) / np.hypot(t, 1.0)
                r = tracer.trace(origin, direction, keep_path=True)
                if r.ok:
                    bundle.append(r.path)
        traced.extend((path, color) for path in bundle)
        if bundle:
            beams.append((bundle, color))

    paths = np.stack([path for path, _ in traced]) if traced else None
    drawn = draw_elements(
        ax,
        system,
        wavelength_um=tracer.wavelength_um,
        material_colors=material_colors,
        paths=paths,
        sensors=image_plane,
    )
    if paths is not None:
        warn_outside_glass(system, paths, drawn)

    for path, color in traced:
        finite = path[np.isfinite(path).all(axis=1)]
        ax.plot(finite[:, 2], finite[:, 1], color=color, lw=0.55, alpha=0.65, zorder=2)
    if launch is None:
        for h_field, color in zip(fields, field_colors):
            ax.plot([object_z], [h_field], "o", color=color, ms=4, zorder=4)

    # -- where the light focuses: convergence points of each traced beam ---
    focus_beyond_frame = False
    for bundle, color in beams:
        for z_f, y_f, _, _, is_final in beam_foci_from_paths(np.stack(bundle)):
            if xlim is not None and not (xlim[0] <= z_f <= xlim[1]):
                focus_beyond_frame |= is_final
                continue
            ax.plot([z_f], [y_f], "o", ms=4.5, mfc="white", mec=color, mew=1.2, zorder=6)
    if focus_beyond_frame:
        ax.annotate(
            f"image at z = {tracer.image_z:g} mm",
            xy=(0.985, 0.04),
            xycoords="axes fraction",
            ha="right",
            fontsize=8,
            style="italic",
            color="#666666",
        )

    stop = stop_index if stop_index is not None else system.stop_index
    if stop is not None and _is_axial_plane(Plane(), system.path[stop].face.pose):
        ax.axvline(tracer.step_z(stop), color="#777777", ls=":", lw=0.8, zorder=0)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("z (mm)")
    ax.grid(alpha=0.15)
    return paths


__all__ = [
    "FIELD_COLORS",
    "BodyOutline",
    "as_tracer",
    "body_colors",
    "body_outlines",
    "body_overlaps",
    "draw_elements",
    "draw_system",
    "element_outline",
    "index_color",
    "medium_index",
    "usable_extents",
    "warn_outside_glass",
]
