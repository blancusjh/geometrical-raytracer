"""Closed lens bodies deduced from the media description, colored by density.

The standard description of a sequential system alternates media and
surfaces — ``Air → Σ1 → Glass → Σ2 → Air`` — so which surface pair encloses
which medium is *deducible*, not declarable: every inter-surface gap whose
medium has index above 1 is a body bounded by the two surfaces around it
(:func:`dense_gaps`; a cemented doublet's glasses fall out as bodies
sharing interfaces). This module turns that deduction into geometry and
drawing:

- :func:`element_outline` builds the closed meridional polygon of one
  body, with **per-face extents**: each face is drawn to its own clear
  extent (semidiameter, clamped to a Cartesian profile's single-valued
  branch), so a ray refracting near a face's rim always refracts *on* the
  drawn surface. Faces that cross below the aperture are joined at the
  intersection height found by bisection — the Ω join of the author's
  ``cartesian-surfaces-stl-generator`` (``intersection_between_curves``);
  otherwise the outline closes with a rim between the two face edges.
- :func:`body_outlines` assembles every body of a system and **cuts
  neighbouring bodies apart** where one lens's back face crosses the next
  lens's front face: both parts are trimmed to the crossing height, so
  bodies never interpenetrate — each part is a well-defined solid, ready
  for future per-lens export. :func:`body_overlaps` reports where such
  cuts were needed; an overlap between *different* media is a genuine
  ambiguity of the system description, worth a design review, not only a
  drawing fix.
- :func:`index_color` maps refractive index to fill colors on the
  celestial-blue ramp of the package's palette: nothing for n = 1, then
  sky blue deepening with density — denser media draw darker,
  automatically, for any system read from its standard description.
- :func:`draw_system` renders a system or stigmatic train on a matplotlib
  axes: bodies, bare surfaces, mirrors (drawn over the sub-aperture the
  rays actually use, via :func:`~raytracer.viz.sag_drawing.mirror_arcs_from_paths`,
  when rays are traced), meridional ray fans, and the image plane — the
  shared meridional renderer for figures and notebooks. A face trimmed by
  an inter-lens cut that still refracts traced rays beyond the cut is
  continued as a dashed curve over the heights the rays reach: the
  sequential model keeps the whole mathematical surface, and the drawing
  says so without pretending the solid extends there.
"""

from __future__ import annotations

import numpy as np

from ..design.rows import SurfaceKind
from ..design.system import OpticalSystem
from ..propagation.fields import chief_ray_slopes, FieldPoint, trace_from_object
from ..propagation.sequential import SequentialTracer
from .sag_drawing import mirror_arcs_from_paths, surface_semidiameter

FIELD_COLORS = ["#d62728", "#2878b5", "#1a6e3c", "#9467bd", "#8c564b"]

#: Celestial-blue ramp endpoints (the package's own palette family:
#: ``#bde0fe``/``#2878b5`` is the DUV figure's SIO2 pair).
_SKY = (0xDF, 0xF1, 0xFF)
_DEEP = (0x28, 0x78, 0xB5)
_EDGE_DEEP = (0x1A, 0x52, 0x76)


def _usable_semidiameter(system: OpticalSystem, index: int, fallback: float) -> float:
    """Clear semidiameter clamped to the profile's single-valued branch."""

    row = system.rows[index]
    semi = surface_semidiameter(row, fallback=fallback)
    usable = getattr(row.profile, "max_usable_height", None)
    if usable is not None and np.isfinite(usable):
        semi = min(semi, 0.98 * usable)
    return semi


def dense_gaps(system: OpticalSystem) -> list[tuple[int, int, float, str]]:
    """``(i, i+1, n, name)`` for every inter-surface gap filled by a dense medium.

    Deduced from the media sequence alone: the gap after surface ``i`` is a
    body whenever its refractive index exceeds 1. Double-passed elements on
    folded paths are deduplicated by geometric signature, like
    :meth:`OpticalSystem.solid_elements`.
    """

    gaps = []
    signatures = set()
    for i in range(len(system.rows) - 1):
        if float(system.n_after[i]) <= 1.0 + 1e-9:
            continue
        first, second = system.rows[i], system.rows[i + 1]
        signature = (
            round(float(system.n_after[i]), 9),
            tuple(sorted([
                (round(float(system.vertices[i]), 6), round(first.radius, 6)),
                (round(float(system.vertices[i + 1]), 6), round(second.radius, 6)),
            ])),
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        gaps.append(
            (i, i + 1, float(system.n_after[i]), first.material_after.name)
        )
    return gaps


def _face(system: OpticalSystem, index: int):
    def curve(h):
        return system.vertices[index] + system.rows[index].profile.sag(np.abs(h))

    return curve


def _crossing_height(
    early, late, upper: float, *, samples: int = 240
) -> float | None:
    """Smallest ``h in (0, upper]`` where curve *late* falls behind *early*.

    ``None`` when the curves keep a positive gap over the whole range. The
    root is refined by bisection — the Ω-join construction.
    """

    probe = np.linspace(0.0, upper, samples)
    gaps = late(probe) - early(probe)
    if gaps[0] <= 0.0:
        raise ValueError("faces already cross on the axis; not a physical body")
    crossing = np.nonzero(gaps <= 0.0)[0]
    if crossing.size == 0:
        return None
    lo, hi = probe[crossing[0] - 1], probe[crossing[0]]
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(late(np.asarray([mid]))[0] - early(np.asarray([mid]))[0]) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def element_outline(
    system: OpticalSystem,
    i: int,
    j: int,
    *,
    front_extent: float | None = None,
    back_extent: float | None = None,
    samples: int = 240,
    fallback_semidiameter: float = 50.0,
) -> tuple[np.ndarray, tuple[float, float], bool]:
    """Closed ``(z, h)`` polygon of the body between faces *i* and *j*.

    Each face runs to its own extent (its clear semidiameter clamped to the
    usable branch, further capped by ``front_extent``/``back_extent`` — the
    hooks :func:`body_outlines` uses for inter-lens cuts). The Ω join is
    sought over the *raw* extents, before the caps apply: a cap on one face
    must not hide the height where the two faces of this body meet, or the
    other face would be drawn past their intersection. Returns
    ``(polygon, (front_edge, back_edge), joined)``: *joined* is True when
    the outline closes at the sharp Ω edge where the faces meet; otherwise
    a rim connects the two face edges.
    """

    e_front = _usable_semidiameter(system, i, fallback_semidiameter)
    e_back = _usable_semidiameter(system, j, fallback_semidiameter)

    front, back = _face(system, i), _face(system, j)
    crossing = _crossing_height(front, back, min(e_front, e_back), samples=samples)
    if crossing is not None:
        e_front = e_back = crossing

    if front_extent is not None:
        e_front = min(e_front, front_extent)
    if back_extent is not None:
        e_back = min(e_back, back_extent)
    joined = crossing is not None and e_front == e_back == crossing

    h_front = np.linspace(-e_front, e_front, samples)
    h_back = np.linspace(e_back, -e_back, samples)
    front_curve = np.column_stack([front(h_front), h_front])
    back_curve = np.column_stack([back(h_back), h_back])
    polygon = np.vstack([front_curve, back_curve, front_curve[:1]])
    return polygon, (e_front, e_back), joined


def body_overlaps(system: OpticalSystem) -> list[tuple[int, int, float]]:
    """Interpenetrations between distinct neighbouring bodies.

    Returns ``[(back_face, front_face, h_cut), ...]``: below ``h_cut`` the
    two parts are separate; above it, body A's back face crosses body B's
    front face. For bodies of *different* media the overlap region is a
    genuine ambiguity of the description (which medium fills it?) — the
    drawing cuts both parts at ``h_cut``, and a design that wants to be
    manufactured should be reworked to remove the overlap entirely.
    Cemented interfaces (shared face) are not overlaps.
    """

    gaps = dense_gaps(system)
    overlaps = []
    for (_, j, _, _), (p, _, _, _) in zip(gaps, gaps[1:]):
        if j == p:  # cemented: the interface is shared, nothing to cut
            continue
        upper = min(
            _usable_semidiameter(system, j, 50.0),
            _usable_semidiameter(system, p, 50.0),
        )
        crossing = _crossing_height(_face(system, j), _face(system, p), upper)
        if crossing is not None:
            overlaps.append((j, p, crossing))
    return overlaps


def body_outlines(system: OpticalSystem):
    """Every body's outline, with neighbouring parts cut apart.

    Yields ``(i, j, n, name, polygon)`` per body. Where
    :func:`body_overlaps` finds body A's back face crossing body B's front
    face, both faces are trimmed to the crossing height, so the returned
    parts never interpenetrate.
    """

    caps: dict[int, float] = {}
    for back_face, front_face, h_cut in body_overlaps(system):
        caps[back_face] = min(caps.get(back_face, np.inf), h_cut)
        caps[front_face] = min(caps.get(front_face, np.inf), h_cut)

    outlines = []
    for i, j, n, name in dense_gaps(system):
        polygon, _, _ = element_outline(
            system, i, j,
            front_extent=caps.get(i), back_extent=caps.get(j),
        )
        outlines.append((i, j, n, name, polygon))
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
        return "#" + "".join(
            f"{round(ca + (cb - ca) * level):02x}" for ca, cb in zip(a, b)
        )

    return blend(_SKY, _DEEP, 0.25 + 0.75 * t), blend(_SKY, _EDGE_DEEP, 0.45 + 0.55 * t)


def draw_system(
    ax,
    system_or_train,
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
    """Standard meridional rendering: bodies, surfaces, ray fans, image plane.

    ``fields`` are object heights traced as cones of numerical aperture
    ``na``, aimed at the axial point ``z = aim_z`` — or, when
    ``stop_index`` is given, centred on each field's solved chief ray
    through that stop. In virtual-object mode (``launch`` set to a
    launch-plane z in front of the lens) each field value is instead the
    *sine of an aim angle* and converging fans aimed at the virtual object
    point are launched from that plane.

    Bodies come from :func:`body_outlines` (neighbouring parts cut apart)
    and are colored by :func:`index_color`; ``material_colors`` (name-keyed
    ``(face, edge)`` pairs) overrides individual media. Unpaired refracting
    surfaces draw as bare curves. Mirrors draw over the sub-aperture the
    traced rays actually hit when there are rays (the honest extent for
    off-axis and ring-field systems), or over their full cap otherwise.

    Returns the traced ray paths, stacked ``(rays, points, 3)``, so
    annotation layers can reuse them (e.g. labeling mirrors along the arcs
    from :func:`~raytracer.viz.sag_drawing.mirror_arcs_from_paths`), or
    ``None`` when no ray survived.
    """

    system = (
        system_or_train
        if isinstance(system_or_train, OpticalSystem)
        else system_or_train.to_system()
    )
    tracer = SequentialTracer(system)
    object_z = system.object_z

    # -- rays first, so mirror arcs can follow the real hit envelope -------
    traced: list[tuple[np.ndarray, str]] = []
    smax = na / np.sqrt(1.0 - na * na)
    for h_field, color in zip(fields, field_colors):
        if launch is None:
            if stop_index is not None:
                _, chief = chief_ray_slopes(
                    tracer, FieldPoint(y=float(h_field)), stop_index=stop_index
                )
            else:
                chief = (0.0 - h_field) / (aim_z - object_z)
            for s in np.linspace(-smax, smax, rays):
                r = trace_from_object(tracer, (0.0, h_field), (0.0, chief + s),
                                      keep_path=True)
                if r.ok:
                    traced.append((r.path, color))
        else:
            t0 = np.tan(np.arcsin(h_field))
            for dt in np.linspace(-0.04, 0.04, rays):
                t = t0 + dt
                origin = np.array([0.0, (object_z - launch) * t, launch])
                direction = np.array([0.0, -t, 1.0]) / np.hypot(t, 1.0)
                r = tracer.trace(origin, direction, keep_path=True)
                if r.ok:
                    traced.append((np.vstack([origin, r.path[1:]]), color))

    # -- bodies, cut apart, colored by density -----------------------------
    drawn_faces: set[int] = set()
    face_edge_color: dict[int, str] = {}
    for i, j, n, name, polygon in body_outlines(system):
        colors = (material_colors or {}).get(name.upper()) or index_color(n)
        if colors is None:
            continue
        face, edge = colors
        ax.fill(polygon[:, 0], polygon[:, 1], facecolor=face, edgecolor=edge,
                lw=1.1, alpha=0.8, zorder=1)
        drawn_faces.update((i, j))
        face_edge_color[i] = face_edge_color[j] = edge

    # Faces trimmed by an inter-lens cut may still be *refracting* rays
    # beyond the cut — the sequential model keeps the whole mathematical
    # surface even where the solid part was trimmed. Continue such a face
    # as a dashed curve over the heights the traced rays actually reach,
    # so every drawn refraction happens on a drawn surface and the cut
    # region is visibly not part of either solid.
    if traced:
        paths_arr = np.stack([path for path, _ in traced])
        hit_tops = np.abs(paths_arr[:, 1:-1, 1]).max(axis=0)
        for back_face, front_face, h_cut in body_overlaps(system):
            for k in (back_face, front_face):
                top = min(
                    _usable_semidiameter(system, k, 50.0),
                    1.02 * float(hit_tops[k]),
                )
                if top <= h_cut:
                    continue
                h = np.linspace(h_cut, top, 60)
                z = system.vertices[k] + system.rows[k].profile.sag(h)
                color = face_edge_color.get(k, "#3a6ea5")
                for sign in (1.0, -1.0):
                    ax.plot(z, sign * h, color=color, lw=1.0, alpha=0.9,
                            ls=(0, (2.6, 2.2)), zorder=2)

    # -- mirrors and unpaired refracting faces -----------------------------
    if system.mirror_indices and traced:
        paths = np.stack([path for path, _ in traced])
        for _, z, h in mirror_arcs_from_paths(system, paths):
            ax.plot(z, h, color="#202020", lw=2.6, zorder=3)
    else:
        for i in system.mirror_indices:
            semi = _usable_semidiameter(system, i, 50.0)
            h = np.linspace(-semi, semi, 200)
            z = system.vertices[i] + system.rows[i].profile.sag(np.abs(h))
            ax.plot(z, h, color="#202020", lw=2.6, zorder=3)
    for i, row in enumerate(system.rows):
        if row.kind is SurfaceKind.REFRACT and i not in drawn_faces:
            if float(system.n_before[i]) == float(system.n_after[i]):
                continue
            semi = _usable_semidiameter(system, i, 50.0)
            h = np.linspace(-semi, semi, 200)
            z = system.vertices[i] + row.profile.sag(np.abs(h))
            ax.plot(z, h, color="#3a6ea5", lw=1.6, zorder=2)

    for path, color in traced:
        finite = path[np.isfinite(path[:, 2])]
        ax.plot(finite[:, 2], finite[:, 1], color=color, lw=0.55, alpha=0.65,
                zorder=2)
    if launch is None:
        for h_field, color in zip(fields, field_colors):
            ax.plot([object_z], [h_field], "o", color=color, ms=4, zorder=4)

    if image_plane:
        ax.axvline(system.image_z, color="#999999", ls=":", lw=0.8)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("z (mm)")
    ax.grid(alpha=0.15)
    return np.stack([path for path, _ in traced]) if traced else None


__all__ = [
    "FIELD_COLORS",
    "body_outlines",
    "body_overlaps",
    "dense_gaps",
    "draw_system",
    "element_outline",
    "index_color",
]
