"""Bodies of a system in the drawing plane, their colours, and the matplotlib drawing of a traced system.

A lens is a body of one material between its two faces, closed by its rim
(:func:`~raytracer.viz.sag_drawing.lens_outline`). Coaxial neighbours not in contact may **interpenetrate**:
one's back face crosses the other's front face off-axis, and the space between them is claimed by two
materials at once — an inconsistency of the design, not of the drawing. :func:`body_outlines` cuts both parts
there and leaves that region hollow. The cut bounds the design's valid aperture: :func:`draw_system` warns when
traced rays refract beyond the drawn glass, where the system defines no medium.
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import NamedTuple

import numpy as np

from ..elements import BeamSplitter, Lens, Mirror, Prism, Sensor, Stop
from ..geometry.surfaces import Plane
from ..materials import Material
from ..system import OpticalSystem
from ..system.faces import TRANSMIT
from ..tracing import FieldPoint, SequentialTracer, TraceStatus, chief_ray_slopes, trace_from_object
from .sag_drawing import (
    beam_foci_from_paths,
    crossing_height,
    element_sections,
    extent_of,
    face_curve,
    lens_outline,
    mirror_arcs,
    polyhedron_section,
    section_offset,
    usable_extents,
)

FIELD_COLORS = ["#d62728", "#2878b5", "#1a6e3c", "#9467bd", "#8c564b"]

#: Wavelength at which a body's index is read for its colour when none is given (d line).
COLOR_WAVELENGTH_UM = 0.58756

#: Name-keyed ``(face, edge)`` colours of the README's DUV figure.
DEFAULT_MATERIAL_COLORS = {
    "SIO2": ("#bde0fe", "#2878b5"),
    "CAF2": ("#c7f9cc", "#278b65"),
    "HIINDEX1": ("#ffe29a", "#c47f00"),
    "HIINDEX2": ("#ffb3a1", "#c1440e"),
}

MIRROR_COLOR = "#202020"
SENSOR_COLOR = "#999999"
_THIN_STYLES = {
    Mirror: dict(color=MIRROR_COLOR, lw=2.6, zorder=3),
    BeamSplitter: dict(color="#555555", lw=1.8, ls="--", zorder=3),
    Stop: dict(color="#444444", lw=2.2, solid_capstyle="butt", zorder=3),
    Sensor: dict(color=SENSOR_COLOR, ls=":", lw=1.2, zorder=2),
}


class BodyOutline(NamedTuple):
    """One drawn body: element name, material, index, closed ``(z, y)`` polygon.

    ``extents`` are the drawn half-chords of a lens's front and back faces (``None`` for a prism); ``exact``
    is False when the pose is not meridional and the polygon is a projection.
    """

    name: str
    material: Material
    n: float
    polygon: np.ndarray
    extents: tuple[float, float] | None
    exact: bool


def body_outlines(system: OpticalSystem, *, wavelength_um: float | None = None, samples: int = 240) -> list[BodyOutline]:
    """Every body of *system* (lenses and prisms), interpenetrating neighbours cut apart.

    Where lens A's back face crosses lens B's front face (:func:`body_overlaps`), the space between the
    crossed faces above the cut is claimed by both, so it goes to *neither*: both crossed faces are trimmed to
    the cut. Each part keeps the rest of its unambiguous glass — A its whole front face while its rim stays
    in front of the region, B its back face over the band behind A's back face, up to where that band closes.
    A check that fails caps at the cut rather than over-claim. ``n`` is read at *wavelength_um* (d line).
    """

    caps = _cuts(system)
    outlines = []
    for element, pose in system.walk():
        lens_caps = (caps.get((element.name, "front")), caps.get((element.name, "back")))
        body = body_outline(element, pose, wavelength_um=wavelength_um, caps=lens_caps, samples=samples)
        if body is not None:
            outlines.append(body)
    return outlines


def body_outline(element, pose, *, wavelength_um=None, caps=(None, None), samples=240) -> BodyOutline | None:
    """The outline of a lens or prism at world *pose* (``None`` for other elements); *caps* trim a lens's faces."""

    n = medium_index(element.material, wavelength_um)
    if isinstance(element, Lens):
        polygon, extents, _ = lens_outline(element, pose, front_extent=caps[0], back_extent=caps[1], samples=samples)
        return BodyOutline(element.name, element.material, n, polygon, extents, section_offset(pose)[1])
    if isinstance(element, Prism):
        polygon, exact = polyhedron_section(element.solid, pose)
        if polygon.shape[0] >= 3:
            return BodyOutline(element.name, element.material, n, np.vstack([polygon, polygon[:1]]), None, exact)
    return None


def body_overlaps(system: OpticalSystem) -> list[tuple[str, str, float]]:
    """``[(upstream, downstream, h_cut), ...]``: neighbouring coaxial lenses that interpenetrate.

    Above ``h_cut`` the upstream back face lies behind the downstream front face; ``h_cut = 0`` means they
    collide on the axis. Lenses in contact (a cemented face) never overlap; only lenses on the z axis,
    unrotated, are compared.
    """

    lenses = sorted(
        ((e, p) for e, p in system.walk() if isinstance(e, Lens) and _coaxial(p)), key=lambda lp: float(lp[1].position[2])
    )
    overlaps = []
    for (a, pa), (b, pb) in zip(lenses, lenses[1:]):
        if any(f"{a.name}.back" in f.names and f"{b.name}.front" in f.names for f in system.faces):
            continue
        upper = min(usable_extents(a, pa)[1], usable_extents(b, pb)[0])
        crossing = _crossing(face_curve(a, "back", z0=float(pa.position[2])), face_curve(b, "front", z0=float(pb.position[2])), upper)
        if crossing is not None:
            overlaps.append((a.name, b.name, crossing))
    return overlaps


def _coaxial(pose) -> bool:
    return bool(np.allclose(pose.position[:2], 0.0, atol=1e-12) and pose.rotation.is_identity(1e-12))


def _crossing(early, late, upper):
    try:
        return crossing_height(early, late, upper)
    except ValueError:  # crossed on the axis
        return 0.0


def _cuts(system) -> dict[tuple[str, str], float]:
    """Half-chord caps ``{(lens, side): h}`` that cut interpenetrating neighbours apart (see :func:`body_outlines`)."""

    placed = {e.name: (e, p) for e, p in system.walk()}
    caps: dict[tuple[str, str], float] = {}

    def curve(name, side):
        lens, pose = placed[name]
        return face_curve(lens, side, z0=float(pose.position[2]))

    def at(name, side, h):
        return float(curve(name, side)(np.asarray([h]))[0])

    def usable(name, side):
        return usable_extents(*placed[name])[side == "back"]

    def cap(name, side, h):
        caps[name, side] = min(caps.get((name, side), np.inf), h)

    def clear(h0, z0, h1, z1, h_top, name, side, sign):
        """True when the chord (z0, h0)→(z1, h1) stays on the *sign* side of a face over [h0, h_top]."""
        hs = np.linspace(h0, h_top, 120)
        chord = z0 + (z1 - z0) * (hs - h0) / (h1 - h0)
        return bool((sign * (chord - curve(name, side)(hs)) >= -1e-9).all())

    for a, b, h_cut in body_overlaps(system):
        if h_cut <= 0.0:
            warnings.warn(f"lenses {a!r} and {b!r} collide on the axis; drawn uncut", stacklevel=3)
            continue
        cap(a, "back", h_cut)
        cap(b, "front", h_cut)

        # B keeps its back face over the band behind A's back face, up to where that band closes
        e_back = usable(b, "back")
        upper = min(usable(a, "back"), e_back)
        h_star = _crossing(curve(a, "back"), curve(b, "back"), upper)
        candidate = min(e_back if h_star is None else h_star, e_back)
        keep = h_cut
        if candidate > h_cut and clear(h_cut, at(b, "front", h_cut), candidate, at(b, "back", candidate),
                                       min(candidate, upper), a, "back", +1.0):
            keep = candidate
        cap(b, "back", keep)

        # A keeps its front face while its rim stays in front of the ambiguous region
        e_front = usable(a, "front")
        upper = min(e_front, usable(a, "back"), usable(b, "front"))
        if e_front > h_cut and upper > h_cut and not clear(
            h_cut, at(b, "front", h_cut), e_front, at(a, "front", e_front), upper, b, "front", -1.0
        ):
            cap(a, "front", h_cut)
    return caps


# -- colour ---------------------------------------------------------------------


def medium_index(material: Material | None, wavelength_um: float | None = None) -> float:
    """Refractive index of *material* for colouring (d line unless told otherwise)."""

    if material is None:
        return 1.0
    for wl in (wavelength_um, COLOR_WAVELENGTH_UM):
        if wl is not None:
            try:
                return float(material.index(wl))
            except (ValueError, TypeError):
                pass
    return 1.5


def glass_colors(material, n: float, overrides: dict | None = None) -> tuple[str, str] | None:
    """``(face, edge)`` of a body of *material*: a name-keyed override, else :func:`index_color`."""

    return (overrides or {}).get((getattr(material, "name", "") or "").upper()) or index_color(n)


def index_color(n: float, *, n_max: float = 2.0) -> tuple[str, str] | None:
    """``(face, edge)`` for a medium of index *n* on the celestial-blue ramp; None for n = 1.

    Sky blue for light media deepening toward the package's ``#2878b5``, saturating at ``n_max``.
    """

    if n <= 1.0 + 1e-9:
        return None
    t = min(max((n - 1.0) / (n_max - 1.0), 0.0), 1.0)

    def blend(b, level, a=(0xDF, 0xF1, 0xFF)):
        return "#" + "".join(f"{round(ca + (cb - ca) * level):02x}" for ca, cb in zip(a, b))

    return blend((0x28, 0x78, 0xB5), 0.25 + 0.75 * t), blend((0x1A, 0x52, 0x76), 0.45 + 0.55 * t)


# -- drawing on matplotlib axes ---------------------------------------------------


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
    """Meridional drawing of a system with traced ray fans; returns the traced paths ``(rays, points, 3)`` or None.

    *subject* is a :class:`~raytracer.tracing.SequentialTracer`, a stigmatic train, or a bare
    :class:`~raytracer.system.OpticalSystem` (drawn without fields). ``fields`` are object heights traced as
    fans of numerical aperture ``na`` aimed at the axial point ``z = aim_z`` — or centred on each field's
    chief ray through path step ``stop_index`` when given. With ``launch`` (a launch-plane z) each field is
    instead the sine of an aim angle, and converging fans aimed at the virtual object point start from that
    plane. Each field's foci are marked in its colour, with a note when the final one lies beyond ``xlim``;
    a coaxial aperture stop draws as a dotted vertical. ``material_colors`` overrides colours by medium name.
    """

    if isinstance(subject, SequentialTracer):
        tracer = subject
    else:
        tracer = SequentialTracer(subject) if isinstance(subject, OpticalSystem) else subject.tracer()
    system, object_z = tracer.system, tracer.object_z
    if fields and launch is None and object_z is None:
        raise ValueError("tracing fields needs the object plane: pass a SequentialTracer with object_z")

    beams = []
    for h, color in zip(fields, field_colors):
        if launch is None:
            if stop_index is not None:
                chief = chief_ray_slopes(tracer, FieldPoint(y=float(h)), stop_index=stop_index)[1]
            else:
                chief = (0.0 - h) / (aim_z - object_z)
            smax = na / np.sqrt(1.0 - na * na)
            fan = [trace_from_object(tracer, (0.0, h), (0.0, chief + s), keep_path=True) for s in np.linspace(-smax, smax, rays)]
        else:
            reference = object_z if object_z is not None else 0.0
            fan = [
                tracer.trace(np.array([0.0, (reference - launch) * t, launch]), np.array([0.0, -t, 1.0]) / np.hypot(t, 1.0), keep_path=True)
                for t in np.tan(np.arcsin(h)) + np.linspace(-0.04, 0.04, rays)
            ]
        beams.append((color, fan))

    paths = draw_traced(ax, tracer, beams, xlim=xlim, ray_style=dict(lw=0.55, alpha=0.65, zorder=2),
                        material_colors=material_colors, sensors=image_plane)
    if launch is None:
        for h, color in zip(fields, field_colors):
            ax.plot([object_z], [h], "o", color=color, ms=4, zorder=4)
    stop = stop_index if stop_index is not None else system.stop_index
    if stop is not None and _is_axial(system.path[stop].face.pose):
        ax.axvline(tracer.step_z(stop), color="#777777", ls=":", lw=0.8, zorder=0)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("z (mm)")
    ax.grid(alpha=0.15)
    return paths


def draw_traced(ax, tracer, beams, *, ray_style: dict, draw_failed: bool = False, xlim=None, **element_options):
    """Draw *tracer*'s system, then each beam ``(color, trace results)``: its rays and where it focuses.

    Mirrors follow the hits of the rays that got through; failed rays are drawn up to where they stopped when
    ``draw_failed``. Returns those rays' paths stacked, or None.
    """

    good = [[r.path for r in fan if r.ok] for _, fan in beams]
    paths = np.stack([p for bundle in good for p in bundle]) if any(good) else None
    drawn = draw_elements(ax, tracer.system, wavelength_um=tracer.wavelength_um, paths=paths, **element_options)
    behind = [r.failed_step for _, fan in beams for r in fan if r.status is TraceStatus.BEHIND]
    warn_outside_glass(tracer.system, paths, drawn, behind=behind, stacklevel=4)

    for color, fan in beams:
        for r in fan:
            if r.ok or (draw_failed and r.path is not None):
                finite = r.path[np.isfinite(r.path).all(axis=1)]
                ax.plot(finite[:, 2], finite[:, 1], color=color, **ray_style)
    beyond = False
    for (color, _), bundle in zip(beams, good):
        if not bundle:
            continue
        for z, y, _, _, is_final in beam_foci_from_paths(np.stack(bundle)):
            if xlim is not None and not xlim[0] <= z <= xlim[1]:
                beyond |= is_final
                continue
            ax.plot([z], [y], "o", ms=4.5, mfc="white", mec=color, mew=1.2, zorder=6)
    if beyond:
        ax.annotate(f"image at z = {tracer.image_z:g} mm", xy=(0.985, 0.04), xycoords="axes fraction", ha="right",
                    fontsize=8, style="italic", color="#666666")
    return paths


def draw_elements(ax, system, *, wavelength_um=None, material_colors=None, alpha=0.8, paths=None,
                  mirror_labels=False, sensors=True) -> dict[str, float]:
    """Draw every element of *system* on *ax*; return the drawn half-chord of each lens face by boundary name.

    Bodies fill by index (:func:`body_outlines`); mirrors span the sub-aperture the traced *paths* use when
    given (:func:`~raytracer.viz.sag_drawing.mirror_arcs`), their whole aperture otherwise; blades, beam
    splitters (dashed) and sensors (dotted) draw as their sections.
    """

    drawn = {}
    for body in body_outlines(system, wavelength_um=wavelength_um):
        colors = glass_colors(body.material, body.n, material_colors)
        if colors is not None:
            ax.fill(body.polygon[:, 0], body.polygon[:, 1], facecolor=colors[0], edgecolor=colors[1], lw=1.1, alpha=alpha, zorder=1)
            if body.extents is not None:
                drawn[f"{body.name}.front"], drawn[f"{body.name}.back"] = body.extents

    scale = extent_of(system.elements)

    def label(curve, name):
        if mirror_labels:
            ax.text(float(np.mean(curve[:, 0])), float(np.max(curve[:, 1])) + 0.04 * scale, name, ha="center", fontsize=9)

    arcs = mirror_arcs(system, paths) if paths is not None and len(paths) else []
    for step, curve in arcs:
        ax.plot(curve[:, 0], curve[:, 1], **_THIN_STYLES[Mirror])
        label(curve, step.face.name)
    with_arc = {step.face.name for step, _ in arcs}

    for element, pose in system.walk():
        style = _THIN_STYLES.get(type(element))
        skip = (isinstance(element, Mirror) and element.name in with_arc) or (isinstance(element, Sensor) and not sensors)
        if style is None or skip:
            continue
        if isinstance(element, Sensor) and element.aperture is None and isinstance(element.surface, Plane) and _is_axial(pose):
            ax.axvline(float(pose.position[2]), color=SENSOR_COLOR, ls=":", lw=0.8)
            continue
        for curve in element_sections(element, pose, fallback=scale):
            ax.plot(curve[:, 0], curve[:, 1], **style)
            if isinstance(element, Mirror):
                label(curve, element.name)
    return drawn


def _is_axial(pose) -> bool:
    """Perpendicular to the z axis and cut exactly: a plane there draws as a vertical line across the axes."""

    return abs(abs(float(pose.axis[2])) - 1.0) < 1e-12 and section_offset(pose)[1]


def warn_outside_glass(system, paths, drawn: dict[str, float], *, behind=(), stacklevel: int = 3) -> None:
    """Warn when traced rays meet a lens face beyond its drawn glass.

    A refraction is physical only where glass lies against the face; beyond a drawn part (past an inter-lens
    cut, or the usable branch of a face) the system defines no medium. *paths* are traced rays (or None);
    *behind* the path steps where rays stopped on a boundary they had already passed — which is what becomes
    of light that crosses into an overlap between two bodies.
    """

    problems = []
    for i, step in enumerate(system.path if paths is not None else ()):
        extents = [drawn[n] for n in step.face.names if n in drawn]
        if step.event != TRANSMIT or not extents:
            continue
        hits = np.asarray(paths, dtype=float)[:, i + 1, :]
        hits = hits[np.isfinite(hits).all(axis=1)]
        if hits.shape[0] == 0:
            continue
        local = step.face.pose.to_local(hits)
        reach = float(np.sqrt(np.max(local[:, 0] ** 2 + local[:, 1] ** 2)))
        limit = float(np.hypot(max(extents), section_offset(step.face.pose)[0]))
        if reach > limit * 1.001:
            problems.append(f"{step.face.name}: hits to h = {reach:.3f} mm, glass ends at {limit:.3f}")
    for i, count in sorted(Counter(behind).items()):
        problems.append(f"{system.path[i].face.name}: {count} rays reach it behind the boundary before it")
    if problems:
        warnings.warn(
            "rays refract beyond the drawn glass — outside the design's valid aperture, "
            f"the system defines no medium there ({'; '.join(problems)})",
            stacklevel=stacklevel,
        )


__all__ = [
    "FIELD_COLORS",
    "DEFAULT_MATERIAL_COLORS",
    "BodyOutline",
    "body_outlines",
    "body_outline",
    "body_overlaps",
    "medium_index",
    "glass_colors",
    "index_color",
    "draw_system",
    "draw_traced",
    "draw_elements",
    "warn_outside_glass",
]
