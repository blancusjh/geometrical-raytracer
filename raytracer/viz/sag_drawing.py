"""The meridional drawing plane: where the plane ``x = 0`` cuts each placed face, body and beam.

Every 2-D figure of the package is the meridional section of the 3-D model: the world plane ``x = 0``, drawn
with the optical axis ``z`` horizontal and the height ``y`` vertical. Drawing coordinates are ``(z, y)``.

A face lives in its element's frame (vertex at the origin, ``z = s(x, y)``) and is placed by a pose. When the
pose keeps the local x axis along the world x axis — every coaxial element, and every tilt or decentre within
the y-z plane — the plane ``x = 0`` is the local line ``x = u0``, and the curve ``(u0, v, s(u0, v))`` over the
chord of the face's aperture is the exact section, for any surface. For any other pose the local meridian
``(0, v, s(0, v))`` is drawn projected (its world x dropped); that is not a section, and
:func:`section_offset` says which case applies.

The figures, the neutral :class:`~raytracer.viz.scene.Drawing` and the 3-D viewer take their geometry from
here, so each fact is computed in one place.
"""

from __future__ import annotations

import numpy as np

from ..elements import Lens, Stop
from ..geometry.apertures import Annulus, Circle, Polygon, Rectangle
from ..system.faces import REFLECT

#: Half-length drawn for a face with no aperture (an unbounded sensor plane).
DEFAULT_SEMIDIAMETER = 50.0


def to_drawing(points) -> np.ndarray:
    """World points ``(..., 3)`` to drawing coordinates ``(..., 2)`` = ``(z, y)``."""

    return np.asarray(points, dtype=float)[..., [2, 1]]


def is_meridional(pose) -> bool:
    """True when the local x axis is the world ±x axis, so the drawing plane cuts the frame along ``x = u0``."""

    return abs(abs(float(pose.rotation.matrix[0, 0])) - 1.0) < 1e-9


def section_offset(pose) -> tuple[float, bool]:
    """``(u0, exact)``: the local x of the drawing plane, and whether the section is exact (else ``u0 = 0``, projected)."""

    if is_meridional(pose):
        return float(-pose.position[0] * pose.rotation.matrix[0, 0]), True
    return 0.0, False


def extent_of(elements) -> float:
    """Largest physical extent of any element: the length drawn for unbounded faces."""

    sizes = [
        e.semi_diameter if isinstance(e, Lens) else e.outer_radius if isinstance(e, Stop) else e.aperture.extent
        for e in elements
        if isinstance(e, (Lens, Stop)) or getattr(e, "aperture", None) is not None
    ]
    return max(sizes, default=DEFAULT_SEMIDIAMETER)


# -- faces ----------------------------------------------------------------------


def face_section(surface, pose, aperture, *, hole=None, fallback=DEFAULT_SEMIDIAMETER, samples=240) -> list[np.ndarray]:
    """The drawn curves ``[(M, 2), ...]`` of one placed face.

    One curve per chord of the aperture (less its *hole*, for a diaphragm blade) on the line ``x = u0``, each
    clamped to the surface's single-valued branch: a mirror gives one curve, a blade with its opening two.
    ``aperture=None`` is an unbounded face, drawn over ``±fallback``.
    """

    u0, _ = section_offset(pose)
    chords = aperture_chords(aperture, u0, fallback=fallback)
    if hole is not None:
        chords = _subtract(chords, aperture_chords(hole, u0, fallback=fallback))
    reach = usable_half_chord(surface, u0, np.inf)
    curves = []
    for lo, hi in chords:
        lo, hi = max(lo, -reach), min(hi, reach)
        if hi > lo:
            curves.append(to_drawing(face_points(surface, pose, u0, np.linspace(lo, hi, samples))))
    return curves


def element_sections(element, pose, *, fallback=DEFAULT_SEMIDIAMETER) -> list[np.ndarray]:
    """The curves of a thin element — mirror, beam splitter, diaphragm blade, sensor — at world *pose*.

    A virtual stop is a reference plane, not hardware, and draws nothing.
    """

    if getattr(element, "virtual", False):
        return []
    return [
        curve
        for face in element.faces()
        for curve in face_section(face.interface.surface, pose @ face.pose, face.extent, hole=face.hole, fallback=fallback)
    ]


def aperture_chords(aperture, u0: float, *, fallback=DEFAULT_SEMIDIAMETER, samples=2001) -> list[tuple[float, float]]:
    """The intervals of local ``y`` where the line ``x = u0`` lies in *aperture*.

    Exact for circles, annuli, rectangles and polygons; other shapes are sampled. ``None`` gives ``±fallback``.
    """

    if aperture is None:
        return [(-float(fallback), float(fallback))]
    cx, cy = aperture.center
    dx = u0 - cx
    if isinstance(aperture, Circle):
        if abs(dx) > aperture.radius:
            return []
        half = float(np.sqrt(aperture.radius**2 - dx * dx))
        return [(cy - half, cy + half)]
    if isinstance(aperture, Annulus):
        if abs(dx) > aperture.outer:
            return []
        outer = float(np.sqrt(aperture.outer**2 - dx * dx))
        chords = [(cy - outer, cy + outer)]
        if abs(dx) < aperture.inner:
            inner = float(np.sqrt(aperture.inner**2 - dx * dx))
            chords = _subtract(chords, [(cy - inner, cy + inner)])
        return chords
    if isinstance(aperture, Rectangle):
        if abs(dx) > aperture.width / 2:
            return []
        return [(cy - aperture.height / 2, cy + aperture.height / 2)]
    if isinstance(aperture, Polygon):
        crossings = sorted(
            y0 + (u0 - x0) * (y1 - y0) / (x1 - x0)
            for (x0, y0), (x1, y1) in zip(aperture.vertices, np.roll(aperture.vertices, -1, axis=0))
            if (x0 - u0) * (x1 - u0) < 0.0 or (x0 == u0 and x1 != u0)
        )
        return [(a, b) for a, b in zip(crossings[0::2], crossings[1::2]) if b > a]
    v = np.linspace(-aperture.extent, aperture.extent, samples)
    inside = aperture.contains(np.full_like(v, u0), v).astype(int)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], inside, [0]])))
    return [(float(v[a]), float(v[b - 1])) for a, b in zip(edges[0::2], edges[1::2])]


def _subtract(intervals, holes):
    for h_lo, h_hi in holes:
        pieces = []
        for a, b in intervals:
            if h_hi <= a or h_lo >= b:
                pieces.append((a, b))
                continue
            if h_lo > a:
                pieces.append((a, h_lo))
            if h_hi < b:
                pieces.append((h_hi, b))
        intervals = pieces
    return [(a, b) for a, b in intervals if b > a]


def usable_half_chord(surface, u0: float, limit: float) -> float:
    """*limit* clamped to the face's single-valued branch along the chord ``x = u0`` (a strong ovoid closes back)."""

    radius = float(surface.max_radius)
    if np.isfinite(radius):
        reach = radius * radius - u0 * u0
        limit = min(limit, float(np.sqrt(reach)) if reach > 0.0 else 0.0)
    return float(limit)


def face_points(surface, pose, u0: float, v, *, z_offset: float = 0.0) -> np.ndarray:
    """World points of the face over the local chord ``(u0, v)``, shifted by ``z_offset`` along local z."""

    v = np.asarray(v, dtype=float)
    u = np.full_like(v, u0)
    return pose.to_world(np.column_stack([u, v, z_offset + surface.sag(u, v)]))


# -- bodies -----------------------------------------------------------------------


def lens_outline(lens, pose=None, *, front_extent=None, back_extent=None, samples=240):
    """``(polygon, (front, back), joined)``: the closed ``(z, y)`` outline of a lens body at world *pose*.

    Each face runs over the semi-diameter's chord, clamped to its single-valued branch, so a ray refracting
    near the rim refracts on the drawn surface. Faces that cross below the rim (a negative edge thickness) close
    at the crossing height — the Ω join, and then *joined* is True; otherwise the rim's section closes the
    outline. ``front_extent`` / ``back_extent`` cap the half-chords further (the cuts between neighbours); the
    join is sought before they apply, so a cap on one face cannot hide where the faces meet. The front's
    ``samples`` points run up, the back's run down: point ``k`` pairs with point ``2·samples − 1 − k``
    (:func:`strip_faces`). *pose* defaults to the lens's own.
    """

    pose = lens.pose if pose is None else pose
    u0, _ = section_offset(pose)
    e_front, e_back = usable_extents(lens, pose)
    crossing = crossing_height(face_curve(lens, "front", u0), face_curve(lens, "back", u0), min(e_front, e_back), samples=samples)
    if crossing is not None:
        e_front = e_back = crossing
    e_front = e_front if front_extent is None else min(e_front, front_extent)
    e_back = e_back if back_extent is None else min(e_back, back_extent)
    joined = crossing is not None and e_front == e_back == crossing
    front = to_drawing(face_points(lens.front.surface, pose, u0, np.linspace(-e_front, e_front, samples)))
    back = to_drawing(face_points(lens.back.surface, pose, u0, np.linspace(e_back, -e_back, samples), z_offset=lens.thickness))
    return np.vstack([front, back, front[:1]]), (e_front, e_back), joined


def usable_extents(lens, pose) -> tuple[float, float]:
    """Half-chords of the front and back faces before any join or cut: the rim's chord, each face's branch."""

    u0, _ = section_offset(pose)
    half = float(np.sqrt(max(lens.semi_diameter**2 - u0 * u0, 0.0)))
    return usable_half_chord(lens.front.surface, u0, half), usable_half_chord(lens.back.surface, u0, half)


def face_curve(lens, side: str, u0: float = 0.0, z0: float = 0.0):
    """``v -> z`` of the ``"front"`` or ``"back"`` face of *lens* along its chord ``x = u0``, plus ``z0``."""

    surface = lens.front.surface if side == "front" else lens.back.surface
    z0 = z0 + (0.0 if side == "front" else lens.thickness)

    def curve(v):
        v = np.asarray(v, dtype=float)
        return z0 + surface.sag(np.full_like(v, u0), v)

    return curve


def crossing_height(early, late, upper: float, *, samples: int = 240) -> float | None:
    """Smallest ``h`` in ``(0, upper]`` where curve *late* falls behind *early* at ``±h``, refined by bisection.

    ``None`` when they keep apart; raises when they already cross on the axis.
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
        lo, hi = (mid, hi) if float(gap(np.asarray([mid]))[0]) > 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


def polyhedron_section(solid, pose) -> tuple[np.ndarray, bool]:
    """``(outline, exact)`` of a placed polyhedron: the convex hull of where the plane ``x = 0`` cuts its edges.

    Exact for a convex solid; a solid the plane misses is drawn as the hull of its projection (``exact`` False).
    """

    points, corners = [], facet_corners(solid, pose)
    for world in corners:
        for a, b in zip(world, np.roll(world, -1, axis=0)):
            if a[0] == 0.0:
                points.append(a)
            if a[0] * b[0] < 0.0:
                points.append(a + a[0] / (a[0] - b[0]) * (b - a))
    if len(points) >= 3:
        hull = convex_outline(to_drawing(np.asarray(points)))
        if hull.shape[0] >= 3:
            return hull, True
    return convex_outline(to_drawing(np.vstack(corners))), False


def facet_corners(solid, pose) -> list[np.ndarray]:
    """World corners ``(K, 3)`` of every facet of a polyhedron placed at *pose*."""

    return [
        (pose @ facet_pose).to_world(np.column_stack([np.asarray(outline.vertices), np.zeros(len(outline.vertices))]))
        for facet_pose, outline in solid.facets()
    ]


def convex_outline(points) -> np.ndarray:
    """Convex hull ``(M, 2)`` of 2-D points, counter-clockwise (Andrew's monotone chain)."""

    pts = np.unique(np.round(np.asarray(points, dtype=float), 12), axis=0)
    if pts.shape[0] < 3:
        return pts

    def half(sequence):
        chain = []
        for p in sequence:
            while len(chain) >= 2 and _turn(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
        return chain[:-1]

    return np.asarray(half(pts) + half(pts[::-1]))


def _turn(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def strip_faces(n_points: int) -> np.ndarray:
    """Triangles ``(K, 3)`` filling an outline laid out as by :func:`lens_outline` (a closing point ignored)."""

    m = n_points // 2
    faces = np.empty((2 * (m - 1), 3), dtype=np.uint32)
    i = np.arange(m - 1, dtype=np.uint32)
    back = np.uint32(2 * m - 1) - i
    faces[0::2] = np.column_stack([i, i + 1, back])
    faces[1::2] = np.column_stack([i + 1, back - 1, back])
    return faces


def fan_faces(n_points: int) -> np.ndarray:
    """Triangles of a convex polygon of *n_points* vertices (a fan from vertex 0)."""

    k = np.arange(1, n_points - 1, dtype=np.uint32)
    return np.column_stack([np.zeros_like(k), k, k + 1]).astype(np.uint32)


# -- traced beams -------------------------------------------------------------------


def mirror_arcs(system, paths, *, pad: float = 0.06, samples: int = 200) -> list:
    """``[(step, curve (M, 2)), ...]``: each reflecting step's arc over the part of its face the rays use.

    An off-axis mirror (a ring field, a folded path) is used away from its vertex, so its whole cap would put
    curve where no light goes. *paths* are sequential ``keep_path`` traces ``(N, n_steps + 1, 3)``; the arc
    spans their hits along the face's local y, extended by *pad* of that span.
    """

    paths = np.asarray(paths, dtype=float)
    arcs = []
    for i, step in enumerate(system.path):
        hits = paths[:, i + 1, :]
        hits = hits[np.isfinite(hits).all(axis=1)]
        if step.event != REFLECT or hits.shape[0] == 0:
            continue
        pose = step.face.pose
        v_hits = pose.to_local(hits)[:, 1]
        lo, hi = float(v_hits.min()), float(v_hits.max())
        margin = pad * max(hi - lo, 1.0)
        v = np.linspace(lo - margin, hi + margin, samples)
        arcs.append((step, to_drawing(face_points(step.face.surface, pose, section_offset(pose)[0], v))))
    return arcs


def beam_foci_from_paths(paths, *, pinch_ratio: float = 0.2, min_rays: int = 3) -> list:
    """``[(z, y, blur_rms, leg, is_final), ...]``: where a traced beam focuses, leg by leg, in the drawing plane.

    On each leg between boundaries the rays are straight lines; the focus is the point closest to all of them
    in the least-squares sense (perpendicular distances, so a leg along y after a fold is like one along z).
    An intermediate leg counts only for a genuine pinch inside the leg — blur below ``pinch_ratio`` of the
    beam's spread at both ends — and reports where the traced beam actually pinches, not the nominal plane.
    The final leg reports any real convergence on or beyond the last boundary; a collimated exit (afocal) or
    a virtual focus reports nothing. *paths* are ``keep_path`` traces ``(rays, points, 3)``.
    """

    paths = np.asarray(paths, dtype=float)
    foci = []
    n_legs = paths.shape[1] - 1
    for leg in range(1, n_legs):
        p0, p1 = to_drawing(paths[:, leg, :]), to_drawing(paths[:, leg + 1, :])
        length = np.linalg.norm(p1 - p0, axis=1)
        keep = np.isfinite(length) & (length > 1e-12)
        if keep.sum() < min_rays:
            continue
        a, b, d = p0[keep], p1[keep], (p1 - p0)[keep] / length[keep, None]
        along = d.mean(axis=0) / np.linalg.norm(d.mean(axis=0))
        across = np.array([-along[1], along[0]])
        normals = np.column_stack([-d[:, 1], d[:, 0]])
        m = np.einsum("ij,ik->jk", normals, normals)  # sum of n nᵀ (x − a) = 0, least squares
        if np.linalg.cond(m) > 1e15 or np.var(d @ across) < 1e-18:
            continue  # collimated leg
        star = np.linalg.solve(m, np.einsum("ij,ik,ik->j", normals, normals, a))
        blur = float(np.sqrt(np.mean(np.einsum("ij,ij->i", normals, star - a) ** 2)))
        is_final = leg == n_legs - 1
        s0, s1, s_star = a @ along, b @ along, float(star @ along)
        s_lo, s_hi = float(np.minimum(s0, s1).min()), float(np.maximum(s0, s1).max())
        if is_final:
            if s_star < s_lo:  # virtual focus
                continue
        else:
            inside = s_lo + 0.01 * (s_hi - s_lo) <= s_star <= s_hi - 0.01 * (s_hi - s_lo)
            spread = min(float(np.std(a @ across)), float(np.std(b @ across)))
            if not inside or blur > pinch_ratio * max(spread, 1e-12):
                continue
        foci.append((float(star[0]), float(star[1]), blur, leg, is_final))
    return foci


__all__ = [
    "DEFAULT_SEMIDIAMETER",
    "to_drawing",
    "is_meridional",
    "section_offset",
    "extent_of",
    "face_section",
    "element_sections",
    "aperture_chords",
    "usable_half_chord",
    "face_points",
    "lens_outline",
    "usable_extents",
    "face_curve",
    "crossing_height",
    "polyhedron_section",
    "facet_corners",
    "convex_outline",
    "strip_faces",
    "fan_faces",
    "mirror_arcs",
    "beam_foci_from_paths",
]
