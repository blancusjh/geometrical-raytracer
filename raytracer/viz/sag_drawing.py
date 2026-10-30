"""The meridional drawing plane: sections of placed faces, mirror arcs, beam foci.

Every 2-D figure of the package is the **meridional section** of the 3-D
model: the plane ``x = 0`` of the world, drawn with the optical axis ``z``
horizontal and the height ``y`` vertical. Drawing coordinates are therefore
``(z, y)`` throughout this package.

A face lives in its element's local frame (vertex at the origin, axis +z,
``z = s(x, y)``) and is placed in the world by a pose. Where the plane
``x = 0`` meets it:

- **Exact section.** When the pose keeps the local x axis along the world x
  axis — every coaxial element, and every element tilted or decentred
  *within* the y-z plane (a fold mirror, a tilted lens, a decentred group) —
  the plane ``x = 0`` is the local plane ``x = u0`` with ``u0`` fixed by the
  pose. The section is then the curve ``(u0, v, s(u0, v))`` over the chord
  of the face's aperture at ``x = u0``, mapped to the world: exact, for any
  surface, revolution-symmetric or not.
- **Projection.** For any other pose (an element turned about the world y
  axis, or out of the plane), the local meridian ``(0, v, s(0, v))`` is
  mapped to the world and *projected* onto the drawing plane (its ``x``
  dropped). That is what the figure shows for such elements; it is not a
  section, and :func:`is_meridional` says which case applies.

The same helpers serve the matplotlib figures (:mod:`raytracer.viz.bodies`,
:mod:`raytracer.viz.plots`), the neutral :class:`~raytracer.viz.scene.Drawing`
and the OpenGL viewers, so a fix (sampling, aperture chords, the usable
branch of a Cartesian face) happens once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..elements import Lens
from ..geometry.apertures import Annulus, Aperture, Circle, Polygon, Rectangle
from ..geometry.transforms import Pose
from ..system.faces import REFLECT

if TYPE_CHECKING:  # pragma: no cover
    from ..geometry.surfaces import Surface
    from ..system import OpticalSystem

#: Half-length drawn for a face with no aperture (an unbounded sensor plane).
DEFAULT_SEMIDIAMETER = 50.0

_ALIGN_TOL = 1e-9


# -- the drawing plane --------------------------------------------------------


def to_drawing(points: np.ndarray) -> np.ndarray:
    """World points ``(..., 3)`` to drawing coordinates ``(..., 2)`` = ``(z, y)``."""

    points = np.asarray(points, dtype=float)
    return points[..., [2, 1]]


def is_meridional(pose: Pose) -> bool:
    """True when the plane ``x = 0`` cuts this frame along a local line ``x = u0``.

    That holds exactly when the local x axis is the world ±x axis: coaxial
    poses, and every tilt or decentre within the y-z plane. Faces placed by
    such a pose are drawn as exact sections; any other pose is projected.
    """

    return abs(abs(float(pose.rotation.matrix[0, 0])) - 1.0) < _ALIGN_TOL


def section_offset(pose: Pose) -> tuple[float, bool]:
    """``(u0, exact)``: the local x of the drawing plane, and whether the section is exact.

    For a non-meridional pose, ``u0 = 0`` (the element's own meridian, to be
    projected) and ``exact`` is False.
    """

    if is_meridional(pose):
        r = pose.rotation.matrix
        return float(-pose.position[0] * r[0, 0]), True
    return 0.0, False


# -- apertures cut by the line x = u0 ----------------------------------------


def _subtract(intervals, holes):
    out = []
    for lo, hi in intervals:
        pieces = [(lo, hi)]
        for h_lo, h_hi in holes:
            nxt = []
            for a, b in pieces:
                if h_hi <= a or h_lo >= b:
                    nxt.append((a, b))
                    continue
                if h_lo > a:
                    nxt.append((a, h_lo))
                if h_hi < b:
                    nxt.append((h_hi, b))
            pieces = nxt
        out.extend(pieces)
    return [(a, b) for a, b in out if b > a]


def aperture_chords(
    aperture: Aperture | None,
    u0: float,
    *,
    fallback: float = DEFAULT_SEMIDIAMETER,
    samples: int = 2001,
) -> list[tuple[float, float]]:
    """The intervals of local ``y`` where the line ``x = u0`` lies in *aperture*.

    Exact for circles, annuli, rectangles and polygons; any other aperture
    is sampled. ``None`` (an unbounded face) gives ``[-fallback, fallback]``.
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
        v = np.asarray(aperture.vertices)
        crossings = []
        for (x0, y0), (x1, y1) in zip(v, np.roll(v, -1, axis=0)):
            if (x0 - u0) * (x1 - u0) < 0.0 or (x0 == u0 and x1 != u0):
                crossings.append(y0 + (u0 - x0) * (y1 - y0) / (x1 - x0))
        crossings.sort()
        return [(a, b) for a, b in zip(crossings[0::2], crossings[1::2]) if b > a]
    extent = aperture.extent
    v = np.linspace(-extent, extent, samples)
    inside = aperture.contains(np.full_like(v, u0), v)
    chords, start = [], None
    for k, flag in enumerate(inside):
        if flag and start is None:
            start = k
        if (not flag or k == len(v) - 1) and start is not None:
            end = k if flag else k - 1
            chords.append((float(v[start]), float(v[end])))
            start = None
    return chords


def usable_half_chord(surface: "Surface", u0: float, limit: float) -> float:
    """*limit* clamped to the face's single-valued branch along the chord ``x = u0``."""

    radius = float(surface.max_radius)
    if np.isfinite(radius):
        reach = radius * radius - u0 * u0
        limit = min(limit, float(np.sqrt(reach)) if reach > 0.0 else 0.0)
    return float(limit)


# -- sections -----------------------------------------------------------------


def face_points(surface: "Surface", pose: Pose, u0: float, v: np.ndarray, *, z_offset: float = 0.0) -> np.ndarray:
    """World points of the face over the local chord ``(u0, v)``; ``z_offset`` shifts it along local z."""

    v = np.asarray(v, dtype=float)
    u = np.full_like(v, u0)
    local = np.column_stack([u, v, z_offset + surface.sag(u, v)])
    return pose.to_world(local)


def face_section(
    surface: "Surface",
    pose: Pose,
    aperture: Aperture | None,
    *,
    hole: Aperture | None = None,
    fallback: float = DEFAULT_SEMIDIAMETER,
    samples: int = 240,
) -> list[np.ndarray]:
    """The drawn curves ``[(M, 2) (z, y), ...]`` of one placed face.

    The face's aperture (less its *hole*, for a diaphragm blade) is cut by
    the drawing plane; each resulting chord, clamped to the surface's
    single-valued branch, is one curve. A circular mirror gives one curve, a
    blade with its opening gives two. Non-meridional poses are projected
    (see the module docstring).
    """

    u0, _ = section_offset(pose)
    chords = aperture_chords(aperture, u0, fallback=fallback)
    if hole is not None:
        chords = _subtract(chords, aperture_chords(hole, u0, fallback=fallback))
    curves = []
    radius = float(surface.max_radius)
    for lo, hi in chords:
        if np.isfinite(radius):
            reach = usable_half_chord(surface, u0, max(abs(lo), abs(hi)))
            lo, hi = max(lo, -reach), min(hi, reach)
            if hi <= lo:
                continue
        v = np.linspace(lo, hi, samples)
        curves.append(to_drawing(face_points(surface, pose, u0, v)))
    return curves


def lens_section(
    lens: Lens,
    pose: Pose | None = None,
    *,
    front_extent: float,
    back_extent: float,
    samples: int = 240,
) -> np.ndarray:
    """Closed ``(z, y)`` outline of a lens body with given face extents.

    Each extent is the half-length of that face's chord in the drawing
    plane. The layout is fixed — ``samples`` front-face points from
    ``-front_extent`` to ``+front_extent``, then ``samples`` back-face points
    from ``+back_extent`` to ``-back_extent``, then the first point again —
    so point ``k`` of the front and point ``2·samples − 1 − k`` of the back
    pair up for a strip triangulation (:func:`strip_faces`). The straight
    edges between the two faces are the section of the cylindrical rim.
    """

    pose = lens.pose if pose is None else pose
    u0, _ = section_offset(pose)
    vf = np.linspace(-front_extent, front_extent, samples)
    vb = np.linspace(back_extent, -back_extent, samples)
    front = to_drawing(face_points(lens.front.surface, pose, u0, vf))
    back = to_drawing(face_points(lens.back.surface, pose, u0, vb, z_offset=lens.thickness))
    return np.vstack([front, back, front[:1]])


def strip_faces(n_points: int) -> np.ndarray:
    """Triangles ``(K, 3)`` filling an outline laid out as by :func:`lens_section`.

    *n_points* counts the outline's points (a closing duplicate is ignored):
    the first half runs along one face, the second half back along the
    other, so vertex ``i`` and vertex ``2m − 1 − i`` pair into a quad strip.
    """

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


def lens_outline(lens: Lens, pose: Pose | None = None, *, samples: int = 220) -> np.ndarray:
    """Closed ``(z, y)`` outline of *lens* out to its semi-diameter (no inter-lens cuts).

    Each face is clamped to its usable branch; :func:`raytracer.viz.bodies.element_outline`
    adds the Ω join of crossing faces and the cuts between neighbours.
    """

    pose = lens.pose if pose is None else pose
    u0, _ = section_offset(pose)
    half = float(np.sqrt(max(lens.semi_diameter**2 - u0 * u0, 0.0)))
    return lens_section(
        lens,
        pose,
        front_extent=usable_half_chord(lens.front.surface, u0, half),
        back_extent=usable_half_chord(lens.back.surface, u0, half),
        samples=samples,
    )


def lens_fill_mesh(lens: Lens, pose: Pose | None = None, *, samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Triangulated body ``(verts (M, 2), faces (K, 3))`` of *lens* in the drawing plane."""

    outline = lens_outline(lens, pose, samples=samples)[:-1]
    return outline, strip_faces(outline.shape[0])


def convex_outline(points: np.ndarray) -> np.ndarray:
    """Convex hull ``(M, 2)`` of 2-D points, counter-clockwise (Andrew's monotone chain)."""

    pts = np.unique(np.round(np.asarray(points, dtype=float), 12), axis=0)
    if pts.shape[0] < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def polyhedron_section(solid, pose: Pose) -> tuple[np.ndarray, bool]:
    """``(outline, exact)`` of a placed polyhedron (a prism) in the drawing plane.

    The plane ``x = 0`` cuts every facet edge it crosses; the convex hull of
    those points is the section — exact for a convex solid. A solid the
    plane misses is drawn as the convex hull of its projection (``exact``
    False).
    """

    points, corners = [], []
    for facet_pose, outline in solid.facets():
        world = (pose @ facet_pose).to_world(np.column_stack([np.asarray(outline.vertices), np.zeros(len(outline.vertices))]))
        corners.append(world)
        for a, b in zip(world, np.roll(world, -1, axis=0)):
            if a[0] == 0.0:
                points.append(a)
            if a[0] * b[0] < 0.0:
                t = a[0] / (a[0] - b[0])
                points.append(a + t * (b - a))
    if len(points) >= 3:
        hull = convex_outline(to_drawing(np.asarray(points)))
        if hull.shape[0] >= 3:
            return hull, True
    return convex_outline(to_drawing(np.vstack(corners))), False


# -- faces of a system ---------------------------------------------------------


def _face_by_name(system: "OpticalSystem", key):
    if isinstance(key, (int, np.integer)):
        return system.path[int(key)].face
    return system.face(str(key))


def sample_profile_curve(
    system: "OpticalSystem",
    face: int | str,
    *,
    semidiameter: float | None = None,
    samples: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    """Open drawing-plane curve ``(z, y)`` of one boundary of *system*.

    *face* is a path index (``system.path[i]``) or a boundary name
    (``"L3.front"``, ``"M1"``). The curve runs over the face's clear extent,
    or ``±semidiameter`` when given.
    """

    f = _face_by_name(system, face)
    region = Circle(float(semidiameter)) if semidiameter is not None else f.clear_aperture
    curves = face_section(f.surface, f.pose, region, samples=samples)
    if not curves:
        return np.empty(0), np.empty(0)
    curve = max(curves, key=len)
    return curve[:, 0], curve[:, 1]


def mirror_arcs_from_paths(
    system: "OpticalSystem",
    paths,
    *,
    pad: float = 0.06,
    samples: int = 200,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Per-mirror drawing-plane arcs spanning the part of each mirror the rays use.

    For off-axis systems (ring fields, folded paths) a mirror's *used*
    sub-aperture is displaced from its vertex, so drawing the whole cap puts
    curve where no light goes. This derives each reflecting step's arc from
    the traced *paths* themselves (sequential ``keep_paths`` output, shape
    ``(N, n_steps + 1, 3)``: column ``i + 1`` is the hit at path step ``i``),
    extended by *pad* (a fraction of the hit span) beyond the hits, along
    the face's local y in the drawing plane.

    Returns ``[(mirror_number, z, y), ...]`` in path order, one per
    reflecting step, ready for ``ax.plot(z, y)``.
    """

    paths = np.asarray(paths, dtype=float)
    arcs: list[tuple[int, np.ndarray, np.ndarray]] = []
    count = 0
    for i, step in enumerate(system.path):
        if step.event != REFLECT:
            continue
        count += 1
        hits = paths[:, i + 1, :]
        hits = hits[np.isfinite(hits).all(axis=1)]
        if hits.shape[0] == 0:
            continue
        face = step.face
        u0, _ = section_offset(face.pose)
        v_hits = face.pose.to_local(hits)[:, 1]
        lo, hi = float(v_hits.min()), float(v_hits.max())
        margin = pad * max(hi - lo, 1.0)
        v = np.linspace(lo - margin, hi + margin, samples)
        zy = to_drawing(face_points(face.surface, face.pose, u0, v))
        arcs.append((count, zy[:, 0], zy[:, 1]))
    return arcs


# -- where a traced beam focuses ----------------------------------------------


def beam_foci_from_paths(
    paths,
    *,
    pinch_ratio: float = 0.2,
    min_rays: int = 3,
) -> list[tuple[float, float, float, int, bool]]:
    """Where a traced beam actually focuses, leg by leg, in the drawing plane.

    Between consecutive boundaries every ray is a straight line; the beam's
    focus on that leg is the point of the drawing plane closest, in the
    least-squares sense, to all those lines (perpendicular distances, so a
    leg running along y after a fold mirror is handled like one along z).
    An *intermediate* leg counts as a focus only when the pinch is genuine
    — residual blur below ``pinch_ratio`` of the beam's spread at both ends
    of the leg (a constriction of 5× or better at the default) and the
    point inside the leg — so mere narrowing is not decorated. An aberrated
    intermediate image reports where the traced beam *actually* pinches,
    which need not be the design's nominal plane. The *final* leg (to the
    detector) is where the system claims to focus, and is reported whenever
    the beam converges to a real point on or beyond the last boundary; a
    collimated exit (afocal system) or a virtual focus reports nothing.

    Returns ``[(z, y, blur_rms, leg_index, is_final), ...]`` for *paths* of
    shape ``(rays, points, 3)`` from ``keep_path`` traces.
    """

    paths = np.asarray(paths, dtype=float)
    foci: list[tuple[float, float, float, int, bool]] = []
    n_legs = paths.shape[1] - 1
    for leg in range(1, n_legs):
        p0 = to_drawing(paths[:, leg, :])
        p1 = to_drawing(paths[:, leg + 1, :])
        delta = p1 - p0
        length = np.linalg.norm(delta, axis=1)
        keep = np.isfinite(length) & (length > 1e-12)
        if keep.sum() < min_rays:
            continue
        a, d = p0[keep], delta[keep] / length[keep, None]
        mean_dir = d.mean(axis=0)
        mean_dir /= np.linalg.norm(mean_dir)
        normals = np.column_stack([-d[:, 1], d[:, 0]])
        # sum over rays of n n^T (x - a) = 0, solved in the least-squares sense
        m = np.einsum("ij,ik->jk", normals, normals)
        rhs = np.einsum("ij,ik,ik->j", normals, normals, a)
        if np.linalg.cond(m) > 1e15 or np.var(d @ np.array([-mean_dir[1], mean_dir[0]])) < 1e-18:
            continue  # collimated leg: no focus
        star = np.linalg.solve(m, rhs)
        distances = np.einsum("ij,ij->i", normals, star - a)
        blur = float(np.sqrt(np.mean(distances**2)))
        is_final = leg == n_legs - 1
        across = np.array([-mean_dir[1], mean_dir[0]])
        s0, s1 = a @ mean_dir, p1[keep] @ mean_dir
        s_star = float(star @ mean_dir)
        s_lo = float(np.minimum(s0, s1).min())
        s_hi = float(np.maximum(s0, s1).max())
        if is_final:
            if s_star < s_lo:  # virtual focus behind the last boundary
                continue
        else:
            span = s_hi - s_lo
            inside = s_lo + 0.01 * span <= s_star <= s_hi - 0.01 * span
            spread = min(float(np.std(a @ across)), float(np.std(p1[keep] @ across)))
            if not inside or blur > pinch_ratio * max(spread, 1e-12):
                continue
        foci.append((float(star[0]), float(star[1]), blur, leg, is_final))
    return foci


__all__ = [
    "DEFAULT_SEMIDIAMETER",
    "to_drawing",
    "is_meridional",
    "section_offset",
    "aperture_chords",
    "usable_half_chord",
    "face_points",
    "face_section",
    "lens_section",
    "lens_outline",
    "lens_fill_mesh",
    "strip_faces",
    "fan_faces",
    "convex_outline",
    "polyhedron_section",
    "sample_profile_curve",
    "mirror_arcs_from_paths",
    "beam_foci_from_paths",
]
