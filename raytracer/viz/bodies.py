"""Closed lens bodies deduced from the media description, colored by density.

The standard description of a sequential system alternates media and
surfaces — ``Air → Σ1 → Glass → Σ2 → Air`` — so which surface pair encloses
which medium is *deducible*, not declarable: every inter-surface gap whose
medium has index above 1 is a body bounded by the two surfaces around it
(:func:`dense_gaps`; a cemented doublet's three glasses fall out as three
bodies sharing interfaces). This module turns that deduction into drawing:

- :func:`element_outline` builds the closed meridional polygon of one body.
  Two strongly curved faces can *cross* below the clear aperture; the
  outline then joins them at the intersection height found by bisection —
  the Ω join of the author's ``cartesian-surfaces-stl-generator``
  (``intersection_between_curves``), which computes exactly this meeting of
  two Σ curves. Faces that never meet are closed by a flat rim at the
  common semidiameter. Profiles exposing ``max_usable_height`` (the
  Cartesian ovals) are clamped to their single-valued branch, so an
  outline never wanders onto the closing branch of an ovoid.
- :func:`index_color` maps a refractive index to a fill color: nothing for
  n = 1, then a blue ramp that darkens with n — the denser the medium, the
  darker the body, automatically, for any system read from its standard
  description.
- :func:`draw_system` renders a system or stigmatic train on a matplotlib
  axes with those bodies, bare surfaces (mirrors and unpaired refracting
  faces), meridional ray fans, and the image plane — the shared meridional
  renderer for figures and notebooks.
"""

from __future__ import annotations

import numpy as np

from ..design.rows import SurfaceKind
from ..design.system import OpticalSystem
from ..propagation.fields import trace_from_object
from ..propagation.sequential import SequentialTracer
from .sag_drawing import surface_semidiameter

FIELD_COLORS = ["#d62728", "#2878b5", "#1a6e3c", "#9467bd", "#8c564b"]


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


def element_outline(
    system: OpticalSystem,
    i: int,
    j: int,
    *,
    semidiameter: float | None = None,
    samples: int = 240,
    fallback_semidiameter: float = 50.0,
) -> tuple[np.ndarray, float, bool]:
    """Closed ``(z, h)`` polygon of the body between faces *i* and *j*.

    Returns ``(polygon, edge_height, joined)``: *joined* is True when the
    faces intersect below the aperture and the outline closes at that sharp
    edge (found by bisection on ``z_j(h) - z_i(h)``); False when it closes
    with a flat rim at the common semidiameter.
    """

    if semidiameter is None:
        semidiameter = min(
            _usable_semidiameter(system, i, fallback_semidiameter),
            _usable_semidiameter(system, j, fallback_semidiameter),
        )

    def front(h):
        return system.vertices[i] + system.rows[i].profile.sag(np.abs(h))

    def back(h):
        return system.vertices[j] + system.rows[j].profile.sag(np.abs(h))

    def gap(h):
        return float(np.ravel(back(np.asarray(h)) - front(np.asarray(h)))[0])

    edge, joined = float(semidiameter), False
    if gap(0.0) <= 0.0:
        raise ValueError(
            f"faces {i} and {j} already cross on the axis "
            f"(z_{j}(0) <= z_{i}(0)); not a physical body"
        )
    probe = np.linspace(0.0, semidiameter, samples)
    gaps = back(probe) - front(probe)
    crossing = np.nonzero(gaps <= 0.0)[0]
    if crossing.size:
        lo, hi = probe[crossing[0] - 1], probe[crossing[0]]
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if gap(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        edge, joined = 0.5 * (lo + hi), True

    h = np.linspace(-edge, edge, samples)
    front_curve = np.column_stack([front(h), h])
    back_curve = np.column_stack([back(h), h])[::-1]
    polygon = np.vstack([front_curve, back_curve, front_curve[:1]])
    return polygon, edge, joined


def index_color(n: float, *, n_max: float = 2.0) -> tuple[str, str] | None:
    """``(face, edge)`` colors for a medium of index *n*; None for n = 1.

    A deterministic blue ramp: the denser the medium, the darker the body.
    Media at or beyond ``n_max`` saturate at the darkest shade.
    """

    if n <= 1.0 + 1e-9:
        return None
    t = min(max((n - 1.0) / (n_max - 1.0), 0.0), 1.0)

    def shade(level):
        # light #e3eefb (227,238,251) -> deep #16365f (22,54,95)
        r = round(227 + (22 - 227) * level)
        g = round(238 + (54 - 238) * level)
        b = round(251 + (95 - 251) * level)
        return f"#{r:02x}{g:02x}{b:02x}"

    return shade(0.15 + 0.75 * t), shade(min(0.35 + 0.75 * t, 1.0))


def draw_system(
    ax,
    system_or_train,
    *,
    fields=(),
    na: float = 0.1,
    rays: int = 7,
    aim_z: float = 0.0,
    launch: float | None = None,
    field_colors=FIELD_COLORS,
    material_colors: dict | None = None,
    xlim=None,
    ylim=None,
    image_plane: bool = True,
) -> None:
    """Standard meridional rendering: bodies, surfaces, ray fans, image plane.

    ``fields`` are object heights traced as cones of numerical aperture
    ``na`` aimed at the axial point ``z = aim_z`` — except in virtual-object
    mode (``launch`` set to a launch-plane z in front of the lens), where
    each field value is the *sine of an aim angle* and converging fans aimed
    at the virtual object point are launched from that plane.

    Bodies are deduced from the media sequence and colored by
    :func:`index_color`; ``material_colors`` (name-keyed ``(face, edge)``
    pairs) overrides individual media. Mirrors and unpaired refracting
    surfaces draw as bare curves.
    """

    system = (
        system_or_train
        if isinstance(system_or_train, OpticalSystem)
        else system_or_train.to_system()
    )
    tracer = SequentialTracer(system)
    object_z = system.object_z

    drawn_faces: set[int] = set()
    for i, j, n, name in dense_gaps(system):
        polygon, _, _ = element_outline(system, i, j)
        colors = (material_colors or {}).get(name) or index_color(n)
        if colors is None:
            continue
        face, edge = colors
        ax.fill(polygon[:, 0], polygon[:, 1], facecolor=face, edgecolor=edge,
                lw=1.1, alpha=0.75, zorder=1)
        drawn_faces.update((i, j))

    for i, row in enumerate(system.rows):
        if row.kind is SurfaceKind.MIRROR:
            semi = _usable_semidiameter(system, i, 50.0)
            h = np.linspace(-semi, semi, 200)
            z = system.vertices[i] + row.profile.sag(np.abs(h))
            ax.plot(z, h, color="#202020", lw=2.6, zorder=3)
        elif i not in drawn_faces and row.kind is SurfaceKind.REFRACT:
            if float(system.n_before[i]) == float(system.n_after[i]):
                continue
            semi = _usable_semidiameter(system, i, 50.0)
            h = np.linspace(-semi, semi, 200)
            z = system.vertices[i] + row.profile.sag(np.abs(h))
            ax.plot(z, h, color="#3a6ea5", lw=1.6, zorder=2)

    smax = na / np.sqrt(1.0 - na * na)
    for h_field, color in zip(fields, field_colors):
        if launch is None:
            chief = (0.0 - h_field) / (aim_z - object_z)
            for s in np.linspace(-smax, smax, rays):
                r = trace_from_object(tracer, (0.0, h_field), (0.0, chief + s),
                                      keep_path=True)
                if r.ok:
                    ax.plot(r.path[:, 2], r.path[:, 1], color=color, lw=0.55,
                            alpha=0.65, zorder=2)
            ax.plot([object_z], [h_field], "o", color=color, ms=4, zorder=4)
        else:
            t0 = np.tan(np.arcsin(h_field))
            for dt in np.linspace(-0.04, 0.04, rays):
                t = t0 + dt
                origin = np.array([0.0, (object_z - launch) * t, launch])
                direction = np.array([0.0, -t, 1.0]) / np.hypot(t, 1.0)
                r = tracer.trace(origin, direction, keep_path=True)
                if r.ok:
                    path = np.vstack([origin, r.path[1:]])
                    ax.plot(path[:, 2], path[:, 1], color=color, lw=0.55,
                            alpha=0.65, zorder=2)

    if image_plane:
        ax.axvline(system.image_z, color="#999999", ls=":", lw=0.8)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("z (mm)")
    ax.grid(alpha=0.15)


__all__ = [
    "FIELD_COLORS",
    "dense_gaps",
    "draw_system",
    "element_outline",
    "index_color",
]
