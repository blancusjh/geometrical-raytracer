"""Bridge between an engineered design and the 2-D branching propagation.

The one function here legitimately depends on both: it reinterprets a
sequential :class:`~raytracer.design.system.OpticalSystem` as the 2-D
``Surface`` objects (:mod:`raytracer.optics`) the branching propagation
needs to explore it non-sequentially (caustics, stray light, ...).
"""

from __future__ import annotations

import numpy as np

from ..optics.profiles import ProfileSurface
from ..optics.surface import Surface
from .surfaces import SurfaceKind


def to_branching_surfaces(system) -> list[Surface]:
    """Meridional (y-z) slice of a sequential system as 2-D surfaces.

    The sequential z axis maps to 2-D x. Every refractive row becomes a
    ``ProfileSurface`` carrying its before/after indices; mirrors become
    reflective faces; stops are skipped (pure markers). Double-passed
    surfaces on folded return paths (duplicated rows with matching vertex
    and radius) collapse to a single physical face.

    Caveat: this bridge is exact for dioptric (unfolded) systems. In a
    folded catadioptric, the branching propagation will let the forward
    beam interact with any mirror whose full aperture crosses its path —
    which sequential ordering deliberately ignores. Clip mirror
    semidiameters to the physically used sub-aperture in that case, or
    analyse folded systems with the sequential propagation instead.
    """

    surfaces: list[Surface] = []
    seen: set[tuple[float, float]] = set()
    for i, row in enumerate(system.rows):
        if row.kind is SurfaceKind.STOP:
            continue
        signature = (round(float(system.vertices[i]), 9), round(row.radius, 9))
        if signature in seen:
            continue  # double-passed surface: keep the first physical copy
        seen.add(signature)
        semidiameter = row.semidiameter if row.semidiameter is not None else 100.0
        face = ProfileSurface(
            profile=row.profile,
            vertex=np.array([system.vertices[i], 0.0]),
            angle=0.0,
            semidiameter=semidiameter,
            surface_id=f"s{i + 1}",
            n_exterior=float(system.n_before[i]),
            n_interior=float(system.n_after[i]),
            interaction="reflect" if row.kind is SurfaceKind.MIRROR else "refract",
        )
        surfaces.append(face)
    return surfaces


__all__ = ["to_branching_surfaces"]
