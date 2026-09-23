"""Bridge between an engineered design and the 2-D branching propagation.

The one function here legitimately depends on both: it reinterprets a
sequential :class:`~raytracer.design.system.OpticalSystem` as the 2-D
``Surface`` objects (:mod:`raytracer.optics`) the branching propagation
needs to explore it non-sequentially (caustics, stray light, ...).
"""

from __future__ import annotations

import numpy as np

from ..surfaces.apertures import CircularAperture
from ..surfaces.profile import ProfileSurface
from ..surfaces.stop import ApertureStopSurface
from ..surfaces.surface import Surface
from .rows import SurfaceKind


def to_branching_surfaces(system) -> list[Surface]:
    """Meridional (y-z) slice of a sequential system as 2-D surfaces.

    The sequential z axis maps to 2-D x. Every refractive row becomes a
    ``ProfileSurface`` carrying its before/after indices; mirrors become
    reflective faces; stops become absorbing masks outside the clear opening. Double-passed
    surfaces on folded return paths (duplicated rows with matching vertex
    profile, aperture and unordered index pair) collapse to a single physical face.

    Caveat: this bridge is exact for dioptric (unfolded) systems. In a
    folded catadioptric, the branching propagation will let the forward
    beam interact with any mirror whose full aperture crosses its path —
    which sequential ordering deliberately ignores. Clip mirror
    semidiameters to the physically used sub-aperture in that case, or
    analyse folded systems with the sequential propagation instead.
    """

    if (
        not system.is_centered
        or np.any(system.frame.origin)
        or not np.allclose(system.frame.rotation, np.eye(3))
    ):
        raise ValueError("the 2-D bridge requires an untransformed centered system")
    surfaces: list[Surface] = []
    seen = set()
    for i, row in enumerate(system.rows):
        if row.kind is SurfaceKind.STOP:
            aperture = row.clear_aperture
            if not isinstance(aperture, CircularAperture):
                raise ValueError("the 2-D bridge requires circular stop apertures")
            surfaces.append(
                ApertureStopSurface(
                    [system.vertices[i], 0.0],
                    aperture.radius,
                    inner_radius=aperture.inner_radius,
                    surface_id=f"stop{i + 1}",
                )
            )
            continue
        if row.aperture is not None:
            raise ValueError("independent apertures on interfaces require sequential tracing")
        signature = (
            float(system.vertices[i]),
            row.profile,
            row.kind,
            row.semidiameter,
            tuple(sorted((system.n_before[i], system.n_after[i]))),
        )
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
