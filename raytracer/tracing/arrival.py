"""What every tracer does where rays meet a boundary.

The surface's normal on its +z side tells which side each ray arrives
from; the boundary's two media then give the index the ray travels in and
the one beyond, at the ray's own wavelength.
"""

from __future__ import annotations

import numpy as np

from ..materials import Material
from ..system import PhysicalFace


def indices(medium: Material, wavelength_um: np.ndarray) -> np.ndarray:
    """The index of *medium* for each ray; a NaN wavelength (none stated) suits only a constant index."""

    values = np.empty(wavelength_um.shape)
    for w in np.unique(wavelength_um):
        same = np.isnan(wavelength_um) if np.isnan(w) else wavelength_um == w
        try:
            values[same] = medium.index(0.0 if np.isnan(w) else float(w))
        except ValueError as exc:
            raise ValueError(f"{exc} (give the rays or the tracer a wavelength_um)") from None
    return values


def arrive(face: PhysicalFace, point: np.ndarray, direction: np.ndarray, n_minus, n_plus):
    """Rays arriving along *direction* at *point* of *face*, in its local frame: ``(normal, n1, n2)``.

    The normal faces the rays, ``n1`` is the index they arrive in and ``n2``
    the one beyond.
    """

    normal = face.surface.normal(point)
    going_plus = np.einsum("ij,ij->i", direction, normal) > 0.0
    n1 = np.where(going_plus, n_minus, n_plus)
    n2 = np.where(going_plus, n_plus, n_minus)
    return np.where(going_plus[:, None], -normal, normal), n1, n2


__all__ = ["indices", "arrive"]
