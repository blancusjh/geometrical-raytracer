"""Interfaces: a surface as an optical boundary, and the laws obeyed there.

An :class:`Interface` is a surface (geometry) together with what makes it
optical *on its own*: its coating and its clear aperture. It deliberately
does not carry media — which materials meet at a boundary follows from
the elements on its two sides, and is resolved by the system that places
them. Nor does it carry an "interaction": whether light reflects or
refracts there follows from the media, the coating and the laws below.

Anywhere an element takes an interface, a bare surface is accepted and
means ``Interface(surface)``: uncoated, clear over the element's full
extent.
"""

from __future__ import annotations

import numpy as np

from ..coatings import Coating, Uncoated
from ..geometry.apertures import Aperture
from ..geometry.surfaces import Surface


class Interface:
    """A surface, its coating, and the part of it that is optically specified."""

    __slots__ = ("surface", "coating", "clear_aperture")

    def __init__(self, surface: Surface, *, coating: Coating | None = None, clear_aperture: Aperture | float | None = None) -> None:
        if not isinstance(surface, Surface):
            raise TypeError(f"an interface is built on a Surface, got {surface!r}")
        object.__setattr__(self, "surface", surface)
        object.__setattr__(self, "coating", coating if coating is not None else Uncoated())
        object.__setattr__(self, "clear_aperture", Aperture.coerce(clear_aperture))

    @classmethod
    def coerce(cls, spec: "Interface | Surface") -> "Interface":
        return spec if isinstance(spec, Interface) else cls(spec)

    def __setattr__(self, name, value):
        raise AttributeError("Interface is immutable; use .replace(...)")

    def replace(self, **changes) -> "Interface":
        values = {"surface": self.surface, "coating": self.coating, "clear_aperture": self.clear_aperture}
        values.update(changes)
        return Interface(values.pop("surface"), **values)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Interface) and (self.surface, self.coating, self.clear_aperture) == (
            other.surface,
            other.coating,
            other.clear_aperture,
        )

    def __hash__(self) -> int:
        return hash((self.surface, self.coating, self.clear_aperture))

    def __repr__(self) -> str:
        parts = [repr(self.surface)]
        if not isinstance(self.coating, Uncoated):
            parts.append(f"coating={self.coating!r}")
        if self.clear_aperture is not None:
            parts.append(f"clear_aperture={self.clear_aperture!r}")
        return f"Interface({', '.join(parts)})"


# -- the laws of geometrical optics at a boundary ------------------------------


def reflect(directions: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Law of reflection for ``(N, 3)`` unit directions and normals (either orientation)."""

    cos_i = -np.einsum("ij,ij->i", directions, normals)
    out = directions + 2.0 * cos_i[:, None] * normals
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def refract(directions: np.ndarray, normals: np.ndarray, n1, n2) -> tuple[np.ndarray, np.ndarray]:
    """Snell's law in vector form; *normals* must face the incoming ray.

    Returns ``(directions_out, tir)``; where ``tir`` is set the light is
    totally reflected and ``directions_out`` is meaningless.
    """

    n_rays = directions.shape[0]
    cos_i = -np.einsum("ij,ij->i", directions, normals)
    eta = np.broadcast_to(np.asarray(n1, dtype=float) / np.asarray(n2, dtype=float), (n_rays,))
    radicand = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    tir = radicand < 0.0
    root = np.sqrt(np.maximum(radicand, 0.0))
    out = eta[:, None] * directions + (eta * cos_i - root)[:, None] * normals
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norm == 0.0, 1.0, norm), tir


__all__ = ["Interface", "reflect", "refract"]
