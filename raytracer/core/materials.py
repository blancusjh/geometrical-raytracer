"""Optical materials and refractive-index lookup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Material(Protocol):
    """Anything exposing a name and a wavelength-dependent refractive index."""

    name: str

    def index(self, wavelength_um: float) -> float: ...


@dataclass(frozen=True)
class ConstantIndex:
    """Material with a wavelength-independent refractive index.

    Sufficient for single-wavelength prescriptions; dispersive models
    (Sellmeier, catalog glasses) can implement the same protocol later.
    """

    name: str
    n: float

    def index(self, wavelength_um: float) -> float:
        return self.n


VACUUM = ConstantIndex("VACUUM", 1.0)
AIR = ConstantIndex("AIR", 1.0)


class MaterialLibrary(dict):
    """Name-keyed material registry with tolerant resolution.

    ``resolve`` accepts an optional *index_hint*: when the token is unknown
    but a numeric index accompanies it (as in prescription CSVs), a
    ``ConstantIndex`` is created and registered on the fly.
    """

    def register(self, material: Material) -> Material:
        self[material.name.upper()] = material
        return material

    def resolve(self, token: str, index_hint: float | None = None) -> Material:
        key = token.strip().upper()
        if key in self:
            return self[key]
        if index_hint is not None:
            return self.register(ConstantIndex(key, float(index_hint)))
        raise KeyError(f"Unknown material {token!r} and no index hint provided")


def default_materials() -> MaterialLibrary:
    """Library preloaded with vacuum/air and the DUV materials at 193.368 nm."""

    lib = MaterialLibrary()
    lib.register(VACUUM)
    lib.register(AIR)
    # Indices from the US 7,557,996 B2 prescription at lambda = 193.368 nm.
    lib.register(ConstantIndex("SIO2", 1.56078570))
    lib.register(ConstantIndex("CAF2", 1.50185255))
    lib.register(ConstantIndex("HIINDEX1", 1.70196985))
    lib.register(ConstantIndex("HIINDEX2", 1.59667693))
    return lib


__all__ = [
    "Material",
    "ConstantIndex",
    "MaterialLibrary",
    "default_materials",
    "VACUUM",
    "AIR",
]
