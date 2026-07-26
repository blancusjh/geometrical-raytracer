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


# Standard reference wavelengths (micrometres) used to define the Abbe
# number: d-line (helium yellow-green), F-line and C-line (blue/red
# hydrogen), the classical crown/flint dispersion reference triplet.
_LAMBDA_D_UM = 0.58756
_LAMBDA_F_UM = 0.48613
_LAMBDA_C_UM = 0.65627


@dataclass(frozen=True)
class AbbeMaterial:
    """Refractive index from (n_d, V_d) via a 2-term Cauchy dispersion curve.

    Real glass dispersion needs a manufacturer's measured Sellmeier
    coefficients to get right; this package doesn't carry a glass catalog,
    so this is a standard textbook-level stand-in: fit a Cauchy curve
    ``n(lambda) = A + B / lambda**2`` to match ``n_d`` exactly at the d
    line (587.6 nm) and reproduce the F-C principal dispersion the Abbe
    number defines (``n_F - n_C = (n_d - 1) / V_d``). That's the same
    "normal glass" approximation classical optics references use when only
    the catalog nd/Vd pair is known, not the manufacturer's own dispersion
    curve — adequate to show axial/lateral color *exists* and roughly how
    it scales, not to certify a real design's chromatic correction.
    """

    name: str
    nd: float
    vd: float

    def index(self, wavelength_um: float) -> float:
        n_f_minus_c = (self.nd - 1.0) / self.vd
        inverse_square_spread = (1.0 / _LAMBDA_F_UM**2) - (1.0 / _LAMBDA_C_UM**2)
        b = n_f_minus_c / inverse_square_spread
        a = self.nd - b / _LAMBDA_D_UM**2
        return a + b / wavelength_um**2


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
    """Library preloaded with vacuum/air, the DUV materials at 193.368 nm,
    and the visible-spectrum glasses used by the Cooke triplet/double-Gauss
    prescriptions.

    The DUV materials are single-wavelength ``ConstantIndex`` (that
    prescription is only ever evaluated at one wavelength). The visible
    glasses are ``AbbeMaterial`` — registering them under their prescription
    names means ``OpticalSystem.from_prescription`` picks up real (if
    approximate) dispersion for those two lenses automatically, with no
    change to the CSV files themselves.
    """

    lib = MaterialLibrary()
    lib.register(VACUUM)
    lib.register(AIR)
    # Indices from the US 7,557,996 B2 prescription at lambda = 193.368 nm.
    lib.register(ConstantIndex("SIO2", 1.56078570))
    lib.register(ConstantIndex("CAF2", 1.50185255))
    lib.register(ConstantIndex("HIINDEX1", 1.70196985))
    lib.register(ConstantIndex("HIINDEX2", 1.59667693))
    # Schott glass catalog nd/Vd (d-line index / Abbe number), used by the
    # Cooke triplet and double-Gauss prescriptions.
    lib.register(AbbeMaterial("SK16", nd=1.62041, vd=60.32))
    lib.register(AbbeMaterial("SK4", nd=1.61272, vd=58.63))
    lib.register(AbbeMaterial("F4", nd=1.61700, vd=36.6))
    return lib


__all__ = [
    "Material",
    "ConstantIndex",
    "AbbeMaterial",
    "MaterialLibrary",
    "default_materials",
    "VACUUM",
    "AIR",
]
