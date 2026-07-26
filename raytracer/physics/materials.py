"""Optical materials and refractive-index lookup."""

from __future__ import annotations

import math
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


def _check_wavelength(wavelength_um: float, material_name: str) -> None:
    """Reject a non-positive wavelength for a dispersive material.

    :class:`ConstantIndex` ignores its wavelength argument, so callers have
    historically been able to get away with passing ``0.0`` — notably
    :meth:`OpticalSystem._rebuild`, which substitutes ``0.0`` when a system
    carries no ``wavelength_um``. A dispersive curve evaluated there would
    silently return a physically meaningless index (the Sellmeier form
    returns exactly 1.0, i.e. vacuum), so refuse it loudly instead.
    """

    if not wavelength_um > 0.0:
        raise ValueError(
            f"{material_name} is dispersive and needs a positive wavelength, got "
            f"{wavelength_um!r}; set wavelength_um on the OpticalSystem"
        )


@dataclass(frozen=True)
class AbbeMaterial:
    """Refractive index from (n_d, V_d) via a 2-term Cauchy dispersion curve.

    Measured Sellmeier coefficients (see :class:`SellmeierMaterial` and
    ``_SELLMEIER_CATALOG``) describe a real glass far better; use this only
    for a glass whose catalog nd/Vd pair is all that's known. It is a
    standard textbook-level stand-in: fit a Cauchy curve
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
        _check_wavelength(wavelength_um, f"AbbeMaterial {self.name!r}")
        n_f_minus_c = (self.nd - 1.0) / self.vd
        inverse_square_spread = (1.0 / _LAMBDA_F_UM**2) - (1.0 / _LAMBDA_C_UM**2)
        b = n_f_minus_c / inverse_square_spread
        a = self.nd - b / _LAMBDA_D_UM**2
        return a + b / wavelength_um**2


@dataclass(frozen=True)
class SellmeierMaterial:
    """Refractive index from a measured 3-term Sellmeier dispersion curve.

    ``n(lambda)^2 - 1 = sum_i B_i * lambda^2 / (lambda^2 - C_i)``, with
    ``lambda`` in micrometres and ``B``/``C`` the manufacturer's fitted
    coefficients (``C`` in um^2) — this is the real catalog dispersion
    curve, unlike :class:`AbbeMaterial`'s nd/Vd approximation. Use this
    whenever a glass's actual Sellmeier coefficients are known.
    """

    name: str
    b: tuple[float, float, float]
    c: tuple[float, float, float]
    #: The catalog's own stated d-line index, kept for display and as an
    #: independent cross-check on the fitted curve (``index(0.58756)`` should
    #: reproduce it). Not used in the computation.
    catalog_nd: float | None = None

    def index(self, wavelength_um: float) -> float:
        _check_wavelength(wavelength_um, f"SellmeierMaterial {self.name!r}")
        lambda2 = wavelength_um * wavelength_um
        n2_minus_1 = 0.0
        for bi, ci in zip(self.b, self.c):
            denominator = lambda2 - ci
            if denominator == 0.0:
                raise ValueError(
                    f"{self.name} evaluated exactly at a Sellmeier resonance "
                    f"(lambda^2 == {ci}); the fit is undefined there"
                )
            n2_minus_1 += bi * lambda2 / denominator
        if n2_minus_1 <= -1.0:
            raise ValueError(
                f"{self.name} Sellmeier fit gives n^2 = {1.0 + n2_minus_1:.4g} at "
                f"{wavelength_um} um; the wavelength is outside the fit's valid range"
            )
        return math.sqrt(1.0 + n2_minus_1)


# Verified Sellmeier coefficients (SCHOTT Zemax catalog 2017-01-20b, via
# refractiveindex.info) for a small, deliberately minimal set of glasses
# spanning the classic crown-to-flint range: N-BK7 (the most common optical
# glass in existence), N-F2 (classic flint), N-SF11 (dense flint / high
# index), N-SK16 and N-SK4 (dense crowns; also the glasses the Cooke
# triplet/double-Gauss prescriptions use), and N-LAK21 (lanthanum crown,
# high index with comparatively low dispersion).
_SELLMEIER_CATALOG: dict[str, tuple[float, tuple[float, float, float], tuple[float, float, float]]] = {
    "N-BK7": (1.5168, (1.03961212, 0.231792344, 1.01046945), (0.00600069867, 0.0200179144, 103.560653)),
    "N-F2": (1.62005, (1.39757037, 0.159201403, 1.2686543), (0.00995906143, 0.0546931752, 119.248346)),
    "N-SF11": (1.78472, (1.73759695, 0.313747346, 1.89878101), (0.013188707, 0.0623068142, 155.23629)),
    "N-SK16": (1.62041, (1.34317774, 0.241144399, 0.994317969), (0.00704687339, 0.0229005, 92.7508526)),
    "N-SK4": (1.61272, (1.32993741, 0.228542996, 0.988465211), (0.00716874107, 0.0246455892, 100.886364)),
    "N-LAK21": (1.64049, (1.22718116, 0.420783743, 1.01284843), (0.00602075682, 0.0196862889, 88.4370099)),
}


def sellmeier_glass(name: str) -> SellmeierMaterial:
    """A :class:`SellmeierMaterial` for one of the built-in catalog glasses.

    Raises ``KeyError`` for anything not in ``_SELLMEIER_CATALOG`` — this
    is a small, curated set (see the module-level comment above it), not a
    general glass database.
    """

    key = name.strip().upper()
    nd, b, c = _SELLMEIER_CATALOG[key]
    return SellmeierMaterial(key, b, c, catalog_nd=nd)


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
    and a small common-glass catalog spanning crown to flint.

    The DUV materials are single-wavelength ``ConstantIndex`` (that
    prescription is only ever evaluated at one wavelength). SK16 and SK4 —
    used by the Cooke triplet/double-Gauss prescriptions — are registered
    with their real measured Sellmeier dispersion (see
    ``_SELLMEIER_CATALOG``), so ``OpticalSystem.from_prescription`` picks
    up accurate chromatic behavior for those two lenses automatically, with
    no change to the CSV files themselves. F4 has no verified Sellmeier
    data available (an older, largely-discontinued lead glass), so it stays
    an ``AbbeMaterial`` approximation from its catalog nd/Vd. N-BK7, N-F2,
    N-SF11, and N-LAK21 are registered too, for prescriptions that want a
    common, accurately-dispersive glass but aren't tied to a specific
    catalog match.
    """

    lib = MaterialLibrary()
    lib.register(VACUUM)
    lib.register(AIR)
    # Indices from the US 7,557,996 B2 prescription at lambda = 193.368 nm.
    lib.register(ConstantIndex("SIO2", 1.56078570))
    lib.register(ConstantIndex("CAF2", 1.50185255))
    lib.register(ConstantIndex("HIINDEX1", 1.70196985))
    lib.register(ConstantIndex("HIINDEX2", 1.59667693))
    # Real Sellmeier dispersion for the common-glass catalog.
    for glass_name in _SELLMEIER_CATALOG:
        lib.register(sellmeier_glass(glass_name))
    # Prescription aliases: the Cooke triplet/double-Gauss CSVs spell these
    # without the "N-" (lead-free) prefix, since that distinction isn't
    # meaningful for this package's purposes.
    lib["SK16"] = lib["N-SK16"]
    lib["SK4"] = lib["N-SK4"]
    # F4: no verified Sellmeier coefficients found (older, largely
    # discontinued lead glass) — approximate from its catalog nd/Vd instead.
    lib.register(AbbeMaterial("F4", nd=1.61700, vd=36.6))
    return lib


__all__ = [
    "Material",
    "ConstantIndex",
    "AbbeMaterial",
    "SellmeierMaterial",
    "sellmeier_glass",
    "MaterialLibrary",
    "default_materials",
    "VACUUM",
    "AIR",
]
