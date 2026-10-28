"""Optical materials and refractive-index lookup."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from .material_metadata import MaterialMetadata


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
    metadata: MaterialMetadata = field(default_factory=MaterialMetadata)

    def __post_init__(self):
        if not math.isfinite(self.n) or self.n <= 0:
            raise ValueError("refractive index must be finite and positive")

    def index(self, wavelength_um: float) -> float:
        self.metadata.check_wavelength(wavelength_um, self.name)
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

    if not math.isfinite(wavelength_um) or not wavelength_um > 0.0:
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
    metadata: MaterialMetadata = field(default_factory=MaterialMetadata)

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (self.nd, self.vd)):
            raise ValueError("nd and vd must be finite and positive")

    def index(self, wavelength_um: float) -> float:
        _check_wavelength(wavelength_um, f"AbbeMaterial {self.name!r}")
        self.metadata.check_wavelength(wavelength_um, self.name)
        n_f_minus_c = (self.nd - 1.0) / self.vd
        inverse_square_spread = (1.0 / _LAMBDA_F_UM**2) - (1.0 / _LAMBDA_C_UM**2)
        b = n_f_minus_c / inverse_square_spread
        a = self.nd - b / _LAMBDA_D_UM**2
        return a + b / wavelength_um**2


@dataclass(frozen=True)
class CauchyMaterial:
    """Refractive index from an explicit Cauchy series.

    ``n(lambda) = c_0 + c_1 / lambda**2 + c_2 / lambda**4 + ...`` with
    ``lambda`` in micrometres — the classic empirical dispersion form for
    weakly absorbing materials in the visible. Where :class:`AbbeMaterial`
    *derives* a two-term Cauchy curve from a catalog nd/Vd pair, this class
    takes the coefficients directly, for materials characterized by a
    measured fit rather than a catalog entry (3-D printing resins,
    immersion liquids, plastics).
    """

    name: str
    coefficients: tuple[float, ...]
    metadata: MaterialMetadata = field(default_factory=MaterialMetadata)

    def __post_init__(self):
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        if not self.coefficients or not all(math.isfinite(c) for c in self.coefficients):
            raise ValueError("Cauchy coefficients must be finite and nonempty")

    def index(self, wavelength_um: float) -> float:
        _check_wavelength(wavelength_um, f"CauchyMaterial {self.name!r}")
        self.metadata.check_wavelength(wavelength_um, self.name)
        return sum(c / wavelength_um ** (2 * k) for k, c in enumerate(self.coefficients))


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
    metadata: MaterialMetadata = field(default_factory=MaterialMetadata)

    def __post_init__(self):
        object.__setattr__(self, "b", tuple(self.b))
        object.__setattr__(self, "c", tuple(self.c))
        if (
            len(self.b) != 3
            or len(self.c) != 3
            or not all(math.isfinite(x) for x in (*self.b, *self.c))
        ):
            raise ValueError("Sellmeier needs three finite B and C coefficients")

    def index(self, wavelength_um: float) -> float:
        _check_wavelength(wavelength_um, f"SellmeierMaterial {self.name!r}")
        self.metadata.check_wavelength(wavelength_um, self.name)
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


@dataclass(frozen=True)
class IndexOffsetMaterial:
    """A wavelength-independent index perturbation of an existing material.

    Intended for sensitivity analysis; this is not a newly measured glass.
    The base model's wavelength-domain checks remain active.
    """

    name: str
    base: Material
    offset: float

    def __post_init__(self):
        if not math.isfinite(self.offset):
            raise ValueError("index offset must be finite")

    def index(self, wavelength_um: float) -> float:
        index = self.base.index(wavelength_um) + self.offset
        if not math.isfinite(index) or index <= 0:
            raise ValueError("perturbed index must remain finite and positive")
        return index


# Verified Sellmeier coefficients (SCHOTT Zemax catalog 2017-01-20b, via
# refractiveindex.info) for a small, deliberately minimal set of glasses
# spanning the classic crown-to-flint range: N-BK7 (the most common optical
# glass in existence), N-F2 (classic flint), N-SF11 (dense flint / high
# index), N-SK16 and N-SK4 (dense crowns; also the glasses the Cooke
# triplet/double-Gauss prescriptions use), and N-LAK21 (lanthanum crown,
# high index with comparatively low dispersion).
_SELLMEIER_CATALOG: dict[
    str, tuple[float, tuple[float, float, float], tuple[float, float, float]]
] = {
    "N-BK7": (
        1.5168,
        (1.03961212, 0.231792344, 1.01046945),
        (0.00600069867, 0.0200179144, 103.560653),
    ),
    "N-F2": (
        1.62005,
        (1.39757037, 0.159201403, 1.2686543),
        (0.00995906143, 0.0546931752, 119.248346),
    ),
    "N-SF11": (
        1.78472,
        (1.73759695, 0.313747346, 1.89878101),
        (0.013188707, 0.0623068142, 155.23629),
    ),
    "N-SK16": (
        1.62041,
        (1.34317774, 0.241144399, 0.994317969),
        (0.00704687339, 0.0229005, 92.7508526),
    ),
    "N-SK4": (
        1.61272,
        (1.32993741, 0.228542996, 0.988465211),
        (0.00716874107, 0.0246455892, 100.886364),
    ),
    "N-LAK21": (
        1.64049,
        (1.22718116, 0.420783743, 1.01284843),
        (0.00602075682, 0.0196862889, 88.4370099),
    ),
}


def sellmeier_glass(name: str) -> SellmeierMaterial:
    """A :class:`SellmeierMaterial` for one of the built-in catalog glasses.

    Raises ``KeyError`` for anything not in ``_SELLMEIER_CATALOG`` — this
    is a small, curated set (see the module-level comment above it), not a
    general glass database.
    """

    key = name.strip().upper()
    nd, b, c = _SELLMEIER_CATALOG[key]
    evidence = {
        "N-BK7": (
            "https://media.schott.com/api/public/content/41e799d0bf874807a0bb8e702fbb75b5?v=54856406",
            (0.3126, 2.3254),
        ),
        "N-F2": (
            "https://media.schott.com/api/public/content/061f3156c83a44ed9220770b0f65a869?v=d69b35e0",
            (0.4047, 2.3254),
        ),
    }
    source, interval = evidence.get(
        key,
        (
            "SCHOTT Zemax catalog 2017-01-20b via refractiveindex.info; primary-source review pending",
            None,
        ),
    )
    metadata = MaterialMetadata(
        source=source,
        wavelength_range_um=interval,
        index_reference="relative to air",
        notes="Supported interval bounded by tabulated refractive-index wavelengths; no absorption or thermal model.",
    )
    return SellmeierMaterial(key, b, c, catalog_nd=nd, metadata=metadata)


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

    DUV constants are restricted to 193.368 nm. Legacy names SK16 and SK4
    retain modern N-SK16/N-SK4 surrogate models; their metadata explicitly
    distinguishes these from verified historical glass matches. F4 uses
    an approximate Abbe model. Inspect each material's metadata for source,
    supported wavelength interval and unresolved catalog provenance.
    """

    lib = MaterialLibrary()
    lib.register(VACUUM)
    lib.register(AIR)
    # Indices from the US 7,557,996 B2 prescription at lambda = 193.368 nm.
    lib.register(
        ConstantIndex(
            "SIO2",
            1.56078570,
            metadata=MaterialMetadata(
                source="US 7,557,996 B2, Fig. 3 / Table 3",
                wavelength_range_um=(0.193368, 0.193368),
                notes="Single-wavelength prescription value; dispersion is not known.",
            ),
        )
    )
    lib.register(
        ConstantIndex(
            "CAF2",
            1.50185255,
            metadata=MaterialMetadata(
                source="US 7,557,996 B2, Fig. 3 / Table 3",
                wavelength_range_um=(0.193368, 0.193368),
                notes="Single-wavelength prescription value; dispersion is not known.",
            ),
        )
    )
    lib.register(
        ConstantIndex(
            "HIINDEX1",
            1.70196985,
            metadata=MaterialMetadata(
                source="US 7,557,996 B2, Fig. 3 / Table 3",
                wavelength_range_um=(0.193368, 0.193368),
                notes="Single-wavelength prescription value; dispersion is not known.",
            ),
        )
    )
    lib.register(
        ConstantIndex(
            "HIINDEX2",
            1.59667693,
            metadata=MaterialMetadata(
                source="US 7,557,996 B2, Fig. 3 / Table 3",
                wavelength_range_um=(0.193368, 0.193368),
                notes="Single-wavelength prescription value; dispersion is not known.",
            ),
        )
    )
    # Real Sellmeier dispersion for the common-glass catalog.
    for glass_name in _SELLMEIER_CATALOG:
        lib.register(sellmeier_glass(glass_name))
    # Prescription aliases: the Cooke triplet/double-Gauss CSVs spell these
    # without the "N-" prefix. These are explicit legacy surrogates, not
    # evidence that the historical and modern glasses are identical.
    for alias in ("SK16", "SK4"):
        modern = lib["N-" + alias]
        lib[alias] = replace(
            modern,
            name=alias,
            metadata=replace(
                modern.metadata,
                notes=f"Legacy prescription surrogate using {modern.name}; historical glass identity is not verified.",
            ),
        )
    # F4: no verified Sellmeier coefficients found (older, largely
    # discontinued lead glass) — approximate from its catalog nd/Vd instead.
    lib.register(
        AbbeMaterial(
            "F4",
            nd=1.61700,
            vd=36.6,
            metadata=MaterialMetadata(
                source="Legacy prescription nd/Vd pair; manufacturer dispersion unavailable",
                notes="Two-term Cauchy approximation, not a measured Sellmeier curve.",
            ),
        )
    )
    return lib


__all__ = [
    "Material",
    "MaterialMetadata",
    "IndexOffsetMaterial",
    "ConstantIndex",
    "AbbeMaterial",
    "CauchyMaterial",
    "SellmeierMaterial",
    "sellmeier_glass",
    "MaterialLibrary",
    "default_materials",
    "VACUUM",
    "AIR",
]
