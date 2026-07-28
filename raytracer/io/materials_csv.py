"""Material catalogs as data files.

A material is a name plus a dispersion model with a handful of
coefficients — data, not code — so a project-specific catalog (a printing
resin, an immersion liquid, a manufacturer's glass melt) belongs in a CSV
next to the prescriptions, read at run time.

Format: comma-separated with a header row, ``#`` lines ignored::

    name,model,c1,c2,c3,c4,c5,c6,source

- ``constant``:  c1 = n  (wavelength-independent)
- ``abbe``:      c1 = n_d, c2 = V_d  (two-term Cauchy derived from them)
- ``cauchy``:    c1.. = coefficients of ``n = c1 + c2/λ² + c3/λ⁴ + ...``,
  λ in µm — for materials characterized by a measured fit
- ``sellmeier``: c1..c3 = B₁..B₃, c4..c6 = C₁..C₃ (µm²)

``source`` is free text (measurement provenance); it is carried into the
material's registry but not interpreted. See
``data/materials/formlabs_resins.csv`` for a real catalog in this format.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..optics.materials import (
    AbbeMaterial,
    CauchyMaterial,
    ConstantIndex,
    Material,
    MaterialLibrary,
    SellmeierMaterial,
)

_COEFF_COLUMNS = ["c1", "c2", "c3", "c4", "c5", "c6"]


def _coefficients(record: dict, *, name: str) -> list[float]:
    values = []
    for column in _COEFF_COLUMNS:
        token = (record.get(column) or "").strip()
        if token:
            values.append(float(token))
        else:
            break
    if not values:
        raise ValueError(f"material {name!r} has no coefficients")
    return values


def _build(name: str, model: str, coefficients: list[float]) -> Material:
    if model == "constant":
        if len(coefficients) != 1:
            raise ValueError(f"material {name!r}: 'constant' takes exactly c1 = n")
        return ConstantIndex(name, coefficients[0])
    if model == "abbe":
        if len(coefficients) != 2:
            raise ValueError(f"material {name!r}: 'abbe' takes c1 = n_d, c2 = V_d")
        return AbbeMaterial(name, nd=coefficients[0], vd=coefficients[1])
    if model == "cauchy":
        return CauchyMaterial(name, tuple(coefficients))
    if model == "sellmeier":
        if len(coefficients) != 6:
            raise ValueError(
                f"material {name!r}: 'sellmeier' takes c1..c3 = B and c4..c6 = C"
            )
        return SellmeierMaterial(
            name, tuple(coefficients[:3]), tuple(coefficients[3:])
        )
    raise ValueError(
        f"material {name!r} has unknown model {model!r}; expected "
        "constant, abbe, cauchy, or sellmeier"
    )


def read_materials(
    path: str | Path, *, into: MaterialLibrary | None = None
) -> MaterialLibrary:
    """Read a material-catalog CSV into a :class:`MaterialLibrary`.

    ``into`` extends an existing library (e.g. ``default_materials()``)
    instead of starting an empty one, so a file catalog and the built-in
    glasses can be resolved through a single registry.
    """

    path = Path(path)
    library = into if into is not None else MaterialLibrary()

    with path.open(newline="", encoding="utf-8") as stream:
        clean = (line for line in stream if not line.lstrip().startswith("#"))
        reader = csv.DictReader(clean)
        for record in reader:
            name = (record.get("name") or "").strip()
            model = (record.get("model") or "").strip().lower()
            if not name:
                raise ValueError(f"{path}: row with empty material name: {record}")
            library.register(_build(name, model, _coefficients(record, name=name)))
    return library


__all__ = ["read_materials"]
