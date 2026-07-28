"""Stigmatic-train designs as data files.

A :class:`~raytracer.design.stigmatic.StigmaticTrain` is a handful of
numbers plus media *names* — a finished design is data, and belongs next to
the prescriptions (``data/optical_systems/<kind>/*.json``) rather than
hard-coded in a notebook. The JSON stores media by name and conjugates at
full precision (some designs sit close to the usable-branch edge, where
rounding the conjugates to a few decimals breaks the marginal ray), plus a
snapshot of the indices at the design wavelength as a consistency check:
loading against a material catalog whose dispersion has drifted fails
loudly instead of silently rebuilding different surfaces.

Media names resolve through a
:class:`~raytracer.optics.materials.MaterialLibrary` — pass
``default_materials()`` extended with any file catalogs the design uses
(e.g. ``read_materials("data/materials/formlabs_resins.csv", into=...)``).
Constant-index media are flagged as such in the file and rebuilt from the
stored index when the library lacks them, since the index *is* their whole
definition; a dispersive medium missing from the library is an error — its
dispersion cannot be reconstructed from one number.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..design.stigmatic import StigmaticTrain
from ..optics.materials import ConstantIndex, Material, MaterialLibrary

_KIND = "stigmatic_train"


def write_train(train: StigmaticTrain, path: str | Path, *, comment: str = "") -> None:
    """Save *train* as a JSON design file."""

    payload = {
        "kind": _KIND,
        "comment": comment,
        "wavelength_um": train.wavelength_um,
        "media": [m.name for m in train.media],
        "constant_media": [isinstance(m, ConstantIndex) for m in train.media],
        "indices_at_design": list(train.indices),
        "vertices": list(train.vertices),
        "conjugates": list(train.conjugates),
        "semidiameter": train.semidiameter,
    }
    Path(path).write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def read_train(
    path: str | Path,
    *,
    materials: MaterialLibrary,
    index_tolerance: float = 1e-6,
) -> StigmaticTrain:
    """Load a design file back into a :class:`StigmaticTrain`.

    Every stored index is checked against the resolved medium's index at the
    design wavelength within *index_tolerance*; a mismatch means the catalog
    no longer describes the material this design was built for.
    """

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != _KIND:
        raise ValueError(f"{path} is not a stigmatic-train file: kind={payload.get('kind')!r}")

    wavelength = float(payload["wavelength_um"])
    media: list[Material] = []
    constant_flags = payload["constant_media"]
    for name, is_constant, stored_index in zip(
        payload["media"], constant_flags, payload["indices_at_design"]
    ):
        key = name.strip().upper()
        if key in materials:
            medium = materials[key]
        elif is_constant:
            medium = ConstantIndex(name, float(stored_index))
        else:
            raise KeyError(
                f"{path}: dispersive medium {name!r} is not in the material "
                "library and cannot be rebuilt from its design index alone; "
                "extend the library with io.read_materials(...) for the "
                "catalog this design uses"
            )
        actual = medium.index(wavelength)
        if abs(actual - stored_index) > index_tolerance:
            raise ValueError(
                f"{path}: medium {name!r} has n = {actual:.8f} at "
                f"{wavelength} um but the design was built with "
                f"{stored_index:.8f}; the material catalog has drifted"
            )
        media.append(medium)

    return StigmaticTrain(
        media=tuple(media),
        vertices=tuple(float(v) for v in payload["vertices"]),
        conjugates=tuple(float(d) for d in payload["conjugates"]),
        semidiameter=(
            None if payload["semidiameter"] is None else float(payload["semidiameter"])
        ),
        wavelength_um=wavelength,
    )


__all__ = ["read_train", "write_train"]
