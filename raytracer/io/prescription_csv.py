"""Prescription-table import/export.

The native CSV dialect matches the US7557996 reference export: columns
``surface, surface_type, radius_mm, thickness_to_next_mm, material_after,
index_<wavelength>_nm, clear_semidiameter_mm, aspheric, K, C1..C6,
vertex_z_mm``. Additional formats can be registered in ``FORMAT_READERS`` /
``FORMAT_WRITERS``.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Callable

from ..design.surfaces import SurfaceKind, SurfaceRow
from ..design.system import OpticalSystem
from ..physics.materials import MaterialLibrary, default_materials
from ..shapes.profile import AsphereProfile

logger = logging.getLogger("raytracer.prescription")

_INDEX_COLUMN = re.compile(r"^index_(?P<nm>[0-9.]+)_nm$")
_COEFF_COLUMNS = ["C1_mm^-3", "C2_mm^-5", "C3_mm^-7", "C4_mm^-9", "C5_mm^-11", "C6_mm^-13"]


def _parse_float(token: str | None, default: float = 0.0) -> float:
    if token is None or token.strip() == "":
        return default
    return float(token)


def read_csv(
    path: str | Path,
    *,
    materials: MaterialLibrary | None = None,
    name: str = "",
) -> OpticalSystem:
    """Load a sequential system from the native CSV dialect."""

    path = Path(path)
    materials = materials if materials is not None else default_materials()

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        wavelength_um = None
        index_column = None
        for column in fieldnames:
            match = _INDEX_COLUMN.match(column)
            if match:
                index_column = column
                wavelength_um = float(match.group("nm")) * 1e-3
                break

        rows: list[SurfaceRow] = []
        expected_vertices: list[float | None] = []
        for record in reader:
            surface_type = record["surface_type"].strip().lower()
            radius = _parse_float(record.get("radius_mm"))
            thickness = _parse_float(record.get("thickness_to_next_mm"))
            semidiameter_token = record.get("clear_semidiameter_mm", "")
            semidiameter = (
                None if semidiameter_token.strip() == "" else float(semidiameter_token)
            )
            conic = _parse_float(record.get("K"))
            coefficients = tuple(
                _parse_float(record.get(column)) for column in _COEFF_COLUMNS
            )
            if not any(coefficients):
                coefficients = ()
            profile = AsphereProfile.from_radius(radius, conic, coefficients)

            material_token = record.get("material_after", "AIR").strip()
            index_hint = (
                _parse_float(record.get(index_column), default=1.0)
                if index_column
                else None
            )

            if surface_type == "mirror" or material_token.upper() == "REFL":
                row = SurfaceRow(
                    profile=profile,
                    thickness=thickness,
                    kind=SurfaceKind.MIRROR,
                    semidiameter=semidiameter,
                )
            elif surface_type == "aperture_stop":
                row = SurfaceRow(
                    profile=profile,
                    thickness=thickness,
                    material_after=materials.resolve(material_token, index_hint),
                    kind=SurfaceKind.STOP,
                    semidiameter=semidiameter,
                )
            else:  # "refractive" and "plane" both refract (a plane may separate media)
                row = SurfaceRow(
                    profile=profile,
                    thickness=thickness,
                    material_after=materials.resolve(material_token, index_hint),
                    kind=SurfaceKind.REFRACT,
                    semidiameter=semidiameter,
                )
            rows.append(row)
            vertex_token = record.get("vertex_z_mm", "")
            expected_vertices.append(
                None if vertex_token.strip() == "" else float(vertex_token)
            )

    system = OpticalSystem(
        rows, wavelength_um=wavelength_um, name=name or path.stem
    )

    # The vertex column is redundant with the thickness accumulation; use it
    # as a consistency check rather than trusting it.
    for i, expected in enumerate(expected_vertices):
        if expected is None:
            continue
        if abs(system.vertices[i] - expected) > 1e-6:
            logger.warning(
                "Vertex mismatch at surface %d: accumulated %.9f vs column %.9f",
                i + 1,
                system.vertices[i],
                expected,
            )
    return system


def write_csv(system: OpticalSystem, path: str | Path) -> None:
    """Export *system* in the native CSV dialect."""

    path = Path(path)
    nm = (system.wavelength_um or 0.0) * 1e3
    index_column = f"index_{nm:g}_nm" if nm else "index_nm"
    fields = [
        "surface",
        "surface_type",
        "radius_mm",
        "thickness_to_next_mm",
        "material_after",
        index_column,
        "clear_semidiameter_mm",
        "aspheric",
        "K",
        *_COEFF_COLUMNS,
        "vertex_z_mm",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(system.rows):
            if row.kind is SurfaceKind.MIRROR:
                kind = "mirror"
                material = "REFL"
            elif row.kind is SurfaceKind.STOP:
                kind = "aperture_stop"
                material = row.material_after.name
            elif row.profile.curvature == 0.0:
                kind = "plane"
                material = row.material_after.name
            else:
                kind = "refractive"
                material = row.material_after.name
            coefficients = list(row.profile.coefficients) + [""] * (
                6 - len(row.profile.coefficients)
            )
            aspheric = bool(row.profile.coefficients) or row.profile.conic != 0.0
            writer.writerow(
                {
                    "surface": i + 1,
                    "surface_type": kind,
                    "radius_mm": f"{row.radius:.9f}",
                    "thickness_to_next_mm": f"{row.thickness:.9f}",
                    "material_after": material,
                    index_column: f"{system.n_after[i]:.8f}",
                    "clear_semidiameter_mm": (
                        "" if row.semidiameter is None else f"{row.semidiameter:.3f}"
                    ),
                    "aspheric": aspheric,
                    "K": row.profile.conic if aspheric else "",
                    **dict(zip(_COEFF_COLUMNS, coefficients)),
                    "vertex_z_mm": f"{system.vertices[i]:.9f}",
                }
            )


FORMAT_READERS: dict[str, Callable[..., OpticalSystem]] = {"csv": read_csv}
FORMAT_WRITERS: dict[str, Callable[[OpticalSystem, str | Path], None]] = {"csv": write_csv}

__all__ = ["read_csv", "write_csv", "FORMAT_READERS", "FORMAT_WRITERS"]
