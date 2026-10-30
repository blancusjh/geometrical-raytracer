"""Prescription tables as CSV.

The native dialect matches the US7557996 reference export: columns
``surface, surface_type, radius_mm, thickness_to_next_mm, material_after,
index_<wavelength>_nm, clear_semidiameter_mm, aspheric, K, C1..C6,
vertex_z_mm``. The rows are read into :class:`~raytracer.system.TableRow` s
and from those into placed elements by
:meth:`~raytracer.system.OpticalSystem.from_table`; the wavelength named by
the index column is returned alongside, since it is a fact of the design
file, not of the hardware.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np

from ..geometry.surfaces import Conic, EvenAsphere, Plane, Surface
from ..materials import MaterialLibrary, default_materials
from ..system import OpticalSystem, TableRow

logger = logging.getLogger("raytracer.prescription")

_INDEX_COLUMN = re.compile(r"^index_(?P<nm>[0-9.]+)_nm$")
_COEFF_COLUMNS = ["C1_mm^-3", "C2_mm^-5", "C3_mm^-7", "C4_mm^-9", "C5_mm^-11", "C6_mm^-13"]
_POWERS = [4, 6, 8, 10, 12, 14]


class Prescription(NamedTuple):
    """A system read from a lens file, with the wavelength the file is tabulated at."""

    system: OpticalSystem
    wavelength_um: float | None

    def tracer(self, **kwargs):
        """A sequential tracer of the system at the file's wavelength."""

        from ..tracing import SequentialTracer

        kwargs.setdefault("wavelength_um", self.wavelength_um)
        return SequentialTracer(self.system, **kwargs)


def _parse_float(token: str | None, default: float = 0.0) -> float:
    if token is None or token.strip() == "":
        return default
    return float(token)


def surface_from_columns(radius: float, conic: float, coefficients: dict[int, float]) -> Surface:
    """The most specific surface for a radius (0 = flat), conic constant and asphere terms."""

    R = np.inf if radius == 0.0 else radius
    if any(coefficients.values()):
        return EvenAsphere(R, conic, coefficients)
    if np.isinf(R):
        return Plane()
    return Conic(R, conic).classify()


def read_rows(path: str | Path, *, materials: MaterialLibrary | None = None) -> tuple[list[TableRow], float | None]:
    """The table rows of a CSV prescription, and its wavelength (from the index column)."""

    path = Path(path)
    materials = materials if materials is not None else default_materials()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        wavelength_um = None
        index_column = None
        for column in reader.fieldnames or []:
            match = _INDEX_COLUMN.match(column)
            if match:
                index_column = column
                wavelength_um = float(match.group("nm")) * 1e-3
                break
        rows: list[TableRow] = []
        z = 0.0
        for number, record in enumerate(reader, 1):
            surface_type = record["surface_type"].strip().lower()
            coefficients = {p: _parse_float(record.get(c)) for p, c in zip(_POWERS, _COEFF_COLUMNS)}
            surface = surface_from_columns(_parse_float(record.get("radius_mm")), _parse_float(record.get("K")), coefficients)
            gap = _parse_float(record.get("thickness_to_next_mm"))
            token = record.get("clear_semidiameter_mm", "").strip()
            semi = None if token == "" else float(token)
            material_token = record.get("material_after", "AIR").strip()
            hint = _parse_float(record.get(index_column), default=1.0) if index_column else None
            if surface_type == "mirror" or material_token.upper() == "REFL":
                rows.append(TableRow(surface, gap, None, semi, "mirror"))
            else:
                kind = "stop" if surface_type == "aperture_stop" else "refract"
                rows.append(TableRow(surface, gap, materials.resolve(material_token, hint), semi, kind))
            vertex = record.get("vertex_z_mm", "").strip()
            if vertex and abs(float(vertex) - z) > 1e-6:
                logger.warning("Vertex mismatch at surface %d: accumulated %.9f vs column %s", number, z, vertex)
            z += gap
    return rows, wavelength_um


def read_csv(path: str | Path, *, materials: MaterialLibrary | None = None, name: str = "") -> Prescription:
    """Read a CSV prescription into placed elements."""

    rows, wavelength_um = read_rows(path, materials=materials)
    system = OpticalSystem.from_table(rows, name=name or Path(path).stem)
    return Prescription(system, wavelength_um)


def write_csv(system: OpticalSystem, path: str | Path, *, wavelength_um: float | None = None) -> None:
    """Write the coaxial path of *system* as a CSV prescription."""

    path = Path(path)
    nm = (wavelength_um or 0.0) * 1e3
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
    z = 0.0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(system.table(), 1):
            surface = row.surface.surface if hasattr(row.surface, "surface") else row.surface
            radius = 0.0 if np.isinf(surface.radius) else surface.radius
            conic = getattr(surface, "k", 0.0)
            terms = surface.coefficients if isinstance(surface, EvenAsphere) else {}
            aspheric = bool(terms) or conic != 0.0
            if row.kind == "mirror":
                kind, material, index = "mirror", "REFL", ""
            else:
                kind = "aperture_stop" if row.kind == "stop" else ("plane" if radius == 0.0 else "refractive")
                material = row.medium.name
                index = f"{row.medium.index(wavelength_um or 0.0):.8f}" if (wavelength_um or row.medium.__class__.__name__ == "ConstantIndex") else ""
            writer.writerow(
                {
                    "surface": i,
                    "surface_type": kind,
                    "radius_mm": f"{radius:.9f}",
                    "thickness_to_next_mm": f"{row.gap:.9f}",
                    "material_after": material,
                    index_column: index,
                    "clear_semidiameter_mm": "" if row.semi_diameter is None else f"{row.semi_diameter:.3f}",
                    "aspheric": aspheric,
                    "K": conic if aspheric else "",
                    **{c: terms.get(p, "") for p, c in zip(_POWERS, _COEFF_COLUMNS)},
                    "vertex_z_mm": f"{z:.9f}",
                }
            )
            z += row.gap


FORMAT_READERS: dict[str, Callable[..., Prescription]] = {"csv": read_csv}
FORMAT_WRITERS: dict[str, Callable[..., None]] = {"csv": write_csv}


def read_prescription(path: str | Path, *, fmt: str | None = None, **kwargs) -> Prescription:
    """Read a lens file by format (default: from its extension)."""

    fmt = fmt or Path(path).suffix.lstrip(".").lower()
    if fmt not in FORMAT_READERS:
        raise ValueError(f"no reader for {fmt!r}; known formats: {sorted(FORMAT_READERS)}")
    return FORMAT_READERS[fmt](path, **kwargs)


__all__ = [
    "Prescription",
    "read_prescription",
    "read_rows",
    "read_csv",
    "write_csv",
    "surface_from_columns",
    "FORMAT_READERS",
    "FORMAT_WRITERS",
]
