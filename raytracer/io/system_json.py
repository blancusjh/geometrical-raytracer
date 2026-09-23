"""Versioned, lossless JSON prescriptions for the supported geometric model."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..design.rows import SurfaceKind, SurfaceRow
from ..design.system import OpticalSystem
from ..math.transforms import RigidTransform
from ..optics.materials import AbbeMaterial, CauchyMaterial, ConstantIndex, SellmeierMaterial
from ..surfaces.apertures import CircularAperture, RectangularAperture
from ..surfaces.cartesian_oval import CartesianOvalProfile
from ..surfaces.profile import AsphereProfile

MATERIAL_TYPES = {
    cls.__name__: cls
    for cls in (
        ConstantIndex,
        AbbeMaterial,
        CauchyMaterial,
        SellmeierMaterial,
    )
}
APERTURE_TYPES = {cls.__name__: cls for cls in (CircularAperture, RectangularAperture)}


def _number(value):
    if value is None or np.isfinite(value):
        return value
    if np.isnan(value):
        raise ValueError("a prescription cannot contain NaN")
    return "+infinity" if value > 0 else "-infinity"


def _restore_number(value):
    if value == "+infinity":
        return np.inf
    if value == "-infinity":
        return -np.inf
    return value


def _frame(frame):
    return {"origin_mm": frame.origin.tolist(), "rotation": frame.rotation.tolist()}


def _read_frame(payload):
    return RigidTransform(payload["origin_mm"], payload["rotation"])


def _material(material):
    kind = type(material).__name__
    if kind not in MATERIAL_TYPES:
        raise TypeError(f"cannot serialize material type {kind}; register an explicit format first")
    return {"type": kind, "parameters": asdict(material)}


def _read_material(payload):
    try:
        cls = MATERIAL_TYPES[payload["type"]]
    except KeyError:
        raise ValueError(f"unsupported material type {payload.get('type')!r}") from None
    values = dict(payload["parameters"])
    for key in ("b", "c", "coefficients"):
        if key in values:
            values[key] = tuple(values[key])
    return cls(**values)


def write_system(system: OpticalSystem, path) -> None:
    """Save geometry, placements, apertures, media and conjugates at full precision."""

    rows = []
    for row in system.rows:
        if isinstance(row.profile, AsphereProfile):
            profile = {"type": "asphere", **asdict(row.profile)}
        elif isinstance(row.profile, CartesianOvalProfile):
            profile = {
                "type": "cartesian_oval",
                **{key: _number(getattr(row.profile, key)) for key in ("n0", "z0", "ni", "zi")},
            }
        else:
            raise TypeError(f"cannot serialize profile type {type(row.profile).__name__}")
        aperture = None
        if row.aperture is not None:
            kind = type(row.aperture).__name__
            if kind not in APERTURE_TYPES:
                raise TypeError(f"cannot serialize aperture type {kind}")
            aperture = {"type": kind, **asdict(row.aperture)}
        rows.append(
            {
                "profile": profile,
                "thickness_mm": row.thickness,
                "material_after": _material(row.material_after),
                "kind": row.kind.value,
                "semidiameter_mm": row.semidiameter,
                "aperture": aperture,
                "is_stop": row.is_stop,
                "placement": _frame(row.placement),
                "comment": row.comment,
            }
        )
    payload = {
        "format": "geometrical-raytracer",
        "version": 1,
        "length_unit": "mm",
        "name": system.name,
        "wavelength_um": system.wavelength_um,
        "object_z_mm": _number(system.object_z),
        "object_space": _material(system.object_space),
        "frame": _frame(system.frame),
        "image_placement": _frame(system.image_placement),
        "surfaces": rows,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def read_system(path) -> OpticalSystem:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (payload.get("format"), payload.get("version"), payload.get("length_unit")) != (
        "geometrical-raytracer",
        1,
        "mm",
    ):
        raise ValueError("unsupported geometric prescription format, version or units")
    rows = []
    for record in payload["surfaces"]:
        values = dict(record["profile"])
        kind = values.pop("type")
        if kind == "asphere":
            values["coefficients"] = tuple(values["coefficients"])
            profile = AsphereProfile(**values)
        elif kind == "cartesian_oval":
            profile = CartesianOvalProfile(
                **{key: _restore_number(value) for key, value in values.items()}
            )
        else:
            raise ValueError(f"unsupported surface type {kind!r}")
        aperture = None
        if record["aperture"] is not None:
            values = dict(record["aperture"])
            kind = values.pop("type")
            if kind not in APERTURE_TYPES:
                raise ValueError(f"unsupported aperture type {kind!r}")
            aperture = APERTURE_TYPES[kind](**values)
        rows.append(
            SurfaceRow(
                profile=profile,
                thickness=record["thickness_mm"],
                material_after=_read_material(record["material_after"]),
                kind=SurfaceKind(record["kind"]),
                semidiameter=record["semidiameter_mm"],
                aperture=aperture,
                is_stop=record["is_stop"],
                placement=_read_frame(record["placement"]),
                comment=record["comment"],
            )
        )
    return OpticalSystem(
        rows,
        wavelength_um=payload["wavelength_um"],
        object_z=_restore_number(payload["object_z_mm"]),
        object_space=_read_material(payload["object_space"]),
        name=payload["name"],
        frame=_read_frame(payload["frame"]),
        image_placement=_read_frame(payload["image_placement"]),
    )
