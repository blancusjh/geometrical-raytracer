"""Explicit geometric perturbations in the nominal system's coordinate frame."""

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np

from ...design.rows import SurfaceKind
from ...design.system import OpticalSystem
from ...math.transforms import RigidTransform
from ...optics.materials import IndexOffsetMaterial
from ...surfaces.profile import AsphereProfile


@dataclass(frozen=True)
class Perturbation:
    """One scalar perturbation of selected prescription rows.

    Translations are in mm; tilts in degrees about fixed system axes. A tilt's
    pivot defaults to the first selected vertex of the current system. Supply
    ``pivot_mm`` to keep a specified pivot under a sequence of perturbations.
    Thickness changes move downstream nominal vertices; the evaluator separately
    specifies whether the detector is fixed. Radius changes preserve asphere
    coefficients. Index changes add a constant to the selected transmitted media.
    """

    kind: str
    surfaces: tuple[int, ...]
    label: str = ""
    pivot_mm: tuple[float, float, float] | None = None

    def __post_init__(self):
        allowed = {
            "decenter_x",
            "decenter_y",
            "decenter_z",
            "tilt_x",
            "tilt_y",
            "tilt_z",
            "thickness",
            "radius",
            "index",
        }
        if self.kind not in allowed:
            raise ValueError(f"unknown perturbation {self.kind!r}")
        indices = tuple(self.surfaces)
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(not isinstance(i, int) or i < 0 for i in indices)
        ):
            raise ValueError("surfaces must be nonempty, distinct, nonnegative integer indices")
        if self.kind in ("radius", "thickness") and len(indices) != 1:
            raise ValueError("radius and thickness perturbations require exactly one surface")
        if self.pivot_mm is not None:
            pivot = np.asarray(self.pivot_mm, dtype=float)
            if pivot.shape != (3,) or not np.all(np.isfinite(pivot)):
                raise ValueError("pivot_mm must be a finite three-dimensional point")
        object.__setattr__(self, "surfaces", indices)

    @property
    def unit(self):
        return (
            "deg" if self.kind.startswith("tilt") else ("index" if self.kind == "index" else "mm")
        )

    def apply(self, system: OpticalSystem, amount: float) -> OpticalSystem:
        """Return an independently modified prescription; never mutate the input."""
        if not np.isfinite(amount):
            raise ValueError("perturbation amount must be finite")
        if max(self.surfaces) >= len(system.rows):
            raise ValueError("perturbation surface is outside the prescription")
        result = deepcopy(system)
        result._rebuild()
        if self.kind.startswith("decenter") or self.kind.startswith("tilt"):
            axis = "xyz".index(self.kind[-1])
            vector = np.zeros(3)
            vector[axis] = amount
            if self.kind.startswith("decenter"):
                motion = RigidTransform(vector, np.eye(3))
            else:
                rotation = RigidTransform.from_euler_xyz(angles_deg=vector).rotation
                pivot = (
                    result.frame.to_local(result.surface_frame(self.surfaces[0]).origin)
                    if self.pivot_mm is None
                    else np.asarray(self.pivot_mm)
                )
                motion = RigidTransform(pivot - rotation @ pivot, rotation)
            result.transform_surfaces(self.surfaces, motion)
        elif self.kind == "thickness":
            row = result.rows[self.surfaces[0]]
            row.thickness += amount
        elif self.kind == "radius":
            row = result.rows[self.surfaces[0]]
            if not isinstance(row.profile, AsphereProfile) or row.radius == 0:
                raise ValueError("radius perturbation requires a curved conic/asphere")
            radius = row.radius + amount
            if radius == 0 or radius * row.radius <= 0:
                raise ValueError("radius perturbation cannot cross zero")
            row.profile = replace(row.profile, curvature=1 / radius)
        else:
            for index in self.surfaces:
                row = result.rows[index]
                if row.kind is not SurfaceKind.REFRACT:
                    raise ValueError(
                        "index perturbations apply to refracting rows' transmitted media"
                    )
                row.material_after = IndexOffsetMaterial(
                    row.material_after.name, row.material_after, amount
                )
        result._rebuild()
        return result
