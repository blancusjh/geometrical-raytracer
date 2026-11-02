"""Gaussian (first-order) optics: what the paraxial marginal and chief rays say about a configuration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..conditions import Configuration
from ..tracing.paraxial import Paraxial, ParaxialPupil


@dataclass(frozen=True)
class FirstOrder:
    """First-order data of a configuration at one wavelength (the primary unless told otherwise).

    Positions ``*_z_mm`` are world ``z``; ``bfl_mm`` and ``ffl_mm`` run
    along the light from the last and first vertex that bends it. ``efl_mm`` is the
    reciprocal of the power. The F-number is ``efl / EPD``; the working
    F-number, ``1 / (2 image_na)``; NAs are ``n |u|`` of the paraxial
    marginal ray. ``image_defocus_mm`` is where the detector stands past the
    paraxial image, along the light.
    """

    configuration: Configuration
    wavelength_um: float
    efl_mm: float
    bfl_mm: float
    ffl_mm: float
    f_number: float
    working_f_number: float
    object_na: float
    image_na: float
    entrance_pupil: ParaxialPupil
    exit_pupil: ParaxialPupil
    stop_radius_mm: float
    principal_planes_z_mm: tuple[float, float]
    nodal_points_z_mm: tuple[float, float]
    focal_points_z_mm: tuple[float, float]
    lagrange_invariant_mm: float
    magnification: float
    paraxial_image_z_mm: float
    paraxial_image_height_mm: float
    image_defocus_mm: float
    object_index: float
    image_index: float

    def __str__(self) -> str:
        rows = [
            ("wavelength", f"{self.wavelength_um:.6g} µm"),
            ("effective focal length", f"{self.efl_mm:.6g} mm"),
            ("back focal length", f"{self.bfl_mm:.6g} mm"),
            ("front focal length", f"{self.ffl_mm:.6g} mm"),
            ("F-number", f"{self.f_number:.6g}"),
            ("working F-number", f"{self.working_f_number:.6g}"),
            ("object-space NA", f"{self.object_na:.6g}"),
            ("image-space NA", f"{self.image_na:.6g}"),
            ("entrance pupil", f"z = {self.entrance_pupil.z_mm:.6g} mm, diameter {self.entrance_pupil.diameter_mm:.6g} mm"),
            ("exit pupil", f"z = {self.exit_pupil.z_mm:.6g} mm, diameter {self.exit_pupil.diameter_mm:.6g} mm"),
            ("stop filled to", f"{self.stop_radius_mm:.6g} mm"),
            ("principal planes", "z = {:.6g}, {:.6g} mm".format(*self.principal_planes_z_mm)),
            ("nodal points", "z = {:.6g}, {:.6g} mm".format(*self.nodal_points_z_mm)),
            ("focal points", "z = {:.6g}, {:.6g} mm".format(*self.focal_points_z_mm)),
            ("Lagrange invariant", f"{self.lagrange_invariant_mm:.6g} mm"),
            ("magnification", f"{self.magnification:.6g}"),
            ("paraxial image", f"z = {self.paraxial_image_z_mm:.6g} mm, height {self.paraxial_image_height_mm:.6g} mm"),
            ("detector past it by", f"{self.image_defocus_mm:.6g} mm"),
            ("object / image index", f"{self.object_index:.6g} / {self.image_index:.6g}"),
        ]
        width = max(len(label) for label, _ in rows)
        title = f"First-order data: {self.configuration.system.name or 'system'}"
        return "\n".join([title] + [f"  {label:<{width}}  {value}" for label, value in rows])


def first_order(configuration: Configuration, wavelength_um: float | None = None) -> FirstOrder:
    """The first-order (paraxial) data of *configuration*: focal lengths and points, pupils, NAs, magnification."""

    p = Paraxial(configuration, wavelength_um)
    k = p.last
    epd, image_na = p.entrance_pupil.diameter_mm, p.image_na
    front_focal = float(p.z[p.first] + p.ffl)
    back_focal = float(p.z[k] + np.sign(p.n_out[k]) * p.bfl)
    image_z = p.image_z
    chief_height = float(p.chief.y[k] + (image_z - p.z[k]) * p.chief.u[k]) if np.isfinite(image_z) else np.inf
    return FirstOrder(
        configuration=configuration,
        wavelength_um=p.wavelength_um,
        efl_mm=float(p.efl),
        bfl_mm=float(p.bfl),
        ffl_mm=float(p.ffl),
        f_number=float(p.efl / epd) if np.isfinite(epd) else np.nan,
        working_f_number=float(0.5 / image_na) if image_na else np.inf,
        object_na=p.object_na,
        image_na=image_na,
        entrance_pupil=p.entrance_pupil,
        exit_pupil=p.exit_pupil,
        stop_radius_mm=float(p.stop_radius),
        principal_planes_z_mm=p.principal_planes,
        nodal_points_z_mm=p.nodal_points,
        focal_points_z_mm=(front_focal, back_focal),
        lagrange_invariant_mm=p.lagrange_invariant,
        magnification=float(p.magnification),
        paraxial_image_z_mm=float(image_z),
        paraxial_image_height_mm=chief_height,
        image_defocus_mm=float((p.z[-1] - image_z) * np.sign(p.n_out[k])),
        object_index=p.n_object,
        image_index=p.n_image,
    )


__all__ = ["FirstOrder", "first_order"]
