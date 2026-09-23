"""Classical monochromatic Seidel sums from two paraxial reference rays.

The five sums are optical-path lengths in mm, normalized to the selected
stop radius and meridional field. See docs/third_order_validation.md for
signs, equations, references and asymptotic tests. No wavefront fit is used.
"""

from dataclasses import dataclass

import numpy as np

from ...design.rows import SurfaceKind
from ...design.system import OpticalSystem
from ...propagation.fields import FieldPoint
from ...propagation.paraxial import ParaxialModel
from ...surfaces.apertures import CircularAperture
from ...surfaces.profile import AsphereProfile

NAMES = ("spherical", "coma", "astigmatism", "petzval", "distortion")


@dataclass
class SeidelCoefficients:
    field: FieldPoint
    pupil_radius_mm: float
    stop_index: int
    spherical_surface_mm: np.ndarray
    aspheric_surface_mm: np.ndarray
    marginal_y_mm: np.ndarray
    chief_y_mm: np.ndarray
    marginal_slope_before: np.ndarray
    marginal_slope_after: np.ndarray
    chief_slope_before: np.ndarray
    chief_slope_after: np.ndarray
    signed_indices_before: np.ndarray
    signed_indices_after: np.ndarray
    invariant_mm: float
    image_z_mm: float
    gaussian_image_z_mm: float
    gaussian_image_height_mm: float
    petzval_curvature_per_mm: float

    @property
    def surface_sums_mm(self):
        return self.spherical_surface_mm + self.aspheric_surface_mm

    @property
    def sums_mm(self):
        return self.surface_sums_mm.sum(axis=0)

    @property
    def named_sums_mm(self):
        return dict(zip(NAMES, self.sums_mm.tolist()))

    def transverse(self, pupil_xy, *, field_scale=1.0):
        """Third-order image error in mm at the Gaussian image plane.

        Pupil x/y are dimensionless paraxial stop coordinates. ``field_scale``
        scales object height or tan(field angle), not angle itself. This is
        the error relative to the Gaussian image, including distortion.
        """
        pupil = np.asarray(pupil_xy, dtype=float)
        if (
            pupil.shape[-1:] != (2,)
            or not np.all(np.isfinite(pupil))
            or not np.isfinite(field_scale)
        ):
            raise ValueError(
                "pupil coordinates must be finite (..., 2); field_scale must be finite"
            )
        n_u = self.signed_indices_after[-1] * self.marginal_slope_after[-1]
        if abs(n_u) < 1e-15:
            raise ValueError("transverse Seidel aberration is undefined for an afocal image")
        spherical, coma, astigmatism, petzval, distortion = self.sums_mm
        x, y = pupil[..., 0], pupil[..., 1]
        h = field_scale
        radius_squared = x * x + y * y
        dx = (
            spherical * x * radius_squared
            + 2 * coma * h * x * y
            + (astigmatism + petzval) * h * h * x
        )
        dy = (
            spherical * y * radius_squared
            + coma * h * (x * x + 3 * y * y)
            + (3 * astigmatism + petzval) * h * h * y
            + distortion * h**3
        )
        return np.stack([dx, dy], axis=-1) / (2 * n_u)

    @property
    def field_curvatures_per_mm(self):
        """Geometric curvatures C with z(h)=z_G+C*h²/2 near the axis.

        Tangential/sagittal curvatures require a nonzero reference field.
        Petzval curvature is computed directly and remains defined on axis.
        """
        if self.invariant_mm == 0:
            return {
                "tangential": np.nan,
                "sagittal": np.nan,
                "petzval": self.petzval_curvature_per_mm,
            }
        astigmatism, petzval = self.sums_mm[2:4]
        factor = -self.signed_indices_after[-1] / self.invariant_mm**2
        return {
            "tangential": float(factor * (3 * astigmatism + petzval)),
            "sagittal": float(factor * (astigmatism + petzval)),
            "petzval": self.petzval_curvature_per_mm,
        }

    def summary(self):
        return "\n".join(f"{name}: {value:+.8g} mm" for name, value in self.named_sums_mm.items())


def seidel_coefficients(
    system: OpticalSystem, field: FieldPoint, *, pupil_radius_mm=None, stop_index=None
):
    """Surface ledger for a centered conic/even-asphere system.

    Reflections use signed indices and signed axial thicknesses. A global
    rigid placement is allowed: all returned coordinates are system-local.
    The detector need not be at Gaussian focus; transverse predictions are.
    Only the h^4 asphere term enters third order; higher sag orders do not.
    Cartesian ovals need an explicit vertex expansion and are rejected here.
    """
    model = ParaxialModel(system)
    if not isinstance(field, FieldPoint) or field.x != 0:
        raise ValueError("Seidel sums require a meridional FieldPoint (x=0)")
    if any(not isinstance(row.profile, AsphereProfile) for row in system.rows):
        raise ValueError("Seidel currently supports conic/even-asphere profiles only")
    stop = system.stop_index if stop_index is None else stop_index
    if stop is None or not 0 <= stop < len(system.rows):
        raise ValueError("Seidel requires a valid stop surface")
    if pupil_radius_mm is None:
        aperture = system.rows[stop].clear_aperture
        if not isinstance(aperture, CircularAperture):
            raise ValueError("supply pupil_radius_mm or a circular stop aperture")
        pupil_radius_mm = aperture.radius
    radius = float(pupil_radius_mm)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("pupil_radius_mm must be finite and positive")
    axial = FieldPoint(kind=field.kind)
    marginal = model.launch_state(axial, radius, stop_index=stop)
    chief = model.launch_state(field, stop_index=stop)
    invariant = marginal[0] * chief[1] - chief[0] * marginal[1]
    initial_chief = chief.copy()
    spherical, aspheric, data = [], [], []
    sign, petzval_sum = 1.0, 0.0
    for index, row in enumerate(system.rows):
        before = sign * system.n_before[index]
        if row.kind is SurfaceKind.MIRROR:
            sign = -sign
        after = sign * system.n_after[index]
        curvature = row.profile.curvature
        height, chief_height = marginal[0], chief[0]
        slope, chief_slope = marginal[1] / before, chief[1] / before
        power = curvature * (after - before)
        marginal[1] -= power * height
        chief[1] -= power * chief_height
        outgoing, chief_outgoing = marginal[1] / after, chief[1] / after
        incidence = before * (slope + curvature * height)
        chief_incidence = before * (chief_slope + curvature * chief_height)
        slope_jump = outgoing / after - slope / before
        reciprocal_power = curvature * (1 / after - 1 / before)
        spherical.append(
            [
                -(incidence**2) * height * slope_jump,
                -incidence * chief_incidence * height * slope_jump,
                -(chief_incidence**2) * height * slope_jump,
                -(invariant**2) * reciprocal_power,
                -chief_incidence
                * (
                    chief_incidence**2 * (1 / after**2 - 1 / before**2) * height
                    - (invariant + chief_incidence * height) * chief_height * reciprocal_power
                ),
            ]
        )
        quartic_departure = row.profile.conic * curvature**3 / 8
        if row.profile.coefficients:
            quartic_departure += row.profile.coefficients[0]
        factor = 8 * quartic_departure * (after - before)
        # Polynomial form remains regular when the marginal height is zero.
        aspheric.append(
            factor
            * np.array(
                [
                    height**4,
                    height**3 * chief_height,
                    height**2 * chief_height**2,
                    0,
                    height * chief_height**3,
                ]
            )
        )
        petzval_sum += reciprocal_power
        data.append(
            [height, chief_height, slope, outgoing, chief_slope, chief_outgoing, before, after]
        )
        marginal[0] += row.thickness * outgoing
        chief[0] += row.thickness * chief_outgoing
    data = np.array(data).T
    object_z = -np.inf if field.kind == "angle" else system.object_z
    gaussian_z = model.image_conjugate_z(object_z)
    if not np.isfinite(gaussian_z):
        raise ValueError("Seidel image diagnostics require a finite Gaussian image")
    final_chief = model.matrix_first_vertex_to_image @ initial_chief
    image_height = (
        final_chief[0] + (gaussian_z - system.image_z) * final_chief[1] / model.n_image_signed
    )
    return SeidelCoefficients(
        field=field,
        pupil_radius_mm=radius,
        stop_index=stop,
        spherical_surface_mm=np.array(spherical),
        aspheric_surface_mm=np.array(aspheric),
        marginal_y_mm=data[0],
        chief_y_mm=data[1],
        marginal_slope_before=data[2],
        marginal_slope_after=data[3],
        chief_slope_before=data[4],
        chief_slope_after=data[5],
        signed_indices_before=data[6],
        signed_indices_after=data[7],
        invariant_mm=float(invariant),
        image_z_mm=system.image_z,
        gaussian_image_z_mm=gaussian_z,
        gaussian_image_height_mm=float(image_height),
        petzval_curvature_per_mm=float(model.n_image_signed * petzval_sum),
    )


__all__ = ["SeidelCoefficients", "seidel_coefficients"]
