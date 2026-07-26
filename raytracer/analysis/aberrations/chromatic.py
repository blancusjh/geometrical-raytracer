"""Axial and lateral color from tracing the same prescription at several
wavelengths.

``OpticalSystem.wavelength_um`` is a single fixed value — this package's
tracer is fundamentally monochromatic per instance — so "chromatic
aberration" here means literally rebuilding the system at each wavelength
of interest (same surfaces, same fixed object plane) and comparing where
rays land. That only produces something meaningful once a material's
``index()`` actually varies with wavelength, i.e. an
:class:`~raytracer.core.materials.AbbeMaterial` rather than a
:class:`~raytracer.core.materials.ConstantIndex` — the DUV materials in
this package are only ever evaluated at one wavelength and have no real
dispersion to show.

Two classical quantities:

- **Axial (longitudinal) color**: the on-axis paraxial focus position
  shifts along z with wavelength, since the system's optical power itself
  is index-, hence wavelength-, dependent. Found from a single differential
  ray pair per wavelength, the same technique
  :func:`~raytracer.sequential.paraxial.differential_conjugates` uses to
  recover an object plane, but solving for the *image*-side focus instead
  since here the object plane is fixed and known.
- **Lateral color**: at the *reference* wavelength's paraxial image plane,
  the chief ray's image height for an off-axis field point still shifts
  slightly with wavelength (color-dependent magnification) — this is
  measured at that fixed plane, not each wavelength's own shifted focus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...sequential.fields import FieldPoint, chief_ray_slopes, trace_from_object
from ...sequential.paraxial import direction_from_slopes
from ...sequential.system import OpticalSystem
from ...sequential.trace import SequentialTracer


@dataclass
class AxialColor:
    """Paraxial on-axis focus position vs. wavelength."""

    wavelengths_um: np.ndarray
    focus_z_mm: np.ndarray
    reference_wavelength_um: float

    @property
    def focus_shift_mm(self) -> np.ndarray:
        """Focus position relative to the reference wavelength's own focus."""

        reference_index = int(np.argmin(np.abs(self.wavelengths_um - self.reference_wavelength_um)))
        return self.focus_z_mm - self.focus_z_mm[reference_index]


@dataclass
class LateralColor:
    """Chief-ray image height vs. wavelength, at the reference focal plane."""

    wavelengths_um: np.ndarray
    field_height_mm: float
    image_height_mm: np.ndarray
    reference_wavelength_um: float

    @property
    def lateral_color_um(self) -> np.ndarray:
        """Image-height deviation from the reference wavelength, in µm."""

        reference_index = int(np.argmin(np.abs(self.wavelengths_um - self.reference_wavelength_um)))
        return (self.image_height_mm - self.image_height_mm[reference_index]) * 1e3


def _system_at_wavelength(system: OpticalSystem, wavelength_um: float) -> OpticalSystem:
    """A same-prescription copy of *system*, rebuilt at a different wavelength."""

    return OpticalSystem(
        system.rows, wavelength_um=wavelength_um,
        object_space=system.object_space, object_z=system.object_z, name=system.name,
    )


def _paraxial_focus_z(tracer: SequentialTracer, *, eps: float = 1e-4) -> float:
    """On-axis paraxial focus z, from one differential (near-axial) ray.

    The axial ray (zero height, zero slope) stays exactly on-axis; a
    second ray differing only by a small slope ``eps`` reaches the tabulated
    image plane at some height and local direction — extrapolating that
    ray's straight line back to the axis (y=0) gives the paraxial focus,
    which may sit in front of or behind the nominal image plane once the
    wavelength no longer matches the design wavelength.
    """

    object_z = tracer.system.object_z
    if object_z is None:
        raise ValueError("system.object_z is unset")
    origin = np.array([0.0, 0.0, object_z])
    angled = tracer.trace(origin, direction_from_slopes(0.0, eps), keep_path=False)
    if not angled.ok:
        raise RuntimeError("Differential ray did not survive the system")
    y2 = angled.image_point[1]
    slope = angled.direction[1] / angled.direction[2]
    return float(tracer.system.image_z - y2 / slope)


def axial_color(
    system: OpticalSystem,
    wavelengths_um,
    *,
    reference_wavelength_um: float,
) -> AxialColor:
    """Paraxial focus position vs. wavelength, object plane held fixed.

    ``system.object_z`` must already be set (e.g. via
    :func:`~raytracer.sequential.paraxial.solve_object_plane` at the
    reference wavelength) — the same physical object plane is reused for
    every wavelength; only the resulting focus position changes.
    """

    wavelengths_um = np.asarray(wavelengths_um, dtype=float)
    focus_z = []
    for wl in wavelengths_um:
        wl_tracer = SequentialTracer(_system_at_wavelength(system, float(wl)))
        focus_z.append(_paraxial_focus_z(wl_tracer))

    return AxialColor(
        wavelengths_um=wavelengths_um,
        focus_z_mm=np.asarray(focus_z),
        reference_wavelength_um=reference_wavelength_um,
    )


def lateral_color(
    system: OpticalSystem,
    wavelengths_um,
    *,
    field_height_mm: float,
    reference_wavelength_um: float,
    stop_index: int | None = None,
) -> LateralColor:
    """Chief-ray image height vs. wavelength, at the fixed nominal image plane.

    Traces the chief ray for the same object field height at each
    wavelength (warm-started from the previous wavelength's solution,
    since they're close together) and reads its height at the system's
    tabulated image plane — deliberately *not* each wavelength's own
    shifted focus (see :func:`axial_color`), since lateral color is a
    statement about magnification differing by wavelength at one common
    plane, not about focus.
    """

    wavelengths_um = np.asarray(wavelengths_um, dtype=float)
    image_heights = []
    guess = None
    field = FieldPoint(y=float(field_height_mm))
    for wl in wavelengths_um:
        wl_system = _system_at_wavelength(system, float(wl))
        wl_tracer = SequentialTracer(wl_system)
        _, chief_slope = chief_ray_slopes(
            wl_tracer, field, stop_index=stop_index, initial_guess=guess
        )
        guess = (0.0, chief_slope)
        result = trace_from_object(wl_tracer, (0.0, field.y), (0.0, chief_slope))
        image_heights.append(float(result.image_point[1]))

    return LateralColor(
        wavelengths_um=wavelengths_um,
        field_height_mm=field_height_mm,
        image_height_mm=np.asarray(image_heights),
        reference_wavelength_um=reference_wavelength_um,
    )


__all__ = ["AxialColor", "LateralColor", "axial_color", "lateral_color"]
