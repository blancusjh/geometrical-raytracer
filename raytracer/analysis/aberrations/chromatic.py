"""Axial and lateral color from tracing the same prescription at several
wavelengths.

``OpticalSystem.wavelength_um`` is a single fixed value — this package's
tracer is fundamentally monochromatic per instance — so "chromatic
aberration" here means literally rebuilding the system at each wavelength
of interest (same surfaces, same fixed object plane) and comparing where
rays land. That only produces something meaningful once a material's
``index()`` actually varies with wavelength, i.e. an
:class:`~raytracer.physics.materials.AbbeMaterial` rather than a
:class:`~raytracer.physics.materials.ConstantIndex` — the DUV materials in
this package are only ever evaluated at one wavelength and have no real
dispersion to show.

Three classical quantities:

- **Axial (longitudinal) color**: the on-axis paraxial focus position
  shifts along z with wavelength, since the system's optical power itself
  is index-, hence wavelength-, dependent. Found from one differential ray
  per wavelength — the same technique
  :func:`~raytracer.propagation.paraxial.differential_conjugates` uses to
  recover an object plane, but solving for the *image*-side focus instead
  since here the object plane is fixed and known.
- **Lateral color**: at the *reference* wavelength's paraxial image plane,
  the chief ray's image height for an off-axis field point still shifts
  slightly with wavelength (color-dependent magnification) — this is
  measured at that fixed plane, not each wavelength's own shifted focus.
- **Chromatic spots**: the geometric blur each wavelength forms at one
  common image plane, all referred to one common origin so the colors can
  be overlaid and compared (see :func:`chromatic_spots`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from ...propagation.fields import (
    FieldPoint,
    PupilSampling,
    chief_ray_slopes,
    trace_from_object,
    trace_pupil,
)
from ...propagation.paraxial import direction_from_slopes
from ...design.system import OpticalSystem
from ...propagation.sequential import SequentialTracer


def _reference_index(wavelengths_um: np.ndarray, reference_wavelength_um: float) -> int:
    """Index of the sample closest to *reference_wavelength_um*.

    Raises if the nearest sample is not actually close, rather than
    silently snapping to whichever end of the sweep happens to be nearest
    and reporting shifts against a reference the caller never chose.
    """

    wavelengths_um = np.asarray(wavelengths_um, dtype=float)
    index = int(np.argmin(np.abs(wavelengths_um - reference_wavelength_um)))
    nearest = wavelengths_um[index]
    spacing = np.diff(np.sort(wavelengths_um))
    tolerance = float(spacing.min()) if spacing.size else 0.0
    if abs(nearest - reference_wavelength_um) > max(tolerance, 1e-9):
        raise ValueError(
            f"reference_wavelength_um={reference_wavelength_um} is not among the "
            f"sampled wavelengths (nearest is {nearest}); include it so shifts are "
            "measured against the reference you actually asked for"
        )
    return index


@dataclass
class AxialColor:
    """Paraxial on-axis focus position vs. wavelength."""

    wavelengths_um: np.ndarray
    focus_z_mm: np.ndarray
    reference_wavelength_um: float

    @property
    def focus_shift_mm(self) -> np.ndarray:
        """Focus position relative to the reference wavelength's own focus."""

        index = _reference_index(self.wavelengths_um, self.reference_wavelength_um)
        return self.focus_z_mm - self.focus_z_mm[index]


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

        index = _reference_index(self.wavelengths_um, self.reference_wavelength_um)
        return (self.image_height_mm - self.image_height_mm[index]) * 1e3


def _system_at_wavelength(system: OpticalSystem, wavelength_um: float) -> OpticalSystem:
    """A same-prescription copy of *system*, rebuilt at a different wavelength.

    The surface rows are deep-copied: ``OpticalSystem`` keeps a shallow
    ``list(rows)``, so sharing them would let a later ``set_thickness`` on
    one wavelength's copy silently mutate the caller's own system.
    """

    return OpticalSystem(
        copy.deepcopy(system.rows), wavelength_um=wavelength_um,
        object_space=system.object_space, object_z=system.object_z, name=system.name,
    )


def _paraxial_focus_z(tracer: SequentialTracer, *, probe_height_mm: float | None = None) -> float:
    """On-axis paraxial focus z, from one differential (near-axial) ray.

    The axial ray (zero height, zero slope) stays exactly on-axis; a second
    ray launched at a small slope reaches the tabulated image plane at some
    height and local direction — extrapolating that ray's straight line
    back to the axis (y=0) gives the paraxial focus, which may sit in front
    of or behind the nominal image plane once the wavelength no longer
    matches the design wavelength.

    The probe ray is specified by the *height* it reaches at the first
    surface rather than by its launch slope, because the slope that keeps a
    ray paraxial depends entirely on how far away the object is: a fixed
    slope that is safely paraxial for an object 80 mm away overflows the
    aperture outright for one 100 m away. ``probe_height_mm`` defaults to
    0.5% of the smallest defined clear semidiameter — well inside the
    paraxial region, and far enough from zero to stay numerically
    well-conditioned — or, for a system that defines no semidiameters at
    all, to a small fraction of the object distance.
    """

    system = tracer.system
    object_z = system.object_z
    if object_z is None:
        raise ValueError("system.object_z is unset")

    first_vertex = float(system.vertices[0])
    reach = first_vertex - object_z
    if probe_height_mm is None:
        semidiameters = [r.semidiameter for r in system.rows if r.semidiameter is not None]
        probe_height_mm = (
            0.005 * min(semidiameters) if semidiameters else 1e-4 * abs(reach)
        )

    origin = np.array([0.0, 0.0, object_z])
    angled = tracer.trace(
        origin, direction_from_slopes(0.0, probe_height_mm / reach), keep_path=False
    )
    if not angled.ok:
        raise RuntimeError(
            f"Differential ray (probe height {probe_height_mm:.4g} mm at the first "
            "surface) did not survive the system; pass an explicit probe_height_mm"
        )
    y2 = angled.image_point[1]
    slope = angled.direction[1] / angled.direction[2]
    if abs(slope) < 1e-12:
        raise RuntimeError(
            "The differential ray leaves the system parallel to the axis, so it "
            "has no finite focus; an afocal system has no axial-color curve"
        )
    return float(system.image_z - y2 / slope)


def axial_color(
    system: OpticalSystem,
    wavelengths_um,
    *,
    reference_wavelength_um: float,
) -> AxialColor:
    """Paraxial focus position vs. wavelength, object plane held fixed.

    ``system.object_z`` must already be set (e.g. via
    :func:`~raytracer.propagation.paraxial.solve_object_plane` at the
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


@dataclass
class ChromaticSpots:
    """Per-wavelength geometric spots sharing one common reference frame.

    ``offsets_um[i]`` is the ``(M_i, 2)`` array of image-plane ray
    intersections for ``wavelengths_um[i]``, in µm, measured from a single
    common origin (the reference wavelength's chief-ray image point) on a
    single common image plane. Both "commons" matter: recentring each
    wavelength on *its own* centroid — what
    :func:`~raytracer.analysis.imaging.spots.spot_data` does, correctly,
    for a monochromatic spot — would subtract out exactly the color
    separation this class exists to show.

    Ray counts differ between wavelengths when vignetting does, so this is
    a list of arrays rather than one rectangular array.
    """

    wavelengths_um: np.ndarray
    offsets_um: list[np.ndarray]
    reference_wavelength_um: float
    field: FieldPoint

    @property
    def rms_radius_um(self) -> np.ndarray:
        """Per-wavelength RMS radius about the *common* origin."""

        return np.array(
            [float(np.sqrt(np.mean(np.sum(o**2, axis=1)))) for o in self.offsets_um]
        )

    @property
    def polychromatic_rms_um(self) -> float:
        """RMS radius of every wavelength's rays pooled together.

        The single number that answers "how big is the white-light blur",
        as opposed to the per-wavelength figures, which each ignore the
        others' displacement.
        """

        pooled = np.vstack(self.offsets_um)
        return float(np.sqrt(np.mean(np.sum(pooled**2, axis=1))))


def chromatic_spots(
    system: OpticalSystem,
    wavelengths_um,
    *,
    na_object_sine: float,
    reference_wavelength_um: float,
    field: FieldPoint | float = 0.0,
    sampling: PupilSampling | None = None,
    stop_index: int | None = None,
) -> ChromaticSpots:
    """Trace one pupil bundle per wavelength onto a shared image plane.

    Every wavelength is traced through the *same* prescription — including
    the same tabulated image plane — so a system with axial color puts each
    color's best focus somewhere else and only one of them lands sharp.
    That defocus blur is the point: it is what axial color looks like on a
    detector, and it is why the reference wavelength's spot can be tight
    while the rest are not.

    The common transverse origin is the reference wavelength's chief-ray
    image point, so on axis the offsets are simply image heights, and off
    axis any bodily displacement of one color's bundle relative to another
    is lateral color.
    """

    if not isinstance(field, FieldPoint):
        field = FieldPoint(y=float(field))
    wavelengths_um = np.asarray(wavelengths_um, dtype=float)
    _reference_index(wavelengths_um, reference_wavelength_um)  # validate early
    sampling = sampling or PupilSampling(kind="rings", radial=8, azimuth=32)

    def bundle(wavelength_um: float, guess):
        tracer = SequentialTracer(_system_at_wavelength(system, wavelength_um))
        slopes = chief_ray_slopes(
            tracer, field, stop_index=stop_index, initial_guess=guess
        )
        pupil = trace_pupil(
            tracer, field, na_object_sine=na_object_sine, sampling=sampling,
            chief_slope=slopes, stop_index=stop_index,
        )
        return pupil, slopes

    reference_pupil, guess = bundle(float(reference_wavelength_um), None)
    origin = reference_pupil.chief.image_point[:2]

    offsets = []
    for wavelength_um in wavelengths_um:
        pupil, _ = bundle(float(wavelength_um), guess)
        offsets.append((pupil.image_points[:, :2] - origin) * 1e3)

    return ChromaticSpots(
        wavelengths_um=wavelengths_um,
        offsets_um=offsets,
        reference_wavelength_um=float(reference_wavelength_um),
        field=field,
    )


__all__ = [
    "AxialColor",
    "LateralColor",
    "ChromaticSpots",
    "axial_color",
    "lateral_color",
    "chromatic_spots",
]
