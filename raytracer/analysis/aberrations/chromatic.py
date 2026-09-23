"""Geometric color at a fixed detector, with explicit spectral and pupil weights."""

from dataclasses import dataclass

import numpy as np

from ...design.system import OpticalSystem
from ...propagation.fields import FieldPoint, PupilSampling, PupilTrace, trace_pupil
from ...propagation.paraxial import ParaxialModel
from ...propagation.sequential import SequentialTracer


def _wavelengths(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError("wavelengths must be a nonempty vector of finite positive values in um")
    if len(np.unique(values)) != len(values):
        raise ValueError("wavelengths must be distinct")
    return values


def _reference_index(wavelengths_um, reference_wavelength_um):
    wavelengths_um = _wavelengths(wavelengths_um)
    matches = np.flatnonzero(
        np.isclose(wavelengths_um, reference_wavelength_um, atol=1e-12, rtol=0)
    )
    if len(matches) != 1:
        raise ValueError("reference_wavelength_um must occur exactly once in wavelengths_um")
    return int(matches[0])


def spectral_weights(values, count):
    """Normalize relative incident powers, including detector response if desired.

    These are discrete band weights, not a sampled spectral density. For a
    density the caller must include the wavelength integration widths.
    """
    weights = np.ones(count) if values is None else np.array(values, dtype=float, copy=True)
    if weights.shape != (count,) or not np.all(np.isfinite(weights) & (weights >= 0)):
        raise ValueError("spectral weights must match wavelengths and be finite and nonnegative")
    if not np.any(weights > 0):
        raise ValueError("at least one spectral weight must be positive")
    weights /= weights.max()
    return weights / weights.sum()


def _tracer(source):
    return source if isinstance(source, SequentialTracer) else SequentialTracer(source)


@dataclass
class AxialColor:
    wavelengths_um: np.ndarray
    focus_z_mm: np.ndarray
    reference_wavelength_um: float

    @property
    def focus_shift_mm(self):
        index = _reference_index(self.wavelengths_um, self.reference_wavelength_um)
        return self.focus_z_mm - self.focus_z_mm[index]


def axial_color(system: OpticalSystem, wavelengths_um, *, reference_wavelength_um: float):
    """Gaussian image conjugate versus wavelength in system coordinates (mm).

    The object stays fixed; +/- infinity denotes a collimated axial input.
    Analytic ABCD avoids mixing finite-aperture spherical aberration into color.
    Undefined/afocal image positions are rejected, not plotted as finite foci.
    """
    wavelengths_um = _wavelengths(wavelengths_um)
    _reference_index(wavelengths_um, reference_wavelength_um)
    if system.object_z is None or np.isnan(system.object_z):
        raise ValueError(
            "axial color requires object_z, including infinity for a collimated object"
        )
    foci = np.array(
        [
            ParaxialModel(system.at_wavelength(wl)).image_conjugate_z(system.object_z)
            for wl in wavelengths_um
        ]
    )
    if not np.all(np.isfinite(foci)):
        raise ValueError("axial color is undefined for an afocal image conjugate")
    return AxialColor(wavelengths_um, foci, reference_wavelength_um)


@dataclass
class LateralColor:
    wavelengths_um: np.ndarray
    field_height_mm: float
    image_height_mm: np.ndarray
    reference_wavelength_um: float

    @property
    def lateral_color_um(self):
        index = _reference_index(self.wavelengths_um, self.reference_wavelength_um)
        return (self.image_height_mm - self.image_height_mm[index]) * 1e3


@dataclass
class ChromaticSpots:
    """All colors expressed in the same local detector frame.

    Offsets are in um about the reference-wavelength chief ray. Pupil weights
    retain their pre-vignetting normalization. ``rms_radius_um`` and
    ``polychromatic_rms_um`` measure about that common reference;
    ``centroid_rms_um`` instead measures about the combined centroid.
    Geometric throughput excludes absorption and Fresnel losses.
    """

    wavelengths_um: np.ndarray
    offsets_um: list[np.ndarray]
    reference_wavelength_um: float
    field: FieldPoint
    pupil_weights: list[np.ndarray]
    spectral_weights: np.ndarray
    chief_offsets_um: np.ndarray
    pupils: tuple[PupilTrace, ...]

    @property
    def wavelength_throughput(self):
        return np.array([weights.sum() for weights in self.pupil_weights])

    @property
    def geometric_throughput(self):
        return float(self.spectral_weights @ self.wavelength_throughput)

    @property
    def detected_spectral_weights(self):
        power = self.spectral_weights * self.wavelength_throughput
        return power / power.sum()

    @property
    def rms_radius_um(self):
        return np.array(
            [
                np.sqrt(np.sum(weights * np.sum(offsets**2, axis=1)) / weights.sum())
                if weights.sum() > 0
                else np.nan
                for offsets, weights in zip(self.offsets_um, self.pupil_weights)
            ]
        )

    def _pooled(self):
        weights = np.concatenate(
            [spectrum * pupil for spectrum, pupil in zip(self.spectral_weights, self.pupil_weights)]
        )
        return np.vstack(self.offsets_um), weights / weights.sum()

    @property
    def polychromatic_rms_um(self):
        points, weights = self._pooled()
        return float(np.sqrt(weights @ np.sum(points**2, axis=1)))

    @property
    def centroid_offset_um(self):
        points, weights = self._pooled()
        return weights @ points

    @property
    def centroid_rms_um(self):
        points, weights = self._pooled()
        centered = points - weights @ points
        return float(np.sqrt(weights @ np.sum(centered**2, axis=1)))


def chromatic_spots(
    system: OpticalSystem | SequentialTracer,
    wavelengths_um,
    *,
    reference_wavelength_um: float,
    na_object_sine: float | None = None,
    field: FieldPoint | float = 0.0,
    sampling: PupilSampling | None = None,
    stop_index: int | None = None,
    wavelength_weights=None,
) -> ChromaticSpots:
    """Polychromatic spots on one fixed detector, including placed 3-D systems.

    Accept a tracer to preserve virtual-path and intersection options. The
    pupil is aimed separately at each wavelength; its declared measure is
    uniform stop area (or the legacy launch-cone measure if explicitly set).
    Spectral weights represent incident power in that measure, not ray counts.
    """
    wavelengths_um = _wavelengths(wavelengths_um)
    reference = _reference_index(wavelengths_um, reference_wavelength_um)
    spectrum = spectral_weights(wavelength_weights, len(wavelengths_um))
    template = _tracer(system)
    field = field if isinstance(field, FieldPoint) else FieldPoint(y=float(field))
    sampling = sampling or PupilSampling(kind="gauss", radial=8, azimuth=32)
    pupils = tuple(
        trace_pupil(
            template.with_system(template.system.at_wavelength(wl)),
            field,
            na_object_sine=na_object_sine,
            sampling=sampling,
            stop_index=stop_index,
        )
        for wl in wavelengths_um
    )
    frame = template.system.image_frame
    chief_points = np.array([frame.to_local(pupil.chief.image_point)[:2] for pupil in pupils])
    if not np.all(np.isfinite(chief_points)):
        raise ValueError("chromatic reference requires a finite chief-ray image at each wavelength")
    origin = chief_points[reference]
    offsets = [(frame.to_local(p.image_points)[:, :2] - origin) * 1e3 for p in pupils]
    weights = [p.sample_weights[p.valid].copy() for p in pupils]
    report = ChromaticSpots(
        wavelengths_um,
        offsets,
        reference_wavelength_um,
        field,
        weights,
        spectrum,
        (chief_points - origin) * 1e3,
        pupils,
    )
    if report.geometric_throughput <= 0:
        raise ValueError("no weighted ray reaches the detector")
    return report


def lateral_color(
    system: OpticalSystem | SequentialTracer,
    wavelengths_um,
    *,
    field_height_mm: float,
    reference_wavelength_um: float,
    stop_index: int | None = None,
) -> LateralColor:
    """Finite-field chief-ray y at the fixed detector, in local detector axes."""
    report = chromatic_spots(
        system,
        wavelengths_um,
        field=FieldPoint(y=field_height_mm),
        reference_wavelength_um=reference_wavelength_um,
        stop_index=stop_index,
        sampling=PupilSampling(kind="gauss", radial=2, azimuth=4),
    )
    frame = _tracer(system).system.image_frame
    heights = np.array([frame.to_local(p.chief.image_point)[1] for p in report.pupils])
    return LateralColor(report.wavelengths_um, field_height_mm, heights, reference_wavelength_um)


__all__ = [
    "AxialColor",
    "LateralColor",
    "ChromaticSpots",
    "axial_color",
    "lateral_color",
    "chromatic_spots",
]
