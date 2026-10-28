"""Geometric aberrations on a specified detector plane, without a wave model."""

from dataclasses import dataclass

import numpy as np

from ...math.transforms import RigidTransform
from ...propagation.fields import PupilTrace


@dataclass(frozen=True)
class GeometricAberrations:
    """Lengths in mm; RMS sizes refer to the area-weighted spot centroid.

    Transverse aberrations refer to ``reference_xy_mm`` (the chief ray by
    default, or an explicitly supplied paraxial image). Best focus is a
    translation along the detector's local z axis, with orientation fixed.
    """

    reference_xy_mm: np.ndarray
    centroid_xy_mm: np.ndarray
    transverse_mm: np.ndarray
    rms_radius_mm: float
    best_focus_shift_mm: float
    best_focus_rms_mm: float
    geometric_throughput: float
    valid_rays: int
    total_rays: int


def geometric_aberrations(
    pupil: PupilTrace,
    *,
    image_frame: RigidTransform | None = None,
    reference_xy_mm=None,
) -> GeometricAberrations:
    """Measure a traced pupil; retain vignetting in the throughput denominator."""

    frame = image_frame or RigidTransform.identity()
    if not pupil.valid.any():
        raise ValueError("geometric aberrations require at least one surviving ray")
    points = frame.to_local(pupil.image_points)[:, :2]
    directions = frame.direction_to_local(pupil.directions)
    if np.any(np.abs(directions[:, 2]) < 1e-14):
        raise ValueError("best-focus analysis requires rays transverse to the detector")
    if reference_xy_mm is None:
        reference = frame.to_local(pupil.chief.image_point)[:2]
    else:
        reference = np.asarray(reference_xy_mm, dtype=float)
    if reference.shape != (2,) or not np.all(np.isfinite(reference)):
        raise ValueError("supply a finite two-dimensional reference image point")
    weights = pupil.weights / pupil.weights.sum()
    centroid = np.sum(points * weights[:, None], axis=0)
    centered = points - centroid
    slopes = directions[:, :2] / directions[:, 2:]
    slope_mean = np.sum(slopes * weights[:, None], axis=0)
    slope_spread = slopes - slope_mean
    denominator = np.sum(weights[:, None] * slope_spread**2)
    shift = (
        np.nan
        if denominator <= np.finfo(float).tiny
        else float(-np.sum(weights[:, None] * centered * slope_spread) / denominator)
    )
    rms = float(np.sqrt(np.sum(weights[:, None] * centered**2)))
    focused_rms = (
        np.nan
        if not np.isfinite(shift)
        else float(np.sqrt(np.sum(weights[:, None] * (centered + shift * slope_spread) ** 2)))
    )
    return GeometricAberrations(
        reference,
        centroid,
        points - reference,
        rms,
        shift,
        focused_rms,
        pupil.geometric_throughput,
        int(pupil.valid.sum()),
        len(pupil.batch),
    )


def axial_intercepts(points, directions, *, frame: RigidTransform | None = None):
    """Return (z, miss distance) at closest approach to the reference z axis.

    A skew ray generally misses the axis; its returned miss distance must not
    be discarded when interpreting longitudinal spherical aberration.
    Axial rays have no unique intercept and return NaN.
    """

    frame = frame or RigidTransform.identity()
    points = frame.to_local(points)
    directions = frame.direction_to_local(directions)
    denominator = np.sum(directions[..., :2] ** 2, axis=-1)
    distance = np.divide(
        -np.sum(points[..., :2] * directions[..., :2], axis=-1),
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 1e-28,
    )
    closest = points + distance[..., None] * directions
    return closest[..., 2], np.linalg.norm(closest[..., :2], axis=-1)
