"""Spot-diagram statistics from traced pupil bundles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...sequential.fields import PupilTrace


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values = np.asarray(values)[order]
    weights = np.asarray(weights)[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(quantile, cumulative, values))


@dataclass
class SpotData:
    """Weighted image-plane spot statistics for one field point."""

    field_y: float
    centroid: np.ndarray  # (2,) image-plane centroid (mm)
    relative_um: np.ndarray  # (M, 2) ray offsets from the centroid (um)
    weights: np.ndarray  # (M,) pupil-area weights (unit sum)
    rms_radius_um: float
    rms_x_um: float
    rms_y_um: float

    def ee_radius_um(self, fraction: float = 0.8) -> float:
        """Encircled-energy radius at *fraction* (weighted quantile)."""

        radius = np.linalg.norm(self.relative_um, axis=1)
        return weighted_quantile(radius, self.weights, fraction)


def _spot_from_arrays(field_y: float, points: np.ndarray, weights: np.ndarray) -> SpotData:
    centroid = np.sum(points * weights[:, None], axis=0)
    relative_um = (points - centroid) * 1e3
    radius_um = np.linalg.norm(relative_um, axis=1)
    return SpotData(
        field_y=field_y,
        centroid=centroid,
        relative_um=relative_um,
        weights=weights,
        rms_radius_um=float(np.sqrt(weighted_mean(radius_um**2, weights))),
        rms_x_um=float(np.sqrt(weighted_mean(relative_um[:, 0] ** 2, weights))),
        rms_y_um=float(np.sqrt(weighted_mean(relative_um[:, 1] ** 2, weights))),
    )


def spot_data(pupil: PupilTrace) -> SpotData:
    """Compute area-weighted spot statistics from a pupil trace."""

    return _spot_from_arrays(pupil.field.y, pupil.image_points[:, :2], pupil.weights)


def spot_data_from_points(
    points: np.ndarray, *, field_label: float = 0.0, weights: np.ndarray | None = None
) -> SpotData:
    """Spot statistics from raw ``(N, 2)`` image-plane points (mm), equal-weighted
    by default. For engines with no :class:`PupilTrace` — e.g. the 2-D
    non-sequential engine's ``Screen2D.coordinates()`` — this is the same
    convergence metric (RMS radius from the centroid, in um) without needing a
    sequential pupil trace.
    """

    points = np.asarray(points, dtype=float)
    if weights is None:
        weights = np.full(len(points), 1.0 / len(points))
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
    return _spot_from_arrays(field_label, points, weights)


__all__ = ["SpotData", "spot_data", "spot_data_from_points", "weighted_mean", "weighted_quantile"]
