"""Per-field aberration metrics: distortion, focus shifts, astigmatism.

Direct port of the reference ``aberration_metrics``: all quantities are
computed from exact rays with pupil-area weights.

Distortion is reported two ways. ``chief_ray_distortion_um`` is the
industry-standard definition (Zemax/OpticStudio convention): the deviation
of the chief ray's image-plane intersection from the paraxial ideal height.
It is unaffected by vignetting/apodization since it depends on a single ray.
``distortion_um`` is the flux-weighted centroid of the full ray bundle
instead; it mixes in coma and vignetting, so it is kept for reference but is
not the default plotted quantity.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..sequential.fields import FieldPoint, PupilSampling, trace_pupil
from ..sequential.trace import SequentialTracer
from .spots import weighted_mean, weighted_quantile


def field_metrics(
    tracer: SequentialTracer,
    fields: Iterable[FieldPoint | float],
    *,
    na_object_sine: float,
    magnification: float,
    sampling: PupilSampling | None = None,
) -> list[dict]:
    """Aberration metrics per field point.

    ``magnification`` supplies the ideal image height ``y * M`` against
    which centroid distortion is measured.
    """

    sampling = sampling or PupilSampling(kind="rings", radial=10, azimuth=48)
    results = []
    for field in fields:
        if not isinstance(field, FieldPoint):
            field = FieldPoint(y=float(field))
        pupil = trace_pupil(
            tracer, field, na_object_sine=na_object_sine, sampling=sampling
        )
        points = pupil.image_points[:, :2]
        directions = pupil.directions
        weights = pupil.weights

        centroid = np.sum(points * weights[:, None], axis=0)
        relative = points - centroid
        radius_um = np.linalg.norm(relative, axis=1) * 1e3
        rms_x_um = np.sqrt(weighted_mean((relative[:, 0] * 1e3) ** 2, weights))
        rms_y_um = np.sqrt(weighted_mean((relative[:, 1] * 1e3) ** 2, weights))
        rms_radius_um = np.sqrt(weighted_mean(radius_um**2, weights))
        ee80_radius_um = weighted_quantile(radius_um, weights, 0.8)

        slopes = directions[:, :2] / directions[:, 2, None]
        mean_slopes = np.sum(slopes * weights[:, None], axis=0)
        slope_relative = slopes - mean_slopes

        def best_focus(axis: int) -> float:
            numerator = np.sum(weights * relative[:, axis] * slope_relative[:, axis])
            denominator = np.sum(weights * slope_relative[:, axis] ** 2)
            return float(-numerator / denominator)

        sagittal_focus = best_focus(0)
        tangential_focus = best_focus(1)
        numerator = np.sum(weights[:, None] * relative * slope_relative)
        denominator = np.sum(weights[:, None] * slope_relative**2)
        best_image_focus = float(-numerator / denominator)
        best_points = points + best_image_focus * slopes
        best_centroid = np.sum(best_points * weights[:, None], axis=0)
        best_relative_um = (best_points - best_centroid) * 1e3
        best_focus_rms_um = float(
            np.sqrt(np.sum(weights * np.sum(best_relative_um**2, axis=1)))
        )

        ideal_y = field.y * magnification
        chief_y = float(pupil.chief.image_point[1])
        results.append(
            {
                "object_height_mm": field.y,
                "paraxial_image_height_mm": ideal_y,
                "centroid_image_height_mm": float(centroid[1]),
                "distortion_um": float((centroid[1] - ideal_y) * 1e3),
                "relative_distortion_ppm": (
                    float((centroid[1] / ideal_y - 1) * 1e6) if ideal_y != 0.0 else 0.0
                ),
                "chief_image_height_mm": chief_y,
                "chief_ray_distortion_um": float((chief_y - ideal_y) * 1e3),
                "chief_ray_relative_distortion_ppm": (
                    float((chief_y / ideal_y - 1) * 1e6) if ideal_y != 0.0 else 0.0
                ),
                "rms_spot_radius_um": float(rms_radius_um),
                "rms_sagittal_um": float(rms_x_um),
                "rms_tangential_um": float(rms_y_um),
                "ee80_radius_um": float(ee80_radius_um),
                "best_focus_shift_mm": best_image_focus,
                "best_focus_rms_spot_um": best_focus_rms_um,
                "sagittal_focus_shift_mm": sagittal_focus,
                "tangential_focus_shift_mm": tangential_focus,
                "astigmatic_separation_mm": tangential_focus - sagittal_focus,
                "unvignetted_rays": int(pupil.valid.sum()),
            }
        )
    return results


def export_metrics_csv(path, metrics: Sequence[dict]) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)


__all__ = ["field_metrics", "export_metrics_csv"]
