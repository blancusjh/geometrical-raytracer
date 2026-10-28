"""Render a reproducible geometric-optics report; this is not a test runner.

Run from any directory with the package installed:
    python -m examples.aberrations.geometrical_validation

The committed RayOptics fixture is read only. Regeneration is a separate,
explicit command in reference/generate_rayoptics_reference.py.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from raytracer.analysis.aberrations import axial_intercepts, geometric_aberrations
from raytracer.design import OpticalSystem, SurfaceRow
from raytracer.optics.materials import ConstantIndex
from raytracer.propagation import FieldPoint, PupilSampling, SequentialTracer, trace_pupil

ROOT = Path(__file__).resolve().parents[2]


def main():
    radius = 100.0
    system = OpticalSystem(
        [SurfaceRow.mirror(radius=-radius, thickness=-radius / 2, semidiameter=8, is_stop=True)]
    )
    heights = np.linspace(0.1, 8, 100)
    origins = np.column_stack([np.zeros_like(heights), heights, np.full_like(heights, -200)])
    directions = np.tile([0.0, 0.0, 1.0], (len(heights), 1))
    batch = SequentialTracer(system).trace_batch(origins, directions)
    intercepts, _ = axial_intercepts(batch.image_points, batch.directions)
    longitudinal = intercepts + radius / 2
    transverse = batch.image_points[:, 1]
    pupil = trace_pupil(
        SequentialTracer(system),
        FieldPoint.angle(),
        sampling=PupilSampling(kind="gauss", radial=10, azimuth=48),
    )
    report = geometric_aberrations(pupil, image_frame=system.image_frame)

    fixture = json.loads((ROOT / "tests/reference/rayoptics_cooke.json").read_text())
    prescription = ROOT / "data/optical_systems/photographic/cooke_triplet_prescription.csv"
    errors = []
    for case in fixture["cases"]:
        cooke = OpticalSystem.from_prescription(prescription)
        for row, index in zip(cooke.rows, case["indices_after"]):
            row.material_after = ConstantIndex("reference", index)
        ray = SequentialTracer(cooke, clip_apertures=False, allow_virtual_segments=True).trace(
            case["origin_mm"], case["direction"]
        )
        if not ray.ok:
            raise RuntimeError(f"Reference ray failed: {case['origin_mm']}")
        errors.append(
            [
                np.max(np.abs(ray.path - case["paths_mm"])),
                np.max(np.abs(ray.direction - case["outgoing_direction"])),
                abs(ray.opl - case["opl_mm"]),
            ]
        )
    maxima = np.max(errors, axis=0)
    metrics = {
        "reference": f"RayOptics {fixture['version']}",
        "reference_rays": len(errors),
        "max_surface_coordinate_error_mm": float(maxima[0]),
        "max_direction_component_error": float(maxima[1]),
        "max_optical_path_error_mm": float(maxima[2]),
        "mirror_radius_mm": radius,
        "mirror_semidiameter_mm": 8,
        "mirror_paraxial_plane_rms_mm": report.rms_radius_mm,
        "mirror_best_focus_shift_mm": report.best_focus_shift_mm,
        "mirror_best_focus_rms_mm": report.best_focus_rms_mm,
        "mirror_geometric_throughput": report.geometric_throughput,
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    axes[0].plot(heights, longitudinal * 1000, label="Trazado exacto")
    axes[0].plot(heights, heights**2 / (4 * radius) * 1000, "--", label="Tercer orden")
    axes[0].set(xlabel="Altura del rayo [mm]", ylabel="LSA [µm]", title="Aberración longitudinal")
    axes[1].plot(heights, transverse * 1000, label="Trazado exacto")
    axes[1].plot(heights, -(heights**3) / (2 * radius**2) * 1000, "--", label="Tercer orden")
    axes[1].set(xlabel="Altura del rayo [mm]", ylabel="TSA [µm]", title="Aberración transversal")
    axes[2].plot(np.arange(1, len(errors) + 1), np.array(errors)[:, 0] * 1e12, ".")
    axes[2].set(
        xlabel="Rayo de referencia",
        ylabel="Error máximo por coordenada [fm]",
        title="Comparación con RayOptics",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Óptica geométrica: teoría de tercer orden y referencia independiente")
    destination = ROOT / "docs/img/geometrical_validation.png"
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    (ROOT / "docs/geometrical_validation_results.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Figure: {destination}")


if __name__ == "__main__":
    main()
