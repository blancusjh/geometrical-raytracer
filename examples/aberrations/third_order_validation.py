"""Regenerate third-order ledgers, field curves and independent residuals."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.aberrations.chromatic_sensitivity import doublet
from raytracer.analysis.aberrations import distortion_map, parabasal_focus, seidel_coefficients
from raytracer.design import OpticalSystem, SurfaceRow
from raytracer.io import write_analysis
from raytracer.optics.materials import ConstantIndex
from raytracer.propagation import FieldPoint, PupilSampling, SequentialTracer
from raytracer.propagation.aiming import aim_stop_targets

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "tests/reference/rayoptics_third_order.json"


def reference_system(case):
    rows = []
    for index, record in enumerate(case["rows"]):
        options = dict(
            radius=record["radius"],
            thickness=record["thickness"],
            conic=record.get("conic", 0.0),
            coefficients=(record.get("a4", 0.0),),
            semidiameter=case["radius"] if index == case["stop"] else 20.0,
            is_stop=index == case["stop"],
        )
        rows.append(
            SurfaceRow.mirror(**options)
            if record.get("mirror")
            else SurfaceRow.refracting(
                **options, material=ConstantIndex("reference", record["index"])
            )
        )
    system = OpticalSystem(rows, object_z=-np.inf if case["object_z"] is None else case["object_z"])
    system.set_thickness(len(rows) - 1, case["gaussian_z_mm"] - system.vertices[-1])
    field = (
        FieldPoint.angle(y_deg=case["field"])
        if case["object_z"] is None
        else FieldPoint(y=case["field"])
    )
    return system, field


def compare_reference(case):
    system, field = reference_system(case)
    tracer = SequentialTracer(system, clip_apertures=False)
    ledger = seidel_coefficients(system, field)
    focus = parabasal_focus(tracer, field)
    distortion = distortion_map(tracer, [field])
    pupil = np.array([[0.0, 0.7], [0.4, 0.2], [0.0, 0.0], [0.5, -0.5]])
    scales = np.array([0.5, 0.25, 0.125])
    errors, focus_errors = [], []
    for scale in scales:
        scaled = (
            (FieldPoint.angle(y_deg=np.rad2deg(np.arctan(scale * np.tan(np.deg2rad(field.y))))))
            if field.kind == "angle"
            else FieldPoint(y=field.y * scale)
        )
        origins, directions, residual = aim_stop_targets(
            tracer, scaled, pupil * ledger.pupil_radius_mm * scale
        )
        if residual.max() > 1e-11:
            raise RuntimeError("asymptotic comparison did not aim to the stop")
        rays = tracer.trace_batch(origins, directions)
        if not rays.valid.all():
            raise RuntimeError("asymptotic comparison lost a ray")
        exact = rays.image_points[:, :2] - [0, scale * ledger.gaussian_image_height_mm]
        errors.append(float(np.max(abs(exact - ledger.transverse(pupil) * scale**3))))
        parabasal = parabasal_focus(tracer, scaled, step_mm=5e-4)
        curvatures = ledger.field_curvatures_per_mm
        predicted = (
            0.5
            * np.array([curvatures["tangential"], curvatures["sagittal"]])
            * (ledger.gaussian_image_height_mm * scale) ** 2
        )
        measured = np.array([parabasal.tangential_shift_mm, parabasal.sagittal_shift_mm])
        focus_errors.append(float(np.max(abs(measured - predicted))))
    return {
        "name": case["name"],
        "max_seidel_surface_error_mm": float(
            np.max(abs(ledger.surface_sums_mm - case["surface_sums_mm"]))
        ),
        "max_parabasal_error_mm": float(
            np.max(
                abs(
                    np.array([focus.tangential_shift_mm, focus.sagittal_shift_mm])
                    - case["parabasal_shifts_mm"]
                )
            )
        ),
        "max_chief_error_mm": float(
            np.max(abs(distortion.chief_xy_mm[0] - case["chief_image_xy_mm"]))
        ),
        "scales": scales,
        "transverse_remainder_mm": errors,
        "transverse_reduction_ratios": np.array(errors[:-1]) / errors[1:],
        "field_focus_remainder_mm": focus_errors,
        "field_focus_reduction_ratios": np.array(focus_errors[:-1]) / focus_errors[1:],
    }


def main():
    reference = json.loads(REFERENCE.read_text())
    comparisons = [compare_reference(case) for case in reference["cases"]]
    system = doublet()
    tracer = SequentialTracer(system)
    angles = np.linspace(0, 8, 17)
    fields = [FieldPoint.angle(y_deg=value) for value in angles]
    sample = PupilSampling(kind="gauss", radial=8, azimuth=32)
    ledger = seidel_coefficients(system, FieldPoint.angle(y_deg=3))
    focuses = [parabasal_focus(tracer, field) for field in fields]
    mapping = distortion_map(tracer, fields, centroid_sampling=sample)
    scale = np.tan(np.deg2rad(angles)) / np.tan(np.deg2rad(3))
    curvature = ledger.field_curvatures_per_mm
    image_height = ledger.gaussian_image_height_mm * scale
    predicted_t = curvature["tangential"] * image_height**2 / 2
    predicted_s = curvature["sagittal"] * image_height**2 / 2
    third_distortion = np.array(
        [ledger.transverse([0, 0], field_scale=value)[1] for value in scale]
    )
    results = {
        "reference_sha256": hashlib.sha256(REFERENCE.read_text().encode()).hexdigest(),
        "independent_comparisons": comparisons,
        "seidel_ledger": ledger,
        "seidel_sums_mm": ledger.named_sums_mm,
        "curvatures_per_mm": curvature,
        "parabasal_focus": focuses,
        "mapping": mapping,
        "chief_distortion_mm": mapping.deviation_mm,
        "centroid_deviation_mm": mapping.centroid_xy_mm - mapping.ideal_xy_mm,
        "third_order_tangential_shift_mm": predicted_t,
        "third_order_sagittal_shift_mm": predicted_s,
        "third_order_distortion_mm": third_distortion,
    }
    write_analysis(
        ROOT / "docs/third_order_results.json",
        system=system,
        settings={
            "fields": fields,
            "sampling": sample,
            "seidel_reference_field": ledger.field,
            "stop_radius_mm": ledger.pupil_radius_mm,
            "parabasal_step_mm": 1e-3,
            "distortion_reference": "paraxial rectilinear at fixed detector",
            "focus_reference": "nominal detector at Gaussian d focus",
        },
        results=results,
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    positions = np.arange(5)
    for surface, values in enumerate(ledger.surface_sums_mm):
        axes[0, 0].bar(
            positions + (surface - 1) * 0.24,
            values * 1000,
            width=0.23,
            label=f"Superficie {surface + 1}",
        )
    axes[0, 0].set_xticks(positions, ["Esférica", "Coma", "Astigmatismo", "Petzval", "Distorsión"])
    axes[0, 0].set(ylabel="Suma de Seidel [µm]", title="Contribuciones a 3°; radio de stop 5 mm")
    axes[0, 0].axhline(0, color="#444444", lw=0.7)
    axes[0, 0].legend(fontsize=8)
    for values, theory, label, color in [
        ([f.tangential_shift_mm for f in focuses], predicted_t, "Tangencial", "#305d91"),
        ([f.sagittal_shift_mm for f in focuses], predicted_s, "Sagital", "#b46336"),
    ]:
        axes[0, 1].plot(angles, values, color=color, label=label + " parabasal")
        axes[0, 1].plot(angles, theory, "--", color=color, label=label + " tercer orden")
    axes[0, 1].set(
        xlabel="Campo [°]",
        ylabel="Desplazamiento focal [mm]",
        title="Curvatura de campo: definiciones separadas",
    )
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(angles, mapping.deviation_mm[:, 1] * 1000, label="Rayo principal")
    axes[1, 0].plot(
        angles,
        (mapping.centroid_xy_mm - mapping.ideal_xy_mm)[:, 1] * 1000,
        label="Centroide de área transmitida",
    )
    axes[1, 0].plot(angles, third_distortion * 1000, "--", label="Distorsión de tercer orden")
    axes[1, 0].set(
        xlabel="Campo [°]",
        ylabel="Desviación respecto del mapa paraxial [µm]",
        title="Detector fijo; centroide y distorsión",
    )
    axes[1, 0].legend(fontsize=8)
    labels = [
        "Lente / infinito",
        "Lente / objeto finito",
        "Asfera / objeto finito",
        "Espejo esférico",
        "Espejo parabólico",
    ]
    for result, label in zip(comparisons, labels):
        axes[1, 1].loglog(result["scales"], result["transverse_remainder_mm"], "o-", label=label)
    axes[1, 1].loglog(
        [0.125, 0.5], np.array([0.125, 0.5]) ** 5 * 1e-3, ":", color="black", label="Pendiente 5"
    )
    axes[1, 1].set(
        xlabel="Escala conjunta de campo y pupila",
        ylabel="Residuo transversal máximo [mm]",
        title="Trazado exacto menos predicción cúbica",
    )
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_xticks([0.125, 0.25, 0.5], ["1/8", "1/4", "1/2"])
    axes[1, 1].set_xticks([], minor=True)
    for ax in axes.flat:
        ax.grid(alpha=0.18)
    fig.suptitle("Tercer orden y geometría de campo — doblete N-BK7 / N-F2, línea d")
    fig.savefig(ROOT / "docs/img/third_order_validation.png", dpi=160)
    plt.close(fig)
    for key in ["max_seidel_surface_error_mm", "max_parabasal_error_mm", "max_chief_error_mm"]:
        print(key, max(result[key] for result in comparisons))
    print("Doublet sums [mm]:", ledger.named_sums_mm)
    print("Doublet curvatures [1/mm]:", curvature)
    print(
        "At 8 degrees: chief/centroid deviation [um]:",
        mapping.deviation_mm[-1, 1] * 1000,
        (mapping.centroid_xy_mm[-1, 1] - mapping.ideal_xy_mm[-1, 1]) * 1000,
    )
    print(
        "Geometric throughput range:",
        mapping.geometric_throughput.min(),
        mapping.geometric_throughput.max(),
    )
    print(
        "Fifth-order ratios:",
        min(np.min(r["transverse_reduction_ratios"]) for r in comparisons),
        max(np.max(r["transverse_reduction_ratios"]) for r in comparisons),
    )


if __name__ == "__main__":
    main()
