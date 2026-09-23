"""Reproduce material, color, alignment and 3-D geometry reports.

Run: python -m examples.aberrations.chromatic_sensitivity
The doublet radii follow the analytic thin-contact achromat equations; its
finite thickness leaves residual color. No design optimization is performed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from raytracer.analysis.aberrations import axial_color, chromatic_spots
from raytracer.analysis.tolerancing import (
    GeometricEvaluator,
    Perturbation,
    Tolerance,
    monte_carlo,
    sensitivity,
)
from raytracer.design import OpticalSystem, SurfaceRow
from raytracer.io import write_analysis
from raytracer.optics.materials import AIR, sellmeier_glass
from raytracer.propagation import FieldPoint, PupilSampling, SequentialTracer, trace_pupil
from raytracer.viz.geometric import layout_3d_figure

ROOT = Path(__file__).resolve().parents[2]
WAVELENGTHS = np.array([0.48613, 0.58756, 0.65627])
SPECTRUM = [1, 2, 1]


def doublet():
    crown, flint = [sellmeier_glass(name) for name in ("N-BK7", "N-F2")]
    # P_d=1/100 mm^-1 and P_F-P_C=0 in the thin-contact limit.
    matrix = [
        [crown.index(WAVELENGTHS[1]) - 1, flint.index(WAVELENGTHS[1]) - 1],
        [
            crown.index(WAVELENGTHS[0]) - crown.index(WAVELENGTHS[2]),
            flint.index(WAVELENGTHS[0]) - flint.index(WAVELENGTHS[2]),
        ],
    ]
    crown_curvature, flint_curvature = np.linalg.solve(matrix, [0.01, 0])
    radii = 1 / np.array(
        [crown_curvature / 2, -crown_curvature / 2, -crown_curvature / 2 - flint_curvature]
    )
    rows = [
        SurfaceRow.refracting(
            radius=radii[0], thickness=4, material=crown, semidiameter=5, is_stop=True
        ),
        SurfaceRow.refracting(radius=radii[1], thickness=2, material=flint, semidiameter=5),
        SurfaceRow.refracting(radius=radii[2], thickness=95, material=AIR, semidiameter=5),
    ]
    system = OpticalSystem(
        rows, wavelength_um=0.58756, object_z=-np.inf, name="N-BK7 / N-F2 reference doublet"
    )
    focus = axial_color(system, WAVELENGTHS, reference_wavelength_um=0.58756).focus_z_mm[1]
    system.set_thickness(2, focus - 6)
    return system


def main():
    system = doublet()
    sample = PupilSampling(kind="gauss", radial=8, azimuth=32)
    tracer = SequentialTracer(system)
    fields = [FieldPoint.angle(), FieldPoint.angle(y_deg=3)]
    evaluator = GeometricEvaluator(
        tracer, fields, WAVELENGTHS, wavelength_weights=SPECTRUM, sampling=sample
    )
    refocused = GeometricEvaluator(
        tracer,
        fields,
        WAVELENGTHS,
        wavelength_weights=SPECTRUM,
        sampling=sample,
        focus_policy="refocus",
    )
    parameters = [
        Perturbation("decenter_x", (0, 1, 2), label="Doublet translation x"),
        Perturbation("tilt_y", (0, 1, 2), label="Doublet tilt y", pivot_mm=(0, 0, 0)),
        Perturbation("index", (0,), label="Crown index offset"),
    ]
    sensitivities = [
        sensitivity(evaluator, parameter, step)
        for parameter, step in zip(parameters, [0.01, 0.01, 1e-4])
    ]
    tolerances = [
        Tolerance(parameters[0], 0.02),
        Tolerance(parameters[1], 0.02),
        Tolerance(parameters[2], 1e-4),
    ]
    trials = monte_carlo(evaluator, tolerances, samples=128, seed=20260923)
    color = axial_color(system, WAVELENGTHS, reference_wavelength_um=0.58756)
    spots = chromatic_spots(
        tracer,
        WAVELENGTHS,
        reference_wavelength_um=0.58756,
        wavelength_weights=SPECTRUM,
        field=fields[1],
        sampling=sample,
    )
    manufacturer = {
        "N-BK7": [1.52238, 1.51680, 1.51432, 1.50731],
        "N-F2": [1.63208, 1.62005, 1.61506, 1.60261],
    }
    residuals = {
        name: [
            sellmeier_glass(name).index(wl) - value
            for wl, value in zip([*WAVELENGTHS, 1.014], values)
        ]
        for name, values in manufacturer.items()
    }
    results = {
        "axial_color": color,
        "spectral_centroid_rms_um_at_3deg": spots.centroid_rms_um,
        "manufacturer_index_residuals": residuals,
        "nominal": evaluator.nominal.metrics(),
        "per_field_refocused": refocused.nominal.metrics(),
        "sensitivities": sensitivities,
        "monte_carlo": trials,
        "conventions": "Source weights uniform in nominal stop area; incident rays and detector held fixed; no Fresnel/absorption.",
    }
    write_analysis(
        ROOT / "docs/chromatic_sensitivity_results.json",
        system=system,
        settings=evaluator.settings(),
        results=results,
    )
    system.to_prescription(
        ROOT / "data/optical_systems/elements/reference_achromatic_doublet.json", fmt="json"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    axes[0, 0].plot(WAVELENGTHS * 1000, color.focus_shift_mm * 1000, "o-")
    axes[0, 0].set(
        xlabel="Longitud de onda [nm]",
        ylabel="Desplazamiento focal [µm]",
        title="Color axial respecto de la línea d",
    )
    colors = ["#306ab2", "#a57b17", "#ba4444"]
    for wavelength, points, c in zip(WAVELENGTHS, spots.offsets_um, colors):
        axes[0, 1].scatter(
            points[:, 0], points[:, 1], s=4, color=c, alpha=0.6, label=f"{wavelength * 1000:.2f} nm"
        )
    axes[0, 1].set(
        xlabel="x [µm]", ylabel="y [µm]", title="Campo 3°: referencia común del rayo principal"
    )
    axes[0, 1].set_aspect("equal", adjustable="datalim")
    axes[0, 1].legend(fontsize=8)
    decenter = np.linspace(-0.03, 0.03, 13)
    shifts = [
        evaluator.evaluate(parameters[0].apply(system, value)).metrics()["field_1.centroid_x_mm"]
        for value in decenter
    ]
    axes[1, 0].plot(decenter * 1000, np.array(shifts) * 1000, "o-")
    axes[1, 0].set(
        xlabel="Descentramiento x [µm]",
        ylabel="Centroide x [µm]",
        title="Haz incidente y detector fijos",
    )
    key = "field_1.reference_rms_mm"
    values = trials.metrics[key] * 1000
    axes[1, 1].hist(values[np.isfinite(values)], bins=16, color="#3a7886", edgecolor="white")
    axes[1, 1].set(
        xlabel="RMS respecto de la referencia nominal [µm]",
        ylabel="Realizaciones",
        title="128 realizaciones; semilla 20260923",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.18)
    fig.suptitle("Doblete N-BK7 / N-F2: cromática y sensibilidad geométrica")
    fig.savefig(ROOT / "docs/img/chromatic_sensitivity.png", dpi=160)
    plt.close(fig)

    moved = parameters[1].apply(system, 0.5)
    pupil = trace_pupil(
        tracer.with_system(moved),
        FieldPoint.angle(y_deg=3),
        sampling=PupilSampling(kind="gauss", radial=3, azimuth=8),
        keep_paths=True,
    )
    fig = layout_3d_figure(
        moved,
        paths=pupil.batch.paths,
        title="Doblete inclinado 0,5°: superficies y rayos en sus marcos reales",
    )
    fig.savefig(ROOT / "docs/img/placed_doublet.png", dpi=160, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(
        "Maximum manufacturer-index residual:",
        max(abs(x) for values in residuals.values() for x in values),
    )
    print("F-C focal separation [um]:", abs(np.ptp(color.focus_z_mm[[0, 2]])) * 1000)
    print("Polychromatic RMS at 3 degrees [um]:", spots.centroid_rms_um)
    print(
        "Fixed/refocused RMS [um]:",
        evaluator.nominal.fields[1].rms_radius_mm * 1000,
        refocused.nominal.fields[1].rms_radius_mm * 1000,
    )
    print(
        "Monte Carlo successes:", len(trials.draws) - len(trials.failures), "/", len(trials.draws)
    )
    print("Reference RMS quantiles [um]:", np.nanquantile(values, [0.05, 0.5, 0.95]))
    print("Sensitivity to x translation:", sensitivities[0].derivative["field_1.centroid_x_mm"])


if __name__ == "__main__":
    main()
