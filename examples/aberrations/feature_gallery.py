"""Reproduce common geometric-optics views from actual traced rays.

Run ``python -m examples.aberrations.feature_gallery``. Energy means uniform
incident stop-area weights; Fresnel, absorption and diffraction are excluded.
The gallery is a reporting example, not a new desktop application.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.aberrations.chromatic_sensitivity import WAVELENGTHS, doublet
from raytracer.analysis.aberrations import distortion_map, geometric_aberrations
from raytracer.design import OpticalSystem, SurfaceRow
from raytracer.io import write_analysis
from raytracer.propagation import FieldPoint, PupilSampling, SequentialTracer, trace_pupil
from raytracer.propagation.aiming import aim_stop_targets
from raytracer.propagation.sequential import TraceStatus
from raytracer.surfaces.apertures import CircularAperture, RectangularAperture

ROOT = Path(__file__).resolve().parents[2]
COLORS = ["#2864b4", "#a37708", "#bf424a"]
FIELDS = [0, 3, 8]


def save(fig, name):
    for ax in fig.axes:
        if ax.axison:
            ax.grid(alpha=0.15)
    fig.savefig(ROOT / f"docs/img/gallery_{name}.png", dpi=160)
    plt.close(fig)


def checked_pupil(tracer, field, sampling, **kwargs):
    pupil = trace_pupil(tracer, field, sampling=sampling, **kwargs)
    if not np.isin(pupil.batch.status, [TraceStatus.OK, TraceStatus.VIGNETTED]).all():
        raise RuntimeError("Unexpected ray failure in the gallery")
    if np.max(pupil.aiming_residual_mm) > 1e-9:
        raise RuntimeError("Pupil aiming did not converge")
    return pupil


def plot_spots(system, bundles):
    frame = system.image_frame
    reports = {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for column, field in enumerate(FIELDS):
        reference = frame.to_local(bundles[field, WAVELENGTHS[1]].chief.image_point)[:2]
        maximum = 0
        for wavelength, color in zip(WAVELENGTHS, COLORS):
            pupil = bundles[field, wavelength]
            xy = frame.to_local(pupil.image_points)[:, :2]
            delta = (xy - reference) * 1000
            maximum = max(maximum, float(np.max(np.abs(delta))))
            axes[0, column].scatter(
                *delta.T, s=2, alpha=0.45, color=color, label=f"{wavelength * 1000:.2f} nm"
            )
            centered = xy - np.sum(xy * pupil.weights[:, None], axis=0)
            radii = np.linalg.norm(centered, axis=1) * 1000
            order = np.argsort(radii)
            axes[1, column].step(
                np.r_[0, radii[order]],
                np.r_[0, np.cumsum(pupil.weights[order])],
                where="post",
                color=color,
            )
        extent = maximum * 1.08
        axes[0, column].set(
            xlim=(-extent, extent),
            ylim=(-extent, extent),
            xlabel="x − principal d [µm]",
            ylabel="y − principal d [µm]",
            title=f"Manchas · campo {field}° · escala propia",
        )
        axes[0, column].set_aspect("equal")
        axes[0, column].legend(fontsize=8, markerscale=3)
        axes[1, column].set(
            xlabel="Radio respecto del centroide de cada λ [µm]",
            ylabel="Fracción geométrica encerrada",
            ylim=(0, 1.02),
            title="Normalizada a los rayos transmitidos",
        )
        report = geometric_aberrations(bundles[field, WAVELENGTHS[1]], image_frame=frame)
        reports[str(field)] = report
    fig.suptitle("Manchas y energía geométrica · detector común · pesos uniformes en área del stop")
    save(fig, "spots")

    return reports


def plot_fans(system, bundles):
    frame = system.image_frame
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), layout="constrained")
    coordinates = np.linspace(-0.999, 0.999, 161)
    for column, field in enumerate(FIELDS):
        for wavelength, color in zip(WAVELENGTHS, COLORS):
            spectral = doublet()
            spectral.wavelength_um = float(wavelength)
            spectral_tracer = SequentialTracer(spectral)
            pupil = bundles[field, wavelength]
            chief = frame.to_local(pupil.chief.image_point)[:2]
            for row, component in [(0, 1), (1, 0)]:
                targets = np.zeros((len(coordinates), 2))
                targets[:, component] = 5 * coordinates
                origins, directions, errors = aim_stop_targets(
                    spectral_tracer, FieldPoint.angle(y_deg=field), targets
                )
                if np.max(errors) > 1e-10:
                    raise RuntimeError("Fan aiming failed")
                batch = spectral_tracer.trace_batch(origins, directions)
                if not np.isin(batch.status, [TraceStatus.OK, TraceStatus.VIGNETTED]).all():
                    raise RuntimeError("Unexpected fan failure")
                delta = (frame.to_local(batch.image_points)[:, component] - chief[component]) * 1000
                delta[~batch.valid] = np.nan
                axes[row, column].plot(
                    coordinates, delta, color=color, label=f"{wavelength * 1000:.2f} nm"
                )
            points = frame.to_local(pupil.image_points)[:, :2]
            directions = frame.direction_to_local(pupil.directions)
            slopes = directions[:, :2] / directions[:, 2:]
            centered = points - np.sum(points * pupil.weights[:, None], axis=0)
            spread = slopes - np.sum(slopes * pupil.weights[:, None], axis=0)
            shifts = np.linspace(-4, 0.5, 180)
            rms = (
                np.sqrt(
                    np.sum(
                        (centered[None, :, :] + shifts[:, None, None] * spread[None, :, :]) ** 2
                        * pupil.weights[None, :, None],
                        axis=(1, 2),
                    )
                )
                * 1000
            )
            axes[2, column].plot(shifts, rms, color=color)
        for row, name in [(0, "Tangencial · Δy"), (1, "Sagital · Δx")]:
            axes[row, column].axhline(0, color="0.5", lw=0.6)
            axes[row, column].set(
                title=f"{name} · campo {field}°",
                xlabel="Coordenada normalizada del stop",
                ylabel="Desviación del principal de cada λ [µm]",
            )
        axes[0, column].legend(fontsize=8)
        axes[2, column].axvline(0, color="0.5", lw=0.6, ls="--")
        axes[2, column].set(
            xlabel="Traslación del detector [mm]",
            ylabel="Radio RMS [µm]",
            title=f"Barrido de foco · campo {field}°",
        )
    fig.suptitle("Abanicos y desenfoque · las interrupciones corresponden a rayos bloqueados")
    save(fig, "fans")


def plot_layout(system, bundles):
    tracer = SequentialTracer(system)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for index, row in enumerate(system.rows):
        height = np.linspace(-5, 5, 250)
        sag = row.profile.sag(np.abs(height)) + system.vertices[index]
        for ax in axes[0, :2]:
            ax.plot(sag, height, color="0.2", lw=1.5)
    for field, color in zip(FIELDS, COLORS):
        pupil = checked_pupil(
            tracer, FieldPoint.angle(y_deg=field), PupilSampling(kind="fan_t", n=9), keep_paths=True
        )
        for path in pupil.batch.paths:
            for ax in axes[0, :2]:
                ax.plot(path[:, 2], path[:, 1], color=color, lw=0.7, alpha=0.65)
        axes[0, 0].plot([], [], color=color, label=f"{field}°")
    axes[0, 0].set(title="Trazado meridional", xlabel="z [mm]", ylabel="y [mm]")
    axes[0, 0].legend()
    axes[0, 1].set(
        title="Detalle de las superficies",
        xlabel="z [mm]",
        ylabel="y [mm]",
        xlim=(-1, 7),
        ylim=(-6, 6),
        aspect="equal",
    )
    wavelengths = np.linspace(0.42, 0.75, 200)
    for row in system.rows[:2]:
        axes[0, 2].plot(
            wavelengths * 1000,
            [row.material_after.index(w) for w in wavelengths],
            label=row.material_after.name,
        )
    axes[0, 2].set(
        title="Dispersión de los materiales",
        xlabel="Longitud de onda [nm]",
        ylabel="Índice de refracción",
    )
    axes[0, 2].legend()
    for index, ax in enumerate(axes[1]):
        angle = np.linspace(0, 2 * np.pi, 250)
        ax.plot(5 * np.cos(angle), 5 * np.sin(angle), color="0.2", lw=1)
        for field, color in zip(FIELDS, COLORS):
            pupil = bundles[field, WAVELENGTHS[1]]
            footprint = system.surface_frame(index).to_local(pupil.batch.paths[:, index + 1])
            ax.scatter(footprint[:, 0], footprint[:, 1], s=2, color=color, alpha=0.3)
        ax.set(
            title=f"Huella en S{index + 1} · λd · hasta el bloqueo",
            xlabel="x local [mm]",
            ylabel="y local [mm]",
            aspect="equal",
        )
    fig.suptitle("Geometría, materiales y huellas · campos 0°, 3° y 8°")
    save(fig, "layout")


def plot_field(system, bundles):
    tracer = SequentialTracer(system)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
    grid = np.linspace(-6, 6, 13)
    fields = [FieldPoint.angle(x_deg=x, y_deg=y) for y in grid for x in grid]
    mapping = distortion_map(tracer, fields)
    ideal = mapping.ideal_xy_mm.reshape(13, 13, 2)
    real = mapping.chief_xy_mm.reshape(13, 13, 2)
    for index in range(13):
        for points, style, color in [(ideal, "--", "0.65"), (real, "-", COLORS[0])]:
            axes[0, 0].plot(*points[index].T, style, color=color, lw=0.7)
            axes[0, 0].plot(*points[:, index].T, style, color=color, lw=0.7)
    axes[0, 0].set(
        title="Retícula real / ideal (gris) · sin amplificar",
        xlabel="x imagen [mm]",
        ylabel="y imagen [mm]",
        aspect="equal",
    )
    delta = real - ideal
    axes[0, 1].quiver(
        ideal[..., 0],
        ideal[..., 1],
        delta[..., 0] * 1000,
        delta[..., 1] * 1000,
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    axes[0, 1].set(
        title="Vectores de distorsión · amplificados ×1000",
        xlabel="x ideal [mm]",
        ylabel="y ideal [mm]",
        aspect="equal",
    )
    scan = np.linspace(0, 10, 26)
    throughput = {}
    for radial, azimuth, style in [(8, 32, "--"), (24, 96, "-")]:
        values = [
            checked_pupil(
                tracer,
                FieldPoint.angle(y_deg=f),
                PupilSampling(kind="gauss", radial=radial, azimuth=azimuth),
            ).geometric_throughput
            for f in scan
        ]
        throughput[str(radial * azimuth)] = values
        axes[1, 0].plot(scan, np.array(values) * 100, style, label=f"{radial * azimuth} rayos")
    axes[1, 0].set(
        title="Viñeteo · comparación de muestreos",
        xlabel="Campo [°]",
        ylabel="Fracción geométrica transmitida [%]",
    )
    axes[1, 0].legend()
    pupil = bundles[8, WAVELENGTHS[1]]
    for valid, color, label in [(True, COLORS[0], "Transmitido"), (False, COLORS[2], "Bloqueado")]:
        xy = pupil.pupil_uv[pupil.valid == valid] * 5
        axes[1, 1].scatter(*xy.T, s=8, color=color, label=label)
    axes[1, 1].set(
        title="Mapa de aceptación del stop · campo 8°",
        xlabel="x del stop [mm]",
        ylabel="y del stop [mm]",
        aspect="equal",
    )
    axes[1, 1].legend()
    fig.suptitle("Distorsión y viñeteo · λd · referencia rectilínea f tan θ")
    save(fig, "field")

    return scan, throughput, float(np.max(np.linalg.norm(delta, axis=2)) * 1000)


def plot_apertures(bundles):
    # Exercise aperture clipping with the identical incident ray ensemble.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    baseline = bundles[0, WAVELENGTHS[1]]
    apertures = [CircularAperture(5), CircularAperture(5, 2), RectangularAperture(4, 3)]
    acceptance = {}
    for ax, aperture, name in zip(axes, apertures, ["Circular", "Anular", "Rectangular"]):
        modified = doublet()
        modified.rows[0].aperture = aperture
        batch = SequentialTracer(modified).trace_batch(
            baseline.launch_origins, baseline.launch_directions
        )
        if not np.isin(batch.status, [TraceStatus.OK, TraceStatus.VIGNETTED]).all():
            raise RuntimeError("Unexpected aperture failure")
        acceptance[name] = float(baseline.sample_weights[batch.valid].sum())
        for valid, color in [(False, "0.82"), (True, COLORS[0])]:
            xy = baseline.pupil_uv[batch.valid == valid] * 5
            ax.scatter(*xy.T, s=6, color=color)
        ax.set(
            title=f"{name} · ≈{acceptance[name]:.1%} del haz circular",
            xlabel="x del stop [mm]",
            ylabel="y del stop [mm]",
            aspect="equal",
        )
    fig.suptitle("Aperturas físicas · mismo haz incidente · gris: rayos bloqueados")
    save(fig, "apertures")
    return acceptance


def plot_mirrors():
    """Compare an exact axial paraboloid focus with a spherical mirror."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for column, (conic, name) in enumerate([(0, "Esférico"), (-1, "Parabólico")]):
        system = OpticalSystem(
            [
                SurfaceRow.mirror(
                    radius=-100, thickness=-50, semidiameter=8, conic=conic, is_stop=True
                )
            ]
        )
        tracer = SequentialTracer(system)
        pupil = checked_pupil(
            tracer, FieldPoint.angle(), PupilSampling(kind="gauss", radial=12, azimuth=48)
        )
        rays = checked_pupil(
            tracer, FieldPoint.angle(), PupilSampling(kind="fan_t", n=13), keep_paths=True
        )
        h = np.linspace(-8, 8, 200)
        axes[0, column].plot(system.rows[0].profile.sag(np.abs(h)), h, color="0.2")
        for path in rays.batch.paths:
            axes[0, column].plot(path[:, 2], path[:, 1], color=COLORS[column], lw=0.8)
        axes[0, column].set(
            title=f"Espejo {name.lower()} · R = −100 mm", xlabel="z [mm]", ylabel="y [mm]"
        )
        points = system.image_frame.to_local(pupil.image_points)[:, :2] * 1000
        axes[1, column].scatter(*points.T, s=4, color=COLORS[column])
        axes[1, column].set(
            xlabel="x [µm]", ylabel="y [µm]", xlim=(-30, 30), ylim=(-30, 30), aspect="equal"
        )
        report = geometric_aberrations(pupil, image_frame=system.image_frame)
        axes[1, column].set_title(f"Plano z = −50 mm · RMS {report.rms_radius_mm * 1000:.3g} µm")
    fig.suptitle("Reflexión y superficies cónicas · mismo radio de vértice y apertura")
    save(fig, "mirrors")


def main():
    system = doublet()
    sample = PupilSampling(kind="gauss", radial=16, azimuth=64)
    bundles = {}
    for field in FIELDS:
        for wavelength in WAVELENGTHS:
            spectral = doublet()
            spectral.wavelength_um = float(wavelength)
            bundles[field, wavelength] = checked_pupil(
                SequentialTracer(spectral),
                FieldPoint.angle(y_deg=field),
                sample,
                keep_paths=True,
            )
    reports = plot_spots(system, bundles)
    plot_fans(system, bundles)
    plot_layout(system, bundles)
    scan, throughput, distortion = plot_field(system, bundles)
    acceptance = plot_apertures(bundles)
    plot_mirrors()
    write_analysis(
        ROOT / "docs/feature_gallery_results.json",
        system=system,
        settings={
            "fields_deg": FIELDS,
            "wavelengths_um": WAVELENGTHS,
            "sampling": "Gauss 16 × 64; uniform incident stop area",
        },
        results={
            "monochromatic_d_metrics": reports,
            "throughput_fields_deg": scan,
            "throughput_sampling": throughput,
            "aperture_acceptance": acceptance,
            "max_grid_distortion_um": distortion,
        },
    )
    print("Gallery generated; aperture acceptance:", acceptance)


if __name__ == "__main__":
    main()
