"""Stigmatic ovoid lenses and the aplanatism condition, end to end.

Three singlets share the same conjugates (-30 mm -> +70 mm), thickness
(40 mm), glass (n = 1.5) and vertex curvatures; they differ only in what
their surfaces are:

- **sphere** — spherical surfaces: neither stigmatic nor aplanatic;
- **stigmatic** — an Omega singlet (two Cartesian surfaces sharing the
  intermediate conjugate d1 = 300): rigorously stigmatic, not aplanatic;
- **aplanat** — the mirror-symmetric Omega singlet (d1 = ξ/2): stigmatic
  *and* exactly aplanatic, since every ray leaves at its entrance angle.

The script verifies each claim on traced rays and prints the numbers: the
axial spot collapse (mm -> double-precision floor), the aplanatism map
M - 1 of Silva-Lora & Torres, the field-linear (coma) versus field-quadratic
(astigmatism) growth of the off-axis blur, and the chief-ray distortion.
It then reproduces the paper's 4-surface cemented triplet by optimizing the
intermediate conjugates, and closes with the flat-field experiment: how many
Cartesian surfaces it takes to put the aplanatic image on a *plane*.

Theory: A. Silva-Lora and R. Torres, "Aplanatism in stigmatic optical
systems," J. Opt. Soc. Am. A 37 (2020); "Superconical aplanatic ovoid
singlet lenses," J. Opt. Soc. Am. A 37, 1155-1165 (2020).

Run::

    python -m examples.stigmatic_surfaces.aplanatic_sol [--save out.png]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

from raytracer.analysis import (
    aplanatic_image_surface,
    aplanatism_report,
    chief_ray_distortion,
    spot_data,
)
from raytracer.design import OpticalSystem, StigmaticTrain, SurfaceRow
from raytracer.optics.materials import AIR, ConstantIndex
from raytracer.optimize import optimize_aplanat
from raytracer.propagation import (
    FieldPoint,
    PupilSampling,
    SequentialTracer,
    trace_pupil,
)

GLASS_INDEX = 1.5
D0, D2, THICKNESS = -30.0, 70.0, 40.0
SEMIDIAMETER, NA = 3.5, 0.08
FIELDS_MM = np.array([0.0, 0.05, 0.1, 0.2])


def build_singlets():
    """The three comparison singlets; the sphere copies the aplanat's vertex radii."""

    aplanat = StigmaticTrain.symmetric_singlet(
        n=GLASS_INDEX, d0=D0, thickness=THICKNESS, semidiameter=SEMIDIAMETER
    )
    stigmatic = StigmaticTrain.singlet(
        n=GLASS_INDEX, d0=D0, d1=300.0, d2=D2, thickness=THICKNESS,
        semidiameter=SEMIDIAMETER,
    )
    radius = 1.0 / aplanat.profiles[0].curvature
    glass = ConstantIndex(f"N{GLASS_INDEX}", GLASS_INDEX)
    sphere = OpticalSystem(
        [
            SurfaceRow.refracting(radius=radius, thickness=THICKNESS,
                                  material=glass, semidiameter=SEMIDIAMETER),
            SurfaceRow.refracting(radius=-radius, thickness=D2 - THICKNESS,
                                  material=AIR, semidiameter=SEMIDIAMETER),
        ],
        object_space=AIR,
        object_z=D0,
    )
    return sphere, stigmatic, aplanat


def _axial_rms_um(tracer: SequentialTracer) -> float:
    pupil = trace_pupil(
        tracer, FieldPoint(y=0.0), na_object_sine=NA,
        sampling=PupilSampling(kind="rings", radial=10, azimuth=48), chief_slope=0.0,
    )
    return spot_data(pupil).rms_radius_um


def _system_image_surface(system: OpticalSystem, heights) -> "np.ndarray":
    """Best-focus blur per field for a plain system (the sphere)."""

    from raytracer.analysis.aberrations.aplanatism import _closest_point_to_rays

    tracer = SequentialTracer(system)
    blur = np.full(len(heights), np.nan)
    for i, h in enumerate(heights):
        pupil = trace_pupil(
            tracer, FieldPoint(y=float(h)), na_object_sine=NA,
            sampling=PupilSampling(kind="rings", radial=6, azimuth=24),
            chief_slope=(0.0 - h) / (0.0 - system.object_z),
        )
        if pupil.valid.sum() >= 3:
            _, blur_mm = _closest_point_to_rays(pupil.image_points, pupil.directions)
            blur[i] = blur_mm * 1e3
    return blur


def field_blur(sphere=None, stigmatic=None, aplanat=None) -> dict:
    """Best-focus blur (um) per field height for the three singlets."""

    if sphere is None:
        sphere, stigmatic, aplanat = build_singlets()
    return {
        "sphere": _system_image_surface(sphere, FIELDS_MM),
        "stigmatic": aplanatic_image_surface(
            stigmatic, FIELDS_MM, na_object_sine=NA).blur_rms_um,
        "aplanat": aplanatic_image_surface(
            aplanat, FIELDS_MM, na_object_sine=NA).blur_rms_um,
    }


def part1_compare_singlets():
    print("=" * 72)
    print("1. Same conjugates, same vertex curvatures, NA %.2f" % NA)
    print("=" * 72)
    sphere, stigmatic, aplanat = build_singlets()

    print(f"\naxial RMS spot ({D0:g} mm -> {D2:g} mm):")
    print(f"  sphere      {_axial_rms_um(SequentialTracer(sphere)):12.1f} um")
    print(f"  stigmatic   {_axial_rms_um(SequentialTracer(stigmatic.to_system())):12.3e} um")
    print(f"  aplanat     {_axial_rms_um(SequentialTracer(aplanat.to_system())):12.3e} um")

    print("\naplanatism map of the two stigmatic trains (M - 1, RMS over the fan):")
    reports = {}
    for name, train in (("stigmatic", stigmatic), ("aplanat", aplanat)):
        reports[name] = aplanatism_report(train, na_object_sine=NA, samples=21)
        print(f"  {name:10s}  (M-1)_RMS = {reports[name].map_rms:9.3e}   "
              f"gt = {reports[name].gt:+.6f}")

    print("\nbest-focus blur vs field height (um) — coma is the linear column:")
    blur = field_blur(sphere, stigmatic, aplanat)
    header = "  h (mm):    " + "".join(f"{h:>10.2f}" for h in FIELDS_MM)
    print(header)
    for name, values in blur.items():
        print(f"  {name:10s}" + "".join(f"{v:10.3f}" for v in values))
    for name, expected in (("stigmatic", 2.0), ("aplanat", 4.0)):
        ratio = blur[name][3] / blur[name][2]
        print(f"  {name}: blur(0.2)/blur(0.1) = {ratio:.2f}  "
              f"(field-{'linear, coma' if expected == 2.0 else 'quadratic, coma-free'})")

    print("\nchief-ray distortion, stop at the front vertex, field 0 -> 2 mm:")
    heights = np.linspace(0.4, 2.0, 5)
    for name, system, m in (
        ("sphere", sphere, None),
        ("stigmatic", stigmatic.to_system(), stigmatic.gt),
        ("aplanat", aplanat.to_system(), aplanat.gt),
    ):
        tracer = SequentialTracer(system)
        if m is None:
            from raytracer.propagation import chief_ray_slopes, trace_from_object

            slopes = chief_ray_slopes(tracer, FieldPoint(y=1e-4), stop_index=0)
            m = trace_from_object(tracer, (0.0, 1e-4), slopes).image_point[1] / 1e-4
        rows = chief_ray_distortion(tracer, heights, magnification=m, stop_index=0)
        worst = max(rows, key=lambda r: abs(r["chief_ray_relative_distortion_ppm"]))
        print(f"  {name:10s}  m0 = {m:+.4f}   worst {worst['chief_ray_distortion_um']:+9.3f} um "
              f"({worst['chief_ray_relative_distortion_ppm']/1e4:+.3f} %) at h = {worst['field']:g} mm")
    return blur


CEMENTED = dict(
    media=(1.0, 1.517122, 1.670591, 1.851280, 1.0),
    vertices=(0.0, 15.0, 25.0, 35.0),
    semidiameter=10.0,
)
AIR_SPACED = dict(
    media=(1.0, 1.516798, 1.0, 1.516798, 1.0, 1.516798, 1.0),
    vertices=(0.0, 15.0, 25.0, 35.0, 45.0, 65.0),
    semidiameter=10.0,
)


def part2_cemented_triplet():
    print()
    print("=" * 72)
    print("2. The paper's cemented triplet: 4 Cartesian surfaces, -100 -> +90 mm")
    print("=" * 72)
    train = StigmaticTrain(conjugates=(-100.0, -150.0, -600.0, 450.0, 90.0), **CEMENTED)
    fit = optimize_aplanat(
        train, na_object_sine=0.195, samples=21,
        bounds=(-3000, 3000), diff_step=1e-4, xtol=1e-14, ftol=1e-14, gtol=1e-14,
    )
    print(fit.summary())
    print("(the paper's own example reaches gt = -0.602188 in a different basin;")
    print(" every basin is exactly stigmatic, the optimizer only buys aplanatism)")
    return fit


def _best_flat_fit(base, *, starts, fields, weight, na, seed=11):
    rng = np.random.default_rng(seed)
    free = len(base["vertices"]) - 1
    best = None
    d0, dn = base.pop("d0"), base.pop("dn")
    for _ in range(starts):
        intermediates = rng.uniform(-900, 900, size=free)
        try:
            train = StigmaticTrain(conjugates=(d0, *intermediates, dn), **base)
            fit = optimize_aplanat(
                train, na_object_sine=na, samples=15,
                field_heights=fields, flat_field_weight=weight,
                bounds=(-3000, 3000), diff_step=1e-4,
                xtol=1e-14, ftol=1e-14, gtol=1e-14,
            )
        except (ValueError, ZeroDivisionError):
            continue
        cost = fit.after.map_rms + weight * abs(fit.image_surface.max_sagitta)
        if best is None or cost < best[0]:
            best = (cost, fit)
    base["d0"], base["dn"] = d0, dn
    return None if best is None else best[1]


def part3_flat_field():
    print()
    print("=" * 72)
    print("3. Aplanatism on a *plane*: how many surfaces does a flat field take?")
    print("=" * 72)
    fields = [0.0, 1.0, 2.0, 3.0]
    print("aplanatic-image-surface sagitta over field 0..3 mm, and (M-1)_RMS:\n")

    # N = 2: the exact aplanat has no freedom left; its image surface is curved.
    aplanat = StigmaticTrain.symmetric_singlet(
        n=GLASS_INDEX, d0=D0, thickness=THICKNESS, semidiameter=SEMIDIAMETER
    )
    surface = aplanatic_image_surface(aplanat, fields, na_object_sine=NA)
    report = aplanatism_report(aplanat, na_object_sine=NA, samples=15)
    print(f"  N=2 (exact aplanat)      (M-1)_RMS = {report.map_rms:9.3e}   "
          f"max sagitta = {surface.max_sagitta:8.4f} mm  <- curved, no DOF left")

    results = {2: (report.map_rms, surface.max_sagitta)}
    for n_surfaces, base, na in (
        (4, dict(CEMENTED, d0=-100.0, dn=90.0), 0.15),
        (6, dict(AIR_SPACED, d0=-100.0, dn=120.0), 0.15),
    ):
        fit = _best_flat_fit(base, starts=4, fields=fields, weight=10.0, na=na)
        if fit is None:
            print(f"  N={n_surfaces}: no traceable optimum found")
            continue
        sagitta = fit.image_surface.max_sagitta
        results[n_surfaces] = (fit.after.map_rms, sagitta)
        print(f"  N={n_surfaces} (flat-field weight)  (M-1)_RMS = {fit.after.map_rms:9.3e}   "
              f"max sagitta = {sagitta:8.4f} mm   d* = "
              f"{tuple(round(float(v), 1) for v in fit.result.x)}")

    tol_map, tol_sag = 1e-4, 0.05
    winners = [n for n, (m, s) in sorted(results.items())
               if m < tol_map and abs(s) < tol_sag]
    print(f"\n  criterion: (M-1)_RMS < {tol_map:g} and |sagitta| < {tol_sag:g} mm")
    if winners:
        print(f"  -> minimal N reaching aplanatism on a plane: {winners[0]}")
    else:
        best_n = min(results, key=lambda n: abs(results[n][1]))
        print(f"  -> none met both; flattest was N={best_n} "
              f"(sagitta {results[best_n][1]:.4f} mm)")
    return results


def figure(blur, save: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, stigmatic, aplanat = build_singlets()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    for name, train, color in (("stigmatic (d1 = 300)", stigmatic, "#c1440e"),
                               ("aplanat (symmetric)", aplanat, "#2878b5")):
        report = aplanatism_report(train, na_object_sine=NA, samples=40)
        axes[0].plot(report.sines_object[report.valid],
                     report.map_values[report.valid] - 1.0, color=color, label=name)
    axes[0].axhline(0.0, color="black", lw=0.7)
    axes[0].set(xlabel="sin u (object side)", ylabel="M − 1",
                title="Aplanatism map (Silva-Lora & Torres):\nM ≡ 1 is the sine condition")
    axes[0].legend(fontsize=8)

    heights = np.linspace(0.0, 2.5, 11)
    for name, train, color in (("stigmatic", stigmatic, "#c1440e"),
                               ("aplanat", aplanat, "#2878b5")):
        locus = aplanatic_image_surface(train, heights, na_object_sine=NA)
        axes[1].plot(locus.points[:, 2], locus.points[:, 1], "o-", ms=3,
                     color=color, label=name)
    axes[1].axvline(D2, color="#999999", ls="--", lw=0.8, label=f"plane z = {D2:g}")
    axes[1].set(xlabel="z (mm)", ylabel="image height (mm)",
                title="Where the off-axis image actually forms\n(the paper's dashed surface)")
    axes[1].legend(fontsize=8)

    h = FIELDS_MM[1:]
    for name, color in (("sphere", "#666666"), ("stigmatic", "#c1440e"),
                        ("aplanat", "#2878b5")):
        axes[2].loglog(h, blur[name][1:], "o-", color=color, label=name)
    axes[2].set(xlabel="field height (mm)", ylabel="best-focus blur RMS (µm)",
                title="Blur growth with field:\nslope 1 = coma, slope 2 = coma-free")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25, which="both")

    for ax in axes[:2]:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nfigure saved to {save}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", metavar="PATH", help="save the summary figure")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    blur = part1_compare_singlets()
    part2_cemented_triplet()
    part3_flat_field()
    if args.save:
        figure(blur, args.save)


if __name__ == "__main__":
    main()
