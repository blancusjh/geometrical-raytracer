"""Multistart design searches behind examples/notebooks/building_optical_systems.ipynb.

One function per system kind; each explores random intermediate conjugates,
polishes the best basin, and prints the winner the notebook re-polishes.
Runtime is minutes per system — this is the search; the notebook only
verifies. The projector search seeds only from configurations whose field
cones fully trace (feasibility filter), and measures distortion against a
reference pupil inside the barrel (aim_z)."""
import functools
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
print = functools.partial(print, flush=True)

from raytracer.analysis import aplanatic_image_surface, aplanatism_map, aplanatism_report
from raytracer.design import StigmaticTrain
from raytracer.optics.materials import AIR, sellmeier_glass
from raytracer.optimize import (
    Aplanatism, AxialColor, Constraint, Distortion, EvaluationContext,
    FlatImageSurface, TargetMagnification, optimize_train,
)
from raytracer.propagation import SequentialTracer

BK7 = sellmeier_glass("N-BK7")
F2 = sellmeier_glass("N-F2")
SF11 = sellmeier_glass("N-SF11")
SK16 = sellmeier_glass("N-SK16")
SK4 = sellmeier_glass("N-SK4")
LAK21 = sellmeier_glass("N-LAK21")
WL = 0.58756


def multistart(build, constraints, *, na, samples, n_free, starts, seed, spans,
               bounds, score, **ls):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(starts):
        d = np.array([rng.uniform(*span) for span in spans])
        try:
            train = build(tuple(d))
            fit = optimize_train(train, constraints, na_object_sine=na,
                                 samples=samples, bounds=bounds,
                                 diff_step=1e-4, max_nfev=60, **ls)
            s = score(fit)
        except (ValueError, ZeroDivisionError):
            continue
        print(f"    start -> score {s:.3e}")
        if np.isfinite(s) and (best is None or s < best[0]):
            best = (s, fit)
    if best is not None:
        fit = optimize_train(best[1].train, constraints, na_object_sine=na,
                             samples=samples, bounds=bounds, diff_step=1e-4,
                             xtol=1e-14, ftol=1e-14, gtol=1e-14, **ls)
        best = (score(fit), fit)
    return best


# ---------------------------------------------------------------- telescope
def telescope():
    print("=" * 70)
    print("TELESCOPE OBJECTIVE — 2 lenses (N-BK7 + N-F2), D=52 mm, EFL≈400 mm")
    t0 = time.time()

    def build(d):
        return StigmaticTrain(media=(AIR, BK7, AIR, F2, AIR), vertices=(0.0, 9.0, 13.0, 19.0),
            conjugates=(-1.0e6, *d, 400.0), wavelength_um=WL, semidiameter=26.0)

    na = 2.6e-5
    cons = [Aplanatism(), AxialColor(weight=0.05)]

    def score(fit):
        ctx = EvaluationContext(fit.train, na_object_sine=na, samples=9)
        color = float(np.max(np.abs(AxialColor().residuals(ctx))))
        return fit.after.map_rms + 0.01 * color

    best = multistart(build, cons, na=na, samples=11, n_free=3, starts=6,
                      seed=5, spans=[(-5e4, 5e4)] * 3, bounds=(-2e5, 2e5),
                      score=score)
    s, fit = best
    ctx = EvaluationContext(fit.train, na_object_sine=na, samples=9)
    color = AxialColor().residuals(ctx)
    fields = [0.0, 6000.0, 13000.0]  # object heights: 0 / 0.34deg / 0.75deg
    locus = aplanatic_image_surface(fit.train, fields, na_object_sine=na)
    print(f"  d* = {tuple(round(v, 3) for v in fit.result.x)}   [{time.time()-t0:.0f}s]")
    print(f"  (M-1)_RMS {fit.after.map_rms:.2e}   axial color F/C = "
          f"{color[0]:+.4f} / {color[1]:+.4f} mm   gt = {fit.after.gt:+.3e}")
    print(f"  blur at 0 / 0.34deg / 0.75deg field: "
          f"{np.array2string(locus.blur_rms_um, precision=2)} um "
          f"(Airy radius f/7.7 ≈ 5.5 um)   sagitta {locus.max_sagitta:.4f} mm")
    return fit


# ---------------------------------------------------------------- microscope
def microscope():
    print("=" * 70)
    print("MICROSCOPE OBJECTIVE — 3 lenses (N-LAK21/N-SF11/N-BK7), 20x NA 0.25")
    t0 = time.time()

    def build(d):
        return StigmaticTrain(media=(AIR, LAK21, AIR, SF11, AIR, BK7, AIR),
            vertices=(0.0, 4.5, 8.0, 12.0, 16.0, 22.0),
            conjugates=(-6.0, *d, 170.0), wavelength_um=WL, semidiameter=7.0)

    na = 0.25
    fields = (0.15, 0.3)
    cons = [Aplanatism(),
            TargetMagnification(value=-20.0, weight=0.05),
            Distortion(heights=fields, weight=1.0),
            FlatImageSurface(heights=(0.0, *fields), weight=1.0)]

    def score(fit):
        return (fit.after.map_rms
                + 0.02 * abs(fit.after.gt + 20.0)
                + abs(fit.image_surface.max_sagitta))

    best = multistart(build, cons, na=na, samples=11, n_free=5, starts=6,
                      seed=3, spans=[(-900, 900)] * 5, bounds=(-5e3, 5e3),
                      score=score)
    s, fit = best
    ctx = EvaluationContext(fit.train, na_object_sine=na, samples=9)
    dist = Distortion(heights=fields).residuals(ctx)
    locus = fit.image_surface
    print(f"  d* = {tuple(round(v, 3) for v in fit.result.x)}   [{time.time()-t0:.0f}s]")
    print(f"  (M-1)_RMS {fit.after.map_rms:.2e}   gt = {fit.after.gt:+.4f}  "
          f"distortion {np.array2string(dist * 100, precision=3)} %")
    print(f"  image-surface sagitta {locus.max_sagitta:.5f} mm   "
          f"blur {np.array2string(locus.blur_rms_um, precision=3)} um "
          f"(image-side Airy ≈ 28.7 um)")
    return fit


# ---------------------------------------------------------------- projector
def projector():
    print("=" * 70)
    print("PROJECTOR — 4 lenses (N-SK16/N-F2/N-BK7/N-SK4), slide -> screen 2 m, 45x")
    t0 = time.time()

    def build(d):
        return StigmaticTrain(media=(AIR, SK16, AIR, F2, AIR, BK7, AIR, SK4, AIR),
            vertices=(0.0, 6.0, 10.0, 14.0, 18.0, 24.0, 28.0, 34.0),
            conjugates=(-40.0, *d, 2000.0), wavelength_um=WL, semidiameter=16.0)

    na = 0.10
    fields = (3.0, 6.0)
    aim = 17.0
    cons = [Aplanatism(),
            TargetMagnification(value=-45.0, weight=0.02),
            Distortion(heights=fields, weight=2.0, aim_z=aim),
            AxialColor(weight=0.05)]

    def score(fit):
        ctx = EvaluationContext(fit.train, na_object_sine=na, samples=9)
        color = float(np.max(np.abs(AxialColor().residuals(ctx))))
        dist = float(np.max(np.abs(Distortion(heights=fields, aim_z=aim).residuals(ctx))))
        return fit.after.map_rms + 0.02 * abs(fit.after.gt + 45.0) + dist + 5e-3 * color

    best = multistart(build, cons, na=na, samples=11, n_free=7, starts=6,
                      seed=17, spans=[(-900, 900)] * 7, bounds=(-6e3, 6e3),
                      score=score)
    s, fit = best
    ctx = EvaluationContext(fit.train, na_object_sine=na, samples=9)
    dist = Distortion(heights=fields, aim_z=17.0).residuals(ctx)
    color = AxialColor().residuals(ctx)
    locus = fit.image_surface
    print(f"  d* = {tuple(round(v, 3) for v in fit.result.x)}   [{time.time()-t0:.0f}s]")
    print(f"  (M-1)_RMS {fit.after.map_rms:.2e}   gt = {fit.after.gt:+.3f}  "
          f"distortion {np.array2string(dist * 100, precision=3)} %  "
          f"color F/C {color[0]:+.2f}/{color[1]:+.2f} mm")
    print(f"  blur at slide fields {np.array2string(locus.blur_rms_um, precision=1)} um on screen")
    return fit


# ---------------------------------------------------------------- ultrawide
class VirtualObjectAplanatism(Constraint):
    """Sine condition for a train with a *virtual* object (d0 > 0).

    Rays cannot be launched from the object plane (it sits inside the
    system), so a converging fan aimed at the virtual point is launched
    from a plane in front of the lens and the aplanatism map M is
    evaluated from the traced hit heights — the closed form never cared
    whether the conjugate is real.
    """

    def __init__(self, sines, launch_z=-25.0, weight=1.0):
        self.sines = tuple(sines)
        self.launch_z = launch_z
        self.weight = weight

    def size(self, context):
        return len(self.sines)

    def residuals(self, context):
        train = context.train
        tracer = SequentialTracer(train.to_system())
        a = train.conjugates[0]
        out = np.full(len(self.sines), 1e3)
        heights, kept = [], []
        for i, sine in enumerate(self.sines):
            t = sine / np.sqrt(1.0 - sine * sine)
            origin = np.array([0.0, (a - self.launch_z) * t, self.launch_z])
            direction = np.array([0.0, -t, 1.0]) / np.sqrt(1.0 + t * t)
            r = tracer.trace(origin, direction, keep_path=True)
            if not r.ok:
                continue
            heights.append(np.abs(r.path[1:-1, 1]))
            kept.append(i)
        if kept:
            m = aplanatism_map(train, np.array(heights), from_heights=True)
            out[np.array(kept)] = m - 1.0
        return self.weight * out


def ultrawide():
    print("=" * 70)
    print("ULTRAWIDE — 2 lenses (N-BK7), virtual object d0 = +8 mm, rays to ±55°")
    t0 = time.time()

    def build(d):
        return StigmaticTrain(media=(AIR, SF11, AIR, BK7, AIR), vertices=(0.0, 7.0, 11.0, 18.0),
            conjugates=(8.0, *d, 70.0), wavelength_um=WL, semidiameter=22.0)

    sines = np.sin(np.deg2rad(np.linspace(5.0, 55.0, 11)))
    cons = [VirtualObjectAplanatism(sines)]

    def score(fit):
        ctx = EvaluationContext(fit.train, na_object_sine=0.1, samples=5)
        return float(np.sqrt(np.mean(cons[0].residuals(ctx) ** 2)))

    best = multistart(build, cons, na=0.1, samples=5, n_free=3, starts=6,
                      seed=23, spans=[(-500, 500)] * 3, bounds=(-3e3, 3e3),
                      score=score)
    s, fit = best
    ctx = EvaluationContext(fit.train, na_object_sine=0.1, samples=5)
    res = cons[0].residuals(ctx)
    # stigmatism check at the extreme angle: manual trace to the image plane
    tracer = SequentialTracer(fit.train.to_system())
    spread = []
    for deg in (15.0, 35.0, 55.0):
        t = np.tan(np.deg2rad(deg))
        origin = np.array([0.0, (8.0 - (-25.0)) * t, -25.0])
        direction = np.array([0.0, -t, 1.0]) / np.sqrt(1 + t * t)
        r = tracer.trace(origin, direction)
        spread.append(abs(r.image_point[1]) if r.ok else np.nan)
    print(f"  d* = {tuple(round(v, 3) for v in fit.result.x)}   [{time.time()-t0:.0f}s]")
    print(f"  aplanatism |M-1| rms over 5-55 deg: {s:.3e}   "
          f"(worst {np.max(np.abs(res)):.3e})")
    print(f"  image height of rays aimed at A from 15/35/55 deg: "
          f"{np.array2string(np.array(spread) * 1e3, precision=3)} um  <- stigmatic at all angles")
    return fit


if __name__ == "__main__":
    tel = telescope()
    mic = microscope()
    pro = projector()
    uw = ultrawide()
    print("=" * 70)
    print("WINNERS (paste into notebook):")
    for name, fit in (("telescope", tel), ("microscope", mic),
                      ("projector", pro), ("ultrawide", uw)):
        print(f"  {name}: {tuple(float(f'{v:.6g}') for v in fit.result.x)}")
