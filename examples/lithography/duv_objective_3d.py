"""3-D view of the US7557996 DUV objective with traced field bundles.

Two ray modes, toggled live with the **m** key:

* **lines** (default) — per-field colored rays, MSAA-antialiased, depth-tested:
  occluded by the mirrors and tinted by the semitransparent glass they cross.
* **beam** — a much denser bundle blended additively: every ray carries the
  spectral color of the operating wavelength (CIE 1931 -> linear sRGB, as in
  diffractsim) scaled by its energy, so overlaps sum into a continuous glow.
  193.368 nm is deep UV, rendered as its violet representation.

Other keys: **w** wireframe, drag orbit, wheel zoom.

Usage:
    python -m examples.lithography.duv_objective_3d                 # interactive (start in lines mode)
    python -m examples.lithography.duv_objective_3d --beam          # start in beam mode
    python -m examples.lithography.duv_objective_3d --save out.png  # offscreen snapshot
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer.design import OpticalSystem
from raytracer.propagation import SequentialTracer, solve_object_plane
from raytracer.viz.gl3d import Viewer3D

FIELDS = [56.0, 62.0, 67.0]
NA_OBJECT = 0.30

# Beam-density multiplier. The additive beam/spectrum modes get smoother and
# more continuous with more rays (cost: ~N² rays). The viewer clamps the drawn
# beam to ~30k rays/field (display cap, visually equivalent beyond that), so
# large N stays fast and the beam never quantizes to invisible. The *lines*
# mode stays at fixed sampling: multiplied alpha traces only turn to spaghetti.
N = 10
LINES_SAMPLING = dict(radial=6, azimuth=48)
BEAM_SAMPLING = dict(radial=14 * N, azimuth=96 * N)


def _bundle(viewer, tracer, label, sampling, **kw):
    import time

    n_est = sampling["radial"] * sampling["azimuth"] * len(FIELDS)
    print(f"[duv_objective_3d] Tracing {label}: ~{n_est:,} rays...", end=" ", flush=True)
    if n_est > 200_000:
        print("(huge sampling, this may take minutes)", end=" ", flush=True)
    t0 = time.time()
    viewer.add_field_bundles(tracer, fields=FIELDS, na_object_sine=NA_OBJECT,
                             **sampling, **kw)
    print(f"{time.time() - t0:.1f}s", flush=True)


def main() -> None:
    print("[duv_objective_3d] Loading the US7557996 prescription (48 surfaces)...", flush=True)
    csv = ROOT / "data" / "optical_systems/lithography/US7557996_Fig3_Table3_prescription.csv"
    system = OpticalSystem.from_prescription(csv)
    tracer = SequentialTracer(system)
    solve_object_plane(tracer)

    viewer = Viewer3D(title="US7557996 — 3D   [m] lines/beam/spectrum  [a] axes  [w] wireframe")
    viewer.add_system(system)
    viewer.add_axes()
    _bundle(viewer, tracer, "lines mode", LINES_SAMPLING)
    _bundle(viewer, tracer, "beam mode (additive, violet)", BEAM_SAMPLING, mode="beam")
    _bundle(viewer, tracer, "spectrum mode (colorimetric)", BEAM_SAMPLING,
            mode="spectrum")
    print("[duv_objective_3d] Opening window — [m] cycles ray mode, [w] wireframe,"
          " drag to orbit.", flush=True)
    if "--beam" in sys.argv:
        viewer.set_ray_mode("beam")
    elif "--spectrum" in sys.argv:
        viewer.set_ray_mode("spectrum")

    if "--save" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--save") + 1])
        image = viewer.snapshot()
        import matplotlib.image as mpimg

        mpimg.imsave(out, image)
        print(f"wrote {out}")
    else:  # pragma: no cover - interactive
        viewer.run()


if __name__ == "__main__":
    main()
