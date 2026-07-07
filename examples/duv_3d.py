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
    python -m examples.duv_3d                 # interactive (start in lines mode)
    python -m examples.duv_3d --beam          # start in beam mode
    python -m examples.duv_3d --save out.png  # offscreen snapshot
"""

from __future__ import annotations

import sys
from importlib import resources
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer.sequential import OpticalSystem, SequentialTracer, solve_object_plane
from raytracer.viz.gl3d import Viewer3D

FIELDS = [56.0, 62.0, 67.0]
NA_OBJECT = 0.30

# Beam-density multiplier. The additive beam/spectrum modes get smoother and
# more continuous with more rays (cost: ~N² rays -> N=3 is already very dense).
# The *lines* mode stays at fixed sampling: individual alpha traces only turn
# into spaghetti when multiplied.
N = 3
LINES_SAMPLING = dict(radial=6, azimuth=48)
BEAM_SAMPLING = dict(radial=14 * N, azimuth=96 * N)


def _bundle(viewer, tracer, label, sampling, **kw):
    import time

    n_est = sampling["radial"] * sampling["azimuth"] * len(FIELDS)
    print(f"[duv_3d] Trazando {label}: ~{n_est:,} rayos...", end=" ", flush=True)
    if n_est > 200_000:
        print("(muestreo ENORME, esto puede tardar minutos)", end=" ", flush=True)
    t0 = time.time()
    viewer.add_field_bundles(tracer, fields=FIELDS, na_object_sine=NA_OBJECT,
                             **sampling, **kw)
    print(f"{time.time() - t0:.1f}s", flush=True)


def main() -> None:
    print("[duv_3d] Cargando prescripción US7557996 (48 superficies)...", flush=True)
    csv = resources.files("raytracer") / "data" / "US7557996_Fig3_Table3_prescription.csv"
    system = OpticalSystem.from_prescription(csv)
    tracer = SequentialTracer(system)
    solve_object_plane(tracer)

    viewer = Viewer3D(title="US7557996 — 3D   [m] lines/beam/spectrum  [w] wireframe")
    viewer.add_system(system)
    _bundle(viewer, tracer, "modo lines", LINES_SAMPLING)
    _bundle(viewer, tracer, "modo beam (aditivo violeta)", BEAM_SAMPLING, mode="beam")
    _bundle(viewer, tracer, "modo spectrum (colorimétrico)", BEAM_SAMPLING,
            mode="spectrum")
    print("[duv_3d] Abriendo ventana — [m] cambia modo de rayos, [w] wireframe,"
          " arrastra para orbitar.", flush=True)
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
