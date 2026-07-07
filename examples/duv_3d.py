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


def main() -> None:
    csv = resources.files("raytracer") / "data" / "US7557996_Fig3_Table3_prescription.csv"
    system = OpticalSystem.from_prescription(csv)
    tracer = SequentialTracer(system)
    solve_object_plane(tracer)

    viewer = Viewer3D(title="US7557996 — 3D   [m] lines/beam/spectrum  [w] wireframe")
    viewer.add_system(system)
    # mode 1: colored line traces (no color summing)
    viewer.add_field_bundles(
        tracer, fields=FIELDS, na_object_sine=NA_OBJECT, radial=6, azimuth=48
    )
    # mode 2: additive violet beam at the DUV wavelength
    viewer.add_field_bundles(
        tracer, fields=FIELDS, na_object_sine=NA_OBJECT, radial=14, azimuth=96,
        mode="beam",
    )
    # mode 3: one wavelength per field, colorimetric additive mixing
    viewer.add_field_bundles(
        tracer, fields=FIELDS, na_object_sine=NA_OBJECT, radial=14, azimuth=96,
        mode="spectrum",
    )
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
