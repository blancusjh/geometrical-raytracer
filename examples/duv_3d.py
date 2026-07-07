"""3-D view of the US7557996 DUV objective with traced field bundles.

Usage:
    python -m examples.duv_3d                 # interactive turntable
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


def main() -> None:
    csv = resources.files("raytracer") / "data" / "US7557996_Fig3_Table3_prescription.csv"
    system = OpticalSystem.from_prescription(csv)
    tracer = SequentialTracer(system)
    solve_object_plane(tracer)

    viewer = Viewer3D(title="US7557996 — 3D")
    viewer.add_system(system)
    viewer.add_field_bundles(
        tracer, fields=[56.0, 62.0, 67.0], na_object_sine=0.30, radial=6, azimuth=48
    )

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
