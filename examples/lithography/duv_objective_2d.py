"""2-D OpenGL view of the US7557996 DUV objective (HDR ray accumulation).

Meridional (z, y) section of the folded catadioptric objective in the
`raytracer.viz.gl` HDR viewer:

* each lens is drawn as its **body** — both faces joined at the rim, filled
  by material — so it reads as a solid enclosing one medium;
* the two mirrors are thick gold arcs;
* a dense fan of exact sequential rays per field fills the full object
  aperture and **terminates on the image sensor** (markers show where each
  ray lands);
* rays accumulate additively ("beam" mode, ``use_solid_rays=False``) so the
  converging cones glow — see ``examples/telescopes/*.py`` for the "geometric"
  (crisp, non-additive) counterpart.

Usage:
    python -m examples.lithography.duv_objective_2d                 # interactive pan/zoom
    python -m examples.lithography.duv_objective_2d --save out.png  # offscreen snapshot
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer.elements import Mirror
from raytracer.io import read_prescription
from raytracer.tracing import FieldPoint, chief_ray_slopes, solve_object_plane, trace_from_object
from raytracer.viz import Drawing, LensBodyItem, RaySegmentsItem, SurfaceItem, show
from raytracer.viz.bodies import DEFAULT_MATERIAL_COLORS
from raytracer.viz.gl.viewer import RenderConfig

FIELDS = [56.0, 62.0, 67.0]
FIELD_RGB = [(0.35, 0.70, 1.0), (0.40, 1.0, 0.55), (1.0, 0.60, 0.30)]
FAN_HALF_SLOPE = 0.305  # ~asin(0.3) marginal slope (object NA 1.2/4) -> full aperture
RAYS_PER_FIELD = 121
MIRROR_COLOR = (1.0, 0.82, 0.35, 1.0)
GLASS_ALPHA = 0.30


def fan_segments(tracer, field_y: float) -> tuple[np.ndarray, np.ndarray]:
    """One field's fan about its chief ray: drawing-plane segments ``(N, 2, 2)`` and where it lands ``(M, 3)``."""

    chief = chief_ray_slopes(tracer, FieldPoint(y=field_y))[1]
    segments, landings = [], []
    for delta in np.linspace(-FAN_HALF_SLOPE, FAN_HALF_SLOPE, RAYS_PER_FIELD):
        result = trace_from_object(tracer, (0.0, field_y), (0.0, chief + delta), keep_path=True)
        path = result.path[np.isfinite(result.path).all(axis=1)]  # a vignetted ray is drawn up to where it stopped
        zy = path[:, [2, 1]]
        segments.append(np.stack([zy[:-1], zy[1:]], axis=1))
        if result.ok:
            landings.append(result.image_point)
    return np.concatenate(segments), np.array(landings)


def build_drawing() -> Drawing:
    print(f"[duv_objective_2d] Loading the prescription and tracing {len(FIELDS) * RAYS_PER_FIELD} exact rays...",
          flush=True)
    prescription = read_prescription(ROOT / "data/optical_systems/lithography/US7557996_Fig3_Table3_prescription.csv")
    tracer = prescription.tracer()
    solve_object_plane(tracer)
    system = tracer.system

    drawing = Drawing()
    landings = []
    for field_y, rgb in zip(FIELDS, FIELD_RGB):
        segments, landed = fan_segments(tracer, field_y)
        n = len(segments)
        drawing.add(RaySegmentsItem(segments, np.tile([*rgb, 1.0], (n, 1)), np.ones(n), np.zeros(n, dtype=bool),
                                    np.zeros((n, 2))))
        landings.append(landed)

    # Lens bodies filled by material, the mirrors in gold, the image sensor with every landing.
    drawing.add_system(system, wavelength_um=tracer.wavelength_um, material_colors=DEFAULT_MATERIAL_COLORS,
                       sensors=False)
    for item in drawing.items:
        if isinstance(item, LensBodyItem):  # light glass, so the beams read through it
            item.facecolor = (*item.facecolor[:3], GLASS_ALPHA)
        elif isinstance(item, SurfaceItem) and isinstance(system[item.name], Mirror):
            item.color, item.width = MIRROR_COLOR, 3.4
    sensor = system["IMG"]
    half = 0.55 * max(e.semi_diameter for e in system.elements if hasattr(e, "semi_diameter"))
    drawing.add_sensor(sensor, hits=np.vstack(landings), fallback=half)
    # The object plane, dim.
    z = tracer.object_z
    drawing.add(SurfaceItem(np.array([[z, -1.1 * half], [z, 1.1 * half]]), (0.5, 0.5, 0.55, 1.0), 1.5))
    return drawing


def main():
    save = Path(sys.argv[sys.argv.index("--save") + 1]) if "--save" in sys.argv else None
    drawing = build_drawing()
    if save is None:
        print("[duv_objective_2d] Opening window — drag to pan, wheel to zoom.", flush=True)
    backend = show(
        drawing, backend="gl", interactive=save is None, size=(1600, 640),
        title="US7557996 DUV objective (2D) — raytracer",
        render_config=RenderConfig(ray_width=0.7, sigma_factor=0.42, min_pixels=1.0, use_solid_rays=False,
                                   background="black"),
    )
    if save is not None:
        backend.save(save)
        print(f"wrote {save}")


if __name__ == "__main__":
    main()
