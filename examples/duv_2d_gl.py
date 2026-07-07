"""2-D OpenGL view of the US7557996 DUV objective (HDR ray accumulation).

Meridional (z, y) layout of the folded catadioptric objective in the
`raytracer.viz.gl` HDR viewer:

* each lens element is drawn as a **closed outline** (both faces joined at the
  rim) so it reads as a solid body enclosing one medium;
* the two mirrors are thick gold caps;
* a dense fan of exact rays per field fills the full object aperture and
  **terminates on a drawn observation screen** at the image plane (markers show
  where each ray lands);
* rays accumulate additively so the converging cones glow.

Usage:
    python -m examples.duv_2d_gl                 # interactive pan/zoom
    python -m examples.duv_2d_gl --save out.png  # offscreen snapshot
"""

from __future__ import annotations

import sys
from importlib import resources
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vispy.color import Color

from raytracer import OpenGLViewer, RenderConfig
from raytracer.sequential import (
    FieldPoint, OpticalSystem, SequentialTracer, chief_ray_slope, solve_object_plane,
)
from raytracer.sequential.fields import trace_from_object
from raytracer.viz.plots import DEFAULT_MATERIAL_COLORS

GLASS_ALPHA = 0.30

FIELDS = [56.0, 62.0, 67.0]
FIELD_RGB = [(0.35, 0.70, 1.0), (0.40, 1.0, 0.55), (1.0, 0.60, 0.30)]
NA_OBJECT = 1.2 / 4.0            # object-space sine = 0.3
FAN_HALF_SLOPE = 0.305          # ~asin(0.3) marginal slope -> full aperture
RAYS_PER_FIELD = 121


def _semi(row, fallback=50.0):
    return row.semidiameter if row.semidiameter is not None else fallback


def surface_curve(system, i, samples=260):
    """Open meridional profile (z, y) of surface ``i``."""
    row = system.rows[i]
    semi = _semi(row)
    h = np.linspace(-semi, semi, samples)
    z = system.vertices[i] + row.profile.sag(np.abs(h))
    return np.column_stack([z, h])


def lens_fill_mesh(system, i, j, samples=200):
    """Strip triangulation (verts, faces) of the element body between faces i and j."""
    ri, rj = system.rows[i], system.rows[j]
    semi = max(_semi(ri), _semi(rj))
    h = np.linspace(-semi, semi, samples)
    zf = system.vertices[i] + ri.profile.sag(np.abs(h))
    zb = system.vertices[j] + rj.profile.sag(np.abs(h))
    verts = np.empty((2 * samples, 2))
    verts[0::2] = np.column_stack([zf, h])   # front-face vertices (even)
    verts[1::2] = np.column_stack([zb, h])   # back-face vertices (odd)
    k = np.arange(samples - 1)
    faces = np.empty((2 * (samples - 1), 3), dtype=np.uint32)
    faces[0::2] = np.column_stack([2 * k, 2 * k + 1, 2 * k + 2])
    faces[1::2] = np.column_stack([2 * k + 1, 2 * k + 3, 2 * k + 2])
    return verts, faces


def material_rgba(material, alpha=GLASS_ALPHA):
    face = DEFAULT_MATERIAL_COLORS.get(material, ("#c9c9c9",))[0]
    r, g, b = Color(face).rgb
    return (float(r), float(g), float(b), alpha)


def lens_outline(system, i, j, samples=220):
    """Closed (z, y) outline of the element spanning surfaces i (front) and j (back)."""
    ri, rj = system.rows[i], system.rows[j]
    semi = max(_semi(ri), _semi(rj))
    h = np.linspace(-semi, semi, samples)
    zf = system.vertices[i] + ri.profile.sag(np.abs(h))
    zb = system.vertices[j] + rj.profile.sag(np.abs(h))
    front = np.column_stack([zf, h])
    back = np.column_stack([zb, h])[::-1]
    return np.vstack([front, back, front[:1]])  # closed loop


def build_viewer():
    csv = resources.files("raytracer") / "data" / "US7557996_Fig3_Table3_prescription.csv"
    system = OpticalSystem.from_prescription(csv)
    tracer = SequentialTracer(system)
    solve_object_plane(tracer)

    verts = np.asarray(system.vertices, dtype=float)
    z_lo = min(system.object_z, verts.min())
    z_hi = max(system.image_z, verts.max())
    y_max = max(_semi(r, 0.0) for r in system.rows) * 1.12
    pad = 0.03 * (z_hi - z_lo)

    viewer = OpenGLViewer(
        x_lims=(z_lo - pad, z_hi + pad), y_lims=(-y_max, y_max),
        size=(1600, 640),
        render_config=RenderConfig(
            ray_width=0.7, sigma_factor=0.42, min_pixels=1.0,
            use_solid_rays=False, background="black",
        ),
    )

    # --- dense ray fans, terminating at the image plane ---
    landings = []
    for field_y, rgb in zip(FIELDS, FIELD_RGB):
        sy0 = chief_ray_slope(tracer, FieldPoint(y=field_y))
        segs, cols = [], []
        for delta in np.linspace(-FAN_HALF_SLOPE, FAN_HALF_SLOPE, RAYS_PER_FIELD):
            res = trace_from_object(tracer, (0.0, field_y), (0.0, sy0 + delta), keep_path=True)
            if res.path is None:
                continue
            path = res.path[~np.isnan(res.path[:, 2])]
            if len(path) < 2:
                continue
            zy = np.column_stack([path[:, 2], path[:, 1]])
            segs.append(np.stack([zy[:-1], zy[1:]], axis=1))
            if abs(zy[-1, 0] - system.image_z) < 1e-3:
                landings.append(zy[-1])
        if segs:
            segments = np.concatenate(segs, axis=0)
            colors = np.tile([*rgb, 1.0], (len(segments), 1))
            viewer.draw_ray_segments(segments, colors=colors)

    # --- lens bodies filled by material (shows the enclosed medium / index) ---
    for i, j, material in system.solid_elements():
        verts, faces = lens_fill_mesh(system, i, j)
        viewer.draw_filled_polygon(verts, faces, color=material_rgba(material))

    # --- lens outlines (closed) and mirrors, drawn over the fills ---
    for i, j, material in system.solid_elements():
        edge = DEFAULT_MATERIAL_COLORS.get(material, ("#c9c9c9", "#888888"))[1]
        er, eg, eb = Color(edge).rgb
        viewer.draw_polyline(lens_outline(system, i, j),
                             color=(float(er), float(eg), float(eb), 1.0), width=1.4)
    for i in system.mirror_indices:
        viewer.draw_polyline(surface_curve(system, i), color=(1.0, 0.82, 0.35, 1.0), width=3.4)

    # --- observation screen at the image plane + landing markers ---
    screen = np.array([[system.image_z, -y_max * 0.55], [system.image_z, y_max * 0.55]])
    viewer.draw_polyline(screen, color=(1.0, 1.0, 1.0, 1.0), width=2.5)
    if landings:
        viewer.draw_markers(np.asarray(landings), color=(1.0, 1.0, 1.0), size=4.0)

    # --- object plane (dim) ---
    obj = np.array([[system.object_z, -y_max * 0.6], [system.object_z, y_max * 0.6]])
    viewer.draw_polyline(obj, color=(0.5, 0.5, 0.55, 1.0), width=1.5)

    return viewer


def main():
    viewer = build_viewer()
    if "--save" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--save") + 1])
        import matplotlib.image as mpimg

        mpimg.imsave(out, viewer.snapshot())
        print(f"wrote {out}")
        return
    viewer.run()


if __name__ == "__main__":
    main()
