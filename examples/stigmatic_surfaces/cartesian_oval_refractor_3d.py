"""3-D Cartesian oval: a stigmatic refracting surface of revolution, in 3-D.

The 3-D counterpart of ``cartesian_oval_refractor_2d.py``: the same GOTS
closed-form surface (:class:`~raytracer.surfaces.cartesian_oval.
CartesianOvalProfile`, ``SurfaceRow.cartesian_oval``), here revolved into its
actual 3-D shape and traversed by a full 3-D pupil bundle traced with the
exact sequential engine — the same viewer as
``examples/lithography/duv_objective_3d.py``.

Two ray modes, toggled live with the **m** key:

* **lines** (default) — a sparse bundle of alpha-blended traces;
* **beam** — a dense additive bundle: overlapping rays sum into a continuous
  glow that collapses to a single bright point at the image inside the glass,
  which is the stigmatism made visible.

The claim is also checked numerically, not just drawn: the exit-pupil OPD and
the geometric spot RMS (:func:`raytracer.analysis.stigmatism_report`) are
printed and read ~0.

Other keys: **a** toggles the meridional scale axes (tick numbers re-label
as you zoom), **w** wireframe, drag orbit, wheel zoom. The green dot marks
the design image point at constant pixel size regardless of zoom.

Note when orbiting near the optical axis: the ray polylines bend at the
surface (the refraction), and under strong foreshortening those bend points
cluster into what can look like a second focus — the axes make the actual
scale and positions unambiguous.

Usage:
    python -m examples.stigmatic_surfaces.cartesian_oval_refractor_3d
    python -m examples.stigmatic_surfaces.cartesian_oval_refractor_3d --beam
    python -m examples.stigmatic_surfaces.cartesian_oval_refractor_3d --save out.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer.analysis import stigmatism_report
from raytracer.optics.materials import AIR, ConstantIndex
from raytracer.design import OpticalSystem, SurfaceRow
from raytracer.propagation import FieldPoint, PupilSampling, SequentialTracer, trace_pupil
from raytracer.viz.gl3d import Viewer3D, spectral_rgb

N0, Z0 = 1.0, -30.0  # object side: air, 30 mm in front of the vertex
NI, ZI = 1.7, 10.0  # image side: dense glass, 10 mm behind the vertex
SEMIDIAMETER = 3.0
NA_OBJECT = 0.1
WAVELENGTH_NM = 550.0  # display color for the beam mode
LINES_SAMPLING = dict(radial=3, azimuth=14)  # sparse: readable individual traces
BEAM_SAMPLING = dict(radial=40, azimuth=240)
BEAM_ENERGY = 12.0


def build_system() -> OpticalSystem:
    glass = ConstantIndex("DENSE_GLASS", NI)
    row = SurfaceRow.cartesian_oval(
        n0=N0, z0=Z0, ni=NI, zi=ZI, thickness=ZI, material=glass,
        semidiameter=SEMIDIAMETER, comment="GOTS Cartesian oval",
    )
    return OpticalSystem([row], object_space=AIR, object_z=Z0)


def _bundle_paths(tracer: SequentialTracer, sampling: dict) -> tuple[np.ndarray, int]:
    """Trace an on-axis pupil bundle (surface = its own stop: chief slope 0)."""

    pupil = trace_pupil(
        tracer, FieldPoint(y=0.0), na_object_sine=NA_OBJECT,
        sampling=PupilSampling(kind="rings", **sampling),
        chief_slope=0.0, keep_paths=True,
    )
    return pupil.batch.paths[pupil.valid], int(pupil.valid.sum())


def main() -> None:
    system = build_system()
    tracer = SequentialTracer(system, restart_offset=0.0)

    print("3-D Cartesian oval refractor: stigmatic imaging")
    print(f"  object: z0={Z0} (air, n={N0}); image: zi={ZI} (glass, n={NI})")
    print(f"  max usable height for this conjugate pair: "
          f"{system.rows[0].profile.max_usable_height:.4f} (aperture = {SEMIDIAMETER})")

    report = stigmatism_report(
        trace_pupil(tracer, FieldPoint(y=0.0), na_object_sine=NA_OBJECT,
                    sampling=PupilSampling(kind="rings", radial=10, azimuth=32),
                    slope_model="sine", chief_slope=0.0),
        wavelength_mm=WAVELENGTH_NM * 1e-6, na_image=0.3, reference_radius=5.0,
    )
    print(report.summary())
    assert report.is_stigmatic(), "expected a stigmatic conjugate pair"

    viewer = Viewer3D(
        title="Cartesian oval (stigmatic) — 3D   [m] lines/beam  [a] axes  [w] wireframe"
    )
    viewer.add_system(system)

    lines_paths, _ = _bundle_paths(tracer, LINES_SAMPLING)
    viewer.add_paths(lines_paths, color=(0.45, 0.80, 1.0, 0.55), group="lines")

    beam_paths, n_beam = _bundle_paths(tracer, BEAM_SAMPLING)
    rgb = spectral_rgb(WAVELENGTH_NM) * (BEAM_ENERGY / max(n_beam, 1))
    viewer.add_paths(beam_paths, color=(*rgb.tolist(), 1.0), width=1.0,
                     group="beam", additive=True)

    viewer.add_marker([0.0, 0.0, ZI])  # design image point (pixel-sized at any zoom)
    viewer.add_axes()

    if "--beam" in sys.argv:
        viewer.set_ray_mode("beam")

    if "--save" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--save") + 1])
        import matplotlib.image as mpimg

        mpimg.imsave(out, viewer.snapshot())
        print(f"wrote {out}")
    else:  # pragma: no cover - interactive
        print("Opening window — [m] cycles ray mode, [w] wireframe, drag to orbit.")
        viewer.run()


if __name__ == "__main__":
    main()
