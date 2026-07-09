"""Galilean telescope: converging objective + diverging eyepiece.

The classic "opera glass" layout — a positive objective and a *negative*
(diverging) eyepiece separated by ``f_obj + f_eye`` (with ``f_eye < 0``),
giving an upright image in a short tube. Compare with the Keplerian design
(``examples/telescopes/keplerian.py``), which forms a real intermediate image
at the cost of a longer tube and an inverted view.

Each lens is a real element (`Lens2D.spherical`): two spherical surfaces
enclosing the glass, closed by absorbing rims at the clear aperture — drawn
as a filled body, exactly like the lithography examples. The input beams are
sized so the compressed output beam fits through the eyepiece aperture.

Being a telescope means being afocal, and that is checked, not assumed:

* the objective-eyepiece air gap is *solved* (:func:`_afocal.solve_afocal_gap`)
  so the effective focal length is exactly infinite for the real
  finite-thickness lenses, not the thin-lens estimate;
* two parallel rays entering at the same field angle are traced through the
  exact sequential engine and confirmed to exit parallel, with the angular
  magnification measured as the output/input angle ratio (positive: upright).

Usage:
    python -m examples.telescopes.galilean
    python -m examples.telescopes.galilean --save out.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer import ParallelSource2D, RayTracer2D, RenderConfig, TraceConfig
from raytracer.core.materials import AIR, ConstantIndex
from raytracer.nonseq.elements import Lens2D
from raytracer.sequential import OpticalSystem, SequentialTracer, SurfaceRow
from raytracer.viz import Scene, show
from examples.telescopes._afocal import solve_afocal_gap

GLASS_N = 1.5168  # BK7
GLASS = ConstantIndex("BK7", GLASS_N)
F_OBJECTIVE = 200.0
F_EYEPIECE = -50.0  # negative: diverging eyepiece
OBJ_THICKNESS, OBJ_SEMI = 5.0, 25.0
EYE_THICKNESS, EYE_SEMI = 3.0, 8.0
BEAM_HALF_WIDTH = 12.0  # compressed |M|x by the telescope: must exit within EYE_SEMI
FIELD_ANGLE = 0.03  # rad, off-axis bundle

ON_AXIS_COLOR = (0.5, 0.85, 1.0, 0.9)
OFF_AXIS_COLOR = (1.0, 0.85, 0.4, 0.9)


def _equiconvex_radius(f: float, n: float = GLASS_N) -> float:
    """Radius of the two spherical faces of an equiconvex/equiconcave lens
    with focal length *f* (thin-lens estimate; the gap solve absorbs the
    thickness error)."""

    return 2.0 * (n - 1.0) * f


R_OBJ = _equiconvex_radius(F_OBJECTIVE)
R_EYE = _equiconvex_radius(F_EYEPIECE)


def build_sequential(gap: float) -> OpticalSystem:
    """Exact on-axis model used for the afocal solve and verification."""

    rows = [
        SurfaceRow.refracting(radius=R_OBJ, thickness=OBJ_THICKNESS, material=GLASS, semidiameter=OBJ_SEMI),
        SurfaceRow.refracting(radius=-R_OBJ, thickness=gap, material=AIR, semidiameter=OBJ_SEMI),
        SurfaceRow.refracting(radius=R_EYE, thickness=EYE_THICKNESS, material=GLASS, semidiameter=EYE_SEMI),
        SurfaceRow.refracting(radius=-R_EYE, thickness=30.0, material=AIR, semidiameter=EYE_SEMI),
    ]
    return OpticalSystem(rows, object_space=AIR)


def angular_magnification(system: OpticalSystem, theta_in: float = 0.01) -> tuple[float, float]:
    """Trace two parallel rays at field angle *theta_in*; return (M, parallelism residual in rad)."""

    tracer = SequentialTracer(system, restart_offset=0.0)
    d = np.array([0.0, np.sin(theta_in), np.cos(theta_in)])
    origins = np.array([[0.0, -5.0, -20.0], [0.0, 5.0, -20.0]])
    result = tracer.trace_batch(origins, np.tile(d, (2, 1)))
    theta_out = np.arctan2(result.directions[:, 1], result.directions[:, 2])
    return float(np.mean(theta_out) / theta_in), float(np.ptp(theta_out))


def build_scene(gap: float) -> Scene:
    objective = Lens2D.spherical(
        R1=R_OBJ, R2=-R_OBJ, thickness=OBJ_THICKNESS, semidiameter=OBJ_SEMI,
        n=GLASS_N, vertex=(0.0, 0.0), name="objective",
    )
    eyepiece = Lens2D.spherical(
        R1=R_EYE, R2=-R_EYE, thickness=EYE_THICKNESS, semidiameter=EYE_SEMI,
        n=GLASS_N, vertex=(OBJ_THICKNESS + gap, 0.0), name="eyepiece",
    )
    surfaces = [*objective.surfaces(), *eyepiece.surfaces()]
    tracer = RayTracer2D(surfaces, TraceConfig(max_generations=8, fresnel_split=False))

    on_axis = ParallelSource2D(
        origin=np.array([-25.0, 0.0]), direction=np.array([1.0, 0.0]),
        width=2.0 * BEAM_HALF_WIDTH, samples=17,
    )
    off_axis = ParallelSource2D(
        origin=np.array([-25.0, -25.0 * np.tan(FIELD_ANGLE)]),
        direction=np.array([np.cos(FIELD_ANGLE), np.sin(FIELD_ANGLE)]),
        width=2.0 * BEAM_HALF_WIDTH, samples=17,
    )

    exit_x = OBJ_THICKNESS + gap + EYE_THICKNESS
    scene = Scene(x_lims=(-30.0, exit_x + 50.0), y_lims=(-30.0, 30.0))
    scene.add_elements([objective, eyepiece])
    scene.add_tree(tracer.trace([on_axis]), default_color=ON_AXIS_COLOR)
    scene.add_tree(tracer.trace([off_axis]), default_color=OFF_AXIS_COLOR)
    return scene


def main() -> None:
    gap, system = solve_afocal_gap(build_sequential, low=100.0, high=200.0)
    magnification, residual = angular_magnification(system)

    print("Galilean telescope")
    print(f"  objective f={F_OBJECTIVE} mm, eyepiece f={F_EYEPIECE} mm")
    print(f"  objective-eyepiece air gap (solved, exact afocal): {gap:.4f} mm")
    print(f"  angular magnification: {magnification:.3f}x "
          f"({'upright' if magnification > 0 else 'inverted'})")
    print(f"  parallelism residual between two field rays: {residual:.2e} rad "
          "(should be ~0: confirms afocal)")

    scene = build_scene(gap)
    render_config = RenderConfig(ray_width=0.25, min_pixels=1.0, use_solid_rays=True)

    save_path = None
    if "--save" in sys.argv:
        save_path = Path(sys.argv[sys.argv.index("--save") + 1])
    backend = show(
        scene, backend="gl", interactive=save_path is None,
        size=(1500, 620), bgcolor="black", render_config=render_config,
        title="Galilean telescope — raytracer",
    )
    if save_path is not None:
        backend.save(save_path)
        print(f"wrote {save_path}")


if __name__ == "__main__":
    main()
