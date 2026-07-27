"""Keplerian telescope: two converging lenses sharing a real intermediate image.

Both the objective and the eyepiece are positive, separated by the sum of
their focal lengths: the objective's rear focal point coincides with the
eyepiece's front focal point, where a real, inverted intermediate image forms
(marked in the diagram). That real image is where a reticle or field stop
goes in a practical eyepiece — the advantage over the Galilean design
(``examples/telescopes/galilean.py``), traded for a longer tube and an
inverted view.

Each lens is a real element (`Lens.spherical`): two spherical surfaces
enclosing the glass, closed by absorbing rims, drawn as a filled body.

As in the Galilean example, being afocal is checked, not assumed: the gap is
solved for an exactly infinite effective focal length
(:func:`_afocal.solve_afocal_gap`), and two traced parallel field rays are
confirmed to exit parallel, with the angular magnification measured as their
output/input angle ratio (negative: inverted).

Usage:
    python -m examples.telescopes.keplerian
    python -m examples.telescopes.keplerian --save out.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raytracer import BranchingTracer, ParallelSource, RenderConfig, TraceConfig
from raytracer.design import OpticalSystem, SurfaceRow
from raytracer.optics import Lens
from raytracer.optics.materials import AIR, ConstantIndex
from raytracer.propagation import SequentialTracer
from raytracer.viz import Scene, show
from raytracer.viz.scene import MarkerItem
from examples.telescopes._afocal import solve_afocal_gap
from examples.telescopes.galilean import GLASS_N, _equiconvex_radius, angular_magnification

GLASS = ConstantIndex("BK7", GLASS_N)
F_OBJECTIVE = 200.0
F_EYEPIECE = 50.0  # positive: converging eyepiece
OBJ_THICKNESS, OBJ_SEMI = 5.0, 25.0
EYE_THICKNESS, EYE_SEMI = 4.0, 12.0
BEAM_HALF_WIDTH = 14.0  # compressed |M|x by the telescope: must exit within EYE_SEMI
FIELD_ANGLE = 0.02  # rad, off-axis bundle

ON_AXIS_COLOR = (0.5, 0.85, 1.0, 0.9)
OFF_AXIS_COLOR = (1.0, 0.85, 0.4, 0.9)

R_OBJ = _equiconvex_radius(F_OBJECTIVE)
R_EYE = _equiconvex_radius(F_EYEPIECE)


def build_sequential(gap: float) -> OpticalSystem:
    """Exact on-axis model used for the afocal solve and verification."""

    rows = [
        SurfaceRow.refracting(radius=R_OBJ, thickness=OBJ_THICKNESS, material=GLASS, semidiameter=OBJ_SEMI),
        SurfaceRow.refracting(radius=-R_OBJ, thickness=gap, material=AIR, semidiameter=OBJ_SEMI),
        SurfaceRow.refracting(radius=R_EYE, thickness=EYE_THICKNESS, material=GLASS, semidiameter=EYE_SEMI),
        SurfaceRow.refracting(radius=-R_EYE, thickness=40.0, material=AIR, semidiameter=EYE_SEMI),
    ]
    return OpticalSystem(rows, object_space=AIR)


def objective_focus_x() -> float:
    """Exact rear focal point of the (finite-thickness) objective alone.

    Traces a marginal parallel ray through just the objective's two surfaces
    and intersects the exiting ray with the axis — no thin-lens estimate.
    """

    rows = [
        SurfaceRow.refracting(radius=R_OBJ, thickness=OBJ_THICKNESS, material=GLASS, semidiameter=OBJ_SEMI),
        SurfaceRow.refracting(radius=-R_OBJ, thickness=300.0, material=AIR, semidiameter=OBJ_SEMI),
    ]
    tracer = SequentialTracer(OpticalSystem(rows, object_space=AIR), restart_offset=0.0)
    result = tracer.trace(np.array([0.0, 1.0, -10.0]), np.array([0.0, 0.0, 1.0]))
    point, direction = result.image_point, result.direction
    t_axis = -point[1] / direction[1]
    return float(point[2] + t_axis * direction[2])


def build_scene(gap: float, intermediate_image_x: float) -> Scene:
    objective = Lens.spherical(
        R1=R_OBJ, R2=-R_OBJ, thickness=OBJ_THICKNESS, semidiameter=OBJ_SEMI,
        n=GLASS_N, vertex=(0.0, 0.0), name="objective",
    )
    eyepiece = Lens.spherical(
        R1=R_EYE, R2=-R_EYE, thickness=EYE_THICKNESS, semidiameter=EYE_SEMI,
        n=GLASS_N, vertex=(OBJ_THICKNESS + gap, 0.0), name="eyepiece",
    )
    surfaces = [*objective.surfaces(), *eyepiece.surfaces()]
    tracer = BranchingTracer(surfaces, TraceConfig(max_generations=8, fresnel_split=False))

    on_axis = ParallelSource(
        origin=np.array([-25.0, 0.0]), direction=np.array([1.0, 0.0]),
        width=2.0 * BEAM_HALF_WIDTH, samples=17,
    )
    off_axis = ParallelSource(
        origin=np.array([-25.0, -25.0 * np.tan(FIELD_ANGLE)]),
        direction=np.array([np.cos(FIELD_ANGLE), np.sin(FIELD_ANGLE)]),
        width=2.0 * BEAM_HALF_WIDTH, samples=17,
    )

    exit_x = OBJ_THICKNESS + gap + EYE_THICKNESS
    scene = Scene(x_lims=(-30.0, exit_x + 60.0), y_lims=(-30.0, 30.0))
    scene.add_elements([objective, eyepiece])
    scene.add_tree(tracer.trace([on_axis]), default_color=ON_AXIS_COLOR)
    scene.add_tree(tracer.trace([off_axis]), default_color=OFF_AXIS_COLOR)
    # Real intermediate image: shared focal point of objective and eyepiece.
    scene.add(MarkerItem(points=np.array([[intermediate_image_x, 0.0]]),
                         color=(0.3, 1.0, 0.4, 1.0), size=8.0))
    return scene


def main() -> None:
    gap, system = solve_afocal_gap(build_sequential, low=200.0, high=300.0)
    magnification, residual = angular_magnification(system)
    intermediate_image_x = objective_focus_x()

    print("Keplerian telescope")
    print(f"  objective f={F_OBJECTIVE} mm, eyepiece f={F_EYEPIECE} mm")
    print(f"  objective-eyepiece air gap (solved, exact afocal): {gap:.4f} mm")
    print(f"  real intermediate image near z={intermediate_image_x:.2f} mm "
          "(shared focal point, marked in green)")
    print(f"  angular magnification: {magnification:.3f}x "
          f"({'upright' if magnification > 0 else 'inverted'})")
    print(f"  parallelism residual between two field rays: {residual:.2e} rad "
          "(should be ~0: confirms afocal)")

    scene = build_scene(gap, intermediate_image_x)
    render_config = RenderConfig(ray_width=0.25, min_pixels=1.0, use_solid_rays=True)

    save_path = None
    if "--save" in sys.argv:
        save_path = Path(sys.argv[sys.argv.index("--save") + 1])
    backend = show(
        scene, backend="gl", interactive=save_path is None,
        size=(1500, 620), bgcolor="black", render_config=render_config,
        title="Keplerian telescope — raytracer",
    )
    if save_path is not None:
        backend.save(save_path)
        print(f"wrote {save_path}")


if __name__ == "__main__":
    main()
