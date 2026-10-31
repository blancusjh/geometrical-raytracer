"""Keplerian telescope: two converging lenses sharing a real intermediate image.

Both the objective and the eyepiece are positive, separated by the sum of
their focal lengths: the objective's rear focal point coincides with the
eyepiece's front focal point, where a real, inverted intermediate image forms
(marked in the diagram). That real image is where a reticle or field stop
goes in a practical eyepiece — the advantage over the Galilean design
(``examples/telescopes/galilean.py``), traded for a longer tube and an
inverted view.

Each lens is a real element (:class:`~raytracer.elements.Lens`): two
spherical surfaces enclosing the glass, closed by an absorbing rim, drawn as
a filled body.

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

from raytracer import Gap, OpticalSystem, Rays
from raytracer.elements import Sensor
from raytracer.tracing import SequentialTracer
from raytracer.viz import MarkerItem
from examples.telescopes._afocal import solve_afocal_gap
from examples.telescopes.galilean import angular_magnification, beams, build_drawing, equiconvex, show_telescope

F_OBJECTIVE = 200.0
F_EYEPIECE = 50.0  # positive: converging eyepiece
OBJ_THICKNESS, OBJ_SEMI = 5.0, 25.0
EYE_THICKNESS, EYE_SEMI = 4.0, 12.0
BEAM_HALF_WIDTH = 14.0  # compressed |M|x by the telescope: must exit within EYE_SEMI
FIELD_ANGLE = 0.02  # rad, off-axis bundle

OBJECTIVE = equiconvex("objective", F_OBJECTIVE, OBJ_THICKNESS, OBJ_SEMI)
EYEPIECE = equiconvex("eyepiece", F_EYEPIECE, EYE_THICKNESS, EYE_SEMI)


def build_sequential(gap: float) -> OpticalSystem:
    """The two lenses *gap* apart, and a sensor behind them: the exact on-axis model."""

    return OpticalSystem.compose(OBJECTIVE, Gap(gap), EYEPIECE, Gap(40.0), Sensor("image"))


def objective_focus_z() -> float:
    """Exact rear focal point of the (finite-thickness) objective alone.

    Traces a marginal parallel ray through just the objective and intersects
    the exiting ray with the axis — no thin-lens estimate.
    """

    tracer = SequentialTracer(OpticalSystem.compose(OBJECTIVE, Gap(300.0), Sensor("image")), restart_offset=0.0)
    ray = tracer.trace(Rays([0.0, 1.0, -10.0], [0.0, 0.0, 1.0]))
    point, direction = ray.position[0], ray.direction[0]
    return float(point[2] - point[1] / direction[1] * direction[2])


def main() -> None:
    gap, system = solve_afocal_gap(build_sequential, low=200.0, high=300.0)
    magnification, residual = angular_magnification(system)
    intermediate_image_z = objective_focus_z()

    print("Keplerian telescope")
    print(f"  objective f={F_OBJECTIVE} mm, eyepiece f={F_EYEPIECE} mm")
    print(f"  objective-eyepiece air gap (solved, exact afocal): {gap:.4f} mm")
    print(f"  real intermediate image near z={intermediate_image_z:.2f} mm "
          "(shared focal point, marked in green)")
    print(f"  angular magnification: {magnification:.3f}x "
          f"({'upright' if magnification > 0 else 'inverted'})")
    print(f"  parallelism residual between two field rays: {residual:.2e} rad "
          "(should be ~0: confirms afocal)")

    exit_z = OBJ_THICKNESS + gap + EYE_THICKNESS
    lenses = OpticalSystem.compose(OBJECTIVE, Gap(gap), EYEPIECE)
    drawing = build_drawing(lenses, beams(BEAM_HALF_WIDTH, FIELD_ANGLE), x_lims=(-30.0, exit_z + 60.0))
    # The real intermediate image: the shared focal point of objective and eyepiece.
    drawing.add(MarkerItem(np.array([[intermediate_image_z, 0.0]]), color=(0.3, 1.0, 0.4, 1.0), size=8.0))
    show_telescope(drawing, "Keplerian telescope — raytracer")


if __name__ == "__main__":
    main()
