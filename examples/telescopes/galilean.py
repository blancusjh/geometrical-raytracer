"""Galilean telescope: converging objective + diverging eyepiece.

The classic "opera glass" layout — a positive objective and a *negative*
(diverging) eyepiece separated by ``f_obj + f_eye`` (with ``f_eye < 0``),
giving an upright image in a short tube. Compare with the Keplerian design
(``examples/telescopes/keplerian.py``), which forms a real intermediate image
at the cost of a longer tube and an inverted view.

Each lens is a real element (:class:`~raytracer.elements.Lens`): two
spherical surfaces enclosing the glass, closed by an absorbing rim at the
clear aperture — drawn as a filled body, exactly like the lithography
examples. The input beams are sized so the compressed output beam fits
through the eyepiece aperture.

Being a telescope means being afocal, and that is checked, not assumed:

* the objective-eyepiece air gap is *solved* (:func:`_afocal.solve_afocal_gap`)
  so the effective focal length is exactly infinite for the real
  finite-thickness lenses, not the thin-lens estimate;
* two parallel rays entering at the same field angle are traced through the
  exact sequential engine and confirmed to exit parallel, with the angular
  magnification measured as the output/input angle ratio (positive: upright).

The figure traces the same lenses non-sequentially with two meridional
beams, on axis and off axis.

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

from raytracer import Gap, OpticalSystem, Rays
from raytracer.elements import Lens, Sensor
from raytracer.geometry.surfaces import Sphere
from raytracer.materials import ConstantIndex
from raytracer.sources import CollimatedSource
from raytracer.tracing import NonSequentialTracer, SequentialTracer
from raytracer.viz import Drawing, show
from raytracer.viz.gl.viewer import RenderConfig
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


def equiconvex(name: str, f: float, thickness: float, semi_diameter: float) -> Lens:
    """An equiconvex/equiconcave lens of focal length *f* (thin-lens radii;
    the gap solve absorbs the thickness error)."""

    radius = 2.0 * (GLASS_N - 1.0) * f
    return Lens(name, surfaces=[Sphere(radius), Sphere(-radius)], thicknesses=[thickness],
                materials=[GLASS], semi_diameter=semi_diameter)


OBJECTIVE = equiconvex("objective", F_OBJECTIVE, OBJ_THICKNESS, OBJ_SEMI)
EYEPIECE = equiconvex("eyepiece", F_EYEPIECE, EYE_THICKNESS, EYE_SEMI)


def build_sequential(gap: float) -> OpticalSystem:
    """The two lenses *gap* apart, and a sensor behind them: the exact on-axis model."""

    return OpticalSystem.compose(OBJECTIVE, Gap(gap), EYEPIECE, Gap(30.0), Sensor("image"))


def angular_magnification(system: OpticalSystem, theta_in: float = 0.01) -> tuple[float, float]:
    """Trace two parallel rays at field angle *theta_in*; return (M, parallelism residual in rad)."""

    tracer = SequentialTracer(system, restart_offset=0.0)
    d = np.array([0.0, np.sin(theta_in), np.cos(theta_in)])
    origins = np.array([[0.0, -5.0, -20.0], [0.0, 5.0, -20.0]])
    trace = tracer.trace(Rays(origins, np.tile(d, (2, 1))))
    theta_out = np.arctan2(trace.direction[:, 1], trace.direction[:, 2])
    return float(np.mean(theta_out) / theta_in), float(np.ptp(theta_out))


def beams(half_width: float, field_angle: float) -> list[CollimatedSource]:
    """Meridional beams 25 mm before the objective: on axis, and at *field_angle* (rad) aimed at its vertex."""

    tilted = (0.0, np.sin(field_angle), np.cos(field_angle))
    return [
        CollimatedSource((0.0, 0.0, -25.0), beam_radius=half_width, rays=17, fan="y"),
        CollimatedSource((0.0, -25.0 * np.tan(field_angle), -25.0), direction=tilted, beam_radius=half_width,
                         rays=17, fan="y"),
    ]


def build_drawing(lenses: OpticalSystem, sources, *, x_lims, y_lims=(-30.0, 30.0)) -> Drawing:
    """The lenses, and each source traced through them in its own colour."""

    tracer = NonSequentialTracer(lenses, max_interactions=8, split=False)
    drawing = Drawing(x_lims=x_lims, y_lims=y_lims).add_system(lenses)
    for source, color in zip(sources, (ON_AXIS_COLOR, OFF_AXIS_COLOR)):
        drawing.add_rays(tracer.trace(source), default_color=color, intensities=1.0)
    return drawing


def show_telescope(drawing: Drawing, title: str) -> None:
    save = Path(sys.argv[sys.argv.index("--save") + 1]) if "--save" in sys.argv else None
    backend = show(
        drawing, backend="gl", interactive=save is None, size=(1500, 620), bgcolor="black", title=title,
        render_config=RenderConfig(ray_width=0.25, min_pixels=1.0, use_solid_rays=True),
    )
    if save is not None:
        backend.save(save)
        print(f"wrote {save}")


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

    exit_z = OBJ_THICKNESS + gap + EYE_THICKNESS
    lenses = OpticalSystem.compose(OBJECTIVE, Gap(gap), EYEPIECE)
    drawing = build_drawing(lenses, beams(BEAM_HALF_WIDTH, FIELD_ANGLE), x_lims=(-30.0, exit_z + 50.0))
    show_telescope(drawing, "Galilean telescope — raytracer")


if __name__ == "__main__":
    main()
